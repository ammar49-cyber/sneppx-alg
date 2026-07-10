#include "../../include/neural_core/architecture/distributed.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ============================================================================
// Distributed Checkpoint Coordinator
// ============================================================================

typedef struct {
    char checkpoint_dir[512];
    int world_size;
    int rank;
    int save_interval_steps;
    int keep_last_n;
    int async_save;
    
    // Async I/O state
    void* save_buffer;
    size_t save_size;
    cudaStream_t save_stream;
    int save_in_progress;
    
    // Metadata
    int current_step;
    char last_checkpoint_path[512];
} SNEPPX_CheckpointCoord;

int sneppx_checkpoint_init(SNEPPX_CheckpointCoord** cp,
                            const char* dir, int world_size, int rank,
                            int save_interval, int keep_last,
                            int async_enabled) {
    if (!cp || !dir) return -1;
    SNEPPX_CheckpointCoord* c = (SNEPPX_CheckpointCoord*)calloc(1, sizeof(SNEPPX_CheckpointCoord));
    if (!c) return -1;
    strncpy(c->checkpoint_dir, dir, 511);
    c->world_size = world_size;
    c->rank = rank;
    c->save_interval_steps = save_interval;
    c->keep_last_n = keep_last;
    c->async_save = async_enabled;
    c->current_step = 0;
    c->save_in_progress = 0;
    if (async_enabled) cudaStreamCreate(&c->save_stream);
    *cp = c;
    return 0;
}

void sneppx_checkpoint_destroy(SNEPPX_CheckpointCoord* cp) {
    if (!cp) return;
    if (cp->async_save) {
        cudaStreamSynchronize(cp->save_stream);
        cudaStreamDestroy(cp->save_stream);
    }
    if (cp->save_buffer) cudaFree(cp->save_buffer);
    free(cp);
}

static void sneppx_build_checkpoint_path(SNEPPX_CheckpointCoord* cp,
                                          int step, char* path, size_t path_size) {
    snprintf(path, path_size, "%s/checkpoint_%d_rank_%d.sneppx",
             cp->checkpoint_dir, step, cp->rank);
}

int sneppx_checkpoint_save(SNEPPX_CheckpointCoord* cp,
                            const void* model_state, size_t state_size,
                            int current_step, cudaStream_t stream) {
    if (!cp || !model_state) return -1;
    cp->current_step = current_step;
    if (current_step % cp->save_interval_steps != 0) return 0;
    if (cp->save_in_progress) {
        cudaStreamSynchronize(cp->save_stream);
    }
    cudaStream_t save_stream = cp->async_save ? cp->save_stream : stream;
    if (!cp->save_buffer || cp->save_size < state_size) {
        if (cp->save_buffer) cudaFree(cp->save_buffer);
        cudaMalloc(&cp->save_buffer, state_size);
        cp->save_size = state_size;
    }
    cudaMemcpyAsync(cp->save_buffer, model_state, state_size,
                    cudaMemcpyDeviceToDevice, save_stream);
    cudaStreamSynchronize(save_stream);
    float* host_buffer = (float*)malloc(state_size);
    cudaMemcpy(host_buffer, cp->save_buffer, state_size, cudaMemcpyDeviceToHost);
    char path[512];
    sneppx_build_checkpoint_path(cp, current_step, path, sizeof(path));
    FILE* f = fopen(path, "wb");
    if (!f) { free(host_buffer); return -1; }
    fwrite(&current_step, sizeof(int), 1, f);
    fwrite(&state_size, sizeof(size_t), 1, f);
    fwrite(host_buffer, 1, state_size, f);
    fclose(f);
    free(host_buffer);
    strncpy(cp->last_checkpoint_path, path, 511);
    if (cp->rank == 0) printf("[SNEPPX Checkpoint] Saved to %s (step %d)\n", path, current_step);
    return 0;
}

int sneppx_checkpoint_load(SNEPPX_CheckpointCoord* cp,
                            void* model_state, size_t state_size,
                            int* loaded_step) {
    if (!cp || !model_state) return -1;
    char path[512];
    if (cp->last_checkpoint_path[0]) {
        strncpy(path, cp->last_checkpoint_path, 511);
    } else {
        sneppx_build_checkpoint_path(cp, cp->current_step, path, sizeof(path));
    }
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int step;
    size_t saved_size;
    fread(&step, sizeof(int), 1, f);
    fread(&saved_size, sizeof(size_t), 1, f);
    if (saved_size != state_size) { fclose(f); return -1; }
    float* host_buffer = (float*)malloc(state_size);
    fread(host_buffer, 1, state_size, f);
    fclose(f);
    cudaMemcpy(model_state, host_buffer, state_size, cudaMemcpyHostToDevice);
    free(host_buffer);
    if (loaded_step) *loaded_step = step;
    cp->current_step = step;
    if (cp->rank == 0) printf("[SNEPPX Checkpoint] Loaded from %s (step %d)\n", path, step);
    return 0;
}

// Barrier-based coordinated save across all ranks
int sneppx_checkpoint_coordinated_save(SNEPPX_CheckpointCoord* cp,
                                         const void* model_state, size_t state_size,
                                         int current_step, cudaStream_t stream,
                                         int (*barrier_fn)(void)) {
    if (!cp || !model_state) return -1;
    if (barrier_fn) barrier_fn();
    int ret = sneppx_checkpoint_save(cp, model_state, state_size, current_step, stream);
    if (barrier_fn) barrier_fn();
    return ret;
}

// ============================================================================
// Fault Tolerance / Elastic Training
// ============================================================================

typedef struct {
    int world_size;
    int rank;
    int num_nodes;
    int heartbeat_interval_ms;
    int timeout_ms;
    int64_t last_heartbeat;
    int* alive_ranks;
    int num_alive;
    int enable_elastic;
    int max_restarts;
    int restart_count;
} SNEPPX_FaultTolerance;

int sneppx_fault_tolerance_init(SNEPPX_FaultTolerance** ft,
                                  int world_size, int rank,
                                  int heartbeat_ms, int timeout_ms) {
    if (!ft) return -1;
    SNEPPX_FaultTolerance* f = (SNEPPX_FaultTolerance*)calloc(1, sizeof(SNEPPX_FaultTolerance));
    if (!f) return -1;
    f->world_size = world_size;
    f->rank = rank;
    f->heartbeat_interval_ms = heartbeat_ms;
    f->timeout_ms = timeout_ms;
    f->alive_ranks = (int*)calloc(world_size, sizeof(int));
    for (int i = 0; i < world_size; i++) f->alive_ranks[i] = 1;
    f->num_alive = world_size;
    f->enable_elastic = 1;
    f->max_restarts = 3;
    f->restart_count = 0;
    *ft = f;
    return 0;
}

void sneppx_fault_tolerance_destroy(SNEPPX_FaultTolerance* ft) {
    if (!ft) return;
    free(ft->alive_ranks);
    free(ft);
}

int sneppx_fault_tolerance_check_health(SNEPPX_FaultTolerance* ft) {
    if (!ft) return 0;
    // In a real implementation, this checks heartbeats from all ranks
    // For now, simplified - assumes all alive
    ft->num_alive = ft->world_size;
    return ft->num_alive;
}

int sneppx_fault_tolerance_handle_failure(SNEPPX_FaultTolerance* ft,
                                           int failed_rank) {
    if (!ft || failed_rank < 0 || failed_rank >= ft->world_size) return -1;
    ft->alive_ranks[failed_rank] = 0;
    ft->num_alive--;
    printf("[SNEPPX FT] Rank %d detected failure of rank %d. %d/%d alive.\n",
           ft->rank, failed_rank, ft->num_alive, ft->world_size);
    if (ft->num_alive < ft->world_size / 2) return -1;
    return 0;
}

// ============================================================================
// Communication Profiling Utilities
// ============================================================================

typedef struct {
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float total_time_ms;
    float total_bytes;
    int num_calls;
    char operation_name[64];
} SNEPPX_CommProfiler;

int sneppx_comm_profiler_init(SNEPPX_CommProfiler** prof,
                               const char* name) {
    if (!prof) return -1;
    SNEPPX_CommProfiler* p = (SNEPPX_CommProfiler*)calloc(1, sizeof(SNEPPX_CommProfiler));
    if (!p) return -1;
    cudaEventCreate(&p->start_event);
    cudaEventCreate(&p->end_event);
    strncpy(p->operation_name, name ? name : "unknown", 63);
    p->total_time_ms = 0.0f;
    p->total_bytes = 0.0f;
    p->num_calls = 0;
    *prof = p;
    return 0;
}

void sneppx_comm_profiler_destroy(SNEPPX_CommProfiler* prof) {
    if (!prof) return;
    cudaEventDestroy(prof->start_event);
    cudaEventDestroy(prof->end_event);
    free(prof);
}

void sneppx_comm_profiler_start(SNEPPX_CommProfiler* prof,
                                 cudaStream_t stream) {
    if (prof) cudaEventRecord(prof->start_event, stream);
}

void sneppx_comm_profiler_stop(SNEPPX_CommProfiler* prof,
                                cudaStream_t stream, float bytes) {
    if (!prof) return;
    cudaEventRecord(prof->end_event, stream);
    cudaEventSynchronize(prof->end_event);
    float ms = 0;
    cudaEventElapsedTime(&ms, prof->start_event, prof->end_event);
    prof->total_time_ms += ms;
    prof->total_bytes += bytes;
    prof->num_calls++;
}

float sneppx_comm_profiler_avg_time_ms(SNEPPX_CommProfiler* prof) {
    if (!prof || prof->num_calls == 0) return 0.0f;
    return prof->total_time_ms / prof->num_calls;
}

float sneppx_comm_profiler_bandwidth_gbps(SNEPPX_CommProfiler* prof) {
    if (!prof || prof->total_time_ms == 0) return 0.0f;
    return (prof->total_bytes * 8.0f) / (prof->total_time_ms * 1e6f);
}