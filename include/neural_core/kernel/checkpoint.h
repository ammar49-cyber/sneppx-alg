#ifndef SNEPPX_CHECKPOINT_COORD_H
#define SNEPPX_CHECKPOINT_COORD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SNEPPX_HAS_CUDA
typedef void* cudaStream_t;
#endif

/* Opaque forward declaration for async state */
typedef struct SNEPPXCkptAsync SNEPPXCkptAsync;

typedef struct {
    char checkpoint_dir[512];
    int world_size;
    int rank;
    int save_interval_steps;
    int keep_last_n;
    int async_save;
    void* _async_state;  /* SNEPPXCkptAsync* */
    int current_step;
    char last_checkpoint_path[512];
    int save_in_progress;
} SNEPPX_CheckpointCoord;

/* Async checkpoint lifecycle */
int  SNEPPX_checkpoint_init(SNEPPX_CheckpointCoord** cp,
                            const char* dir, int world_size, int rank,
                            int save_interval, int keep_last,
                            int async_enabled);
void SNEPPX_checkpoint_destroy(SNEPPX_CheckpointCoord* cp);
int  SNEPPX_checkpoint_save(SNEPPX_CheckpointCoord* cp,
                            const void* model_state, size_t state_size,
                            int current_step, cudaStream_t stream);
int  SNEPPX_checkpoint_load(SNEPPX_CheckpointCoord* cp,
                            void* model_state, size_t state_size,
                            int* loaded_step);
int  SNEPPX_checkpoint_coordinated_save(SNEPPX_CheckpointCoord* cp,
                                        const void* model_state, size_t state_size,
                                        int current_step, cudaStream_t stream,
                                        int (*barrier_fn)(void));

/* Fault tolerance */
typedef struct SNEPPX_FaultTolerance SNEPPX_FaultTolerance;

int  SNEPPX_fault_tolerance_init(SNEPPX_FaultTolerance** ft,
                                 int world_size, int rank,
                                 int heartbeat_ms, int timeout_ms);
void SNEPPX_fault_tolerance_destroy(SNEPPX_FaultTolerance* ft);
int  SNEPPX_fault_tolerance_check_health(SNEPPX_FaultTolerance* ft);
int  SNEPPX_fault_tolerance_handle_failure(SNEPPX_FaultTolerance* ft,
                                           int failed_rank);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_CHECKPOINT_COORD_H */
