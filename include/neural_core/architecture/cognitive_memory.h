#ifndef SNEPPX_COGNITIVE_MEMORY_H
#define SNEPPX_COGNITIVE_MEMORY_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
#include <stdint.h>

#define SNEPPX_WORKING_MEMORY_MIN_SLOTS 5
#define SNEPPX_WORKING_MEMORY_MAX_SLOTS 9
#define SNEPPX_WORKING_MEMORY_DEFAULT_SLOTS 7
#define SNEPPX_MAX_CONCEPT_KEY_LEN 128

typedef struct {
    size_t capacity;
    size_t max_priority_samples;
    uint64_t consolidation_interval;
} SNEPPXEpisodicConfig;

typedef struct {
    size_t max_concepts;
    size_t max_edges;
    size_t embedding_dim;
    size_t lru_size;
} SNEPPXSemanticConfig;

typedef struct {
    size_t num_slots;
    size_t slot_dim;
} SNEPPXWorkingConfig;

typedef struct {
    size_t cache_size;
    size_t state_dim;
    size_t skill_dim;
    size_t compilation_threshold;
} SNEPPXProceduralConfig;

typedef struct {
    SNEPPXEpisodicConfig episodic;
    SNEPPXSemanticConfig semantic;
    SNEPPXWorkingConfig working;
    SNEPPXProceduralConfig procedural;
} SNEPPXCognitiveMemoryConfig;

typedef struct SNEPPXCognitiveMemory SNEPPXCognitiveMemory;

SNEPPXCognitiveMemoryConfig SNEPPX_cognitive_memory_config_default(void);

SNEPPXCognitiveMemory* SNEPPX_cognitive_memory_create(
    const SNEPPXCognitiveMemoryConfig* config, unsigned int seed);
void SNEPPX_cognitive_memory_destroy(SNEPPXCognitiveMemory* cmem);

int SNEPPX_cognitive_memory_forward(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* input,
    SNEPPXTensor** output);

int SNEPPX_episodic_record(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* action,
    float reward,
    const SNEPPXTensor* next_state);

int SNEPPX_episodic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    uint64_t time_start,
    uint64_t time_end,
    SNEPPXTensor** output);

int SNEPPX_episodic_sample(
    SNEPPXCognitiveMemory* cmem,
    size_t batch_size,
    SNEPPXTensor** states,
    SNEPPXTensor** actions,
    float* rewards,
    SNEPPXTensor** next_states);

int SNEPPX_episodic_consolidate(SNEPPXCognitiveMemory* cmem);

int SNEPPX_semantic_store(
    SNEPPXCognitiveMemory* cmem,
    const char* key,
    const SNEPPXTensor* embedding);

int SNEPPX_semantic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    size_t top_k,
    SNEPPXTensor** output,
    char*** out_keys);

int SNEPPX_semantic_relate(
    SNEPPXCognitiveMemory* cmem,
    const char* from_key,
    const char* to_key,
    float weight);

int SNEPPX_semantic_forget(
    SNEPPXCognitiveMemory* cmem,
    const char* key);

int SNEPPX_working_write(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    const SNEPPXTensor* content);

int SNEPPX_working_read(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    SNEPPXTensor** output);

int SNEPPX_working_attend(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    SNEPPXTensor** output);

int SNEPPX_working_clear(SNEPPXCognitiveMemory* cmem);

int SNEPPX_procedural_learn(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* skill);

int SNEPPX_procedural_recall(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    SNEPPXTensor** output);

int SNEPPX_procedural_compile(SNEPPXCognitiveMemory* cmem);

size_t SNEPPX_cognitive_memory_get_params(
    const SNEPPXCognitiveMemory* cmem,
    SNEPPXTensor** out,
    size_t max);

int SNEPPX_cognitive_memory_build_train_graph(
    SNEPPXCognitiveMemory* cmem,
    SNEPPXTape* tape,
    SNEPPXVariable* input_var,
    SNEPPXVariable** weight_vars,
    size_t num_weights,
    SNEPPXVariable** output_var);

#endif
