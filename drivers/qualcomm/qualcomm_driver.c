#include "qualcomm_driver.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>

int SNEPPX_qualcomm_register(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_qualcomm_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_qualcomm_get_device_props(int dev_id, char* name, size_t name_max, unsigned long long* total_mem) { (void)dev_id; if (name) snprintf(name, name_max, "QNN Device %d", dev_id); if (total_mem) *total_mem = 8ULL*1024*1024*1024; return 0; }
void* SNEPPX_qualcomm_create_context(const char* model_path) { (void)model_path; return calloc(1, 8); }
void SNEPPX_qualcomm_destroy_context(void* ctx) { free(ctx); }
int SNEPPX_qualcomm_set_input(void* ctx, const char* name, const float* data, size_t size) { (void)ctx; (void)name; (void)data; (void)size; return 0; }
int SNEPPX_qualcomm_run_inference(void* ctx) { (void)ctx; return 0; }
int SNEPPX_qualcomm_get_output(void* ctx, const char* name, float* data, size_t size) { (void)ctx; (void)name; (void)data; (void)size; return 0; }
