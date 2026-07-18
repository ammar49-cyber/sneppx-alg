#include "sgx_enclave.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>

int SNEPPX_sgx_init(const char* enclave_path) { (void)enclave_path; return SNEPPX_DRIVER_UNSUPPORTED; }
void SNEPPX_sgx_destroy(void) {}
int SNEPPX_sgx_create_enclave(const char* name, size_t heap_size, size_t stack_size) { (void)name; (void)heap_size; (void)stack_size; return 0; }
int SNEPPX_sgx_destroy_enclave(void) { return 0; }
int SNEPPX_sgx_call(const char* func_name, void* input, size_t input_len, void* output, size_t output_len) { (void)func_name; (void)input; (void)input_len; (void)output; (void)output_len; return 0; }
int SNEPPX_sgx_seal_data(const unsigned char* data, size_t data_len, unsigned char* sealed, size_t* sealed_len) { (void)data; (void)data_len; (void)sealed; (void)sealed_len; return 0; }
int SNEPPX_sgx_unseal_data(const unsigned char* sealed, size_t sealed_len, unsigned char* data, size_t* data_len) { (void)sealed; (void)sealed_len; (void)data; (void)data_len; return 0; }
