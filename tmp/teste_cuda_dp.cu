#include <stdio.h>
#include <cuda_device_runtime_api.h>

/* Compilar com: -DCUDA_FORCE_CDP1_IF_SUPPORTED -rdc=true -lcudadevrt */

__global__ void filho() {
    printf("  [filho] Thread %d executando via CUDA DP\n", threadIdx.x);
}

__global__ void pai() {
    printf("[pai] Lançando kernel filho via CUDA DP\n");
    filho<<<1, 2>>>();
    cudaDeviceSynchronize();
    printf("[pai] Kernel filho concluído\n");
}

int main() {
    printf("Teste CUDA Dynamic Parallelism (sm_61)\n");
    pai<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
