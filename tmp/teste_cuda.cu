#include <stdio.h>

__global__ void kernel() {
    printf("  Thread %d de %d em execução\n",
           threadIdx.x, blockDim.x);
}

int main() {
    printf("Executando kernel CUDA: 1 bloco, 4 threads\n");
    kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
