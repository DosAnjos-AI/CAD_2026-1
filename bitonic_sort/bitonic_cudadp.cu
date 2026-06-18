#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Kernel filho: um passo do bitonic sort, identico ao bitonic_cuda.cu.
   Cada thread trata exatamente um par (i, parceiro), com parceiro = i ^ j. */
__global__ void bitonic_step(int32_t *arr, int n, int k, int j) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (tid / j) * (j * 2) + (tid % j);
    int parceiro = i ^ j;

    if ((i & k) == 0) {
        if (arr[i] > arr[parceiro]) {
            int32_t tmp = arr[i];
            arr[i] = arr[parceiro];
            arr[parceiro] = tmp;
        }
    } else {
        if (arr[i] < arr[parceiro]) {
            int32_t tmp = arr[i];
            arr[i] = arr[parceiro];
            arr[parceiro] = tmp;
        }
    }
}

/* Kernel pai: controla o loop de passos (k, j) inteiramente no device,
   lancando um kernel filho por passo e eliminando o round-trip CPU->GPU.
   Nota tecnica: no modelo CDP2 (padrao a partir do CUDA 12), lancamentos
   sucessivos feitos pela mesma thread sem stream explicito caem na stream
   default por-thread, que serializa a execucao na ordem de lancamento.
   Isso garante que cada passo termine antes do proximo iniciar, sem
   necessidade de cudaDeviceSynchronize() explicito dentro do kernel pai
   (alem de cudaDeviceSynchronize() no device estar depreciado no CDP2 e
   gerar warning de compilacao). */
__global__ void bitonic_sort_dp(int32_t *arr, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int blocos = n / 2 / BLOCK_SIZE;
            bitonic_step<<<blocos, BLOCK_SIZE>>>(arr, n, k, j);
        }
    }
}

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/* Preenche arr conforme o cenario: crescente (ordenado) ou decrescente (invertido) */
static void gerar_vetor(int32_t *arr, int n, const char *cenario) {
    if (strcmp(cenario, "ordenado") == 0) {
        for (int i = 0; i < n; i++)
            arr[i] = i;
    } else {
        for (int i = 0; i < n; i++)
            arr[i] = n - 1 - i;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <N> <cenario> [iteracoes]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    const char *cenario = argv[2];
    int iteracoes = (argc == 4) ? atoi(argv[3]) : 5;

    if (strcmp(cenario, "ordenado") != 0 && strcmp(cenario, "invertido") != 0) {
        fprintf(stderr, "Erro: cenario deve ser 'ordenado' ou 'invertido' (recebido: %s)\n", cenario);
        return 1;
    }

    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "Erro: N deve ser potencia de 2 (recebido: %d)\n", n);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int32_t *arr = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!arr || !ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(arr);
        free(ref);
        return 1;
    }

    int32_t *d_arr;
    if (cudaMalloc(&d_arr, (size_t)n * sizeof(int32_t)) != cudaSuccess) {
        fprintf(stderr, "Erro: falha ao alocar memoria na GPU\n");
        free(arr);
        free(ref);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* 1 execucao de warmup, sem saida */
    for (int w = 0; w < 1; w++) {
        gerar_vetor(arr, n, cenario);

        cudaMemcpy(d_arr, arr, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice);
        bitonic_sort_dp<<<1, 1>>>(d_arr, n);
        cudaDeviceSynchronize();
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 4; exec++) {
            gerar_vetor(arr, n, cenario);

            if (exec == 0) {
                memcpy(ref, arr, (size_t)n * sizeof(int32_t));
                qsort(ref, (size_t)n, sizeof(int32_t), cmp_int32);
            }

            cudaMemcpy(d_arr, arr, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice);

            cudaEventRecord(start);
            bitonic_sort_dp<<<1, 1>>>(d_arr, n);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            soma += (double)ms / 1000.0;

            cudaMemcpy(arr, d_arr, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToHost);

            if (exec == 0) {
                for (int i = 0; i < n; i++) {
                    if (arr[i] != ref[i]) {
                        corretude = 0;
                        break;
                    }
                }
            }
        }

        double tempo_s = soma / 4.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("bitonic_sort|cudadp|%s|%d|%d|%s|%d|1x1_dp\n",
               cenario, n, iter, tempo_str, corretude);
        fflush(stdout);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(arr);
    free(ref);
    return 0;
}
