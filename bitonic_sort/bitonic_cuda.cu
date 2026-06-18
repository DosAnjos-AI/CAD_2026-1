#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Um passo do bitonic sort: cada thread trata exatamente um par (i, parceiro).
   i e calculado a partir do indice global da thread de forma que o bit em
   posicao j esteja sempre zerado, garantindo parceiro = i ^ j = i + j. */
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
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Uso: %s <N> <cenario> [iteracoes] [warmup]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    const char *cenario = argv[2];
    int iteracoes = (argc >= 4) ? atoi(argv[3]) : 5;
    int warmup = (argc == 5) ? atoi(argv[4]) : 1;

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

    int blocos = n / 2 / BLOCK_SIZE;

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

    /* Execucoes de warmup (default 1), sem saida */
    for (int w = 0; w < warmup; w++) {
        gerar_vetor(arr, n, cenario);

        cudaMemcpy(d_arr, arr, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice);
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_step<<<blocos, BLOCK_SIZE>>>(d_arr, n, k, j);
            }
        }
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
            for (int k = 2; k <= n; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    bitonic_step<<<blocos, BLOCK_SIZE>>>(d_arr, n, k, j);
                }
            }
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

        printf("bitonic_sort|cuda|%s|%d|%d|%s|%d|256x%d\n",
               cenario, n, iter, tempo_str, corretude, blocos);
        fflush(stdout);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    free(arr);
    free(ref);
    return 0;
}
