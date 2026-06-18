#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Co-rank: dado o rank k (posicao 0-indexada) na saida do merge de A[0..m)
   com B[0..n), encontra i tal que os primeiros i elementos de A e os
   primeiros (k-i) elementos de B sao exatamente os k menores elementos
   combinados. Busca binaria classica de merge path. */
__device__ int co_rank(int k, const int32_t *A, int m, const int32_t *B, int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;

    while (1) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            break;
        }
    }

    return i;
}

/* Cada thread calcula, via merge path, o elemento que ocupa a posicao idx
   na saida do merge dos subarrays src[merge_start..mid) e src[mid..right). */
__global__ void merge_kernel(const int32_t *src, int32_t *dst, int n, int sub_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int merge_size = sub_size * 2;
    int merge_start = (idx / merge_size) * merge_size;
    int left  = merge_start;
    int mid   = min(merge_start + sub_size, n);
    int right = min(merge_start + merge_size, n);

    int sizeL = mid - left;
    int sizeR = right - mid;
    int k = idx - left;

    const int32_t *A = src + left;
    const int32_t *B = src + mid;

    int i = co_rank(k, A, sizeL, B, sizeR);
    int j = k - i;

    if (i < sizeL && (j >= sizeR || A[i] <= B[j]))
        dst[idx] = A[i];
    else
        dst[idx] = B[j];
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

    if (n <= 0) {
        fprintf(stderr, "Erro: N deve ser positivo (recebido: %d)\n", n);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int blocos = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int32_t *arr = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!arr || !ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(arr);
        free(ref);
        return 1;
    }

    int32_t *d_arr, *d_tmp;
    if (cudaMalloc(&d_arr, (size_t)n * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_tmp, (size_t)n * sizeof(int32_t)) != cudaSuccess) {
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

        int32_t *d_src = d_arr, *d_dst = d_tmp;
        int flipflop = 0;
        for (int sub_size = 1; sub_size < n; sub_size <<= 1) {
            merge_kernel<<<blocos, BLOCK_SIZE>>>(d_src, d_dst, n, sub_size);
            int32_t *t = d_src; d_src = d_dst; d_dst = t;
            flipflop = !flipflop;
        }
        if (flipflop)
            cudaMemcpy(d_arr, d_tmp, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);

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

            int32_t *d_src = d_arr, *d_dst = d_tmp;
            int flipflop = 0;
            for (int sub_size = 1; sub_size < n; sub_size <<= 1) {
                merge_kernel<<<blocos, BLOCK_SIZE>>>(d_src, d_dst, n, sub_size);
                int32_t *t = d_src; d_src = d_dst; d_dst = t;
                flipflop = !flipflop;
            }
            if (flipflop)
                cudaMemcpy(d_arr, d_tmp, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);

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

        printf("merge_sort|cuda|%s|%d|%d|%s|%d|256x%d\n",
               cenario, n, iter, tempo_str, corretude, blocos);
        fflush(stdout);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    cudaFree(d_tmp);
    free(arr);
    free(ref);
    return 0;
}
