#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUTOFF 1024
#define BLOCK_SIZE 256

/* Merge sequencial no device: intercala arr[left..mid] e arr[mid+1..right]
   usando tmp como buffer auxiliar, depois copia o resultado de volta */
__device__ void merge_device(int32_t *arr, int32_t *tmp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            tmp[k++] = arr[i++];
        else
            tmp[k++] = arr[j++];
    }
    while (i <= mid)
        tmp[k++] = arr[i++];
    while (j <= right)
        tmp[k++] = arr[j++];

    for (int x = left; x <= right; x++)
        arr[x] = tmp[x];
}

/* Merge sort sequencial no device, usado abaixo do cutoff (thread unica).
   Bottom-up iterativo: evita recursao no device, o que impediria o linker
   de determinar estaticamente o tamanho da pilha do kernel */
__device__ void merge_sort_seq_device(int32_t *arr, int32_t *tmp, int left, int right) {
    int n = right - left + 1;

    for (int width = 1; width < n; width <<= 1) {
        for (int i = left; i <= right; i += 2 * width) {
            int mid = min(i + width - 1, right);
            int hi = min(i + 2 * width - 1, right);
            if (mid < hi)
                merge_device(arr, tmp, i, mid, hi);
        }
    }
}

/* Kernel auxiliar que executa o merge final apos os dois filhos concluirem.
   Lancado no stream cudaStreamTailLaunch (CDP2): so comeca a executar quando
   o grid atual e todos os seus descendentes (os dois filhos abaixo) tiverem
   terminado, dispensando cudaDeviceSynchronize() explicito no device */
__global__ void merge_tail_kernel(int32_t *arr, int32_t *tmp, int left, int mid, int right) {
    merge_device(arr, tmp, left, mid, right);
}

/* Kernel recursivo: divide o intervalo lancando dois kernels filhos e agenda
   o merge do resultado via tail launch. Abaixo do cutoff, ordena
   sequencialmente no device sem lancar filhos */
__global__ void merge_sort_dp(int32_t *arr, int32_t *tmp, int left, int right) {
    if (right - left <= CUTOFF) {
        merge_sort_seq_device(arr, tmp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        merge_sort_dp<<<1, 1>>>(arr, tmp, left, mid);
        merge_sort_dp<<<1, 1>>>(arr, tmp, mid + 1, right);
        merge_tail_kernel<<<1, 1, 0, cudaStreamTailLaunch>>>(arr, tmp, left, mid, right);
    }
}

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <N> [iteracoes]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int iteracoes = (argc == 3) ? atoi(argv[2]) : 10;

    if (n <= 0) {
        fprintf(stderr, "Erro: N deve ser positivo (recebido: %d)\n", n);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    /* Aumenta o limite de lancamentos pendentes do device runtime: a arvore
       de recursao do CDP pode gerar mais filhos simultaneos que o padrao
       (2048) para N grande */
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 1 << 20);

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

    /* 3 execucoes de warmup, sem saida */
    for (int w = 0; w < 3; w++) {
        srand(42);
        for (int i = 0; i < n; i++)
            arr[i] = rand();

        cudaMemcpy(d_arr, arr, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice);

        merge_sort_dp<<<1, 1>>>(d_arr, d_tmp, 0, n - 1);
        cudaDeviceSynchronize();
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 10; exec++) {
            srand(42);
            for (int i = 0; i < n; i++)
                arr[i] = rand();

            if (exec == 0) {
                memcpy(ref, arr, (size_t)n * sizeof(int32_t));
                qsort(ref, (size_t)n, sizeof(int32_t), cmp_int32);
            }

            cudaMemcpy(d_arr, arr, (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice);

            cudaEventRecord(start);

            merge_sort_dp<<<1, 1>>>(d_arr, d_tmp, 0, n - 1);
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

        double tempo_s = soma / 10.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("merge_sort|cudadp|aleatorio|%d|%d|%s|%d|1x1_dp\n",
               n, iter, tempo_str, corretude);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    cudaFree(d_tmp);
    free(arr);
    free(ref);
    return 0;
}
