#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define THREADS_POR_BLOCO 256

/* Ordenação por inserção in-place na GPU */
__device__ void insertion_sort_device(int *v, int esq, int dir) {
    for (int i = esq + 1; i <= dir; i++) {
        int chave = v[i];
        int j = i - 1;
        while (j >= esq && v[j] > chave) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = chave;
    }
}

/* Conta elementos em arr[lo..hi] estritamente menores que val */
__device__ int lower_bound(int *arr, int lo, int hi, int val) {
    int l = lo, r = hi + 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] < val) l = mid + 1;
        else r = mid;
    }
    return l - lo;
}

/* Conta elementos em arr[lo..hi] menores ou iguais a val */
__device__ int upper_bound(int *arr, int lo, int hi, int val) {
    int l = lo, r = hi + 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] <= val) l = mid + 1;
        else r = mid;
    }
    return l - lo;
}

/*
 * Kernel filho lançado via Dynamic Parallelism.
 * Cada thread encontra sua posição em dst via busca binária,
 * paralelizando o merge entre src[esq..meio] e src[meio+1..dir].
 * Esquerdo usa lower_bound; direito usa upper_bound — posições únicas
 * mesmo com elementos duplicados.
 */
__global__ void binary_search_merge(int *src, int *dst, int esq, int meio, int dir) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total     = dir - esq + 1;

    if (thread_id >= total) return;

    int esq_size = meio - esq + 1;
    int posicao;
    int elemento;

    if (thread_id < esq_size) {
        /* Elemento do subvetor esquerdo */
        elemento = src[esq + thread_id];
        int num_menores = lower_bound(src, meio + 1, dir, elemento);
        posicao = thread_id + num_menores;
    } else {
        /* Elemento do subvetor direito */
        int idx_dir = thread_id - esq_size;
        elemento = src[meio + 1 + idx_dir];
        int num_menores_iguais = upper_bound(src, esq, meio, elemento);
        posicao = idx_dir + num_menores_iguais;
    }

    dst[esq + posicao] = elemento;
}

/*
 * Kernel principal com Dynamic Parallelism.
 * Subproblemas com > 16384 e < 1048576 elementos lançam binary_search_merge
 * via DP com stream NonBlocking. Os demais usam insertion_sort diretamente.
 */
__global__ void kernel_mergesort_dp(int *src, int *dst, int n, int largura) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int esq = thread_id * largura * 2;

    if (esq >= n) return;

    int meio         = min(esq + largura - 1, n - 1);
    int dir          = min(esq + largura * 2 - 1, n - 1);
    int num_elementos = dir - esq + 1;

    if (num_elementos > 16384 && num_elementos < 1048576) {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        int blocos = (num_elementos + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
        binary_search_merge<<<blocos, THREADS_POR_BLOCO, 0, stream>>>(src, dst, esq, meio, dir);
        cudaStreamDestroy(stream);
    } else {
        insertion_sort_device(src, esq, dir);
        for (int i = esq; i <= dir; i++)
            dst[i] = src[i];
    }
}

/* Gera vetor de n inteiros aleatórios */
void gerar_vetor(int *v, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        v[i] = rand();
}

/* Retorna 1 se o vetor está ordenado em ordem não-decrescente, 0 caso contrário */
int validar_ordenacao(int *v, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (v[i] > v[i + 1])
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s N [--runs N]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "N deve ser um inteiro positivo\n");
        return 1;
    }

    int runs = 10000;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[i + 1]);
    }

    int *h_original = (int *)malloc(n * sizeof(int));
    int *h_v        = (int *)malloc(n * sizeof(int));
    if (!h_original || !h_v) {
        fprintf(stderr, "Erro ao alocar memoria no host\n");
        free(h_original); free(h_v);
        return 1;
    }

    /* d_a e d_b: buffers permanentes alternados no sort bottom-up */
    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMalloc((void **)&d_b, n * sizeof(int));

    gerar_vetor(h_original, n);

    /* warm-up: inclui transferência H→D, kernels DP e D→H para validar */
    {
        int *src = d_a, *dst = d_b;
        cudaMemcpy(src, h_original, n * sizeof(int), cudaMemcpyHostToDevice);
        for (int largura = 1; largura < n; largura *= 2) {
            int num_threads = (n + largura * 2 - 1) / (largura * 2);
            int num_blocos  = (num_threads + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
            kernel_mergesort_dp<<<num_blocos, THREADS_POR_BLOCO>>>(src, dst, n, largura);
            cudaDeviceSynchronize();
            int *tmp = src; src = dst; dst = tmp;
        }
        cudaMemcpy(h_v, src, n * sizeof(int), cudaMemcpyDeviceToHost);
    }
    const char *corretude = validar_ordenacao(h_v, n) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        int *src = d_a, *dst = d_b;
        cudaMemcpy(src, h_original, n * sizeof(int), cudaMemcpyHostToDevice);
        for (int largura = 1; largura < n; largura *= 2) {
            int num_threads = (n + largura * 2 - 1) / (largura * 2);
            int num_blocos  = (num_threads + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
            kernel_mergesort_dp<<<num_blocos, THREADS_POR_BLOCO>>>(src, dst, n, largura);
            cudaDeviceSynchronize();
            int *tmp = src; src = dst; dst = tmp;
        }
        /* corretude validada no warm-up — sem D→H a cada run */
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("mergesort,cuda_dp,%d,%d,%.6f,%s\n", n, runs, tempo_total, corretude);

    free(h_original); free(h_v);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
