#include <stdio.h>
#include <stdlib.h>
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
        fprintf(stderr, "Uso: %s N\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "N deve ser um inteiro positivo\n");
        return 1;
    }

    int *h_v = (int *)malloc(n * sizeof(int));
    if (!h_v) {
        fprintf(stderr, "Erro ao alocar memoria no host\n");
        return 1;
    }

    int *d_src, *d_dst;
    cudaMalloc((void **)&d_src, n * sizeof(int));
    cudaMalloc((void **)&d_dst, n * sizeof(int));

    gerar_vetor(h_v, n);

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    cudaMemcpy(d_src, h_v, n * sizeof(int), cudaMemcpyHostToDevice);

    /* Laço bottom-up: dobra a largura a cada iteração */
    for (int largura = 1; largura < n; largura *= 2) {
        int num_threads = (n + largura * 2 - 1) / (largura * 2);
        int num_blocos  = (num_threads + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
        kernel_mergesort_dp<<<num_blocos, THREADS_POR_BLOCO>>>(d_src, d_dst, n, largura);
        cudaDeviceSynchronize();
        /* Resultado em d_dst — swap coloca em d_src para a próxima iteração */
        int *tmp = d_src;
        d_src    = d_dst;
        d_dst    = tmp;
    }

    cudaMemcpy(h_v, d_src, n * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&fim, NULL);

    double tempo = (fim.tv_sec  - inicio.tv_sec) +
                   (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("mergesort,cuda_dp,%d,%.6f,%s\n", n, tempo,
           validar_ordenacao(h_v, n) ? "OK" : "ERRO");

    free(h_v);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
