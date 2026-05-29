#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define THREADS_POR_BLOCO 256

/* Ordenação por inserção in-place na GPU — usada nas folhas (largura == 1) */
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

/* Merge sequencial: mescla src[esq..meio] e src[meio+1..dir] em dst */
__device__ void gpu_bottomup_merge(int *src, int *dst, int esq, int meio, int dir) {
    int i = esq, j = meio + 1, k = esq;
    while (i <= meio && j <= dir) {
        if (src[i] <= src[j])
            dst[k++] = src[i++];
        else
            dst[k++] = src[j++];
    }
    while (i <= meio) dst[k++] = src[i++];
    while (j <= dir)  dst[k++] = src[j++];
}

/*
 * Kernel bottom-up: cada thread processa uma fatia de 2*largura elementos.
 * Na primeira iteração (largura == 1), ordena par com insertion_sort e copia para dst.
 * Nas demais, realiza merge de dois subvetores adjacentes de tamanho largura.
 */
__global__ void kernel_mergesort(int *src, int *dst, int n, int largura) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int esq = thread_id * largura * 2;

    if (esq >= n) return;

    int meio = min(esq + largura - 1, n - 1);
    int dir  = min(esq + largura * 2 - 1, n - 1);

    if (largura == 1) {
        /* Ordena par in-place em src, depois copia para dst */
        insertion_sort_device(src, esq, dir);
        for (int i = esq; i <= dir; i++)
            dst[i] = src[i];
    } else {
        gpu_bottomup_merge(src, dst, esq, meio, dir);
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

    /* warm-up: inclui transferência H→D, kernels e D→H para validar */
    {
        int *src = d_a, *dst = d_b;
        cudaMemcpy(src, h_original, n * sizeof(int), cudaMemcpyHostToDevice);
        for (int largura = 1; largura < n; largura *= 2) {
            int num_threads = (n + largura * 2 - 1) / (largura * 2);
            int num_blocos  = (num_threads + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
            kernel_mergesort<<<num_blocos, THREADS_POR_BLOCO>>>(src, dst, n, largura);
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
            kernel_mergesort<<<num_blocos, THREADS_POR_BLOCO>>>(src, dst, n, largura);
            cudaDeviceSynchronize();
            int *tmp = src; src = dst; dst = tmp;
        }
        /* corretude validada no warm-up — sem D→H a cada run */
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("mergesort,cuda,%d,%d,%.6f,%s\n", n, runs, tempo_total, corretude);

    free(h_original); free(h_v);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
