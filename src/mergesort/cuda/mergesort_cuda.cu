#include <stdio.h>
#include <stdlib.h>
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
        /* Número de threads necessárias = número de pares de subvetores */
        int num_threads = (n + largura * 2 - 1) / (largura * 2);
        int num_blocos  = (num_threads + THREADS_POR_BLOCO - 1) / THREADS_POR_BLOCO;
        kernel_mergesort<<<num_blocos, THREADS_POR_BLOCO>>>(d_src, d_dst, n, largura);
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

    printf("mergesort,cuda,%d,%.6f,%s\n", n, tempo,
           validar_ordenacao(h_v, n) ? "OK" : "ERRO");

    free(h_v);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
