#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define THRESHOLD 32
#define PILHA_MAX 4096  /* suficiente para N=100000 com dados rand() */

/* Ordenação por inserção in-place na GPU — usada nos subproblemas pequenos */
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

/* Particionamento de Lomuto: pivô = v[dir], retorna índice final do pivô */
__device__ int particionar_device(int *v, int esq, int dir) {
    int pivo = v[dir];
    int i    = esq - 1;
    for (int j = esq; j < dir; j++) {
        if (v[j] <= pivo) {
            i++;
            int aux = v[i]; v[i] = v[j]; v[j] = aux;
        }
    }
    int aux = v[i + 1]; v[i + 1] = v[dir]; v[dir] = aux;
    return i + 1;
}

/*
 * Kernel iterativo com pilha dupla alternada.
 * Cada thread processa um subproblema da pilha atual.
 * Subproblemas <= THRESHOLD são ordenados com insertion_sort.
 * Os demais são particionados e os subvetores resultantes empurrados
 * na pilha próxima via atomicAdd — posições de escrita únicas por thread.
 */
__global__ void kernel_quicksort(int *v,
                                  int *esq_atual, int *dir_atual, int tam_atual,
                                  int *esq_prox,  int *dir_prox,  int *tam_prox) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= tam_atual) return;

    int esq = esq_atual[thread_id];
    int dir = dir_atual[thread_id];

    if (esq >= dir) return;

    if ((dir - esq) <= THRESHOLD) {
        insertion_sort_device(v, esq, dir);
        return;
    }

    int p = particionar_device(v, esq, dir);

    if (p - 1 > esq) {
        int idx = atomicAdd(tam_prox, 1);
        esq_prox[idx] = esq;
        dir_prox[idx] = p - 1;
    }

    if (p + 1 < dir) {
        int idx = atomicAdd(tam_prox, 1);
        esq_prox[idx] = p + 1;
        dir_prox[idx] = dir;
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

    int *d_v;
    int *d_esq_a, *d_dir_a;
    int *d_esq_b, *d_dir_b;
    int *d_tam_prox;

    cudaMalloc((void **)&d_v,        n         * sizeof(int));
    cudaMalloc((void **)&d_esq_a,    PILHA_MAX * sizeof(int));
    cudaMalloc((void **)&d_dir_a,    PILHA_MAX * sizeof(int));
    cudaMalloc((void **)&d_esq_b,    PILHA_MAX * sizeof(int));
    cudaMalloc((void **)&d_dir_b,    PILHA_MAX * sizeof(int));
    cudaMalloc((void **)&d_tam_prox, sizeof(int));

    gerar_vetor(h_original, n);

    /* Executa um run completo do quicksort com reset da pilha */
    #define EXECUTAR_RUN() do { \
        int ini_esq = 0, ini_dir = n - 1; \
        cudaMemcpy(d_v,     h_original, n * sizeof(int), cudaMemcpyHostToDevice); \
        cudaMemcpy(d_esq_a, &ini_esq,   sizeof(int),     cudaMemcpyHostToDevice); \
        cudaMemcpy(d_dir_a, &ini_dir,   sizeof(int),     cudaMemcpyHostToDevice); \
        int *pe = d_esq_a, *pd = d_dir_a, *pe2 = d_esq_b, *pd2 = d_dir_b; \
        int tam = 1, tam_p = 0; \
        while (tam > 0) { \
            cudaMemset(d_tam_prox, 0, sizeof(int)); \
            int nb = (tam + 255) / 256; \
            kernel_quicksort<<<nb, 256>>>(d_v, pe, pd, tam, pe2, pd2, d_tam_prox); \
            cudaDeviceSynchronize(); \
            cudaMemcpy(&tam_p, d_tam_prox, sizeof(int), cudaMemcpyDeviceToHost); \
            tam = tam_p; \
            int *t; t = pe; pe = pe2; pe2 = t; t = pd; pd = pd2; pd2 = t; \
        } \
    } while (0)

    /* warm-up */
    EXECUTAR_RUN();
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);
    const char *corretude = validar_ordenacao(h_v, n) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        EXECUTAR_RUN();
    }

    gettimeofday(&fim, NULL);

    #undef EXECUTAR_RUN

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("quicksort,cuda,%d,%d,%.6f,%s\n", n, runs, tempo_total, corretude);

    free(h_original); free(h_v);
    cudaFree(d_v);
    cudaFree(d_esq_a);
    cudaFree(d_dir_a);
    cudaFree(d_esq_b);
    cudaFree(d_dir_b);
    cudaFree(d_tam_prox);
    return 0;
}
