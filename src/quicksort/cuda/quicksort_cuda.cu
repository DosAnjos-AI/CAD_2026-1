#include <stdio.h>
#include <stdlib.h>
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

    gerar_vetor(h_v, n);

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    cudaMemcpy(d_v, h_v, n * sizeof(int), cudaMemcpyHostToDevice);

    /* Inicializa pilha A com o subproblema completo [0, n-1] */
    int ini_esq = 0, ini_dir = n - 1;
    cudaMemcpy(d_esq_a, &ini_esq, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dir_a, &ini_dir, sizeof(int), cudaMemcpyHostToDevice);

    int *pilha_esq_atual = d_esq_a, *pilha_dir_atual = d_dir_a;
    int *pilha_esq_prox  = d_esq_b, *pilha_dir_prox  = d_dir_b;
    int  tam_atual = 1;
    int  tam_prox  = 0;

    while (tam_atual > 0) {
        cudaMemset(d_tam_prox, 0, sizeof(int));
        int num_blocos = (tam_atual + 255) / 256;
        kernel_quicksort<<<num_blocos, 256>>>(
            d_v,
            pilha_esq_atual, pilha_dir_atual, tam_atual,
            pilha_esq_prox,  pilha_dir_prox,  d_tam_prox
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&tam_prox, d_tam_prox, sizeof(int), cudaMemcpyDeviceToHost);
        tam_atual = tam_prox;
        /* Alterna as pilhas: próxima torna-se atual na próxima iteração */
        int *tmp;
        tmp = pilha_esq_atual; pilha_esq_atual = pilha_esq_prox; pilha_esq_prox = tmp;
        tmp = pilha_dir_atual; pilha_dir_atual = pilha_dir_prox; pilha_dir_prox = tmp;
    }

    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&fim, NULL);

    double tempo = (fim.tv_sec  - inicio.tv_sec) +
                   (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("quicksort,cuda,%d,%.6f,%s\n", n, tempo,
           validar_ordenacao(h_v, n) ? "OK" : "ERRO");

    free(h_v);
    cudaFree(d_v);
    cudaFree(d_esq_a);
    cudaFree(d_dir_a);
    cudaFree(d_esq_b);
    cudaFree(d_dir_b);
    cudaFree(d_tam_prox);
    return 0;
}
