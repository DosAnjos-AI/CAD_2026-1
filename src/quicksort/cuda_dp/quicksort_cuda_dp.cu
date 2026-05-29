#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define PROFUNDIDADE_MAX 24
#define THRESHOLD        32

/* Selection sort in-place — usado quando a profundidade ou tamanho atingem o limite */
__device__ void selection_sort_device(int *v, int esq, int dir) {
    for (int i = esq; i < dir; i++) {
        int min_idx = i;
        for (int j = i + 1; j <= dir; j++) {
            if (v[j] < v[min_idx])
                min_idx = j;
        }
        if (min_idx != i) {
            int aux = v[i]; v[i] = v[min_idx]; v[min_idx] = aux;
        }
    }
}

/*
 * Kernel recursivo via Dynamic Parallelism.
 * Usa particionamento de Hoare com pivô no centro.
 * Para ao atingir a profundidade máxima ou subproblema <= THRESHOLD,
 * ordenando com selection_sort. Filhos lançados em streams NonBlocking.
 */
__global__ void cdp_quicksort(int *v, int esq, int dir, int profundidade) {
    if (profundidade >= PROFUNDIDADE_MAX || (dir - esq) <= THRESHOLD) {
        selection_sort_device(v, esq, dir);
        return;
    }

    /* Particionamento de Hoare: pivô no meio do subvetor */
    int  pivo = v[(esq + dir) / 2];
    int *lptr = v + esq;
    int *rptr = v + dir;

    while (lptr <= rptr) {
        while (*lptr < pivo) lptr++;
        while (*rptr > pivo) rptr--;
        if (lptr <= rptr) {
            int aux = *lptr; *lptr = *rptr; *rptr = aux;
            lptr++;
            rptr--;
        }
    }

    int nright = rptr - v;  /* último índice do subvetor esquerdo */
    int nleft  = lptr - v;  /* primeiro índice do subvetor direito */

    /* Lança kernel filho esquerdo se o subvetor não for vazio */
    if (esq < nright) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_quicksort<<<1, 1, 0, s>>>(v, esq, nright, profundidade + 1);
        cudaStreamDestroy(s);
    }

    /* Lança kernel filho direito se o subvetor não for vazio */
    if (nleft < dir) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_quicksort<<<1, 1, 0, s1>>>(v, nleft, dir, profundidade + 1);
        cudaStreamDestroy(s1);
    }
}

/* Gera vetor de n inteiros aleatórios diretamente no buffer unificado */
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

    /* Memória unificada: acessível pela CPU e GPU sem cudaMemcpy */
    int *v;
    cudaMallocManaged((void **)&v, n * sizeof(int));

    int *h_original = (int *)malloc(n * sizeof(int));
    if (!h_original) {
        fprintf(stderr, "Erro ao alocar memoria no host\n");
        cudaFree(v);
        return 1;
    }

    gerar_vetor(h_original, n);

    /* Configura a profundidade máxima de sincronização do device runtime */
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, PROFUNDIDADE_MAX);

    /* warm-up: memcpy para memória unificada, executa e valida */
    memcpy(v, h_original, n * sizeof(int));
    cdp_quicksort<<<1, 1>>>(v, 0, n - 1, 0);
    cudaDeviceSynchronize();
    const char *corretude = validar_ordenacao(v, n) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        memcpy(v, h_original, n * sizeof(int));
        cdp_quicksort<<<1, 1>>>(v, 0, n - 1, 0);
        cudaDeviceSynchronize();
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("quicksort,cuda_dp,%d,%d,%.6f,%s\n", n, runs, tempo_total, corretude);

    free(h_original);
    cudaFree(v);
    return 0;
}
