#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

/* Subvetores com até THRESHOLD elementos vão direto para selection_sort */
#define THRESHOLD 1024

/* Ordenação sequencial O(n²) usada nas folhas da recursão */
void selection_sort(int *v, int n) {
    for (int i = 0; i < n - 1; i++) {
        int min = i;
        for (int j = i + 1; j < n; j++) {
            if (v[j] < v[min])
                min = j;
        }
        if (min != i) {
            int aux = v[i];
            v[i]   = v[min];
            v[min] = aux;
        }
    }
}

/* Merge sequencial usando buffer temporário tmp */
void merge(int *v, int *tmp, int esq, int meio, int dir) {
    int i = esq, j = meio + 1, k = esq;

    while (i <= meio && j <= dir) {
        if (v[i] <= v[j])
            tmp[k++] = v[i++];
        else
            tmp[k++] = v[j++];
    }
    while (i <= meio)
        tmp[k++] = v[i++];
    while (j <= dir)
        tmp[k++] = v[j++];

    for (int idx = esq; idx <= dir; idx++)
        v[idx] = tmp[idx];
}

/*
 * Mergesort recursivo com tasks OpenMP.
 * Cada metade é lançada como task independente.
 * Subvetores <= THRESHOLD são ordenados com selection_sort sem criar task.
 */
void mergesort_omp(int *v, int *tmp, int esq, int dir) {
    if (esq >= dir)
        return;

    int tamanho = dir - esq + 1;

    if (tamanho <= THRESHOLD) {
        selection_sort(v + esq, tamanho);
        return;
    }

    int meio = esq + (dir - esq) / 2;

    #pragma omp task shared(v, tmp)
    mergesort_omp(v, tmp, esq, meio);

    #pragma omp task shared(v, tmp)
    mergesort_omp(v, tmp, meio + 1, dir);

    #pragma omp taskwait

    merge(v, tmp, esq, meio, dir);
}

/* Gera vetor de N inteiros aleatórios */
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

    int *v   = malloc(n * sizeof(int));
    int *tmp = malloc(n * sizeof(int));
    if (!v || !tmp) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        free(v);
        free(tmp);
        return 1;
    }

    gerar_vetor(v, n);

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    #pragma omp parallel
    {
        #pragma omp single
        mergesort_omp(v, tmp, 0, n - 1);
    }

    gettimeofday(&fim, NULL);

    double tempo = (fim.tv_sec  - inicio.tv_sec) +
                   (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("mergesort,openmp,%d,%.6f,%s\n", n, tempo,
           validar_ordenacao(v, n) ? "OK" : "ERRO");

    free(v);
    free(tmp);
    return 0;
}
