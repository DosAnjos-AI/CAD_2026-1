#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define THRESHOLD 1024

/* Particionamento de Lomuto: pivô = v[dir], retorna índice final do pivô */
static int particionar(int *v, int esq, int dir) {
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

/* Quicksort sequencial puro — usado dentro do threshold, sem overhead de task */
static void quicksort_seq(int *v, int esq, int dir) {
    if (esq >= dir) return;
    int p = particionar(v, esq, dir);
    quicksort_seq(v, esq, p - 1);
    quicksort_seq(v, p + 1, dir);
}

/*
 * Quicksort recursivo com tasks OpenMP.
 * Subvetores com <= THRESHOLD elementos são ordenados sequencialmente
 * sem criar tasks para evitar overhead.
 */
static void quicksort_omp(int *v, int esq, int dir) {
    if (esq >= dir) return;

    if ((dir - esq) <= THRESHOLD) {
        quicksort_seq(v, esq, dir);
        return;
    }

    int p = particionar(v, esq, dir);

    #pragma omp task shared(v)
    quicksort_omp(v, esq, p - 1);

    #pragma omp task shared(v)
    quicksort_omp(v, p + 1, dir);

    #pragma omp taskwait
}

/* Gera vetor de n inteiros aleatórios */
static void gerar_vetor(int *v, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        v[i] = rand();
}

/* Retorna 1 se o vetor está ordenado em ordem não-decrescente, 0 caso contrário */
static int validar_ordenacao(int *v, int n) {
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

    int *v_orig = malloc(n * sizeof(int));
    int *v      = malloc(n * sizeof(int));
    if (!v_orig || !v) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        free(v_orig); free(v);
        return 1;
    }

    gerar_vetor(v_orig, n);

    /* warm-up: execução descartada para eliminar overhead de inicialização */
    memcpy(v, v_orig, n * sizeof(int));
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_omp(v, 0, n - 1);
    }
    const char *corretude = validar_ordenacao(v, n) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        memcpy(v, v_orig, n * sizeof(int));
        #pragma omp parallel
        {
            #pragma omp single
            quicksort_omp(v, 0, n - 1);
        }
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("quicksort,openmp,%d,%d,%.6f,%s\n", n, runs, tempo_total, corretude);

    free(v_orig); free(v);
    return 0;
}
