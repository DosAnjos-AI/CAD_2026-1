#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define CUTOFF 1024

/* Merge iterativo: intercala arr[left..mid] e arr[mid+1..right] usando tmp */
static void merge(int32_t *arr, int32_t *tmp, int left, int mid, int right) {
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

    memcpy(arr + left, tmp + left, (size_t)(right - left + 1) * sizeof(int32_t));
}

/* Merge sort sequencial puro, usado abaixo do cutoff */
static void merge_sort_seq(int32_t *arr, int32_t *tmp, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;
    merge_sort_seq(arr, tmp, left, mid);
    merge_sort_seq(arr, tmp, mid + 1, right);
    merge(arr, tmp, left, mid, right);
}

/* Merge sort paralelo: divide com tasks OpenMP, serializa abaixo do cutoff */
static void merge_sort_omp(int32_t *arr, int32_t *tmp, int left, int right) {
    if (right - left < CUTOFF) {
        merge_sort_seq(arr, tmp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task
    merge_sort_omp(arr, tmp, left, mid);
    #pragma omp task
    merge_sort_omp(arr, tmp, mid + 1, right);
    #pragma omp taskwait

    merge(arr, tmp, left, mid, right);
}

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/* Preenche arr conforme o cenario: crescente (ordenado) ou decrescente (invertido) */
static void gerar_vetor(int32_t *arr, int n, const char *cenario) {
    if (strcmp(cenario, "ordenado") == 0) {
        for (int i = 0; i < n; i++)
            arr[i] = i;
    } else {
        for (int i = 0; i < n; i++)
            arr[i] = n - 1 - i;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <N> <cenario> [iteracoes]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(NUM_THREADS);

    int n = atoi(argv[1]);
    const char *cenario = argv[2];
    int iteracoes = (argc == 4) ? atoi(argv[3]) : 5;

    if (strcmp(cenario, "ordenado") != 0 && strcmp(cenario, "invertido") != 0) {
        fprintf(stderr, "Erro: cenario deve ser 'ordenado' ou 'invertido' (recebido: %s)\n", cenario);
        return 1;
    }

    if (n <= 0) {
        fprintf(stderr, "Erro: N deve ser positivo (recebido: %d)\n", n);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int32_t *arr = malloc((size_t)n * sizeof(int32_t));
    int32_t *tmp = malloc((size_t)n * sizeof(int32_t));
    int32_t *ref = malloc((size_t)n * sizeof(int32_t));
    if (!arr || !tmp || !ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(arr);
        free(tmp);
        free(ref);
        return 1;
    }

    /* 1 execucao de warmup, sem saida */
    for (int w = 0; w < 1; w++) {
        gerar_vetor(arr, n, cenario);

        #pragma omp parallel
        {
            #pragma omp single
            merge_sort_omp(arr, tmp, 0, n - 1);
        }
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 4; exec++) {
            gerar_vetor(arr, n, cenario);

            if (exec == 0) {
                memcpy(ref, arr, (size_t)n * sizeof(int32_t));
                qsort(ref, (size_t)n, sizeof(int32_t), cmp_int32);
            }

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            #pragma omp parallel
            {
                #pragma omp single
                merge_sort_omp(arr, tmp, 0, n - 1);
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);

            soma += (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

            if (exec == 0) {
                for (int i = 0; i < n; i++) {
                    if (arr[i] != ref[i]) {
                        corretude = 0;
                        break;
                    }
                }
            }
        }

        double tempo_s = soma / 4.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("merge_sort|openmp|%s|%d|%d|%s|%d|%dx1\n",
               cenario, n, iter, tempo_str, corretude, omp_get_max_threads());
        fflush(stdout);
    }

    free(arr);
    free(tmp);
    free(ref);
    return 0;
}
