#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

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

/* Merge sort top-down recursivo */
static void merge_sort(int32_t *arr, int32_t *tmp, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;
    merge_sort(arr, tmp, left, mid);
    merge_sort(arr, tmp, mid + 1, right);
    merge(arr, tmp, left, mid, right);
}

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <N> [iteracoes]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int iteracoes = (argc == 3) ? atoi(argv[2]) : 10;

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

    /* 3 execucoes de warmup, sem saida */
    for (int w = 0; w < 3; w++) {
        srand(42);
        for (int i = 0; i < n; i++)
            arr[i] = rand();
        merge_sort(arr, tmp, 0, n - 1);
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 10; exec++) {
            srand(42);
            for (int i = 0; i < n; i++)
                arr[i] = rand();

            if (exec == 0) {
                memcpy(ref, arr, (size_t)n * sizeof(int32_t));
                qsort(ref, (size_t)n, sizeof(int32_t), cmp_int32);
            }

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            merge_sort(arr, tmp, 0, n - 1);
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

        double tempo_s = soma / 10.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("merge_sort|cpu|aleatorio|%d|%d|%s|%d|1x1\n",
               n, iter, tempo_str, corretude);
    }

    free(arr);
    free(tmp);
    free(ref);
    return 0;
}
