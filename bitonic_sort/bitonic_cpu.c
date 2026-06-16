#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Bitonic sort iterativo, ordenacao crescente */
static void bitonic_sort(int32_t *arr, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; i++) {
                int ij = i ^ j;
                if (ij > i) {
                    if ((i & k) == 0 && arr[i] > arr[ij]) {
                        int32_t tmp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = tmp;
                    } else if ((i & k) != 0 && arr[i] < arr[ij]) {
                        int32_t tmp = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = tmp;
                    }
                }
            }
        }
    }
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

    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "Erro: N deve ser potencia de 2 (recebido: %d)\n", n);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int32_t *arr = malloc((size_t)n * sizeof(int32_t));
    int32_t *ref = malloc((size_t)n * sizeof(int32_t));
    if (!arr || !ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(arr);
        free(ref);
        return 1;
    }

    /* 3 execucoes de warmup, sem saida */
    for (int w = 0; w < 3; w++) {
        srand(42);
        for (int i = 0; i < n; i++)
            arr[i] = rand();
        bitonic_sort(arr, n);
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
            bitonic_sort(arr, n);
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

        printf("bitonic_sort|cpu|aleatorio|%d|%d|%s|%d|1x1\n",
               n, iter, tempo_str, corretude);
    }

    free(arr);
    free(ref);
    return 0;
}
