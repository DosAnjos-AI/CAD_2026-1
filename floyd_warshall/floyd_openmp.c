#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define INF 1000000000

/* Gera matriz de adjacencia densa V x V, row-major, com semente fixa */
static void gerar_matriz(int32_t *dist, int v) {
    srand(42);
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < v; j++) {
            if (i == j) {
                dist[i * v + j] = 0;
            } else if ((rand() % 2) == 0) {
                dist[i * v + j] = (rand() % 1000) + 1;
            } else {
                dist[i * v + j] = INF;
            }
        }
    }
}

/* Floyd-Warshall sequencial puro, usado apenas como oraculo de corretude */
static void floyd_warshall_seq(int32_t *dist, int v) {
    for (int k = 0; k < v; k++)
        for (int i = 0; i < v; i++)
            if (dist[i * v + k] < INF)
                for (int j = 0; j < v; j++)
                    if (dist[k * v + j] < INF)
                        if (dist[i * v + k] + dist[k * v + j] < dist[i * v + j])
                            dist[i * v + j] = dist[i * v + k] + dist[k * v + j];
}

/* Floyd-Warshall paralelo com OpenMP: loop k sequencial (dependencia entre
   iteracoes), loop i paralelizado entre threads, loop j sequencial por thread */
static void floyd_warshall_openmp(int32_t *dist, int v) {
    for (int k = 0; k < v; k++) {
        int j;
        #pragma omp parallel for private(j)
        for (int i = 0; i < v; i++)
            for (j = 0; j < v; j++)
                if (dist[i * v + k] < INF && dist[k * v + j] < INF)
                    if (dist[i * v + k] + dist[k * v + j] < dist[i * v + j])
                        dist[i * v + j] = dist[i * v + k] + dist[k * v + j];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <V> [iteracoes]\n", argv[0]);
        return 1;
    }

    int v = atoi(argv[1]);
    int iteracoes = (argc == 3) ? atoi(argv[2]) : 10;

    if (v <= 0) {
        fprintf(stderr, "Erro: V deve ser positivo (recebido: %d)\n", v);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    omp_set_num_threads(NUM_THREADS);

    int32_t *dist = malloc((size_t)v * (size_t)v * sizeof(int32_t));
    int32_t *dist_ref = malloc((size_t)v * (size_t)v * sizeof(int32_t));
    if (!dist || !dist_ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(dist);
        free(dist_ref);
        return 1;
    }

    /* Oraculo: executa uma vez antes do loop principal */
    gerar_matriz(dist_ref, v);
    floyd_warshall_seq(dist_ref, v);

    /* 3 execucoes de warmup, sem saida */
    for (int w = 0; w < 3; w++) {
        gerar_matriz(dist, v);
        floyd_warshall_openmp(dist, v);
    }

    int nthreads = omp_get_max_threads();

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 10; exec++) {
            gerar_matriz(dist, v);

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            floyd_warshall_openmp(dist, v);
            clock_gettime(CLOCK_MONOTONIC, &t1);

            soma += (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

            if (iter == 1) {
                for (int i = 0; i < v * v; i++) {
                    if (dist[i] != dist_ref[i]) {
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

        printf("floyd_warshall|openmp|aleatorio|%d|%d|%s|%d|%dx1\n",
               v, iter, tempo_str, corretude, nthreads);
    }

    free(dist);
    free(dist_ref);
    return 0;
}
