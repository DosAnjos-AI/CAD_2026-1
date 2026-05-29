#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

/*
 * Gera M arestas direcionadas aleatórias com pesos entre 1 e 100.
 * Garante que origem != destino via deslocamento circular.
 */
static void gerar_grafo_ponderado(int N, int M,
                                   int *origem, int *destino, int *custo) {
    for (int i = 0; i < M; i++) {
        int u     = rand() % N;
        int v     = (u + 1 + rand() % (N - 1)) % N;
        origem[i]  = u;
        destino[i] = v;
        custo[i]   = 1 + rand() % 100;
    }
}

/*
 * Bellman-Ford paralelo com OpenMP.
 * Cada thread itera sobre um subconjunto de arestas atribuído pelo
 * scheduler. Convergência antecipada via reduction(&:finished).
 * Race benigna em dist[v]: Bellman-Ford converge em N-1 iterações
 * independentemente da ordem de relaxamento das arestas.
 */
static void sssp_omp(int N, int M,
                     int *origem, int *destino, int *custo,
                     int *dist, int *preNode) {
    for (int i = 0; i < N; i++) {
        dist[i]    = INT_MAX;
        preNode[i] = -1;
    }
    dist[0] = 0;

    for (int iter = 0; iter < N - 1; iter++) {
        int finished = 1;

        #pragma omp parallel for schedule(static) reduction(&:finished)
        for (int i = 0; i < M; i++) {
            int u = origem[i];
            int v = destino[i];
            int w = custo[i];
            /* Guard contra overflow: dist[u]+w quando dist[u]==INT_MAX */
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v]    = dist[u] + w;
                preNode[v] = u;
                finished   = 0;
            }
        }

        if (finished) break;
    }
}

/*
 * Verifica que dist[0]==0 e nenhum nó tem distância negativa.
 * Nós não alcançáveis (INT_MAX) são aceitos como corretos.
 */
static int validar_sssp(int N, int *dist) {
    if (dist[0] != 0)
        return 0;
    for (int i = 0; i < N; i++) {
        if (dist[i] < 0)
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s N M [--runs N]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    if (N <= 0 || M <= 0) {
        fprintf(stderr, "N e M devem ser inteiros positivos\n");
        return 1;
    }

    int runs = 10000;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[i + 1]);
    }

    int *origem  = malloc(M * sizeof(int));
    int *destino = malloc(M * sizeof(int));
    int *custo   = malloc(M * sizeof(int));
    int *dist    = malloc(N * sizeof(int));
    int *preNode = malloc(N * sizeof(int));

    if (!origem || !destino || !custo || !dist || !preNode) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    srand(time(NULL));
    gerar_grafo_ponderado(N, M, origem, destino, custo);

    /* warm-up: sssp_omp reinicializa dist/preNode internamente */
    sssp_omp(N, M, origem, destino, custo, dist, preNode);
    const char *corretude = validar_sssp(N, dist) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++)
        sssp_omp(N, M, origem, destino, custo, dist, preNode);

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("sssp,openmp,%dx%d,%d,%.6f,%s\n", N, M, runs, tempo_total, corretude);

    free(origem);
    free(destino);
    free(custo);
    free(dist);
    free(preNode);
    return 0;
}
