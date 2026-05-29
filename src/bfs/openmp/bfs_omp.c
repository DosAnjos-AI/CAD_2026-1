#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

/*
 * Gera grafo aleatório com N vértices e M arestas em formato CSR.
 * Para cada aresta, escolhe u aleatório e v != u via deslocamento circular.
 * offset[i] = início das arestas do nó i em adj[]
 * size[i]   = número de arestas saindo do nó i
 */
static void gerar_grafo(int N, int M, int *adj, int *offset, int *size) {
    int *esrc = malloc(M * sizeof(int));
    int *edst = malloc(M * sizeof(int));

    memset(size, 0, N * sizeof(int));
    for (int i = 0; i < M; i++) {
        int u    = rand() % N;
        int v    = (u + 1 + rand() % (N - 1)) % N;
        esrc[i]  = u;
        edst[i]  = v;
        size[u]++;
    }

    /* Prefix sum: offset[i] acumula os graus de 0..i-1 */
    offset[0] = 0;
    for (int i = 0; i < N; i++)
        offset[i + 1] = offset[i] + size[i];

    /* Preenche adj[] usando posição auxiliar para cada nó */
    int *pos = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        pos[i] = offset[i];
    for (int i = 0; i < M; i++)
        adj[pos[esrc[i]]++] = edst[i];

    free(esrc);
    free(edst);
    free(pos);
}

/*
 * BFS level-synchronous com OpenMP.
 * Laço externo avança um nível por vez; laço interno paralelo
 * percorre todos os nós e atualiza vizinhos ainda não visitados.
 * `changed` com reduction(|:) evita race condition na flag de parada.
 * Race benigna em distance[v]: múltiplas threads podem escrever
 * o mesmo valor (nivel+1), o resultado final é correto.
 */
static void bfs_omp(int N, int *adj, int *offset, int *size,
                    int *distance, int *parent) {
    for (int i = 0; i < N; i++) {
        distance[i] = INT_MAX;
        parent[i]   = -1;
    }
    distance[0] = 0;

    int nivel   = 0;
    int changed = 1;

    while (changed) {
        changed = 0;
        #pragma omp parallel for schedule(dynamic) reduction(|:changed)
        for (int u = 0; u < N; u++) {
            if (distance[u] == nivel) {
                for (int j = offset[u]; j < offset[u] + size[u]; j++) {
                    int v = adj[j];
                    if (distance[v] == INT_MAX) {
                        distance[v] = nivel + 1;
                        parent[v]   = u;
                        changed     = 1;
                    }
                }
            }
        }
        nivel++;
    }
}

/*
 * Verifica que nenhum nó alcançável tem distância inválida.
 * Nós não alcançáveis (INT_MAX) são aceitos como corretos.
 */
static int validar_bfs(int N, int *distance) {
    for (int i = 0; i < N; i++) {
        if (distance[i] != INT_MAX && (distance[i] < 0 || distance[i] >= N))
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

    int *adj      = malloc(M       * sizeof(int));
    int *offset   = malloc((N + 1) * sizeof(int));
    int *size     = malloc(N       * sizeof(int));
    int *distance = malloc(N       * sizeof(int));
    int *parent   = malloc(N       * sizeof(int));

    if (!adj || !offset || !size || !distance || !parent) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    srand(time(NULL));
    gerar_grafo(N, M, adj, offset, size);

    /* warm-up: bfs_omp reinicializa distance/parent internamente */
    bfs_omp(N, adj, offset, size, distance, parent);
    const char *corretude = validar_bfs(N, distance) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++)
        bfs_omp(N, adj, offset, size, distance, parent);

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("bfs,openmp,%dx%d,%d,%.6f,%s\n", N, M, runs, tempo_total, corretude);

    free(adj);
    free(offset);
    free(size);
    free(distance);
    free(parent);
    return 0;
}
