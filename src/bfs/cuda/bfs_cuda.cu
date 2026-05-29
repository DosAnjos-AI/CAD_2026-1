#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

/* Sentinela compatível com cudaMemset(d_distance, 0xFF, ...) */
#define NAO_VISITADO 0xFFFFFFFFu

/*
 * Gera grafo aleatório com N vértices e M arestas em formato CSR.
 * Idêntica à versão OpenMP.
 */
static void gerar_grafo(int N, int M, int *adj, int *offset, int *size) {
    int *esrc = (int *)malloc(M * sizeof(int));
    int *edst = (int *)malloc(M * sizeof(int));

    memset(size, 0, N * sizeof(int));
    for (int i = 0; i < M; i++) {
        int u   = rand() % N;
        int v   = (u + 1 + rand() % (N - 1)) % N;
        esrc[i] = u;
        edst[i] = v;
        size[u]++;
    }

    offset[0] = 0;
    for (int i = 0; i < N; i++)
        offset[i + 1] = offset[i] + size[i];

    int *pos = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        pos[i] = offset[i];
    for (int i = 0; i < M; i++)
        adj[pos[esrc[i]]++] = edst[i];

    free(esrc);
    free(edst);
    free(pos);
}

/*
 * Kernel BFS com fila — um thread por nó da fila atual.
 * atomicCAS garante que cada nó é inserido na fila seguinte apenas
 * uma vez, eliminando duplicatas sem janela de corrida.
 */
__global__ void kernel_bfs(int nivel,
                            int *d_adj, int *d_offset, int *d_size,
                            unsigned int *d_distance, int *d_parent,
                            int queueSize, int *d_nextQueueSize,
                            int *d_currentQueue, int *d_nextQueue) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= queueSize) return;

    int u = d_currentQueue[thread_id];

    for (int j = d_offset[u]; j < d_offset[u] + d_size[u]; j++) {
        int v = d_adj[j];
        /* Descobre v atomicamente: apenas o primeiro thread vence */
        unsigned int old = atomicCAS(&d_distance[v], NAO_VISITADO,
                                     (unsigned int)(nivel + 1));
        if (old == NAO_VISITADO) {
            d_parent[v] = u;
            int posicao  = atomicAdd(d_nextQueueSize, 1);
            d_nextQueue[posicao] = v;
        }
    }
}

/*
 * Verifica que nenhum nó alcançável tem distância >= N.
 * Nós não alcançáveis (NAO_VISITADO) são aceitos como corretos.
 */
static int validar_bfs(int N, unsigned int *distance) {
    for (int i = 0; i < N; i++) {
        if (distance[i] != NAO_VISITADO && distance[i] >= (unsigned int)N)
            return 0;
    }
    return 1;
}

/* Executa o laço BFS sobre os buffers de fila fornecidos */
static void executar_bfs(int N, int *d_adj, int *d_offset, int *d_size,
                          unsigned int *d_distance, int *d_parent,
                          int *d_qa, int *d_qb, int *d_nextQueueSize) {
    unsigned int zero_dist = 0;
    int no_origem = 0;

    cudaMemset(d_distance, 0xFF, N * sizeof(unsigned int));
    cudaMemcpy(d_distance, &zero_dist, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qa, &no_origem, sizeof(int), cudaMemcpyHostToDevice);

    int *curQ = d_qa, *nxtQ = d_qb;
    int nivel = 0, queueSize = 1, nextQueueSize = 0;

    while (queueSize > 0) {
        cudaMemset(d_nextQueueSize, 0, sizeof(int));
        int num_blocos = (queueSize + 255) / 256;
        kernel_bfs<<<num_blocos, 256>>>(
            nivel, d_adj, d_offset, d_size,
            d_distance, d_parent,
            queueSize, d_nextQueueSize, curQ, nxtQ
        );
        cudaDeviceSynchronize();
        cudaMemcpy(&nextQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
        int *tmp = curQ; curQ = nxtQ; nxtQ = tmp;
        queueSize = nextQueueSize;
        nivel++;
    }
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

    /* Alocações no host */
    int          *h_adj      = (int *)malloc(M       * sizeof(int));
    int          *h_offset   = (int *)malloc((N + 1) * sizeof(int));
    int          *h_size     = (int *)malloc(N       * sizeof(int));
    unsigned int *h_distance = (unsigned int *)malloc(N * sizeof(unsigned int));

    if (!h_adj || !h_offset || !h_size || !h_distance) {
        fprintf(stderr, "Erro ao alocar memoria no host\n");
        return 1;
    }

    srand(time(NULL));
    gerar_grafo(N, M, h_adj, h_offset, h_size);

    /* Alocações na GPU */
    int          *d_adj, *d_offset, *d_size, *d_parent;
    unsigned int *d_distance;
    int          *d_qa, *d_qb, *d_nextQueueSize;

    cudaMalloc((void **)&d_adj,           M       * sizeof(int));
    cudaMalloc((void **)&d_offset,        (N + 1) * sizeof(int));
    cudaMalloc((void **)&d_size,          N       * sizeof(int));
    cudaMalloc((void **)&d_distance,      N       * sizeof(unsigned int));
    cudaMalloc((void **)&d_parent,        N       * sizeof(int));
    cudaMalloc((void **)&d_qa,            N       * sizeof(int));
    cudaMalloc((void **)&d_qb,            N       * sizeof(int));
    cudaMalloc((void **)&d_nextQueueSize, sizeof(int));

    /* Transfere grafo para a GPU uma vez — somente-leitura nos kernels */
    cudaMemcpy(d_adj,    h_adj,    M       * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size,   h_size,   N       * sizeof(int), cudaMemcpyHostToDevice);

    /* warm-up: reseta estado e executa BFS completo para validar */
    executar_bfs(N, d_adj, d_offset, d_size, d_distance, d_parent,
                 d_qa, d_qb, d_nextQueueSize);
    cudaMemcpy(h_distance, d_distance, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    const char *corretude = validar_bfs(N, h_distance) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        executar_bfs(N, d_adj, d_offset, d_size, d_distance, d_parent,
                     d_qa, d_qb, d_nextQueueSize);
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("bfs,cuda,%dx%d,%d,%.6f,%s\n", N, M, runs, tempo_total, corretude);

    free(h_adj);
    free(h_offset);
    free(h_size);
    free(h_distance);
    cudaFree(d_adj);
    cudaFree(d_offset);
    cudaFree(d_size);
    cudaFree(d_distance);
    cudaFree(d_parent);
    cudaFree(d_qa);
    cudaFree(d_qb);
    cudaFree(d_nextQueueSize);
    return 0;
}
