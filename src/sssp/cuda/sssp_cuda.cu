#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

/*
 * Gera M arestas direcionadas aleatórias com pesos entre 1 e 100.
 * Idêntica à versão OpenMP.
 */
static void gerar_grafo_ponderado(int N, int M,
                                   int *origem, int *destino, int *custo) {
    for (int i = 0; i < M; i++) {
        int u      = rand() % N;
        int v      = (u + 1 + rand() % (N - 1)) % N;
        origem[i]  = u;
        destino[i] = v;
        custo[i]   = 1 + rand() % 100;
    }
}

/*
 * Kernel Bellman-Ford: uma thread por aresta.
 * atomicMin garante que o menor valor vence em d_dist[v]
 * quando múltiplas threads tentam atualizar o mesmo destino.
 */
__global__ void kernel_sssp(int M,
                             int *d_origem, int *d_destino, int *d_custo,
                             int *d_dist, int *d_preNode, int *d_finished) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= M) return;

    int u = d_origem[thread_id];
    int v = d_destino[thread_id];
    int w = d_custo[thread_id];

    /* Guard contra overflow: dist[u]+w quando dist[u]==INT_MAX */
    if (d_dist[u] != INT_MAX && d_dist[u] + w < d_dist[v]) {
        atomicMin(&d_dist[v], d_dist[u] + w);
        d_preNode[v]  = u;
        *d_finished   = 0;
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

/* Executa laço Bellman-Ford com convergência antecipada */
static void executar_sssp(int N, int M, int *d_origem, int *d_destino, int *d_custo,
                           int *h_dist_init, int *d_dist, int *d_preNode,
                           int *d_finished) {
    int num_blocos = (M + 255) / 256;
    int h_finished;

    cudaMemcpy(d_dist, h_dist_init, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int iter = 0; iter < N - 1; iter++) {
        h_finished = 1;
        cudaMemcpy(d_finished, &h_finished, sizeof(int), cudaMemcpyHostToDevice);
        kernel_sssp<<<num_blocos, 256>>>(M, d_origem, d_destino, d_custo,
                                          d_dist, d_preNode, d_finished);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_finished, d_finished, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_finished) break;
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
    int *h_origem    = (int *)malloc(M * sizeof(int));
    int *h_destino   = (int *)malloc(M * sizeof(int));
    int *h_custo     = (int *)malloc(M * sizeof(int));
    int *h_dist      = (int *)malloc(N * sizeof(int));
    int *h_preNode   = (int *)malloc(N * sizeof(int));
    /* dist inicial: INT_MAX em todos, exceto nó 0 */
    int *h_dist_init = (int *)malloc(N * sizeof(int));

    if (!h_origem || !h_destino || !h_custo || !h_dist || !h_preNode || !h_dist_init) {
        fprintf(stderr, "Erro ao alocar memoria no host\n");
        return 1;
    }

    srand(time(NULL));
    gerar_grafo_ponderado(N, M, h_origem, h_destino, h_custo);

    for (int i = 0; i < N; i++) h_dist_init[i] = INT_MAX;
    h_dist_init[0] = 0;

    /* Alocações na GPU */
    int *d_origem, *d_destino, *d_custo, *d_dist, *d_preNode, *d_finished;

    cudaMalloc((void **)&d_origem,   M * sizeof(int));
    cudaMalloc((void **)&d_destino,  M * sizeof(int));
    cudaMalloc((void **)&d_custo,    M * sizeof(int));
    cudaMalloc((void **)&d_dist,     N * sizeof(int));
    cudaMalloc((void **)&d_preNode,  N * sizeof(int));
    cudaMalloc((void **)&d_finished, sizeof(int));

    /* Transfere grafo para a GPU uma vez — somente-leitura nos kernels */
    cudaMemcpy(d_origem,  h_origem,  M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_destino, h_destino, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_custo,   h_custo,   M * sizeof(int), cudaMemcpyHostToDevice);

    /* warm-up: reseta d_dist e executa SSSP completo para validar */
    executar_sssp(N, M, d_origem, d_destino, d_custo,
                  h_dist_init, d_dist, d_preNode, d_finished);
    cudaMemcpy(h_dist, d_dist, N * sizeof(int), cudaMemcpyDeviceToHost);
    const char *corretude = validar_sssp(N, h_dist) ? "OK" : "ERRO";

    struct timeval inicio, fim;
    gettimeofday(&inicio, NULL);

    for (int r = 0; r < runs; r++) {
        executar_sssp(N, M, d_origem, d_destino, d_custo,
                      h_dist_init, d_dist, d_preNode, d_finished);
    }

    gettimeofday(&fim, NULL);

    double tempo_total = (fim.tv_sec  - inicio.tv_sec) +
                         (fim.tv_usec - inicio.tv_usec) / 1e6;

    printf("sssp,cuda,%dx%d,%d,%.6f,%s\n", N, M, runs, tempo_total, corretude);

    free(h_origem); free(h_destino); free(h_custo);
    free(h_dist); free(h_preNode); free(h_dist_init);
    cudaFree(d_origem); cudaFree(d_destino); cudaFree(d_custo);
    cudaFree(d_dist); cudaFree(d_preNode); cudaFree(d_finished);
    return 0;
}
