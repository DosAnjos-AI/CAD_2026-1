#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Gera grafo Erdos-Renyi nao-dirigido em formato CSR.
 * Para cada par (u, w) com u < w, aresta existe com probabilidade p = 16.0/V.
 * Construcao em duas passagens: 1) conta grau de cada vertice, 2) preenche col_idx.
 * Aloca *col_idx_out internamente; quem chamar deve liberar o ponteiro retornado.
 * Retorna o numero total de arestas (contando as duas direcoes).
 * Identica ao bfs_cpu.c. */
static int32_t gerar_grafo_csr(int32_t v_count, int32_t *row_ptr, int32_t **col_idx_out) {
    double p = 16.0 / (double)v_count;

    int32_t *grau = (int32_t *)calloc((size_t)v_count, sizeof(int32_t));
    if (!grau) {
        fprintf(stderr, "Erro: falha ao alocar memoria (grau)\n");
        exit(1);
    }

    srand(42);
    for (int32_t u = 0; u < v_count; u++) {
        for (int32_t w = u + 1; w < v_count; w++) {
            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            if (r < p) {
                grau[u]++;
                grau[w]++;
            }
        }
    }

    row_ptr[0] = 0;
    for (int32_t i = 0; i < v_count; i++)
        row_ptr[i + 1] = row_ptr[i] + grau[i];

    int32_t e_total = row_ptr[v_count];
    int32_t *col_idx = (int32_t *)malloc((size_t)e_total * sizeof(int32_t));
    int32_t *offset = (int32_t *)malloc((size_t)v_count * sizeof(int32_t));
    if (!col_idx || !offset) {
        fprintf(stderr, "Erro: falha ao alocar memoria (col_idx/offset)\n");
        exit(1);
    }
    memcpy(offset, row_ptr, (size_t)v_count * sizeof(int32_t));

    srand(42);
    for (int32_t u = 0; u < v_count; u++) {
        for (int32_t w = u + 1; w < v_count; w++) {
            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            if (r < p) {
                col_idx[offset[u]++] = w;
                col_idx[offset[w]++] = u;
            }
        }
    }

    free(grau);
    free(offset);

    *col_idx_out = col_idx;
    return e_total;
}

/* BFS sequencial puro por nivel, usado apenas como oraculo para validar
 * a versao CUDA. Identica ao bfs_cpu.c. */
static void bfs_seq(const int32_t *row_ptr, const int32_t *col_idx, int32_t v_count,
                     int32_t fonte, int32_t *dist, int32_t *frontier, int32_t *next) {
    for (int32_t i = 0; i < v_count; i++)
        dist[i] = -1;
    dist[fonte] = 0;

    int32_t tam_frontier = 1;
    frontier[0] = fonte;

    while (tam_frontier > 0) {
        int32_t tam_next = 0;

        for (int32_t i = 0; i < tam_frontier; i++) {
            int32_t u = frontier[i];
            for (int32_t k = row_ptr[u]; k < row_ptr[u + 1]; k++) {
                int32_t w = col_idx[k];
                if (dist[w] == -1) {
                    dist[w] = dist[u] + 1;
                    next[tam_next++] = w;
                }
            }
        }

        int32_t *tmp = frontier;
        frontier = next;
        next = tmp;
        tam_frontier = tam_next;
    }
}

/* BFS vertex-centric: um thread por vertice, ativo apenas se estiver na
 * fronteira atual. Cada thread expande os vizinhos do seu vertice e usa
 * atomicCAS para reivindicar dist[u] == -1 com exclusividade, evitando que
 * o mesmo vizinho seja inserido mais de uma vez na proxima fronteira. */
__global__ void bfs_kernel(
    int32_t *dist, int32_t V,
    int32_t *row_ptr, int32_t *col_idx,
    int32_t *frontier, int32_t *frontier_next,
    int32_t nivel, int32_t *frontier_vazia)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    if (!frontier[v]) return;

    frontier[v] = 0;
    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        if (atomicCAS(&dist[u], -1, nivel + 1) == -1) {
            frontier_next[u] = 1;
            *frontier_vazia = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <V> [iteracoes]\n", argv[0]);
        return 1;
    }

    int32_t v = atoi(argv[1]);
    int iteracoes = (argc == 3) ? atoi(argv[2]) : 5;

    if (v <= 0) {
        fprintf(stderr, "Erro: V deve ser positivo (recebido: %d)\n", v);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int blocos = (v + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int32_t *row_ptr = (int32_t *)malloc((size_t)(v + 1) * sizeof(int32_t));
    int32_t *col_idx = NULL;
    int32_t *dist = (int32_t *)malloc((size_t)v * sizeof(int32_t));
    int32_t *dist_ref = (int32_t *)malloc((size_t)v * sizeof(int32_t));
    int32_t *frontier_host = (int32_t *)malloc((size_t)v * sizeof(int32_t));
    int32_t *next_host = (int32_t *)malloc((size_t)v * sizeof(int32_t));
    if (!row_ptr || !dist || !dist_ref || !frontier_host || !next_host) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(row_ptr);
        free(dist);
        free(dist_ref);
        free(frontier_host);
        free(next_host);
        return 1;
    }

    /* Oraculo: BFS sequencial executado uma vez antes do loop de medicao.
     * A geracao do grafo e deterministica (srand(42) reiniciado a cada
     * chamada), entao e_total e o conteudo de col_idx sao identicos em
     * todas as regeneracoes seguintes. */
    int32_t e_total = gerar_grafo_csr(v, row_ptr, &col_idx);
    bfs_seq(row_ptr, col_idx, v, 0, dist_ref, frontier_host, next_host);
    free(col_idx);
    col_idx = NULL;

    int32_t *d_row_ptr, *d_col_idx, *d_dist, *d_frontier, *d_frontier_next, *d_frontier_vazia;
    if (cudaMalloc(&d_row_ptr, (size_t)(v + 1) * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_col_idx, (size_t)e_total * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_dist, (size_t)v * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_frontier, (size_t)v * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_frontier_next, (size_t)v * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_frontier_vazia, sizeof(int32_t)) != cudaSuccess) {
        fprintf(stderr, "Erro: falha ao alocar memoria na GPU\n");
        free(row_ptr);
        free(dist);
        free(dist_ref);
        free(frontier_host);
        free(next_host);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int32_t um = 1;

    /* 1 execucao de warmup, sem saida */
    for (int w = 0; w < 1; w++) {
        gerar_grafo_csr(v, row_ptr, &col_idx);
        cudaMemcpy(d_row_ptr, row_ptr, (size_t)(v + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, col_idx, (size_t)e_total * sizeof(int32_t), cudaMemcpyHostToDevice);

        cudaMemset(d_dist, 0xFF, (size_t)v * sizeof(int32_t));
        cudaMemset(d_dist, 0, sizeof(int32_t));
        cudaMemset(d_frontier, 0, (size_t)v * sizeof(int32_t));
        cudaMemcpy(d_frontier, &um, sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_frontier_next, 0, (size_t)v * sizeof(int32_t));

        int32_t nivel = 0;
        int32_t h_vazia;
        while (1) {
            h_vazia = 1;
            cudaMemcpy(d_frontier_vazia, &h_vazia, sizeof(int32_t), cudaMemcpyHostToDevice);
            bfs_kernel<<<blocos, BLOCK_SIZE>>>(d_dist, v, d_row_ptr, d_col_idx,
                                                d_frontier, d_frontier_next, nivel, d_frontier_vazia);
            int32_t *tmp = d_frontier; d_frontier = d_frontier_next; d_frontier_next = tmp;
            cudaMemset(d_frontier_next, 0, (size_t)v * sizeof(int32_t));
            cudaMemcpy(&h_vazia, d_frontier_vazia, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (h_vazia) break;
            nivel++;
        }
        cudaDeviceSynchronize();
        free(col_idx);
        col_idx = NULL;
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 5; exec++) {
            gerar_grafo_csr(v, row_ptr, &col_idx);
            cudaMemcpy(d_row_ptr, row_ptr, (size_t)(v + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_col_idx, col_idx, (size_t)e_total * sizeof(int32_t), cudaMemcpyHostToDevice);

            cudaEventRecord(start);

            cudaMemset(d_dist, 0xFF, (size_t)v * sizeof(int32_t));
            cudaMemset(d_dist, 0, sizeof(int32_t));
            cudaMemset(d_frontier, 0, (size_t)v * sizeof(int32_t));
            cudaMemcpy(d_frontier, &um, sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_frontier_next, 0, (size_t)v * sizeof(int32_t));

            int32_t nivel = 0;
            int32_t h_vazia;
            while (1) {
                h_vazia = 1;
                cudaMemcpy(d_frontier_vazia, &h_vazia, sizeof(int32_t), cudaMemcpyHostToDevice);
                bfs_kernel<<<blocos, BLOCK_SIZE>>>(d_dist, v, d_row_ptr, d_col_idx,
                                                    d_frontier, d_frontier_next, nivel, d_frontier_vazia);
                int32_t *tmp = d_frontier; d_frontier = d_frontier_next; d_frontier_next = tmp;
                cudaMemset(d_frontier_next, 0, (size_t)v * sizeof(int32_t));
                cudaMemcpy(&h_vazia, d_frontier_vazia, sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (h_vazia) break;
                nivel++;
            }
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            soma += (double)ms / 1000.0;

            cudaMemcpy(dist, d_dist, (size_t)v * sizeof(int32_t), cudaMemcpyDeviceToHost);

            if (exec == 0) {
                for (int32_t i = 0; i < v; i++) {
                    if (dist[i] != dist_ref[i]) {
                        corretude = 0;
                        break;
                    }
                }
            }

            free(col_idx);
            col_idx = NULL;
        }

        double tempo_s = soma / 5.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("bfs|cuda|aleatorio|%d|%d|%s|%d|256x%d\n",
               v, iter, tempo_str, corretude, blocos);
        fflush(stdout);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_frontier_next);
    cudaFree(d_frontier_vazia);
    free(row_ptr);
    free(dist);
    free(dist_ref);
    free(frontier_host);
    free(next_host);
    return 0;
}
