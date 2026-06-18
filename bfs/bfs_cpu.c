#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Gera grafo CSR nao-dirigido conforme o cenario:
 *   ordenado  -> estrela: vertice 0 ligado a todos os demais (1 nivel BFS)
 *   invertido -> cadeia: 0-1-2-...-(V-1) (V-1 niveis BFS, pior caso)
 * Construcao em duas passagens: 1) conta grau de cada vertice, 2) preenche col_idx.
 * Aloca *col_idx_out internamente; quem chamar deve liberar o ponteiro retornado.
 * Grafos deterministicos, sem rand. Retorna o numero total de arestas (duas direcoes). */
static int32_t gerar_grafo_csr(int32_t v_count, int32_t *row_ptr, int32_t **col_idx_out,
                                const char *cenario) {
    int estrela = (strcmp(cenario, "ordenado") == 0);

    int32_t *grau = calloc((size_t)v_count, sizeof(int32_t));
    if (!grau) {
        fprintf(stderr, "Erro: falha ao alocar memoria (grau)\n");
        exit(1);
    }

    if (estrela) {
        grau[0] = v_count - 1;
        for (int32_t i = 1; i < v_count; i++)
            grau[i] = 1;
    } else {
        for (int32_t i = 0; i < v_count; i++)
            grau[i] = (i == 0 || i == v_count - 1) ? 1 : 2;
    }

    row_ptr[0] = 0;
    for (int32_t i = 0; i < v_count; i++)
        row_ptr[i + 1] = row_ptr[i] + grau[i];

    int32_t e_total = row_ptr[v_count];
    int32_t *col_idx = malloc((size_t)e_total * sizeof(int32_t));
    int32_t *offset = malloc((size_t)v_count * sizeof(int32_t));
    if (!col_idx || !offset) {
        fprintf(stderr, "Erro: falha ao alocar memoria (col_idx/offset)\n");
        exit(1);
    }
    memcpy(offset, row_ptr, (size_t)v_count * sizeof(int32_t));

    if (estrela) {
        for (int32_t i = 1; i < v_count; i++) {
            col_idx[offset[0]++] = i;
            col_idx[offset[i]++] = 0;
        }
    } else {
        for (int32_t i = 0; i < v_count - 1; i++) {
            col_idx[offset[i]++] = i + 1;
            col_idx[offset[i + 1]++] = i;
        }
    }

    free(grau);
    free(offset);

    *col_idx_out = col_idx;
    return e_total;
}

/* BFS por nivel a partir de fonte, usando dois arrays de fronteira pre-alocados. */
static void bfs(const int32_t *row_ptr, const int32_t *col_idx, int32_t v_count,
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

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <V> <cenario> [iteracoes]\n", argv[0]);
        return 1;
    }

    int32_t v = atoi(argv[1]);
    const char *cenario = argv[2];
    int iteracoes = (argc == 4) ? atoi(argv[3]) : 5;

    if (strcmp(cenario, "ordenado") != 0 && strcmp(cenario, "invertido") != 0) {
        fprintf(stderr, "Erro: cenario deve ser 'ordenado' ou 'invertido' (recebido: %s)\n", cenario);
        return 1;
    }

    if (v <= 0) {
        fprintf(stderr, "Erro: V deve ser positivo (recebido: %d)\n", v);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int32_t *row_ptr = malloc((size_t)(v + 1) * sizeof(int32_t));
    int32_t *col_idx = NULL;
    int32_t *dist = malloc((size_t)v * sizeof(int32_t));
    int32_t *dist_ref = malloc((size_t)v * sizeof(int32_t));
    int32_t *frontier = malloc((size_t)v * sizeof(int32_t));
    int32_t *next = malloc((size_t)v * sizeof(int32_t));
    if (!row_ptr || !dist || !dist_ref || !frontier || !next) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(row_ptr);
        free(dist);
        free(dist_ref);
        free(frontier);
        free(next);
        return 1;
    }

    /* Oraculo: BFS sequencial executado uma vez antes do loop de medicao */
    gerar_grafo_csr(v, row_ptr, &col_idx, cenario);
    bfs(row_ptr, col_idx, v, 0, dist_ref, frontier, next);
    free(col_idx);
    col_idx = NULL;

    /* 1 execucao de warmup, sem saida */
    for (int w = 0; w < 1; w++) {
        gerar_grafo_csr(v, row_ptr, &col_idx, cenario);
        bfs(row_ptr, col_idx, v, 0, dist, frontier, next);
        free(col_idx);
        col_idx = NULL;
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 4; exec++) {
            gerar_grafo_csr(v, row_ptr, &col_idx, cenario);

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            bfs(row_ptr, col_idx, v, 0, dist, frontier, next);
            clock_gettime(CLOCK_MONOTONIC, &t1);

            soma += (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

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

        double tempo_s = soma / 4.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("bfs|cpu|%s|%d|%d|%s|%d|1x1\n",
               cenario, v, iter, tempo_str, corretude);
        fflush(stdout);
    }

    free(row_ptr);
    free(dist);
    free(dist_ref);
    free(frontier);
    free(next);
    return 0;
}
