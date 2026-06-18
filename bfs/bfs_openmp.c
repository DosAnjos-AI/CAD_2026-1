#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Tamanho do buffer local de descobertas por thread antes do merge na
 * frontier_prox global. Evita alocar um buffer do tamanho do grafo por
 * thread: ao enchar, a thread faz flush sob secao critica e reinicia. */
#define LOCAL_BUFFER_SIZE 1024

/* Gera grafo CSR nao-dirigido conforme o cenario:
 *   aleatorio -> Erdos-Renyi grau medio 16 via amostragem geometrica
 *                (Batagelj-Brandes): em vez de testar todos os O(V^2) pares,
 *                salta direto para a proxima aresta sorteando o intervalo de
 *                uma distribuicao geometrica, resultando em custo O(V + E).
 *                Para cada par (v, w) com w < v, aresta com prob p = 16.0/V.
 *                As duas passagens reiniciam srand(42) e geram o mesmo grafo.
 *   ordenado  -> estrela: vertice 0 ligado a todos os demais (1 nivel BFS)
 *   invertido -> cadeia: 0-1-2-...-(V-1) (V-1 niveis BFS, pior caso)
 * Construcao em duas passagens: 1) conta grau de cada vertice, 2) preenche col_idx.
 * Aloca *col_idx_out internamente; quem chamar deve liberar o ponteiro retornado.
 * Retorna o numero total de arestas (contando as duas direcoes).
 * Identica ao bfs_cpu.c. */
static int32_t gerar_grafo_csr(int32_t v_count, int32_t *row_ptr, int32_t **col_idx_out,
                                const char *cenario) {
    int aleatorio = (strcmp(cenario, "aleatorio") == 0);
    int estrela = (strcmp(cenario, "ordenado") == 0);
    double p = 16.0 / (double)v_count;
    double log_um_menos_p = log(1.0 - p);

    int32_t *grau = calloc((size_t)v_count, sizeof(int32_t));
    if (!grau) {
        fprintf(stderr, "Erro: falha ao alocar memoria (grau)\n");
        exit(1);
    }

    /* Passagem 1: conta o grau de cada vertice */
    if (aleatorio) {
        srand(42);
        int32_t v = 1;
        int32_t w = -1;
        while (v < v_count) {
            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            w += 1 + (int32_t)(log(1.0 - r) / log_um_menos_p);
            while (w >= v && v < v_count) {
                w -= v;
                v++;
            }
            if (v < v_count) {
                grau[v]++;
                grau[w]++;
            }
        }
    } else if (estrela) {
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

    /* Passagem 2: preenche col_idx com a mesma sequencia de arestas */
    if (aleatorio) {
        srand(42);
        int32_t v = 1;
        int32_t w = -1;
        while (v < v_count) {
            double r = (double)rand() / ((double)RAND_MAX + 1.0);
            w += 1 + (int32_t)(log(1.0 - r) / log_um_menos_p);
            while (w >= v && v < v_count) {
                w -= v;
                v++;
            }
            if (v < v_count) {
                col_idx[offset[v]++] = w;
                col_idx[offset[w]++] = v;
            }
        }
    } else if (estrela) {
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

/* BFS sequencial puro por nivel, usado apenas como oraculo para validar
 * a versao paralela. Identico ao bfs_cpu.c. */
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

/* BFS por nivel paralelizado com OpenMP. O loop sobre os vertices da
 * fronteira atual e dividido entre as threads. A atualizacao de dist[v]
 * usa CAS atomico para evitar race condition: apenas a thread vencedora
 * adiciona v ao seu buffer local. Cada thread acumula descobertas em um
 * buffer local e faz merge na frontier_prox global sob secao critica
 * (no fim do loop ou quando o buffer enche). O barrier implicito do
 * parallel for garante a sincronizacao entre niveis. */
static void bfs_openmp(const int32_t *row_ptr, const int32_t *col_idx, int32_t v_count,
                        int32_t fonte, int32_t *dist, int32_t *frontier, int32_t *next) {
    for (int32_t i = 0; i < v_count; i++)
        dist[i] = -1;
    dist[fonte] = 0;

    int32_t tam_frontier = 1;
    frontier[0] = fonte;

    while (tam_frontier > 0) {
        int32_t tam_next = 0;

        #pragma omp parallel
        {
            int32_t next_local[LOCAL_BUFFER_SIZE];
            int32_t next_local_cnt = 0;

            #pragma omp for
            for (int32_t i = 0; i < tam_frontier; i++) {
                int32_t u = frontier[i];
                for (int32_t k = row_ptr[u]; k < row_ptr[u + 1]; k++) {
                    int32_t w = col_idx[k];
                    if (dist[w] == -1 && __sync_bool_compare_and_swap(&dist[w], -1, dist[u] + 1)) {
                        next_local[next_local_cnt++] = w;
                        if (next_local_cnt == LOCAL_BUFFER_SIZE) {
                            #pragma omp critical
                            {
                                memcpy(&next[tam_next], next_local, (size_t)next_local_cnt * sizeof(int32_t));
                                tam_next += next_local_cnt;
                            }
                            next_local_cnt = 0;
                        }
                    }
                }
            }

            if (next_local_cnt > 0) {
                #pragma omp critical
                {
                    memcpy(&next[tam_next], next_local, (size_t)next_local_cnt * sizeof(int32_t));
                    tam_next += next_local_cnt;
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
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Uso: %s <V> <cenario> [iteracoes] [warmup]\n", argv[0]);
        return 1;
    }

    int32_t v = atoi(argv[1]);
    const char *cenario = argv[2];
    int iteracoes = (argc >= 4) ? atoi(argv[3]) : 5;
    int warmup = (argc == 5) ? atoi(argv[4]) : 1;

    if (strcmp(cenario, "aleatorio") != 0 && strcmp(cenario, "ordenado") != 0 &&
        strcmp(cenario, "invertido") != 0) {
        fprintf(stderr, "Erro: cenario deve ser 'aleatorio', 'ordenado' ou 'invertido' (recebido: %s)\n", cenario);
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

    omp_set_num_threads(NUM_THREADS);

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
    bfs_seq(row_ptr, col_idx, v, 0, dist_ref, frontier, next);
    free(col_idx);
    col_idx = NULL;

    /* Execucoes de warmup (default 1), sem saida */
    for (int w = 0; w < warmup; w++) {
        gerar_grafo_csr(v, row_ptr, &col_idx, cenario);
        bfs_openmp(row_ptr, col_idx, v, 0, dist, frontier, next);
        free(col_idx);
        col_idx = NULL;
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 5; exec++) {
            gerar_grafo_csr(v, row_ptr, &col_idx, cenario);

            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            bfs_openmp(row_ptr, col_idx, v, 0, dist, frontier, next);
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

        double tempo_s = soma / 5.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("bfs|openmp|%s|%d|%d|%s|%d|%dx1\n",
               cenario, v, iter, tempo_str, corretude, omp_get_max_threads());
        fflush(stdout);
    }

    free(row_ptr);
    free(dist);
    free(dist_ref);
    free(frontier);
    free(next);
    return 0;
}
