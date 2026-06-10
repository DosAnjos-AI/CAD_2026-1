/* bfs_artigo_openmp.c
 * Implementacao fiel ao Algoritmo 8 do apendice de Nogueira et al. (SSCAD 2024).
 * Vertex-centric com #pragma omp parallel for; escrita idempotente em valueChange.
 * Uso: ./bfs_artigo_openmp [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <omp.h>

#define CSV_PATH  "results/benchmark.csv"
#define LOG_PATH  "results/benchmark.log"
#define RAPL_PATH "/sys/class/powercap/intel-rapl:0/energy_uj"

typedef enum { ENERGIA_PERF, ENERGIA_RAPL, ENERGIA_NONE } MetodoEnergia;

static char          hardware_label[32] = "desconhecido";
static MetodoEnergia metodo_energia     = ENERGIA_NONE;
static int           energia_warn_feito = 0;

/* ---------- hardware ---------- */

static void identificar_hardware(void) {
    FILE *fp = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (!fp) return;
    char buf[256] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        if      (strstr(buf, "MX350")) snprintf(hardware_label, sizeof(hardware_label), "mx350");
        else if (strstr(buf, "4090"))  snprintf(hardware_label, sizeof(hardware_label), "rtx4090");
    }
    pclose(fp);
    FILE *f = fopen("/proc/device-tree/model", "r");
    if (f) {
        char model[256] = {0};
        if (fgets(model, sizeof(model), f) && strstr(model, "Jetson"))
            snprintf(hardware_label, sizeof(hardware_label), "jetson");
        fclose(f);
    }
}

/* ---------- energia ---------- */

static MetodoEnergia detectar_energia(void) {
    if (system("perf stat -e power/energy-pkg/ -- sleep 0 2>/dev/null") == 0)
        return ENERGIA_PERF;
    FILE *f = fopen(RAPL_PATH, "r");
    if (f) { fclose(f); return ENERGIA_RAPL; }
    return ENERGIA_NONE;
}

static long long rapl_ler(void) {
    FILE *f = fopen(RAPL_PATH, "r");
    if (!f) return -1LL;
    long long val = 0;
    if (fscanf(f, "%lld", &val) != 1) val = -1LL;
    fclose(f);
    return val;
}

/* ---------- log ---------- */

static void log_msg(const char *nivel, const char *msg) {
    FILE *f = fopen(LOG_PATH, "a");
    if (!f) return;
    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm_info);
    fprintf(f, "[%-5s] %s %s\n", nivel, ts, msg);
    fclose(f);
}

/* ---------- CSV ---------- */

static void double_para_str(char *buf, size_t n, double val) {
    snprintf(buf, n, "%.6f", val);
    for (size_t i = 0; i < n && buf[i]; i++)
        if (buf[i] == '.') buf[i] = ',';
}

static int csv_linha_existe(const char *algo, const char *api, const char *versao,
                             const char *hw, int tam, int iter) {
    FILE *f = fopen(CSV_PATH, "r");
    if (!f) return 0;
    char linha[512], chave[256];
    snprintf(chave, sizeof(chave), "%s|%s|%s|%s|%d|%d|", algo, api, versao, hw, tam, iter);
    int achou = 0;
    while (fgets(linha, sizeof(linha), f))
        if (strncmp(linha, chave, strlen(chave)) == 0) { achou = 1; break; }
    fclose(f);
    return achou;
}

static void csv_append(const char *algo, const char *api, const char *versao,
                        const char *hw, int tam, int iter, double tempo,
                        const char *egpu, const char *ecpu,
                        const char *corretude) {
    if (csv_linha_existe(algo, api, versao, hw, tam, iter)) return;
    FILE *f = fopen(CSV_PATH, "a");
    if (!f) return;
    char tempo_str[32];
    double_para_str(tempo_str, sizeof(tempo_str), tempo);
    fprintf(f, "%s|%s|%s|%s|%d|%d|%s|%s|%s|%s\n",
            algo, api, versao, hw, tam, iter, tempo_str, egpu, ecpu, corretude);
    fclose(f);
}

/* ---------- geracao do grafo ---------- */

/* grafo aleatorio com N nos e E arestas; sem duplicatas, sem self-loops */
static void gerar_grafo(int N, int E, int **adj_out, int **offset_out, int **size_out) {
    /* contagem de grau por no */
    int *grau = (int *)calloc(N, sizeof(int));

    /* gerar arestas sem duplicatas usando conjunto simples de pares */
    int  *us   = (int *)malloc(E * sizeof(int));
    int  *vs   = (int *)malloc(E * sizeof(int));
    int   cont = 0;

    srand(42);
    while (cont < E) {
        int u = rand() % N;
        int v = rand() % N;
        if (u == v) continue;
        /* verificar duplicata nos ultimos inseridos (aproximado mas suficiente) */
        int dup = 0;
        for (int k = cont - 1; k >= 0 && k >= cont - 20; k--)
            if (us[k] == u && vs[k] == v) { dup = 1; break; }
        if (dup) continue;
        us[cont] = u;
        vs[cont] = v;
        grau[u]++;
        cont++;
    }

    /* calcular offsets */
    int *offset = (int *)malloc(N * sizeof(int));
    int *size   = (int *)malloc(N * sizeof(int));
    offset[0] = 0;
    for (int i = 1; i < N; i++) offset[i] = offset[i-1] + grau[i-1];
    for (int i = 0; i < N; i++) size[i] = grau[i];

    /* preencher lista de adjacencia */
    int *adj = (int *)malloc(E * sizeof(int));
    int *pos = (int *)calloc(N, sizeof(int));
    for (int k = 0; k < E; k++) {
        int u = us[k];
        adj[offset[u] + pos[u]] = vs[k];
        pos[u]++;
    }

    free(grau); free(us); free(vs); free(pos);
    *adj_out    = adj;
    *offset_out = offset;
    *size_out   = size;
}

/* ---------- bfs sequencial de referencia ---------- */

static void bfs_sequencial(int N, int *adj, int *offset, int *size, int *dist_ref) {
    for (int i = 0; i < N; i++) dist_ref[i] = INT_MAX;
    dist_ref[0] = 0;

    int *fila = (int *)malloc(N * sizeof(int));
    int head = 0, tail = 0;
    fila[tail++] = 0;

    while (head < tail) {
        int u = fila[head++];
        for (int i = offset[u]; i < offset[u] + size[u]; i++) {
            int v = adj[i];
            if (dist_ref[v] == INT_MAX) {
                dist_ref[v] = dist_ref[u] + 1;
                fila[tail++] = v;
            }
        }
    }
    free(fila);
}

/* ---------- bfs artigo (Algoritmo 8) ---------- */

static void bfs_openmp(int N, int *adj, int *offset, int *size, int *distancia) {
    for (int i = 0; i < N; i++) distancia[i] = INT_MAX;
    distancia[0] = 0;

    int changed = 1;
    int nivel   = 0;
    while (changed == 1) {
        int valueChange = 0;
        changed = 0;
        #pragma omp parallel for shared(changed, valueChange)
        for (int j = 0; j < N; j++) {
            if (distancia[j] == nivel) {
                for (int i = offset[j]; i < offset[j] + size[j]; i++) {
                    int v = adj[i];
                    if (nivel + 1 < distancia[v]) {
                        distancia[v] = nivel + 1;
                        valueChange  = 1;
                    }
                }
            }
            if (valueChange)
                changed = valueChange;
        }
        nivel++;
    }
}

/* ---------- corretude ---------- */

static int verificar_corretude(int N, int *dist, int *dist_ref) {
    for (int i = 0; i < N; i++)
        if (dist[i] != dist_ref[i]) return 0;
    return 1;
}

/* ---------- modos ---------- */

static void modo_teste(void) {
    int N = 10000, E = 30000;
    int *adj, *offset, *size;
    gerar_grafo(N, E, &adj, &offset, &size);

    int *distancia = (int *)malloc(N * sizeof(int));
    int *dist_ref  = (int *)malloc(N * sizeof(int));
    bfs_sequencial(N, adj, offset, size, dist_ref);

    for (int exec = 1; exec <= 2; exec++) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        bfs_openmp(N, adj, offset, size, distancia);
        gettimeofday(&t1, NULL);
        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
        const char *corr = verificar_corretude(N, distancia, dist_ref) ? "OK" : "FAIL";
        printf("[TESTE] bfs openmp N=%d E=%d exec=%d tempo=%.6fs corretude=%s\n",
               N, E, exec, tempo, corr);
    }

    free(adj); free(offset); free(size); free(distancia); free(dist_ref);
}

static void modo_benchmark(void) {
    const int nos[]    = {10000,   100000,   500000};
    const int arestas[] = {30000,  300000,  1000000};
    const int n_tam    = 3;
    const int N_ITER   = 5;
    const int N_REPS   = 50;

    for (int ti = 0; ti < n_tam; ti++) {
        int N = nos[ti];
        int E = arestas[ti];

        int *adj, *offset, *size;
        gerar_grafo(N, E, &adj, &offset, &size);

        int *distancia = (int *)malloc(N * sizeof(int));
        int *dist_ref  = (int *)malloc(N * sizeof(int));
        bfs_sequencial(N, adj, offset, size, dist_ref);

        /* warmup */
        bfs_openmp(N, adj, offset, size, distancia);

        sleep(30);

        for (int iter = 1; iter <= N_ITER; iter++) {
            double    tempo_acum  = 0.0;
            long long rapl_antes  = -1LL;
            long long rapl_depois = -1LL;
            int       corr_ok     = 1;

            for (int rep = 0; rep < N_REPS; rep++) {
                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_antes = rapl_ler();

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
                bfs_openmp(N, adj, offset, size, distancia);
                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corr_ok = verificar_corretude(N, distancia, dist_ref);
            }

            double tempo_medio = tempo_acum / N_REPS;

            char ecpu[32] = "NA";
            if (metodo_energia != ENERGIA_NONE
                    && rapl_antes >= 0 && rapl_depois >= 0) {
                double joules = (double)(rapl_depois - rapl_antes) / 1e6;
                double_para_str(ecpu, sizeof(ecpu), joules);
            }

            const char *corr_str = corr_ok ? "OK" : "FAIL";
            csv_append("bfs", "openmp", "artigo", hardware_label,
                       N, iter, tempo_medio, "NA", ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "bfs openmp %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }
            if (!corr_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg), "FAIL bfs openmp N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        free(adj); free(offset); free(size); free(distancia); free(dist_ref);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg), "inicio bfs openmp hardware=%s", hardware_label);
        log_msg("INFO", msg);
    }
    if (!energia_warn_feito) {
        char msg[64];
        snprintf(msg, sizeof(msg), "metodo energia CPU selecionado: %s",
                 metodo_energia == ENERGIA_PERF ? "perf (RAPL)" :
                 metodo_energia == ENERGIA_RAPL ? "RAPL"        : "NONE");
        log_msg("WARN", msg);
        energia_warn_feito = 1;
    }
    log_msg("WARN", "energia GPU: MX350 sem sensor INA, registrando NA");

    if (teste)
        modo_teste();
    else
        modo_benchmark();

    return 0;
}
