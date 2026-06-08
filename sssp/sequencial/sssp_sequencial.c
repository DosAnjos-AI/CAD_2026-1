/* sssp_sequencial.c
 * Bellman-Ford sequencial puro — baseline sem paralelismo.
 * Uso: ./sssp_sequencial [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>

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

static void gerar_grafo(int N, int E, int *origens, int *destinos, int *custos) {
    int cont = 0;
    srand(42);
    while (cont < E) {
        int u = rand() % N;
        int v = rand() % N;
        if (u == v) continue;
        int dup = 0;
        for (int k = cont - 1; k >= 0 && k >= cont - 20; k--)
            if (origens[k] == u && destinos[k] == v) { dup = 1; break; }
        if (dup) continue;
        origens[cont]  = u;
        destinos[cont] = v;
        custos[cont]   = rand() % 100 + 1;
        cont++;
    }
}

/* ---------- sssp: Bellman-Ford sequencial ---------- */

static void sssp(int N, int E, int *origens, int *destinos, int *custos, int *dist) {
    for (int i = 0; i < N; i++) dist[i] = INT_MAX;
    dist[0] = 0;
    for (int k = 0; k < N - 1; k++) {
        int atualizado = 0;
        for (int i = 0; i < E; i++) {
            int o = origens[i], d = destinos[i], c = custos[i];
            if (dist[o] != INT_MAX && dist[o] + c < dist[d]) {
                dist[d]    = dist[o] + c;
                atualizado = 1;
            }
        }
        if (!atualizado) break;
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
    int N = 1000, E = 4000;

    int *origens  = (int *)malloc(E * sizeof(int));
    int *destinos = (int *)malloc(E * sizeof(int));
    int *custos   = (int *)malloc(E * sizeof(int));
    gerar_grafo(N, E, origens, destinos, custos);

    int *dist     = (int *)malloc(N * sizeof(int));
    int *dist_ref = (int *)malloc(N * sizeof(int));
    /* referencia: mesma funcao, mesma entrada — baseline deterministico */
    sssp(N, E, origens, destinos, custos, dist_ref);

    for (int exec = 1; exec <= 2; exec++) {
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        sssp(N, E, origens, destinos, custos, dist);
        gettimeofday(&t1, NULL);
        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
        const char *corr = verificar_corretude(N, dist, dist_ref) ? "OK" : "FAIL";
        printf("[TESTE] sssp sequencial N=%d E=%d exec=%d tempo=%.6fs corretude=%s\n",
               N, E, exec, tempo, corr);
    }

    free(origens); free(destinos); free(custos); free(dist); free(dist_ref);
}

static void modo_benchmark(void) {
    const int nos[]     = {1000,   10000,  100000,  200000};
    const int arestas[] = {4000,   30000,  300000,  400000};
    const int n_tam     = 4;
    const int N_ITER    = 25;
    const int N_REPS    = 400;

    for (int ti = 0; ti < n_tam; ti++) {
        int N = nos[ti];
        int E = arestas[ti];

        int *origens  = (int *)malloc(E * sizeof(int));
        int *destinos = (int *)malloc(E * sizeof(int));
        int *custos   = (int *)malloc(E * sizeof(int));
        gerar_grafo(N, E, origens, destinos, custos);

        int *dist     = (int *)malloc(N * sizeof(int));
        int *dist_ref = (int *)malloc(N * sizeof(int));
        sssp(N, E, origens, destinos, custos, dist_ref);

        /* warmup */
        sssp(N, E, origens, destinos, custos, dist);

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
                sssp(N, E, origens, destinos, custos, dist);
                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corr_ok = verificar_corretude(N, dist, dist_ref);
            }

            double tempo_medio = tempo_acum / N_REPS;

            char ecpu[32] = "NA";
            if (metodo_energia != ENERGIA_NONE
                    && rapl_antes >= 0 && rapl_depois >= 0) {
                double joules = (double)(rapl_depois - rapl_antes) / 1e6;
                double_para_str(ecpu, sizeof(ecpu), joules);
            }

            const char *corr_str = corr_ok ? "OK" : "FAIL";
            csv_append("sssp", "sequencial", "sequencial", hardware_label,
                       N, iter, tempo_medio, "NA", ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "sssp sequencial %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }
            if (!corr_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg), "FAIL sssp sequencial N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        free(origens); free(destinos); free(custos); free(dist); free(dist_ref);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg), "inicio sssp sequencial hardware=%s", hardware_label);
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
    log_msg("WARN", "energia GPU: sem GPU utilizada, registrando NA");

    if (teste)
        modo_teste();
    else
        modo_benchmark();

    return 0;
}
