/* mergesort_artigo_openmp.c
 * Implementacao fiel ao Algoritmo 6 do apendice de Nogueira et al. (SSCAD 2024).
 * Divisao recursiva com #pragma omp task; conquista via selection sort O(n^2).
 * Uso: ./mergesort_artigo_openmp [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define CSV_PATH  "results/benchmark.csv"
#define LOG_PATH  "results/benchmark.log"
#define RAPL_PATH "/sys/class/powercap/intel-rapl:0/energy_uj"

typedef enum { ENERGIA_PERF, ENERGIA_RAPL, ENERGIA_NONE } MetodoEnergia;

static char         hardware_label[32]  = "desconhecido";
static MetodoEnergia metodo_energia     = ENERGIA_NONE;
static int          energia_warn_feito  = 0;

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
    /* verificar Jetson */
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

/* ---------- corretude ---------- */

static int verificar_corretude(const int *v, int n) {
    for (int i = 0; i < n - 1; i++)
        if (v[i] > v[i + 1]) return 0;
    return 1;
}

/* ---------- algoritmo ---------- */

/* etapa de conquista: selection sort O(n^2) sobre tmp com indices absolutos
 * (indices absolutos evitam corrida quando tasks paralelas compartilham tmp) */
static void selection_sort(int *vetor, int inicio, int fim, int separar, int *tmp) {
    (void)separar; /* parametro da interface do artigo; nao usado pelo selection sort */
    for (int k = inicio; k <= fim; k++) tmp[k] = vetor[k];
    for (int i = inicio; i < fim; i++) {
        int min_idx = i;
        for (int j = i + 1; j <= fim; j++)
            if (tmp[j] < tmp[min_idx]) min_idx = j;
        if (min_idx != i) {
            int t = tmp[i]; tmp[i] = tmp[min_idx]; tmp[min_idx] = t;
        }
    }
    for (int k = inicio; k <= fim; k++) vetor[k] = tmp[k];
}

/* divisao recursiva com tasks OpenMP — Algoritmo 6 do apendice */
static void mergesort(int *vetor, int inicio, int fim, int *tmp) {
    if (inicio < fim) {
        int separar = (inicio + fim) / 2;
        #pragma omp task
        mergesort(vetor, inicio, separar, tmp);
        #pragma omp task
        mergesort(vetor, separar + 1, fim, tmp);
        #pragma omp taskwait
        selection_sort(vetor, inicio, fim, separar, tmp);
    }
}

/* ---------- modos ---------- */

static void modo_teste(void) {
    int N = 100;
    int *original = malloc(N * sizeof(int));
    int *trabalho = malloc(N * sizeof(int));
    int *tmp      = malloc(N * sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) original[i] = rand() % 100000;

    for (int exec = 1; exec <= 2; exec++) {
        memcpy(trabalho, original, N * sizeof(int));
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        #pragma omp parallel
        {
            #pragma omp single
            mergesort(trabalho, 0, N - 1, tmp);
        }
        gettimeofday(&t1, NULL);
        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
        const char *corr = verificar_corretude(trabalho, N) ? "OK" : "FAIL";
        printf("[TESTE] mergesort openmp N=%d exec=%d tempo=%.6fs corretude=%s\n",
               N, exec, tempo, corr);
    }

    free(original); free(trabalho); free(tmp);
}

static void modo_benchmark(void) {
    const int tamanhos[]  = {100, 10000, 100000};
    const int n_tamanhos  = 3;
    const int N_ITER      = 5;
    const int N_REPS      = 50;

    for (int ti = 0; ti < n_tamanhos; ti++) {
        int N = tamanhos[ti];

        int *original = malloc(N * sizeof(int));
        int *trabalho = malloc(N * sizeof(int));
        int *tmp      = malloc(N * sizeof(int));

        srand(42);
        for (int i = 0; i < N; i++) original[i] = rand() % 100000;

        /* warmup */
        memcpy(trabalho, original, N * sizeof(int));
        #pragma omp parallel
        {
            #pragma omp single
            mergesort(trabalho, 0, N - 1, tmp);
        }

        sleep(30);

        for (int iter = 1; iter <= N_ITER; iter++) {
            double      tempo_acum   = 0.0;
            long long   rapl_antes   = -1LL;
            long long   rapl_depois  = -1LL;
            int         corretude_ok = 1;

            for (int rep = 0; rep < N_REPS; rep++) {
                memcpy(trabalho, original, N * sizeof(int));

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_antes = rapl_ler();

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
                #pragma omp parallel
                {
                    #pragma omp single
                    mergesort(trabalho, 0, N - 1, tmp);
                }
                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corretude_ok = verificar_corretude(trabalho, N);
            }

            double tempo_medio = tempo_acum / N_REPS;

            char ecpu[32] = "NA";
            if (metodo_energia != ENERGIA_NONE
                    && rapl_antes >= 0 && rapl_depois >= 0) {
                double joules = (double)(rapl_depois - rapl_antes) / 1e6;
                double_para_str(ecpu, sizeof(ecpu), joules);
            }

            const char *corr_str = corretude_ok ? "OK" : "FAIL";

            csv_append("mergesort", "openmp", "artigo", hardware_label,
                       N, iter, tempo_medio, "NA", ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "mergesort openmp %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }

            if (!corretude_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                         "FAIL mergesort openmp N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        free(original); free(trabalho); free(tmp);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg), "inicio mergesort openmp hardware=%s", hardware_label);
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
