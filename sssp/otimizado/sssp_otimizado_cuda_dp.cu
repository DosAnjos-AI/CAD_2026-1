/* sssp_otimizado_cuda_dp.cu
 * Versao otimizada do Algoritmo 4 de Nogueira et al. (SSCAD 2024).
 * OPT-1: kernel secundario lancado com numTPB=128 e numBlocks proporcional
 * a numEdges, em vez de <<<1,1>>> (artigo) ou <<<numBlocks,256>>> (artigo impl).
 * CUDA Dynamic Parallelism: kernel principal lanca kernel secundario via DP.
 * Lista de arestas; memoria unificada (cudaMallocManaged).
 * NOTA: startId = threadId * numEdges conforme pseudocodigo — apenas thread 0
 * processa arestas no artigo. Fiel ao Alg. 4 exceto pelo lancamento do filho.
 * NOTA: __syncthreads() no kernel principal exige que todos os threads do bloco
 * cheguem ao ponto de sincronizacao; por isso, usa-se clamping em vez de return
 * antecipado para threads sem trabalho (early return causaria deadlock).
 * Uso: ./sssp_otimizado_cuda_dp [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#define CSV_PATH  "results/benchmark.csv"
#define LOG_PATH  "results/benchmark.log"
#define RAPL_PATH "/sys/class/powercap/intel-rapl:0/energy_uj"

typedef enum { ENERGIA_PERF, ENERGIA_RAPL, ENERGIA_NONE } MetodoEnergia;

static char          hardware_label[32] = "desconhecido";
static MetodoEnergia metodo_energia     = ENERGIA_NONE;
static int           energia_warn_feito = 0;

typedef enum { GPU_ENERGIA_NVML, GPU_ENERGIA_TEGRA, GPU_ENERGIA_NONE } MetodoEnergiaGPU;

static MetodoEnergiaGPU metodo_energia_gpu = GPU_ENERGIA_NONE;
static int              gpu_warn_feito     = 0;
static char             dmon_logfile[64];
static char             tegra_logfile[64];

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

/* ---------- energia GPU ---------- */

static MetodoEnergiaGPU detectar_energia_gpu(void) {
    FILE *fp = popen("nvidia-smi dmon -s p -d 1 -c 1 2>/dev/null", "r");
    if (fp) {
        char buf[256];
        while (fgets(buf, sizeof(buf), fp)) {
            if (buf[0] == ' ' && buf[1] >= '0' && buf[1] <= '9') {
                char *tok = strtok(buf, " ");
                tok = strtok(NULL, " ");
                if (tok && tok[0] != '-') {
                    pclose(fp);
                    return GPU_ENERGIA_NVML;
                }
                break;
            }
        }
        pclose(fp);
    }
    FILE *f = fopen("/proc/device-tree/model", "r");
    if (f) {
        char model[256] = {0};
        if (fgets(model, sizeof(model), f) && strstr(model, "Jetson")) {
            fclose(f);
            return GPU_ENERGIA_TEGRA;
        }
        fclose(f);
    }
    return GPU_ENERGIA_NONE;
}

static void gpu_nvml_iniciar(int iter) {
    snprintf(dmon_logfile, sizeof(dmon_logfile),
             "/tmp/dmon_%d_%d.log", (int)getpid(), iter);
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi dmon -s p -d 100 > %s 2>/dev/null &", dmon_logfile);
    system(cmd);
    usleep(200000);
}

static double gpu_nvml_parar(void) {
    system("pkill -f 'nvidia-smi dmon' 2>/dev/null");
    usleep(100000);
    FILE *f = fopen(dmon_logfile, "r");
    if (!f) return -1.0;
    double energia_j = 0.0;
    char linha[256];
    while (fgets(linha, sizeof(linha), f)) {
        if (linha[0] == '#' || linha[0] == '\n') continue;
        int gpu_id;
        double pwr_w;
        if (sscanf(linha, "%d %lf", &gpu_id, &pwr_w) == 2 && pwr_w > 0)
            energia_j += pwr_w * 0.1;
    }
    fclose(f);
    remove(dmon_logfile);
    return energia_j;
}

static void gpu_tegra_iniciar(int iter) {
    snprintf(tegra_logfile, sizeof(tegra_logfile),
             "/tmp/tegra_%d_%d.log", (int)getpid(), iter);
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "tegrastats --interval 100 --logfile %s &", tegra_logfile);
    system(cmd);
    usleep(200000);
}

static double gpu_tegra_parar(void) {
    system("pkill -f tegrastats 2>/dev/null");
    usleep(100000);
    FILE *f = fopen(tegra_logfile, "r");
    if (!f) return -1.0;
    double energia_j = 0.0;
    char linha[1024];
    while (fgets(linha, sizeof(linha), f)) {
        char *ptr = strstr(linha, "VDD_GPU_SOC");
        if (!ptr) ptr = strstr(linha, "GPU");
        if (ptr) {
            double mw = 0.0;
            if (sscanf(ptr + 4, "%lf", &mw) == 1 && mw > 0)
                energia_j += (mw / 1000.0) * 0.1;
        }
    }
    fclose(f);
    remove(tegra_logfile);
    return energia_j;
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

/* ---------- sssp sequencial de referencia (Bellman-Ford) ---------- */

static void sssp_sequencial(int N, int E, int *origens, int *destinos, int *custos,
                             int *dist_ref) {
    for (int i = 0; i < N; i++) dist_ref[i] = INT_MAX;
    dist_ref[0] = 0;
    for (int k = 0; k < N - 1; k++) {
        int atualizado = 0;
        for (int i = 0; i < E; i++) {
            int o = origens[i], d = destinos[i], c = custos[i];
            if (dist_ref[o] != INT_MAX && dist_ref[o] + c < dist_ref[d]) {
                dist_ref[d] = dist_ref[o] + c;
                atualizado  = 1;
            }
        }
        if (!atualizado) break;
    }
}

/* ---------- kernels SSSP com Dynamic Parallelism ---------- */

/* kernel secundario — mesma logica de relaxacao; sem lancamentos recursivos */
__global__ void sssp_kernel_secundario(int numEdges, int *origens, int *destinos,
                                        int *custos, int *dist, int *preNode) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startId  = threadId * numEdges;
    int endId    = (threadId + 1) * numEdges;

    if (startId >= numEdges) return;
    if (endId   >  numEdges) endId = numEdges;

    for (int i = startId; i < endId; i++) {
        int origem  = origens[i];
        int destino = destinos[i];
        int custo   = custos[i];
        if (dist[origem] != INT_MAX && dist[origem] + custo < dist[destino]) {
            dist[destino]    = dist[origem] + custo;
            preNode[destino] = origem;
        }
    }
}

/* kernel principal — detecta mudanca e lanca kernel secundario via DP;
 * usa clamping em vez de return antecipado para nao causar deadlock no
 * __syncthreads() que precede o lancamento do filho.
 * OPT-1: kernel secundario lancado com numTPB=128 e numBlocks proporcional
 * a numEdges, maximizando paralelismo no filho. */
__global__ void sssp_kernel_dp(int numEdges, int *origens, int *destinos,
                                int *custos, int *dist, int *preNode,
                                int *finished) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startId  = threadId * numEdges;
    int endId    = (threadId + 1) * numEdges;

    /* clamping sem early return para preservar __syncthreads() abaixo */
    if (startId >= numEdges) { startId = numEdges; endId = numEdges; }
    else if (endId > numEdges) endId = numEdges;

    int mudou = 0;
    for (int i = startId; i < endId; i++) {
        int origem  = origens[i];
        int destino = destinos[i];
        int custo   = custos[i];
        if (dist[origem] != INT_MAX && dist[origem] + custo < dist[destino]) {
            dist[destino]    = dist[origem] + custo;
            preNode[destino] = origem;
            mudou            = 1;
        }
    }

    /* sincronizacao antes de verificar mudanca */
    __syncthreads();

    /* OPT-1: kernel secundario lancado com paralelismo proporcional a numEdges */
    if (mudou && threadIdx.x == 0 && blockIdx.x == 0) {
        *finished = 0;
        int numTPB    = 128;
        int numBlocks = (numEdges + numTPB - 1) / numTPB;
        sssp_kernel_secundario<<<numBlocks, numTPB>>>(numEdges, origens, destinos,
                                                       custos, dist, preNode);
        cudaDeviceSynchronize();
    }
}

/* ---------- corretude ---------- */

static int verificar_corretude(int N, int *dist, int *dist_ref) {
    for (int i = 0; i < N; i++)
        if (dist[i] != dist_ref[i]) return 0;
    return 1;
}

/* ---------- sssp cuda dp: execucao ---------- */

static void sssp_cuda_dp(int N, int numEdges, int *origens, int *destinos, int *custos,
                          int *dist, int *preNode, int *finished) {
    int numBlocks = (numEdges + 255) / 256;
    *finished = 0;

    while (!(*finished)) {
        *finished = 1;
        sssp_kernel_dp<<<numBlocks, 256>>>(numEdges, origens, destinos, custos,
                                            dist, preNode, finished);
        cudaDeviceSynchronize();
    }
    (void)N;
}

/* ---------- modos ---------- */

static void modo_teste(void) {
    int N = 1000, E = 4000;

    int *origens, *destinos, *custos, *dist, *preNode, *finished;
    cudaMallocManaged(&origens,  E * sizeof(int));
    cudaMallocManaged(&destinos, E * sizeof(int));
    cudaMallocManaged(&custos,   E * sizeof(int));
    cudaMallocManaged(&dist,     N * sizeof(int));
    cudaMallocManaged(&preNode,  N * sizeof(int));
    cudaMallocManaged(&finished, sizeof(int));

    gerar_grafo(N, E, origens, destinos, custos);

    int *dist_ref = (int *)malloc(N * sizeof(int));
    sssp_sequencial(N, E, origens, destinos, custos, dist_ref);

    if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(0);
    else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(0);

    for (int exec = 1; exec <= 2; exec++) {
        for (int i = 0; i < N; i++) { dist[i] = INT_MAX; preNode[i] = -1; }
        dist[0] = 0;

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        sssp_cuda_dp(N, E, origens, destinos, custos, dist, preNode, finished);
        gettimeofday(&t1, NULL);
        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
        const char *corr = verificar_corretude(N, dist, dist_ref) ? "OK" : "FAIL";
        printf("[TESTE] sssp cuda_dp otimizado N=%d E=%d exec=%d tempo=%.6fs corretude=%s\n",
               N, E, exec, tempo, corr);
    }
    {
        char egpu_t[32] = "NA";
        if (metodo_energia_gpu == GPU_ENERGIA_NVML) {
            double e = gpu_nvml_parar();
            if (e >= 0.0) double_para_str(egpu_t, sizeof(egpu_t), e / 2.0);
        } else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) {
            double e = gpu_tegra_parar();
            if (e >= 0.0) double_para_str(egpu_t, sizeof(egpu_t), e / 2.0);
        }
        printf("[TESTE] energia_gpu=%s%s\n", egpu_t, strcmp(egpu_t, "NA") == 0 ? "" : "J");
    }

    cudaFree(origens); cudaFree(destinos); cudaFree(custos);
    cudaFree(dist); cudaFree(preNode); cudaFree(finished);
    free(dist_ref);
}

static void modo_benchmark(void) {
    const int nos[]     = {1000,   10000,  100000,  200000};
    const int arestas[] = {4000,   30000,  300000,  400000};
    const int n_tam     = 4;
    const int N_ITER    = 20;
    const int N_REPS    = 250;

    for (int ti = 0; ti < n_tam; ti++) {
        int N = nos[ti];
        int E = arestas[ti];

        int *origens, *destinos, *custos, *dist, *preNode, *finished;
        cudaMallocManaged(&origens,  E * sizeof(int));
        cudaMallocManaged(&destinos, E * sizeof(int));
        cudaMallocManaged(&custos,   E * sizeof(int));
        cudaMallocManaged(&dist,     N * sizeof(int));
        cudaMallocManaged(&preNode,  N * sizeof(int));
        cudaMallocManaged(&finished, sizeof(int));

        gerar_grafo(N, E, origens, destinos, custos);

        int *dist_ref = (int *)malloc(N * sizeof(int));
        sssp_sequencial(N, E, origens, destinos, custos, dist_ref);

        /* warmup */
        for (int i = 0; i < N; i++) { dist[i] = INT_MAX; preNode[i] = -1; }
        dist[0] = 0;
        sssp_cuda_dp(N, E, origens, destinos, custos, dist, preNode, finished);

        sleep(30);

        for (int iter = 1; iter <= N_ITER; iter++) {
            double    tempo_acum  = 0.0;
            long long rapl_antes  = -1LL;
            long long rapl_depois = -1LL;
            int       corr_ok     = 1;

            if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(iter);
            else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(iter);

            for (int rep = 0; rep < N_REPS; rep++) {
                for (int i = 0; i < N; i++) { dist[i] = INT_MAX; preNode[i] = -1; }
                dist[0] = 0;

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_antes = rapl_ler();

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
                sssp_cuda_dp(N, E, origens, destinos, custos, dist, preNode, finished);
                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corr_ok = verificar_corretude(N, dist, dist_ref);
            }

            double tempo_medio = tempo_acum / N_REPS;

            char egpu[32] = "NA";
            if (metodo_energia_gpu == GPU_ENERGIA_NVML) {
                double e = gpu_nvml_parar();
                if (e >= 0.0) double_para_str(egpu, sizeof(egpu), e / N_REPS);
            } else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) {
                double e = gpu_tegra_parar();
                if (e >= 0.0) double_para_str(egpu, sizeof(egpu), e / N_REPS);
            }

            char ecpu[32] = "NA";
            if (metodo_energia != ENERGIA_NONE
                    && rapl_antes >= 0 && rapl_depois >= 0) {
                double joules = (double)(rapl_depois - rapl_antes) / 1e6;
                double_para_str(ecpu, sizeof(ecpu), joules);
            }

            const char *corr_str = corr_ok ? "OK" : "FAIL";
            csv_append("sssp", "cuda_dp", "otimizado", hardware_label,
                       N, iter, tempo_medio, egpu, ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "sssp cuda_dp otimizado %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }
            if (!corr_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg), "FAIL sssp cuda_dp otimizado N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        cudaFree(origens); cudaFree(destinos); cudaFree(custos);
        cudaFree(dist); cudaFree(preNode); cudaFree(finished);
        free(dist_ref);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg), "inicio sssp cuda_dp otimizado hardware=%s", hardware_label);
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
    metodo_energia_gpu = detectar_energia_gpu();
    if (!gpu_warn_feito) {
        char msg[64];
        snprintf(msg, sizeof(msg), "metodo energia GPU: %s",
                 metodo_energia_gpu == GPU_ENERGIA_NVML  ? "nvidia-smi dmon" :
                 metodo_energia_gpu == GPU_ENERGIA_TEGRA ? "tegrastats"      : "NONE");
        log_msg("WARN", msg);
        gpu_warn_feito = 1;
    }

    if (teste)
        modo_teste();
    else
        modo_benchmark();

    return 0;
}
