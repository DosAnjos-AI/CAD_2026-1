/* mergesort_otimizado_cuda_dp.cu
 * Versao otimizada do mergesort CUDA DP.
 * Base: JoeyOhman/GPUMergeSort (parallel.cu)
 * OPT-1: merge sequencial O(n) no ramo else (substitui insertion_sort O(n^2) do artigo)
 * OPT-2: ping-pong de ponteiros (elimina cudaMemcpy por iteracao do loop bottom-up)
 * OPT-3: numThreadsPerBlock=128 no loop externo (era 32 no repositorio base)
 * Compilar:
 *   nvcc -O2 -arch=sm_61 -ccbin g++-12 -rdc=true \
 *        -DCUDA_FORCE_CDP1_IF_SUPPORTED \
 *        -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
 *        -lcudadevrt mergesort_otimizado_cuda_dp.cu -o mergesort_otimizado_cuda_dp
 * Uso: ./mergesort_otimizado_cuda_dp [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
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
    snprintf(chave, sizeof(chave), "%s|%s|%s|%s|%d|%d|",
             algo, api, versao, hw, tam, iter);
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

/* ---------- corretude (host) ---------- */

static int verificar_corretude(const int *v, int n) {
    for (int i = 0; i < n - 1; i++)
        if (v[i] > v[i + 1]) return 0;
    return 1;
}

/* ---------- algoritmo (base: JoeyOhman/GPUMergeSort parallel.cu) ---------- */

/* busca binaria recursiva — retorna posicao de insercao de val em arr[low..high] */
__device__ static int binarySearch(int *arr, int val, int low, int high) {
    if (high <= low)
        return (val > arr[low]) ? (low + 1) : low;
    int mid = (low + high) / 2;
    if (val > arr[mid])
        return binarySearch(arr, val, mid + 1, high);
    return binarySearch(arr, val, low, mid);
}

/* retorna indice de destino do elemento aux[low+ownIndex] apos o merge */
__device__ static int getIndex(int *subAux, int ownIndex, int nLow, int nTot) {
    int scanIndex;
    int upperBound;
    int partOfFirstArr = (ownIndex < nLow);

    if (partOfFirstArr) {
        scanIndex  = nLow;
        upperBound = nTot;
    } else {
        scanIndex  = 0;
        upperBound = nLow;
    }
    scanIndex = binarySearch(subAux, subAux[ownIndex], scanIndex, upperBound - 1);
    return ownIndex + scanIndex - nLow;
}

/* kernel filho — merge paralelo via busca binaria
 * arr = destino, aux = fonte */
__global__ void mergeKernel(int *arr, int *aux, int low, int mid, int high) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int nLow  = mid - low + 1;
    int nHigh = high - mid;
    int nTot  = nLow + nHigh;

    if (idx >= nTot) return;

    int arrIndex = getIndex(&aux[low], idx, nLow, nTot);
    arr[low + arrIndex] = aux[low + idx];
}

/* OPT-1: merge sequencial O(n) — substitui insertion_sort O(n^2) do artigo
 * le de aux[low..high], escreve resultado em arr[low..high] */
__device__ static void merge(int *arr, int *aux, int low, int mid, int high) {
    int i = 0, j = 0;
    int mergedIndex = low;
    int nLow  = mid - low + 1;
    int nHigh = high - mid;

    while (i < nLow && j < nHigh) {
        if (aux[low + i] < aux[mid + 1 + j]) {
            arr[mergedIndex] = aux[low + i];
            i++;
        } else {
            arr[mergedIndex] = aux[mid + 1 + j];
            j++;
        }
        mergedIndex++;
    }
    while (i < nLow)  { arr[mergedIndex++] = aux[low     + i++]; }
    while (j < nHigh) { arr[mergedIndex++] = aux[mid + 1 + j++]; }
}

/* kernel principal — lanca mergeKernel (DP) ou merge sequencial por subproblema
 * arr = destino, aux = fonte (padrao ping-pong, OPT-2) */
__global__ void mergeSort(int *arr, int *aux, int currentSize, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int low = idx * width;

    /* skipa apenas quando completamente fora dos limites */
    if (low >= n || low < 0) return;

    /* clampear mid ao final do array — caso sem metade direita, merge apenas copia */
    int mid  = min(low + currentSize - 1, n - 1);
    int high = min(low + width - 1, n - 1);
    int nTot = high - low + 1;

    if (nTot > 16384) {
        int numTPB    = 1024;
        int numBlocks = (nTot + numTPB - 1) / numTPB;
        mergeKernel<<<numBlocks, numTPB>>>(arr, aux, low, mid, high);
        cudaDeviceSynchronize();
    } else {
        /* OPT-1: merge O(n) no lugar de insertion_sort O(n^2) */
        merge(arr, aux, low, mid, high);
    }
}

/* ---------- modos ---------- */

static void modo_teste(void) {
    int N = 100;
    int *data, *buf;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&buf,  N * sizeof(int));

    int *original = (int *)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) original[i] = rand() % 100000;

    if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(0);
    else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(0);

    for (int exec = 1; exec <= 2; exec++) {
        memcpy(data, original, N * sizeof(int));

        /* OPT-2: ping-pong — A=fonte, B=destino; troca a cada nivel */
        int *A = data, *B = buf;
        for (int currentSize = 1; currentSize < N; currentSize *= 2) {
            int width    = currentSize * 2;
            int numSorts = (N + width - 1) / width;
            /* OPT-3: 128 threads por bloco (era 32) */
            int numTPB   = 128;
            int numBlocks = (numSorts + numTPB - 1) / numTPB;
            mergeSort<<<numBlocks, numTPB>>>(B, A, currentSize, N, width);
            cudaDeviceSynchronize();
            int *tmp = A; A = B; B = tmp;
        }
        /* apos ultimo swap, resultado esta em A */
        const char *corr = verificar_corretude(A, N) ? "OK" : "FAIL";
        printf("[TESTE] mergesort cuda_dp_otimizado N=%d exec=%d corretude=%s\n",
               N, exec, corr);
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

    free(original);
    cudaFree(data);
    cudaFree(buf);
}

static void modo_benchmark(void) {
    const int tamanhos[] = {100, 10000, 100000};
    const int n_tamanhos = 3;
    const int N_ITER     = 25;
    const int N_REPS     = 400;

    for (int ti = 0; ti < n_tamanhos; ti++) {
        int N = tamanhos[ti];

        int *data, *buf;
        cudaMallocManaged(&data, N * sizeof(int));
        cudaMallocManaged(&buf,  N * sizeof(int));

        int *original = (int *)malloc(N * sizeof(int));
        srand(42);
        for (int i = 0; i < N; i++) original[i] = rand() % 100000;

        /* warmup — 1 iteracao descartada */
        {
            memcpy(data, original, N * sizeof(int));
            int *A = data, *B = buf;
            for (int currentSize = 1; currentSize < N; currentSize *= 2) {
                int width    = currentSize * 2;
                int numSorts = (N + width - 1) / width;
                int numTPB   = 128;
                int numBlocks = (numSorts + numTPB - 1) / numTPB;
                mergeSort<<<numBlocks, numTPB>>>(B, A, currentSize, N, width);
                cudaDeviceSynchronize();
                int *tmp = A; A = B; B = tmp;
            }
        }

        sleep(30);

        for (int iter = 1; iter <= N_ITER; iter++) {
            double    tempo_acum  = 0.0;
            long long rapl_antes  = -1LL;
            long long rapl_depois = -1LL;
            int       corr_ok     = 1;

            if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(iter);
            else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(iter);

            for (int rep = 0; rep < N_REPS; rep++) {
                memcpy(data, original, N * sizeof(int));

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_antes = rapl_ler();

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);

                /* OPT-2: ping-pong — A=fonte, B=destino; troca a cada nivel */
                int *A = data, *B = buf;
                for (int currentSize = 1; currentSize < N; currentSize *= 2) {
                    int width    = currentSize * 2;
                    int numSorts = (N + width - 1) / width;
                    int numTPB   = 128;
                    int numBlocks = (numSorts + numTPB - 1) / numTPB;
                    mergeSort<<<numBlocks, numTPB>>>(B, A, currentSize, N, width);
                    cudaDeviceSynchronize();
                    int *tmp = A; A = B; B = tmp;
                }
                /* apos ultimo swap, resultado esta em A */

                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec  - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corr_ok = verificar_corretude(A, N);
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
            csv_append("mergesort", "cuda_dp", "otimizado", hardware_label,
                       N, iter, tempo_medio, egpu, ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "mergesort cuda_dp_otimizado %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }
            if (!corr_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                         "FAIL mergesort cuda_dp_otimizado N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        free(original);
        cudaFree(data);
        cudaFree(buf);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "inicio mergesort cuda_dp_otimizado hardware=%s", hardware_label);
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
