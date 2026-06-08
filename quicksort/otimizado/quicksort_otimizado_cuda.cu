/* quicksort_otimizado_cuda.cu
 * Versao otimizada do Quicksort CUDA.
 * Base: repositorio gongminaaa/GPU_Parallel_Computing.
 * OPT-1: dimensionamento dinamico de blocos por iteracao (numThreadsPerBlock=128).
 * Memoria unificada (cudaMallocManaged). Tipo int.
 * Uso: ./quicksort_otimizado_cuda [--teste]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CSV_PATH  "results/benchmark.csv"
#define LOG_PATH  "results/benchmark.log"
#define RAPL_PATH "/sys/class/powercap/intel-rapl:0/energy_uj"

#define PICK_PIVOT      1
#define PIVOT_CANDIDATE 3

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
                        const char *egpu, const char *ecpu, const char *corretude) {
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

/* ---------- kernels — base: gongminaaa/GPU_Parallel_Computing ---------- */

__device__ static void gpuSwap(int *a, int *b) {
    int t = *a; *a = *b; *b = t;
}

/* cada thread compara seu elemento com o pivo e o deposita no buffer de saida */
__global__ void quickSort(int n, int *before_sort_array, int *after_sort_array,
                           int *pivot_queue, int *working_queue, int *pivot_arr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int q_tid = 2 * tid;
    int pid   = 2 * working_queue[q_tid];
    int pivot = before_sort_array[working_queue[q_tid]];
    int value = before_sort_array[tid];

    if (pivot < value) {
        int tail = atomicSub(&pivot_queue[pid + 1], 1);
        after_sort_array[tail] = value;
    } else if (pivot >= value) {
        if (pid != q_tid) {
            int head = atomicAdd(&pivot_queue[pid], 1);
            after_sort_array[head] = value;
        }
    }

    /* elemento responsavel pelo pivo salva o valor em pivot_arr */
    if (pivot == value && pid == q_tid) {
        if (pivot_arr[tid] != pivot)
            pivot_arr[tid] = pivot;
    }
}

/* reorganiza filas, deposita pivos e opcionalmente seleciona melhor pivo */
__global__ void arrangeQueueAndPivot(int *new_pivot_queue, int *old_pivot_queue,
                                      int n, int *sort_array, int *pivot_arr,
                                      int *new_working_queue, int *old_working_queue,
                                      bool *done) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    __shared__ int shared_old_working_queue[128];

    int q_tid = 2 * tid;
    shared_old_working_queue[threadIdx.x] = old_working_queue[q_tid];

    int head           = 2 * shared_old_working_queue[threadIdx.x];
    int my_pivot_index = old_pivot_queue[head];

    /* depositar pivo na posicao correta do vetor ordenado */
    if (pivot_arr[tid] != -1)
        sort_array[my_pivot_index] = pivot_arr[tid];

    /* reorganizar filas para proxima iteracao */
    if (tid == my_pivot_index) {
        new_pivot_queue[q_tid]       = tid;
        new_pivot_queue[q_tid + 1]   = tid;
        new_working_queue[q_tid]     = tid;
        new_working_queue[q_tid + 1] = tid;
    } else if (tid == shared_old_working_queue[threadIdx.x]) {
        new_pivot_queue[q_tid]       = shared_old_working_queue[threadIdx.x];
        new_pivot_queue[q_tid + 1]   = my_pivot_index - 1;
        new_working_queue[q_tid]     = shared_old_working_queue[threadIdx.x];
        new_working_queue[q_tid + 1] = my_pivot_index - 1;
    } else if (tid == my_pivot_index + 1) {
        new_pivot_queue[q_tid]       = my_pivot_index + 1;
        new_pivot_queue[q_tid + 1]   = old_working_queue[head + 1];
        new_working_queue[q_tid]     = my_pivot_index + 1;
        new_working_queue[q_tid + 1] = old_working_queue[head + 1];
    } else {
        if (tid > my_pivot_index) {
            new_pivot_queue[q_tid]       = my_pivot_index + 1;
            new_pivot_queue[q_tid + 1]   = -1;
            new_working_queue[q_tid]     = my_pivot_index + 1;
            new_working_queue[q_tid + 1] = -1;
        } else {
            new_pivot_queue[q_tid]       = shared_old_working_queue[threadIdx.x];
            new_pivot_queue[q_tid + 1]   = -1;
            new_working_queue[q_tid]     = shared_old_working_queue[threadIdx.x];
            new_working_queue[q_tid + 1] = -1;
        }
    }

#if PICK_PIVOT
    /* selecionar melhor pivo (mediana de 3 candidatos) */
    int pid2 = 2 * new_working_queue[q_tid];
    if (pid2 == q_tid) {
        int start = new_working_queue[pid2];
        int end   = new_working_queue[pid2 + 1];
        if (end - start >= PIVOT_CANDIDATE) {
            int pivot_cand1 = sort_array[start];
            int pivot_cand2 = sort_array[(end + start) / 2];
            int pivot_cand3 = sort_array[end];
            for (int i = 0; i < PIVOT_CANDIDATE; i++) {
                int pv = sort_array[new_working_queue[pid2] + i];
                bool found = false;
                switch (i) {
                    case 0:
                        found = ((pv >= pivot_cand2 && pv <= pivot_cand3) ||
                                 (pv >= pivot_cand3 && pv <= pivot_cand2));
                        break;
                    case 1:
                        found = ((pv >= pivot_cand1 && pv <= pivot_cand3) ||
                                 (pv >= pivot_cand3 && pv <= pivot_cand1));
                        break;
                    case 2:
                        found = ((pv >= pivot_cand1 && pv <= pivot_cand2) ||
                                 (pv >= pivot_cand2 && pv <= pivot_cand1));
                        break;
                }
                if (found) {
                    gpuSwap(&sort_array[new_working_queue[pid2] + i],
                            &sort_array[new_working_queue[pid2]]);
                    break;
                }
            }
        }
    }
#endif

    if (new_pivot_queue[q_tid] != new_pivot_queue[q_tid + 1])
        *done = false;
}

/* ---------- auxiliares de execucao ---------- */

static void resetar_estado(int *list1, const int *original, int *pivot_arr,
                            int *q, int *q2, int *q_copy, int *q_copy2,
                            bool *done, int N) {
    memcpy(list1, original, N * sizeof(int));
    for (int i = 0; i < N; i++) pivot_arr[i] = -1;
    q[0] = 0; q[1] = N - 1;
    for (int i = 2; i < 2 * N; i++) q[i] = (i % 2 == 0) ? 0 : -1;
    memcpy(q2,     q, 2 * N * sizeof(int));
    memcpy(q_copy,  q, 2 * N * sizeof(int));
    memcpy(q_copy2, q, 2 * N * sizeof(int));
    *done = false;
}

/*
 * OPT-1: numBlocks calculado dinamicamente a cada iteracao com numThreadsPerBlock=128.
 *
 * Investigacao do queueSize: o repositorio base nao rastreia um contador de fila
 * separado — usa cudaMallocManaged para todos os buffers (list1, list2, q, q_copy,
 * q2, q_copy2, pivot_arr, done). Como todos os N elementos precisam ser processados
 * em cada iteracao (o mecanismo ping-pong exige que cada elemento seja copiado para
 * o buffer corrente a cada passo), queueSize = N e constante. O acesso seria direto
 * no host apos cudaDeviceSynchronize() sem cudaMemcpy adicional (primeira opcao da
 * especificacao). Como queueSize nao varia, nao foi alocado como variavel separada
 * para evitar codigo morto — numBlocks e calculado diretamente de N.
 */
static int *executar_quicksort(int *list1, int *list2,
                                int *q, int *q_copy, int *q2, int *q_copy2,
                                int *pivot_arr, bool *done, int N) {
    const int numThreadsPerBlock = 128;
    int count = 0;

    while (!(*done)) {
        *done = true;

        /* dimensionamento dinamico: queueSize = N neste algoritmo */
        int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
        if (numBlocks < 1) numBlocks = 1;

        if (count % 2 == 0) {
            quickSort<<<numBlocks, numThreadsPerBlock>>>(N, list1, list2, q, q_copy, pivot_arr);
            cudaDeviceSynchronize();
            arrangeQueueAndPivot<<<numBlocks, numThreadsPerBlock>>>(q2, q, N, list2, pivot_arr,
                                                                     q_copy2, q_copy, done);
            cudaDeviceSynchronize();
        } else {
            quickSort<<<numBlocks, numThreadsPerBlock>>>(N, list2, list1, q2, q_copy2, pivot_arr);
            cudaDeviceSynchronize();
            arrangeQueueAndPivot<<<numBlocks, numThreadsPerBlock>>>(q, q2, N, list1, pivot_arr,
                                                                     q_copy, q_copy2, done);
            cudaDeviceSynchronize();
        }
        count++;
    }
    return (count % 2 == 0) ? list1 : list2;
}

/* ---------- modos ---------- */

static void modo_teste(void) {
    int N = 100;

    int  *list1, *list2, *pivot_arr, *q, *q_copy, *q2, *q_copy2;
    bool *done;
    cudaMallocManaged(&list1,     N * sizeof(int));
    cudaMallocManaged(&list2,     N * sizeof(int));
    cudaMallocManaged(&pivot_arr, N * sizeof(int));
    cudaMallocManaged(&q,      2 * N * sizeof(int));
    cudaMallocManaged(&q_copy,  2 * N * sizeof(int));
    cudaMallocManaged(&q2,     2 * N * sizeof(int));
    cudaMallocManaged(&q_copy2, 2 * N * sizeof(int));
    cudaMallocManaged(&done, sizeof(bool));

    int *original = (int *)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) original[i] = rand() % 100000;

    if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(0);
    else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(0);

    for (int exec = 1; exec <= 2; exec++) {
        resetar_estado(list1, original, pivot_arr, q, q2, q_copy, q_copy2, done, N);
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        int *res = executar_quicksort(list1, list2, q, q_copy, q2, q_copy2,
                                      pivot_arr, done, N);
        gettimeofday(&t1, NULL);
        double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
        const char *corr = verificar_corretude(res, N) ? "OK" : "FAIL";
        printf("[TESTE] quicksort cuda otimizado N=%d exec=%d tempo=%.6fs corretude=%s\n",
               N, exec, tempo, corr);
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
    cudaFree(list1); cudaFree(list2); cudaFree(pivot_arr);
    cudaFree(q); cudaFree(q_copy); cudaFree(q2); cudaFree(q_copy2);
    cudaFree(done);
}

static void modo_benchmark(void) {
    const int tamanhos[] = {100, 10000, 100000};
    const int n_tamanhos = 3;
    const int N_ITER     = 20;
    const int N_REPS     = 250;

    for (int ti = 0; ti < n_tamanhos; ti++) {
        int N = tamanhos[ti];

        int  *list1, *list2, *pivot_arr, *q, *q_copy, *q2, *q_copy2;
        bool *done;
        cudaMallocManaged(&list1,     N * sizeof(int));
        cudaMallocManaged(&list2,     N * sizeof(int));
        cudaMallocManaged(&pivot_arr, N * sizeof(int));
        cudaMallocManaged(&q,      2 * N * sizeof(int));
        cudaMallocManaged(&q_copy,  2 * N * sizeof(int));
        cudaMallocManaged(&q2,     2 * N * sizeof(int));
        cudaMallocManaged(&q_copy2, 2 * N * sizeof(int));
        cudaMallocManaged(&done, sizeof(bool));

        int *original = (int *)malloc(N * sizeof(int));
        srand(42);
        for (int i = 0; i < N; i++) original[i] = rand() % 100000;

        /* warmup */
        resetar_estado(list1, original, pivot_arr, q, q2, q_copy, q_copy2, done, N);
        executar_quicksort(list1, list2, q, q_copy, q2, q_copy2, pivot_arr, done, N);

        sleep(30);

        for (int iter = 1; iter <= N_ITER; iter++) {
            double    tempo_acum   = 0.0;
            long long rapl_antes   = -1LL;
            long long rapl_depois  = -1LL;
            int       corretude_ok = 1;

            if (metodo_energia_gpu == GPU_ENERGIA_NVML)       gpu_nvml_iniciar(iter);
            else if (metodo_energia_gpu == GPU_ENERGIA_TEGRA) gpu_tegra_iniciar(iter);

            for (int rep = 0; rep < N_REPS; rep++) {
                resetar_estado(list1, original, pivot_arr, q, q2, q_copy, q_copy2, done, N);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_antes = rapl_ler();

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
                int *res = executar_quicksort(list1, list2, q, q_copy, q2, q_copy2,
                                              pivot_arr, done, N);
                gettimeofday(&t1, NULL);

                if (rep == 0 && metodo_energia != ENERGIA_NONE)
                    rapl_depois = rapl_ler();

                tempo_acum += (t1.tv_sec - t0.tv_sec)
                            + (t1.tv_usec - t0.tv_usec) * 1e-6;

                if (rep == 0)
                    corretude_ok = verificar_corretude(res, N);
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

            const char *corr_str = corretude_ok ? "OK" : "FAIL";

            csv_append("quicksort", "cuda", "otimizado", hardware_label,
                       N, iter, tempo_medio, egpu, ecpu, corr_str);

            {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "quicksort cuda otimizado %s N=%d iter=%d tempo=%.6fs corretude=%s",
                         hardware_label, N, iter, tempo_medio, corr_str);
                log_msg("INFO", msg);
            }

            if (!corretude_ok) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                         "FAIL quicksort cuda otimizado N=%d iter=%d", N, iter);
                log_msg("ERROR", msg);
            }
        }

        sleep(30);
        free(original);
        cudaFree(list1); cudaFree(list2); cudaFree(pivot_arr);
        cudaFree(q); cudaFree(q_copy); cudaFree(q2); cudaFree(q_copy2);
        cudaFree(done);
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    int teste = (argc > 1 && strcmp(argv[1], "--teste") == 0);

    identificar_hardware();
    metodo_energia = detectar_energia();

    {
        char msg[128];
        snprintf(msg, sizeof(msg), "inicio quicksort cuda otimizado hardware=%s", hardware_label);
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
