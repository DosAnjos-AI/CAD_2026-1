#!/usr/bin/env bash
set -uo pipefail

# interceptar SIGINT antes de qualquer operacao
trap 'log_warn "execucao interrompida pelo usuario (SIGINT)"; exit 1' INT

# --- CONFIGURACAO ---
MODO_TESTE=0             # 1 = modo teste (--teste), 0 = benchmark completo
SLEEP_ENTRE_BINARIOS=20  # segundos entre binarios
# --- FIM CONFIGURACAO ---

# detectar GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)

if echo "$GPU_NAME" | grep -q "MX350"; then
    HARDWARE="mx350"
    CUDA_ARCH="sm_61"
elif echo "$GPU_NAME" | grep -q "4090"; then
    HARDWARE="rtx4090"
    CUDA_ARCH="sm_89"
elif [ -f /proc/device-tree/model ] && grep -q "Jetson" /proc/device-tree/model 2>/dev/null; then
    HARDWARE="jetson"
    CUDA_ARCH="sm_87"
else
    HARDWARE="desconhecido"
    CUDA_ARCH="sm_61"
fi

# detectar compilador C++ compativel com nvcc
CCBIN=""
for CXX in g++-12 g++-11 g++-10 g++; do
    if command -v "$CXX" &>/dev/null; then
        CCBIN="$CXX"
        break
    fi
done

if [ -z "$CCBIN" ]; then
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') nenhum compilador g++ encontrado" >&2
    exit 1
fi

# flags de compilacao
FLAGS_OMP="-O2 -fopenmp"
FLAGS_SEQ="-O2"
FLAGS_CUDA="-O2 -arch=${CUDA_ARCH} -ccbin ${CCBIN}"
FLAGS_CUDA_DP="${FLAGS_CUDA} -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -lcudadevrt"

LOG_FILE="results/run_all.log"

log_info()  { echo "[INFO ] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"; }
log_warn()  { echo "[WARN ] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"; }

# verificacoes pre-execucao
mkdir -p results
touch results/benchmark.csv results/benchmark.log results/run_all.log

if [ ! -s results/benchmark.csv ]; then
    echo "algoritmo|api|versao|hardware|tamanho|iteracao|tempo_total_s|energia_gpu_j|energia_cpu_j|corretude" \
        > results/benchmark.csv
fi

if cat /sys/class/powercap/intel-rapl:0/energy_uj &>/dev/null; then
    log_info "RAPL disponivel — energia CPU sera coletada"
else
    log_warn "RAPL indisponivel — energia CPU sera NA (executar: sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj)"
fi

LOAD=$(awk '{print $1}' /proc/loadavg)
log_info "load average: $LOAD"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "NA")
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || echo "NA")
log_info "hardware=$HARDWARE arch=$CUDA_ARCH driver=$DRIVER cuda=$CUDA_VER ccbin=$CCBIN"

compilar() {
    local ARQUIVO=$1
    local SAIDA=$2
    local TIPO=$3
    local CMD

    case $TIPO in
        omp)     CMD="gcc $FLAGS_OMP $ARQUIVO -o $SAIDA" ;;
        seq)     CMD="gcc $FLAGS_SEQ $ARQUIVO -o $SAIDA" ;;
        cuda)    CMD="nvcc $FLAGS_CUDA $ARQUIVO -o $SAIDA" ;;
        cuda_dp) CMD="nvcc $FLAGS_CUDA_DP $ARQUIVO -o $SAIDA" ;;
    esac

    log_info "compilando: $ARQUIVO"
    if eval "$CMD" 2>>/tmp/compile_err.log; then
        log_info "OK: $SAIDA"
        return 0
    else
        log_error "FALHA: $ARQUIVO — ver /tmp/compile_err.log"
        return 1
    fi
}

executar() {
    local BINARIO=$1
    local LABEL=$2

    if [ ! -f "$BINARIO" ]; then
        log_error "binario nao encontrado: $BINARIO — pulando"
        return 1
    fi

    log_info "iniciando: $LABEL"
    local T0
    T0=$(date +%s)

    if [ "$MODO_TESTE" -eq 1 ]; then
        "$BINARIO" --teste
    else
        "$BINARIO"
    fi

    local STATUS=$?
    local T1
    T1=$(date +%s)
    local ELAPSED=$((T1 - T0))

    if [ $STATUS -eq 0 ]; then
        log_info "concluido: $LABEL tempo=${ELAPSED}s"
    else
        log_error "erro (exit=$STATUS): $LABEL tempo=${ELAPSED}s"
    fi

    sleep $SLEEP_ENTRE_BINARIOS
}

log_info "=== FASE DE COMPILACAO ==="

# mergesort
compilar mergesort/artigo/mergesort_artigo_openmp.c        mergesort/artigo/mergesort_artigo_openmp        omp
compilar mergesort/artigo/mergesort_artigo_cuda.cu         mergesort/artigo/mergesort_artigo_cuda          cuda
compilar mergesort/artigo/mergesort_artigo_cuda_dp.cu      mergesort/artigo/mergesort_artigo_cuda_dp       cuda_dp
compilar mergesort/otimizado/mergesort_otimizado_openmp.c  mergesort/otimizado/mergesort_otimizado_openmp  omp
compilar mergesort/otimizado/mergesort_otimizado_cuda.cu   mergesort/otimizado/mergesort_otimizado_cuda    cuda
compilar mergesort/otimizado/mergesort_otimizado_cuda_dp.cu mergesort/otimizado/mergesort_otimizado_cuda_dp cuda_dp
compilar mergesort/sequencial/mergesort_sequencial.c       mergesort/sequencial/mergesort_sequencial       seq

# quicksort
compilar quicksort/artigo/quicksort_artigo_openmp.c        quicksort/artigo/quicksort_artigo_openmp        omp
compilar quicksort/artigo/quicksort_artigo_cuda.cu         quicksort/artigo/quicksort_artigo_cuda          cuda
compilar quicksort/artigo/quicksort_artigo_cuda_dp.cu      quicksort/artigo/quicksort_artigo_cuda_dp       cuda_dp
compilar quicksort/otimizado/quicksort_otimizado_openmp.c  quicksort/otimizado/quicksort_otimizado_openmp  omp
compilar quicksort/otimizado/quicksort_otimizado_cuda.cu   quicksort/otimizado/quicksort_otimizado_cuda    cuda
compilar quicksort/otimizado/quicksort_otimizado_cuda_dp.cu quicksort/otimizado/quicksort_otimizado_cuda_dp cuda_dp
compilar quicksort/sequencial/quicksort_sequencial.c       quicksort/sequencial/quicksort_sequencial       seq

# bfs
compilar bfs/artigo/bfs_artigo_openmp.c                    bfs/artigo/bfs_artigo_openmp                    omp
compilar bfs/artigo/bfs_artigo_cuda.cu                     bfs/artigo/bfs_artigo_cuda                      cuda
compilar bfs/artigo/bfs_artigo_cuda_dp.cu                  bfs/artigo/bfs_artigo_cuda_dp                   cuda_dp
compilar bfs/otimizado/bfs_otimizado_openmp.c              bfs/otimizado/bfs_otimizado_openmp              omp
compilar bfs/otimizado/bfs_otimizado_cuda.cu               bfs/otimizado/bfs_otimizado_cuda                cuda
compilar bfs/otimizado/bfs_otimizado_cuda_dp.cu            bfs/otimizado/bfs_otimizado_cuda_dp             cuda_dp
compilar bfs/sequencial/bfs_sequencial.c                   bfs/sequencial/bfs_sequencial                   seq

# sssp
compilar sssp/artigo/sssp_artigo_openmp.c                  sssp/artigo/sssp_artigo_openmp                  omp
compilar sssp/artigo/sssp_artigo_cuda.cu                   sssp/artigo/sssp_artigo_cuda                    cuda
compilar sssp/artigo/sssp_artigo_cuda_dp.cu                sssp/artigo/sssp_artigo_cuda_dp                 cuda_dp
compilar sssp/otimizado/sssp_otimizado_openmp.c            sssp/otimizado/sssp_otimizado_openmp            omp
compilar sssp/otimizado/sssp_otimizado_cuda.cu             sssp/otimizado/sssp_otimizado_cuda              cuda
compilar sssp/otimizado/sssp_otimizado_cuda_dp.cu          sssp/otimizado/sssp_otimizado_cuda_dp           cuda_dp
compilar sssp/sequencial/sssp_sequencial.c                 sssp/sequencial/sssp_sequencial                 seq

log_info "=== COMPILACAO CONCLUIDA ==="

log_info "=== FASE DE EXECUCAO ==="
log_info "modo: $([ $MODO_TESTE -eq 1 ] && echo TESTE || echo BENCHMARK)"
log_info "inicio: $(date)"

# mergesort
executar mergesort/artigo/mergesort_artigo_openmp        "mergesort openmp artigo"
executar mergesort/artigo/mergesort_artigo_cuda_dp       "mergesort cuda_dp artigo"
executar mergesort/otimizado/mergesort_otimizado_openmp  "mergesort openmp otimizado"
executar mergesort/otimizado/mergesort_otimizado_cuda    "mergesort cuda otimizado"
executar mergesort/otimizado/mergesort_otimizado_cuda_dp "mergesort cuda_dp otimizado"
executar mergesort/sequencial/mergesort_sequencial       "mergesort sequencial"

# quicksort
executar quicksort/artigo/quicksort_artigo_openmp        "quicksort openmp artigo"
executar quicksort/artigo/quicksort_artigo_cuda          "quicksort cuda artigo"
executar quicksort/artigo/quicksort_artigo_cuda_dp       "quicksort cuda_dp artigo"
executar quicksort/otimizado/quicksort_otimizado_openmp  "quicksort openmp otimizado"
executar quicksort/otimizado/quicksort_otimizado_cuda    "quicksort cuda otimizado"
executar quicksort/otimizado/quicksort_otimizado_cuda_dp "quicksort cuda_dp otimizado"
executar quicksort/sequencial/quicksort_sequencial       "quicksort sequencial"

# bfs
executar bfs/artigo/bfs_artigo_openmp                    "bfs openmp artigo"
executar bfs/artigo/bfs_artigo_cuda                      "bfs cuda artigo"
executar bfs/artigo/bfs_artigo_cuda_dp                   "bfs cuda_dp artigo"
executar bfs/otimizado/bfs_otimizado_openmp              "bfs openmp otimizado"
executar bfs/otimizado/bfs_otimizado_cuda                "bfs cuda otimizado"
executar bfs/otimizado/bfs_otimizado_cuda_dp             "bfs cuda_dp otimizado"
executar bfs/sequencial/bfs_sequencial                   "bfs sequencial"

# sssp
executar sssp/artigo/sssp_artigo_openmp                  "sssp openmp artigo"
executar sssp/artigo/sssp_artigo_cuda                    "sssp cuda artigo"
executar sssp/artigo/sssp_artigo_cuda_dp                 "sssp cuda_dp artigo"
executar sssp/otimizado/sssp_otimizado_openmp            "sssp openmp otimizado"
executar sssp/otimizado/sssp_otimizado_cuda              "sssp cuda otimizado"
executar sssp/otimizado/sssp_otimizado_cuda_dp           "sssp cuda_dp otimizado"
executar sssp/sequencial/sssp_sequencial                 "sssp sequencial"

# executado por ultimo — N=100K leva ~21h na MX350
executar mergesort/artigo/mergesort_artigo_cuda          "mergesort cuda artigo"

log_info "fim: $(date)"
log_info "=== EXECUCAO CONCLUIDA ==="
