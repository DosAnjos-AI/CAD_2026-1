#!/usr/bin/env bash
set -euo pipefail

# ======================================================
# CONFIGURAÇÃO — editar antes de executar
# Opções de HARDWARE: mx350 | rtx4090 | jetson_agx_orin
# Requer: sudo (para perf energy-pkg e tegrastats no Jetson)
# ======================================================
HARDWARE="${HARDWARE:-mx350}"
ITERACOES="${ITERACOES:-10}"
RUNS="${RUNS:-1500}"

# Navega para o diretório raiz do repositório
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ======================================================
# Detecção de ambiente com base em HARDWARE
# ======================================================
case "$HARDWARE" in
    mx350)
        SM_NUM=61
        COLETA_ENERGIA="nvidia_smi"
        CCBIN="-ccbin gcc-12"
        ;;
    rtx4090)
        SM_NUM=89
        COLETA_ENERGIA="nvidia_smi"
        CCBIN="-ccbin gcc-12"
        ;;
    jetson_agx_orin)
        SM_NUM=87
        COLETA_ENERGIA="tegrastats"
        CCBIN=""
        ;;
    *)
        echo "ERRO: HARDWARE inválido: '$HARDWARE'"
        echo "      Use: mx350 | rtx4090 | jetson_agx_orin"
        exit 1
        ;;
esac

# Detecta nvcc (ordem de preferência: cuda-12.2, cuda genérico, PATH)
if [ -x "/usr/local/cuda-12.2/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda-12.2/bin/nvcc"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
elif command -v nvcc &>/dev/null; then
    NVCC_PATH="$(command -v nvcc)"
else
    echo "ERRO: nvcc não encontrado"
    exit 1
fi

# ======================================================
# Flags de compilação por API
# ======================================================
OMP_FLAGS="-O2 -fopenmp"
CUDA_FLAGS="-O2 ${CCBIN:+$CCBIN }-arch=compute_${SM_NUM} -code=sm_${SM_NUM}"
CUDADP_FLAGS="${CUDA_FLAGS} -rdc=true -lcudadevrt -DCUDA_FORCE_CDP1_IF_SUPPORTED -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING"

RESULTS_DIR="results/${HARDWARE}"
mkdir -p "$RESULTS_DIR"

# Redireciona stdout para terminal E arquivo de log simultaneamente
LOG_FILE="${RESULTS_DIR}/benchmark_log.txt"
exec 1> >(tee -a "$LOG_FILE")

# ======================================================
# Trap: limpeza em saída normal, erro ou Ctrl+C
# ======================================================
POWER_PID=""
TEGRA_PID=""
POWER_LOG="/tmp/power_$$.log"
TEGRA_LOG="/tmp/tegrastats_$$.log"
PERF_LOG="/tmp/perf_$$.log"

cleanup() {
    [ -n "$POWER_PID" ] && kill "$POWER_PID" 2>/dev/null || true
    [ -n "$TEGRA_PID" ] && kill "$TEGRA_PID" 2>/dev/null || true
    wait "$POWER_PID" 2>/dev/null || true
    wait "$TEGRA_PID" 2>/dev/null || true
    rm -f "$POWER_LOG" "$TEGRA_LOG" "$PERF_LOG" "/tmp/make_err_$$.log"
}
trap cleanup EXIT INT TERM

# ======================================================
# Compilação
# ======================================================
compilar() {
    local alg=$1 api=$2
    local dir="src/${alg}/${api}"
    printf "[compilando] %-20s ... " "${alg}/${api}"
    case "$api" in
        openmp)
            make -C "$dir" CC=gcc CFLAGS="$OMP_FLAGS" -s \
                2>"/tmp/make_err_$$.log" || {
                echo "FALHOU"
                cat "/tmp/make_err_$$.log"
                echo "ERRO: falha ao compilar ${alg}/${api}"
                exit 1
            }
            ;;
        cuda)
            make -C "$dir" NVCC="$NVCC_PATH" NVCCFLAGS="$CUDA_FLAGS" -s \
                2>"/tmp/make_err_$$.log" || {
                echo "FALHOU"
                cat "/tmp/make_err_$$.log"
                echo "ERRO: falha ao compilar ${alg}/${api}"
                exit 1
            }
            ;;
        cuda_dp)
            make -C "$dir" NVCC="$NVCC_PATH" NVCCFLAGS="$CUDADP_FLAGS" -s \
                2>"/tmp/make_err_$$.log" || {
                echo "FALHOU"
                cat "/tmp/make_err_$$.log"
                echo "ERRO: falha ao compilar ${alg}/${api}"
                exit 1
            }
            ;;
    esac
    echo "OK"
}

# ======================================================
# Compilar todos os 12 binários
# ======================================================
echo "======================================"
echo "Compilando 12 binários para $HARDWARE"
echo "======================================"
for alg in mergesort quicksort bfs sssp; do
    for api in openmp cuda cuda_dp; do
        compilar "$alg" "$api"
    done
done
echo "[ok] todos os binários compilados"
echo ""

# ======================================================
# Validar que todos os binários existem e são executáveis
# ======================================================
BINARIOS=(
    src/mergesort/openmp/mergesort_omp
    src/mergesort/cuda/mergesort_cuda
    src/mergesort/cuda_dp/mergesort_cuda_dp
    src/quicksort/openmp/quicksort_omp
    src/quicksort/cuda/quicksort_cuda
    src/quicksort/cuda_dp/quicksort_cuda_dp
    src/bfs/openmp/bfs_omp
    src/bfs/cuda/bfs_cuda
    src/bfs/cuda_dp/bfs_cuda_dp
    src/sssp/openmp/sssp_omp
    src/sssp/cuda/sssp_cuda
    src/sssp/cuda_dp/sssp_cuda_dp
)
for bin in "${BINARIOS[@]}"; do
    [ -x "$bin" ] || { echo "ERRO: binário não encontrado: $bin"; exit 1; }
done
echo "[ok] todos os 12 binários validados"
echo ""

# Limpa CSVs anteriores para evitar mistura com dados corretos
echo "[limpeza] removendo CSVs anteriores em $RESULTS_DIR..."
rm -f "${RESULTS_DIR}"/*.csv
echo "[limpeza] concluída"
echo ""

# ======================================================
# Coleta de energia — Plano A (com sudo)
# GPU: nvidia-smi em background
# CPU (x86): sudo perf stat -e power/energy-pkg/ envolvendo a execução
# Jetson: sudo tegrastats em background (GPU+CPU juntos)
# ======================================================
iniciar_coleta_gpu() {
    if [ "$COLETA_ENERGIA" = "nvidia_smi" ]; then
        > "$POWER_LOG"
        nvidia-smi --query-gpu=power.draw \
            --format=csv,noheader,nounits -l 1 >> "$POWER_LOG" 2>/dev/null &
        POWER_PID=$!
    elif [ "$COLETA_ENERGIA" = "tegrastats" ]; then
        if ! command -v tegrastats &>/dev/null; then
            echo "AVISO: tegrastats não encontrado — energia será N/A" >&2
            return
        fi
        > "$TEGRA_LOG"
        sudo tegrastats --interval 1000 >> "$TEGRA_LOG" 2>/dev/null &
        TEGRA_PID=$!
    fi
}

parar_coleta_gpu() {
    if [ -n "$POWER_PID" ]; then
        kill "$POWER_PID" 2>/dev/null || true
        wait "$POWER_PID" 2>/dev/null || true
        POWER_PID=""
    fi
    if [ -n "$TEGRA_PID" ]; then
        sudo kill "$TEGRA_PID" 2>/dev/null || true
        wait "$TEGRA_PID" 2>/dev/null || true
        TEGRA_PID=""
    fi
}

calcular_energia_gpu() {
    local tempo_s=$1
    if [ "$COLETA_ENERGIA" = "nvidia_smi" ]; then
        [ -s "$POWER_LOG" ] || { echo "N/A"; return; }
        LC_NUMERIC=C awk -v t="$tempo_s" \
            '{sum+=$1; n++} END{ if(n>0) printf "%.4f",(sum/n)*t; else print "N/A" }' \
            "$POWER_LOG"
    else
        [ -s "$TEGRA_LOG" ] || { echo "N/A"; return; }
        gawk -v t="$tempo_s" '
        {
            gpu=0
            if (match($0, /VDD_GPU_SOC ([0-9]+)mW/, a)) gpu=a[1]
            if (gpu>0) { sum+=gpu/1000.0; n++ }
        }
        END { if(n>0) printf "%.4f",(sum/n)*t; else print "N/A" }
        ' "$TEGRA_LOG"
    fi
}

# Extrai Joules do CPU do log do perf stat (linha: "X,XX Joules power/energy-pkg/")
extrair_energia_cpu_perf() {
    local log=$1
    # perf stat imprime algo como: "12,34 Joules power/energy-pkg/"
    local joules
    joules=$(grep -oP '[\d,]+(?=\s+Joules\s+power/energy-pkg/)' "$log" 2>/dev/null | head -1 | tr ',' '.' || true)
    [ -n "$joules" ] && echo "$joules" || echo "N/A"
}

extrair_energia_cpu_tegrastats() {
    local tempo_s=$1
    [ -s "$TEGRA_LOG" ] || { echo "N/A"; return; }
    gawk -v t="$tempo_s" '
    {
        cpu=0
        if (match($0, /VDD_CPU_CV ([0-9]+)mW/, b)) cpu=b[1]
        if (cpu>0) { sum+=cpu/1000.0; n++ }
    }
    END { if(n>0) printf "%.4f",(sum/n)*t; else print "N/A" }
    ' "$TEGRA_LOG"
}

# ======================================================
# Execução de um benchmark com coleta completa de energia
# ======================================================
executar_benchmark() {
    local alg=$1 api=$2 label=$3
    shift 3
    local args="$*"

    local suffix
    case "$api" in
        openmp)  suffix="omp"     ;;
        cuda)    suffix="cuda"    ;;
        cuda_dp) suffix="cuda_dp" ;;
    esac
    local bin="src/${alg}/${api}/${alg}_${suffix}"
    local csv="${RESULTS_DIR}/${alg}_${api}.csv"

    [ -s "$csv" ] || echo "algoritmo,api,hardware,tamanho,iteracao,tempo_total_s,energia_gpu_j,energia_cpu_j,corretude" > "$csv"

    local iter
    for iter in $(seq 1 "$ITERACOES"); do
        > "$POWER_LOG"; > "$TEGRA_LOG"; > "$PERF_LOG"

        iniciar_coleta_gpu

        if [ "$COLETA_ENERGIA" = "nvidia_smi" ]; then
            # x86: usa perf stat para coleta de energia CPU
            # shellcheck disable=SC2086
            output=$(sudo perf stat -e power/energy-pkg/ \
                "$bin" $args --runs "$RUNS" 2>"$PERF_LOG") \
                || output="${alg},${api},${label},${RUNS},0,ERRO"
            parar_coleta_gpu
            tempo_total_s=$(echo "$output" | cut -d',' -f5)
            corretude=$(echo "$output"     | cut -d',' -f6)
            energia_gpu_j=$(calcular_energia_gpu "${tempo_total_s:-0}")
            energia_cpu_j=$(extrair_energia_cpu_perf "$PERF_LOG")
        else
            # Jetson: tegrastats cobre GPU+CPU; não usa perf
            # shellcheck disable=SC2086
            output=$("$bin" $args --runs "$RUNS" 2>/dev/null) \
                || output="${alg},${api},${label},${RUNS},0,ERRO"
            parar_coleta_gpu
            tempo_total_s=$(echo "$output" | cut -d',' -f5)
            corretude=$(echo "$output"     | cut -d',' -f6)
            energia_gpu_j=$(calcular_energia_gpu "${tempo_total_s:-0}")
            energia_cpu_j=$(extrair_energia_cpu_tegrastats "${tempo_total_s:-0}")
        fi

        echo "${alg},${api},${HARDWARE},${label},${iter},${tempo_total_s:-0},${energia_gpu_j},${energia_cpu_j},${corretude}" >> "$csv"
        printf "[%s/%s] N=%-18s runs=%s iter %2d/%d  tempo_total=%ss  gpu=%sJ  cpu=%sJ  %s\n" \
            "$alg" "$api" "$label" "$RUNS" "$iter" "$ITERACOES" \
            "${tempo_total_s:-?}" "${energia_gpu_j}" "${energia_cpu_j}" "${corretude:-?}"
    done
}

# ======================================================
# Loop principal de benchmarks
# ======================================================
echo "======================================"
echo "Benchmark: $HARDWARE  |  SM=$SM_NUM  [MODO SUDO]"
echo "Iterações: $ITERACOES  |  Runs internos: $RUNS"
echo "Resultados: $RESULTS_DIR"
echo "======================================"
echo ""

for api in openmp cuda cuda_dp; do
    echo "--- mergesort / $api ---"
    for n in 100 10000 100000; do
        executar_benchmark mergesort "$api" "$n" "$n"
    done

    echo "--- quicksort / $api ---"
    for n in 100 10000 100000; do
        executar_benchmark quicksort "$api" "$n" "$n"
    done

    echo "--- bfs / $api ---"
    for tamanho in "10000 30000" "100000 300000" "500000 1000000"; do
        n_label=$(echo "$tamanho" | tr ' ' 'x')
        # shellcheck disable=SC2086
        executar_benchmark bfs "$api" "$n_label" $tamanho
    done

    echo "--- sssp / $api ---"
    for tamanho in "1000 4000" "10000 30000" "100000 300000" "200000 400000"; do
        n_label=$(echo "$tamanho" | tr ' ' 'x')
        # shellcheck disable=SC2086
        executar_benchmark sssp "$api" "$n_label" $tamanho
    done

    echo ""
done

echo "======================================"
echo "Benchmark concluído."
echo "Resultados salvos em: $RESULTS_DIR"
echo "======================================"
