#!/bin/bash
# Script de sanidade da branch CAD_2026_v3_cenarios.
# Compila os 12 binarios (bitonic, merge, bfs) e executa cada um com tamanho
# minimo, 1 iteracao e 0 warmup, para ambos os cenarios (ordenado e invertido).
# Registra o resultado em results/sanity_check.csv e reporta o total de
# combinacoes que passaram (corretude=1) e falharam (corretude=0 ou erro).
#
# Uso: bash sanity_check.sh
#
# Nao usar "set -e": binarios ausentes ou com falha devem apenas ser contados,
# sem abortar a sanidade.
set -u -o pipefail

cd "$(dirname "$0")"

mkdir -p results

CSV="results/sanity_check.csv"

# Cenarios validados
CENARIOS="ordenado invertido"

# Tamanhos minimos por categoria de algoritmo
N_VETOR=1024
N_BFS=1024

# ---------------------------------------------------------------------------
# Deteccao dinamica de hardware (igual ao run_cenarios_A.sh)
# ---------------------------------------------------------------------------
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH="sm_${SM}"
NTHREADS=$(nproc)

log_msg() {
    echo "[$(date +%H:%M:%S)] $1"
}

log_msg "Hardware detectado: ARCH=${ARCH} NTHREADS=${NTHREADS}"

# ---------------------------------------------------------------------------
# Compilacao dos binarios (falha em uma compilacao nao aborta o script)
# ---------------------------------------------------------------------------
compilar() {
    local descricao="$1"
    shift
    log_msg "Compilando ${descricao}"
    if ! "$@"; then
        log_msg "[ERRO] Falha ao compilar ${descricao}"
    fi
}

compilar "bitonic_cpu" gcc -O3 -march=native -Wall -Wextra -o bitonic_sort/bitonic_cpu bitonic_sort/bitonic_cpu.c
compilar "bitonic_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o bitonic_sort/bitonic_openmp bitonic_sort/bitonic_openmp.c
compilar "bitonic_cuda" nvcc -O3 -arch="${ARCH}" -o bitonic_sort/bitonic_cuda bitonic_sort/bitonic_cuda.cu
compilar "bitonic_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o bitonic_sort/bitonic_cudadp bitonic_sort/bitonic_cudadp.cu

compilar "merge_cpu" gcc -O3 -march=native -Wall -Wextra -o merge_sort/merge_cpu merge_sort/merge_cpu.c
compilar "merge_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o merge_sort/merge_openmp merge_sort/merge_openmp.c
compilar "merge_cuda" nvcc -O3 -arch="${ARCH}" -o merge_sort/merge_cuda merge_sort/merge_cuda.cu
compilar "merge_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o merge_sort/merge_cudadp merge_sort/merge_cudadp.cu

compilar "bfs_cpu" gcc -O3 -march=native -Wall -Wextra -o bfs/bfs_cpu bfs/bfs_cpu.c
compilar "bfs_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o bfs/bfs_openmp bfs/bfs_openmp.c
compilar "bfs_cuda" nvcc -O3 -arch="${ARCH}" -o bfs/bfs_cuda bfs/bfs_cuda.cu
compilar "bfs_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o bfs/bfs_cudadp bfs/bfs_cudadp.cu

# ---------------------------------------------------------------------------
# Inicializacao do CSV de sanidade (sempre recriado do zero)
# ---------------------------------------------------------------------------
echo "sep=|" > "$CSV"
echo "algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos" >> "$CSV"

# ---------------------------------------------------------------------------
# Execucao de uma combinacao: 1 iteracao, 0 warmup
# Combinacoes com falha sao registradas como linha de corretude=0.
# ---------------------------------------------------------------------------
FALHAS=""

testar() {
    local binario=$1 algo=$2 api=$3 cenario=$4 tamanho=$5
    local linha
    if [ ! -f "$binario" ]; then
        log_msg "[ERRO] Binario nao encontrado: $binario"
        echo "${algo}|${api}|${cenario}|${tamanho}|1|0,000000|0|0x0" >> "$CSV"
        FALHAS="${FALHAS}${algo}|${api}|${cenario}|${tamanho}\n"
        return
    fi
    log_msg "Testando ${algo}|${api}|${cenario}|${tamanho}"
    linha=$("$binario" "$tamanho" "$cenario" 1 0 2>/dev/null)
    if [ -z "$linha" ]; then
        log_msg "[ERRO] Execucao sem saida: ${algo}|${api}|${cenario}|${tamanho}"
        echo "${algo}|${api}|${cenario}|${tamanho}|1|0,000000|0|0x0" >> "$CSV"
        FALHAS="${FALHAS}${algo}|${api}|${cenario}|${tamanho}\n"
        return
    fi
    echo "$linha" >> "$CSV"
    # A corretude e o penultimo campo (separador pipe)
    if ! echo "$linha" | grep -q "|1|[^|]*$"; then
        FALHAS="${FALHAS}${algo}|${api}|${cenario}|${tamanho}\n"
    fi
}

# ---------------------------------------------------------------------------
# Execucao: para cada algoritmo -> api -> cenario
# ---------------------------------------------------------------------------
for cenario in $CENARIOS; do
    for api in cpu openmp cuda cudadp; do
        testar "bitonic_sort/bitonic_${api}" "bitonic_sort" "$api" "$cenario" "$N_VETOR"
    done
    for api in cpu openmp cuda cudadp; do
        testar "merge_sort/merge_${api}" "merge_sort" "$api" "$cenario" "$N_VETOR"
    done
    for api in cpu openmp cuda cudadp; do
        testar "bfs/bfs_${api}" "bfs" "$api" "$cenario" "$N_BFS"
    done
done

# ---------------------------------------------------------------------------
# Relatorio final
# ---------------------------------------------------------------------------
TOTAL=$(grep -cE "^(bitonic_sort|merge_sort|bfs)\|" "$CSV") || TOTAL=0
PASSOU=$(grep -c "|1|[^|]*$" "$CSV") || PASSOU=0
FALHOU=$((TOTAL - PASSOU))

echo ""
echo "=== SANIDADE CONCLUIDA ==="
echo "Passou: ${PASSOU}/${TOTAL}"
echo "Falhou: ${FALHOU}/${TOTAL}"
if [ "$FALHOU" -gt 0 ]; then
    echo "Combinacoes com falha:"
    printf "%b" "$FALHAS"
fi
