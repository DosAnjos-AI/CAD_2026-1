#!/bin/bash
# Orquestrador de teste dedicado ao BFS (branch CAD_2026_v3_bfs_test).
# Compila apenas os binarios BFS (openmp, cuda, cudadp) e executa cada
# combinacao bfs|api|tamanho, pulando combinacoes ja completas (10 linhas)
# no CSV de resultados de teste.
#
# Uso: nohup ./run_bfs_test.sh > results/log_bfs_test.txt 2>&1 &
#
# Nao usar "set -e": combinacoes com skip ou binarios ausentes podem
# retornar codigo != 0 e o script deve continuar mesmo assim.
set -u -o pipefail

cd "$(dirname "$0")"

mkdir -p results

CSV="results/resultados_bfs_test.csv"

# ---------------------------------------------------------------------------
# Deteccao dinamica de hardware
# ---------------------------------------------------------------------------
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${SM}"
NTHREADS=$(nproc)

log_msg() {
    echo "[$(date +%H:%M:%S)] $1"
}

log_msg "Hardware detectado: ARCH=${ARCH} NTHREADS=${NTHREADS}"

# ---------------------------------------------------------------------------
# Compilacao dos binarios BFS (falha em uma compilacao nao aborta o script)
# ---------------------------------------------------------------------------
compilar() {
    local descricao="$1"
    shift
    log_msg "Compilando ${descricao}"
    if ! "$@"; then
        log_msg "[ERRO] Falha ao compilar ${descricao}"
    fi
}

compilar "bfs_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o bfs/bfs_openmp bfs/bfs_openmp.c
compilar "bfs_cuda" nvcc -O3 -arch="${ARCH}" -o bfs/bfs_cuda bfs/bfs_cuda.cu
compilar "bfs_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o bfs/bfs_cudadp bfs/bfs_cudadp.cu

# ---------------------------------------------------------------------------
# Inicializacao do CSV de resultados
# ---------------------------------------------------------------------------
mkdir -p results
if [ ! -f "$CSV" ]; then
    echo "sep=|" > "$CSV"
    echo "algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos" >> "$CSV"
fi

# ---------------------------------------------------------------------------
# Funcoes de controle de skip por combinacao completa
# ---------------------------------------------------------------------------
combinacao_completa() {
    local algo=$1 api=$2 tamanho=$3
    local count
    count=$(grep -c "^${algo}|${api}|aleatorio|${tamanho}|" "$CSV" 2>/dev/null) || count=0
    [ "$count" -ge 5 ]
}

limpar_parcial() {
    local algo=$1 api=$2 tamanho=$3
    local tmp
    tmp=$(mktemp)
    grep -v "^${algo}|${api}|aleatorio|${tamanho}|" "$CSV" > "$tmp"
    mv "$tmp" "$CSV"
}

executar() {
    local binario=$1 algo=$2 api=$3 tamanho=$4
    if [ ! -f "$binario" ]; then
        log_msg "[ERRO] Binario nao encontrado: $binario"
        return
    fi
    if combinacao_completa "$algo" "$api" "$tamanho"; then
        log_msg "SKIP ${algo}|${api}|${tamanho} (completo)"
        return
    fi
    limpar_parcial "$algo" "$api" "$tamanho"
    log_msg "Executando ${algo}|${api}|${tamanho}"
    "$binario" "$tamanho" >> "$CSV"
}

# ---------------------------------------------------------------------------
# Tamanhos do BFS para o teste (2^10 a 2^18)
# ---------------------------------------------------------------------------
TAMANHOS_BFS="1024 4096 16384 65536 262144"

# ---------------------------------------------------------------------------
# Loop principal: para o BFS, percorre api -> tamanhos crescentes
# ---------------------------------------------------------------------------
processar_algoritmo() {
    local dir=$1 prefixo=$2 algo_csv=$3 tamanhos=$4
    local api tamanho
    for api in openmp cuda cudadp; do
        for tamanho in $tamanhos; do
            executar "${dir}/${prefixo}_${api}" "$algo_csv" "$api" "$tamanho"
        done
    done
}

# Script dedicado ao BFS: executa somente o algoritmo bfs.
processar_algoritmo "bfs" "bfs" "bfs" "$TAMANHOS_BFS"

log_msg "Benchmark concluido"
