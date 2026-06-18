#!/bin/bash
# Orquestrador de teste dedicado ao BFS (branch CAD_2026_v3_bfs_test_v3).
# Compila apenas os binarios BFS CUDA (cuda, cudadp) e executa cada
# combinacao bfs|api|tamanho, pulando combinacoes ja completas (5 linhas)
# no CSV de resultados de teste.
#
# Ordem de execucao: cenario e o loop externo, api o interno.
#   aleatorio -> cuda -> todos os tamanhos
#   aleatorio -> cudadp -> todos os tamanhos
#   ordenado  -> cuda -> todos os tamanhos
#   ordenado  -> cudadp -> todos os tamanhos
#   invertido -> cuda -> todos os tamanhos
#   invertido -> cudadp -> todos os tamanhos
#
# Uso: nohup ./run_bfs_test_v3.sh > results/log_bfs_test_v3.txt 2>&1 &
#
# Nao usar "set -e": combinacoes com skip ou binarios ausentes podem
# retornar codigo != 0 e o script deve continuar mesmo assim.
set -u -o pipefail

cd "$(dirname "$0")"

mkdir -p results

CSV="results/resultados_bfs_test_v3.csv"

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
# Compilacao dos binarios BFS CUDA (falha em uma compilacao nao aborta o script)
# ---------------------------------------------------------------------------
compilar() {
    local descricao="$1"
    shift
    log_msg "Compilando ${descricao}"
    if ! "$@"; then
        log_msg "[ERRO] Falha ao compilar ${descricao}"
    fi
}

compilar "bfs_cuda" nvcc -O3 -arch="${ARCH}" -o bfs/bfs_cuda bfs/bfs_cuda.cu -lm
compilar "bfs_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o bfs/bfs_cudadp bfs/bfs_cudadp.cu -lm

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
    local algo=$1 api=$2 cenario=$3 tamanho=$4
    local count
    count=$(grep -c "^${algo}|${api}|${cenario}|${tamanho}|" "$CSV" 2>/dev/null) || count=0
    [ "$count" -ge 5 ]
}

limpar_parcial() {
    local algo=$1 api=$2 cenario=$3 tamanho=$4
    local tmp
    tmp=$(mktemp)
    grep -v "^${algo}|${api}|${cenario}|${tamanho}|" "$CSV" > "$tmp"
    mv "$tmp" "$CSV"
}

executar() {
    local binario=$1 algo=$2 api=$3 cenario=$4 tamanho=$5
    if [ ! -f "$binario" ]; then
        log_msg "[ERRO] Binario nao encontrado: $binario"
        return
    fi
    if combinacao_completa "$algo" "$api" "$cenario" "$tamanho"; then
        log_msg "SKIP ${algo}|${api}|${cenario}|${tamanho} (completo)"
        return
    fi
    limpar_parcial "$algo" "$api" "$cenario" "$tamanho"
    log_msg "Executando ${algo}|${api}|${cenario}|${tamanho}"
    "$binario" "$tamanho" "$cenario" >> "$CSV"
}

# ---------------------------------------------------------------------------
# Tamanhos e cenarios do BFS para o teste (2^10 a 2^22)
# ---------------------------------------------------------------------------
TAMANHOS_BFS="1024 4096 16384 65536 262144 1048576 4194304"
CENARIOS="aleatorio ordenado invertido"

# ---------------------------------------------------------------------------
# Loop principal: cenario externo, api interna, tamanhos crescentes
# ---------------------------------------------------------------------------
processar_algoritmo() {
    local dir=$1 prefixo=$2 algo_csv=$3 tamanhos=$4
    local api cenario tamanho
    for cenario in $CENARIOS; do
        for api in cuda cudadp; do
            for tamanho in $tamanhos; do
                executar "${dir}/${prefixo}_${api}" "$algo_csv" "$api" "$cenario" "$tamanho"
            done
        done
    done
}

# Script dedicado ao BFS: executa somente o algoritmo bfs.
processar_algoritmo "bfs" "bfs" "bfs" "$TAMANHOS_BFS"

log_msg "Benchmark concluido"
