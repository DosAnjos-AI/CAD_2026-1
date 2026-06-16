#!/bin/bash
# Orquestrador do benchmark CAD_2026_v3.
# Compila todos os binarios e executa cada combinacao algoritmo|api|tamanho,
# pulando combinacoes ja completas (10 linhas) no CSV de resultados.
#
# Uso: nohup ./run_all.sh > results/log.txt 2>&1 &
#
# Nao usar "set -e": combinacoes com skip ou binarios ausentes podem
# retornar codigo != 0 e o script deve continuar mesmo assim.
set -u -o pipefail

cd "$(dirname "$0")"

CSV="results/resultados.csv"

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
# Compilacao de todos os binarios (falha em uma compilacao nao aborta o script)
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

compilar "floyd_cpu" gcc -O3 -march=native -Wall -Wextra -o floyd_warshall/floyd_cpu floyd_warshall/floyd_cpu.c
compilar "floyd_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o floyd_warshall/floyd_openmp floyd_warshall/floyd_openmp.c
compilar "floyd_cuda" nvcc -O3 -arch="${ARCH}" -o floyd_warshall/floyd_cuda floyd_warshall/floyd_cuda.cu
compilar "floyd_cudadp" nvcc -O3 -arch="${ARCH}" -rdc=true -o floyd_warshall/floyd_cudadp floyd_warshall/floyd_cudadp.cu

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
    [ "$count" -ge 10 ]
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
# Tamanhos por categoria de algoritmo
# ---------------------------------------------------------------------------
TAMANHOS_VETOR="1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864"
TAMANHOS_BFS="1024 4096 16384 65536 262144 1048576 4194304 16777216"
TAMANHOS_FW="32 64 128 256 512 1024 2048 4096 8192"

# ---------------------------------------------------------------------------
# Loop principal: para cada algoritmo, percorre api -> tamanhos crescentes
# ---------------------------------------------------------------------------
processar_algoritmo() {
    local dir=$1 prefixo=$2 algo_csv=$3 tamanhos=$4
    local api tamanho
    for api in cpu openmp cuda cudadp; do
        for tamanho in $tamanhos; do
            executar "${dir}/${prefixo}_${api}" "$algo_csv" "$api" "$tamanho"
        done
    done
}

# Ordem de execucao (branch CAD_2026_v3): bitonic_sort -> merge_sort -> bfs -> floyd_warshall
# Sleep de 3s entre algoritmos distintos para isolamento termico.
processar_algoritmo "bitonic_sort" "bitonic" "bitonic_sort" "$TAMANHOS_VETOR"
sleep 3
processar_algoritmo "merge_sort" "merge" "merge_sort" "$TAMANHOS_VETOR"
sleep 3
processar_algoritmo "bfs" "bfs" "bfs" "$TAMANHOS_BFS"
sleep 3
processar_algoritmo "floyd_warshall" "floyd" "floyd_warshall" "$TAMANHOS_FW"

log_msg "Benchmark concluido"
