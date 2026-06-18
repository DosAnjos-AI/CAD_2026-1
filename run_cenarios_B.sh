#!/bin/bash
# Orquestrador dos cenarios ordenado/invertido (Maquina B) da branch CAD_2026_v3_cenarios.
# Compila os binarios de bitonic_sort, merge_sort e bfs e executa cada combinacao
# algoritmo|api|cenario|tamanho, pulando combinacoes ja completas (5 linhas) no CSV.
# Floyd-Warshall nao participa desta branch. Cenario aleatorio ja existe no v3.
# Diferenca para o run_cenarios_A.sh: ordem dos algoritmos e dos cenarios invertida.
#
# Uso: nohup ./run_cenarios_B.sh > results/log_cenarios_B.txt 2>&1 &
#
# Nao usar "set -e": combinacoes com skip ou binarios ausentes podem
# retornar codigo != 0 e o script deve continuar mesmo assim.
set -u -o pipefail

cd "$(dirname "$0")"

mkdir -p results

CSV="results/resultados_cenarios.csv"

# Ordem dos cenarios (Maquina B): invertido -> ordenado (bitonic/merge)
CENARIOS="invertido ordenado"
# O BFS suporta tambem o cenario aleatorio (Erdos-Renyi via Batagelj-Brandes);
# bitonic/merge so possuem ordenado/invertido, por isso a lista do BFS e separada.
CENARIOS_BFS="invertido ordenado aleatorio"

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

compilar "bfs_cpu" gcc -O3 -march=native -Wall -Wextra -o bfs/bfs_cpu bfs/bfs_cpu.c -lm
compilar "bfs_openmp" gcc -O3 -march=native -fopenmp -Wall -Wextra -DNUM_THREADS="${NTHREADS}" -o bfs/bfs_openmp bfs/bfs_openmp.c -lm
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
# Funcoes de controle de skip por combinacao completa (5 iteracoes)
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
# Tamanhos por categoria de algoritmo
# ---------------------------------------------------------------------------
TAMANHOS_VETOR="67108864 16777216 4194304 1048576 262144 65536 16384 4096 1024"
TAMANHOS_BFS="65536 16384 4096 1024"

# ---------------------------------------------------------------------------
# Loop principal: para cada algoritmo, percorre cenario -> api -> tamanhos decrescentes
# ---------------------------------------------------------------------------
processar_algoritmo() {
    local dir=$1 prefixo=$2 algo_csv=$3 tamanhos=$4 cenarios=$5
    local cenario api tamanho
    for cenario in $cenarios; do
        for api in cudadp cuda openmp cpu; do
            for tamanho in $tamanhos; do
                executar "${dir}/${prefixo}_${api}" "$algo_csv" "$api" "$cenario" "$tamanho"
            done
        done
    done
}

# Ordem de execucao (Maquina B): bfs -> merge_sort -> bitonic_sort
# Sleep de 3s entre algoritmos distintos para isolamento termico.
processar_algoritmo "bfs" "bfs" "bfs" "$TAMANHOS_BFS" "$CENARIOS_BFS"
sleep 3
processar_algoritmo "merge_sort" "merge" "merge_sort" "$TAMANHOS_VETOR" "$CENARIOS"
sleep 3
processar_algoritmo "bitonic_sort" "bitonic" "bitonic_sort" "$TAMANHOS_VETOR" "$CENARIOS"

log_msg "Benchmark de cenarios concluido"
