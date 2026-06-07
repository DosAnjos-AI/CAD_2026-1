#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define FATOR_COARSENING 4

/* Conta quantos elementos de arr[low..high] sao estritamente menores que valor */
__device__ int busca_binaria_estrita(int *arr, int valor, int low, int high) {
    int l = low, r = high + 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] < valor) l = mid + 1;
        else r = mid;
    }
    return l - low;
}

/* Conta quantos elementos de arr[low..high] sao menores ou iguais a valor */
__device__ int busca_binaria_nao_estrita(int *arr, int valor, int low, int high) {
    int l = low, r = high + 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] <= valor) l = mid + 1;
        else r = mid;
    }
    return l - low;
}

/* Mescla arr[inicio..meio] e arr[meio+1..fim] em O(n) usando aux como buffer */
__device__ void merge_sequencial(int *arr, int *aux, int inicio, int meio, int fim) {
    for (int i = inicio; i <= fim; i++)
        aux[i] = arr[i];

    int i = inicio, j = meio + 1, k = inicio;
    while (i <= meio && j <= fim) {
        if (aux[i] <= aux[j])
            arr[k++] = aux[i++];
        else
            arr[k++] = aux[j++];
    }
    while (i <= meio) arr[k++] = aux[i++];
    while (j <= fim)  arr[k++] = aux[j++];
}

/*
 * Kernel filho lancado via CUDA DP pelo kernel pai.
 * Cada thread processa FATOR_COARSENING elementos.
 * Le de aux, escreve em arr. Merge estavel sem condicao de corrida via
 * getIndex_estavel (inline): busca_binaria_estrita para metade esquerda,
 * busca_binaria_nao_estrita para metade direita.
 */
__global__ void kernel_merge_paralelo(int *arr, int *aux, int inicio, int meio, int fim) {
    int total   = fim - inicio + 1;
    int esq_tam = meio - inicio + 1;
    int base    = (blockIdx.x * blockDim.x + threadIdx.x) * FATOR_COARSENING;

    for (int c = 0; c < FATOR_COARSENING; c++) {
        int local_idx = base + c;
        if (local_idx >= total) return;

        int pos_global = inicio + local_idx;
        int elemento   = aux[pos_global];
        int destino;

        /* getIndex_estavel: determina posicao final no array mesclado */
        if (local_idx < esq_tam) {
            /* metade esquerda: quantos da direita sao estritamente menores */
            int num_menores = busca_binaria_estrita(aux, elemento, meio + 1, fim);
            destino = inicio + local_idx + num_menores;
        } else {
            /* metade direita: quantos da esquerda sao menores ou iguais */
            int pos_dir = local_idx - esq_tam;
            int num_leq = busca_binaria_nao_estrita(aux, elemento, inicio, meio);
            destino = inicio + pos_dir + num_leq;
        }

        arr[destino] = elemento;
    }
}

/*
 * Kernel pai: uma thread por par de subarrays.
 * Decide entre merge paralelo via CUDA DP ou merge sequencial conforme threshold.
 */
__global__ void mergesort_kernel(int *arr, int *aux,
                                 int tamanho_atual, int largura,
                                 int size, int threshold) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int inicio = idx * largura;

    if (inicio >= size) return;

    int meio = min(inicio + tamanho_atual - 1, size - 1);
    int fim  = min(inicio + largura - 1, size - 1);

    /* Sem metade direita: nada a mesclar */
    if (meio >= fim) return;

    int num_elementos = fim - inicio + 1;

    if (num_elementos > threshold) {
        /* Copia secao de arr para aux antes de lançar kernel filho */
        for (int i = inicio; i <= fim; i++)
            aux[i] = arr[i];

        int numBlocos = (num_elementos / FATOR_COARSENING + 1023) / 1024;
        kernel_merge_paralelo<<<numBlocos, 1024>>>(arr, aux, inicio, meio, fim);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("ERRO no kernel filho: %s\n", cudaGetErrorString(err));
    } else {
        merge_sequencial(arr, aux, inicio, meio, fim);
    }
}

/* Retorna 1 se v[0..n-1] esta em ordem nao-decrescente, 0 caso contrario */
static int validar_ordenacao(int *v, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (v[i] > v[i + 1])
            return 0;
    }
    return 1;
}

int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threshold = prop.multiProcessorCount
                  * (prop.maxThreadsPerMultiProcessor / 2)
                  * FATOR_COARSENING;

    printf("GPU: %s\n\n", prop.name);

    int tamanhos[]   = {100, 10000, 100000, 1000000};
    int num_tamanhos = 4;

    printf("%-12s| %-8s| %s\n", "Tamanho", "Iteracao", "Tempo medio (ms)");
    printf("------------|----------|------------------\n");

    /* medias_it[t][it]: media das 10 execucoes da iteracao it para o tamanho t */
    double medias_it[4][5];

    for (int t = 0; t < num_tamanhos; t++) {
        int size = tamanhos[t];

        int *h_arr = (int *)malloc(size * sizeof(int));
        int *h_out = (int *)malloc(size * sizeof(int));
        int *d_arr, *d_aux;
        cudaMalloc((void **)&d_arr, size * sizeof(int));
        cudaMalloc((void **)&d_aux, size * sizeof(int));

        /* Gera vetor aleatorio uma vez por tamanho */
        for (int i = 0; i < size; i++)
            h_arr[i] = rand();

        /* Warm-up: aquece a GPU e verifica corretude antes de medir */
        {
            cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);
            for (int tamanho_atual = 1; tamanho_atual < size; tamanho_atual *= 2) {
                int largura   = tamanho_atual * 2;
                int num_sorts = (size + largura - 1) / largura;
                int threads   = 256;
                int blocos    = (num_sorts + threads - 1) / threads;
                mergesort_kernel<<<blocos, threads>>>(d_arr, d_aux,
                                                      tamanho_atual, largura,
                                                      size, threshold);
                cudaDeviceSynchronize();
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                    printf("ERRO no kernel pai: %s\n", cudaGetErrorString(err));
            }
            cudaMemcpy(h_out, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
            if (!validar_ordenacao(h_out, size)) {
                printf("ERRO: vetor nao ordenado (tamanho=%d)\n", size);
                free(h_arr); free(h_out);
                cudaFree(d_arr); cudaFree(d_aux);
                return 1;
            }
        }

        for (int it = 0; it < 5; it++) {
            double soma_iter = 0.0;

            for (int ex = 0; ex < 10; ex++) {
                /* Copia host -> device fora da medicao de tempo */
                cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

                struct timeval t0, t1;
                gettimeofday(&t0, NULL);

                /* Loop bottom-up: unico trecho cronometrado */
                for (int tamanho_atual = 1; tamanho_atual < size; tamanho_atual *= 2) {
                    int largura   = tamanho_atual * 2;
                    int num_sorts = (size + largura - 1) / largura;
                    int threads   = 256;
                    int blocos    = (num_sorts + threads - 1) / threads;
                    mergesort_kernel<<<blocos, threads>>>(d_arr, d_aux,
                                                          tamanho_atual, largura,
                                                          size, threshold);
                    cudaDeviceSynchronize();

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                        printf("ERRO no kernel pai: %s\n", cudaGetErrorString(err));
                }

                gettimeofday(&t1, NULL);
                soma_iter += ((t1.tv_sec  - t0.tv_sec) +
                              (t1.tv_usec - t0.tv_usec) / 1e6) * 1000.0;
            }

            medias_it[t][it] = soma_iter / 10;
            printf("%-12d| %8d | %.6f\n", size, it + 1, medias_it[t][it]);
        }

        free(h_arr);
        free(h_out);
        cudaFree(d_arr);
        cudaFree(d_aux);
    }

    printf("\n");
    printf("%-12s| %-18s| %s\n", "Tamanho", "Media geral (ms)", "Desvio padrao (ms)");
    printf("------------|------------------|--------------------\n");

    for (int t = 0; t < num_tamanhos; t++) {
        double soma = 0.0;
        for (int it = 0; it < 5; it++)
            soma += medias_it[t][it];
        double media = soma / 5;

        double var = 0.0;
        for (int it = 0; it < 5; it++) {
            double d = medias_it[t][it] - media;
            var += d * d;
        }
        double desvio = sqrt(var / 5);

        printf("%-12d| %-18.6f| %.6f\n", tamanhos[t], media, desvio);
    }

    return 0;
}
