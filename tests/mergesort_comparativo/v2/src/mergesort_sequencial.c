#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define ITERACOES  5
#define EXECUCOES  10

/* Le energia do pacote CPU em micro-joules via RAPL */
static long long ler_energia_cpu_uj(void) {
    FILE *f = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r");
    if (!f) return -1LL;
    long long val = -1LL;
    if (fscanf(f, "%lld", &val) != 1) val = -1LL;
    fclose(f);
    return val;
}

/* Mescla v[esq..meio] e v[meio+1..dir] usando aux como buffer */
static void merge(int *v, int *aux, int esq, int meio, int dir) {
    for (int i = esq; i <= dir; i++)
        aux[i] = v[i];
    int i = esq, j = meio + 1, k = esq;
    while (i <= meio && j <= dir) {
        if (aux[i] <= aux[j]) v[k++] = aux[i++];
        else                   v[k++] = aux[j++];
    }
    while (i <= meio) v[k++] = aux[i++];
    while (j <= dir)  v[k++] = aux[j++];
}

/* Mergesort recursivo em v[esq..dir] usando aux como buffer auxiliar */
static void mergesort(int *v, int *aux, int esq, int dir) {
    if (esq >= dir) return;
    int meio = esq + (dir - esq) / 2;
    mergesort(v, aux, esq, meio);
    mergesort(v, aux, meio + 1, dir);
    merge(v, aux, esq, meio, dir);
}

/* Retorna 1 se v[0..n-1] esta em ordem nao-decrescente */
static int validar_ordenacao(int *v, int n) {
    for (int i = 0; i < n - 1; i++)
        if (v[i] > v[i + 1]) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s N\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "N deve ser um inteiro positivo\n");
        return 1;
    }

    int *original = (int *)malloc(n * sizeof(int));
    int *trabalho = (int *)malloc(n * sizeof(int));
    int *aux      = (int *)malloc(n * sizeof(int));
    if (!original || !trabalho || !aux) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        free(original); free(trabalho); free(aux);
        return 1;
    }

    /* Gera vetor aleatorio uma vez com semente fixa — mesmo dado em todas as execucoes */
    srand(42);
    for (int i = 0; i < n; i++)
        original[i] = rand();

    /* Warm-up: 3 sorts completos sem medicao; corretude validada no primeiro */
    const char *corretude = "OK";
    for (int w = 0; w < 3; w++) {
        memcpy(trabalho, original, n * sizeof(int));
        mergesort(trabalho, aux, 0, n - 1);
        if (w == 0 && !validar_ordenacao(trabalho, n))
            corretude = "ERRO";
    }

    /* 5 iteracoes x 10 execucoes — unico trecho medido */
    for (int it = 0; it < ITERACOES; it++) {
        long long e_antes = ler_energia_cpu_uj();

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        for (int ex = 0; ex < EXECUCOES; ex++) {
            memcpy(trabalho, original, n * sizeof(int));
            mergesort(trabalho, aux, 0, n - 1);
        }

        gettimeofday(&t1, NULL);
        long long e_depois = ler_energia_cpu_uj();

        double tempo   = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;
        double energia = 0.0;
        if (e_antes >= 0 && e_depois >= 0)
            energia = (e_depois - e_antes) / 1e6;

        printf("mergesort,sequencial,%d,%d,%d,%.6f,%.6f,%.6f,%s\n",
               n, it + 1, EXECUCOES, tempo, energia, 0.0, corretude);
        fflush(stdout);
    }

    free(original);
    free(trabalho);
    free(aux);
    return 0;
}
