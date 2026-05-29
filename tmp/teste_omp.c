#include <stdio.h>
#include <omp.h>

int main() {
    printf("Threads disponíveis (máximo): %d\n", omp_get_max_threads());

    #pragma omp parallel
    {
        printf("  Thread %d de %d em execução\n",
               omp_get_thread_num(), omp_get_num_threads());
    }

    return 0;
}
