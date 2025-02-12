#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 1000       
#define SPACE_SIZE 100.0 
#define K_MAX 20     

typedef struct {
    double x, y, z;
    int label;       
} DataPoint;

double euclidean_distance(DataPoint p1, DataPoint p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
}

int compare_distances(const void *a, const void *b) {
    return (*(double *)a > *(double *)b) - (*(double *)a < *(double *)b);
}

void generate_random_point(DataPoint *point) {
    point->x = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->y = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->z = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->label = rand() % 100;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start, end;
    DataPoint training_data[N];
    double *local_distances, *global_distances;
    int i, j, k;

    // Inizializzazione MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points_per_process = N / size;  // each process get a portion of the dataset
    DataPoint local_points[num_points_per_process];
    local_distances = (double *)malloc(num_points_per_process * N * sizeof(double));  
    global_distances = (rank == 0) ? (double *)malloc(N * size * sizeof(double)) : NULL; 

    if (local_distances == NULL || (rank == 0 && global_distances == NULL)) {
        fprintf(stderr, "Errore nell'allocazione della memoria\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (i = 0; i < N; i++) {
            generate_random_point(&training_data[i]);
        }
    }

    start = MPI_Wtime();

    MPI_Scatter(training_data, num_points_per_process * sizeof(DataPoint), MPI_BYTE,
                local_points, num_points_per_process * sizeof(DataPoint), MPI_BYTE,
                0, MPI_COMM_WORLD);

    for (i = 0; i < num_points_per_process; i++) {
        for (j = 0; j < N; j++) {
            local_distances[i * N + j] = euclidean_distance(local_points[i], training_data[j]);
        }

        MPI_Gather(local_distances + i * N, N, MPI_DOUBLE, global_distances, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            qsort(global_distances, N * size, sizeof(double), compare_distances);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = MPI_Wtime();
        printf("Elapsed time: %.2f seconds.\n", (end - start));
    }

    free(local_distances);
    if (rank == 0) free(global_distances);
    MPI_Finalize();
    return 0;

    // mpicc -o glob knn_red_glob_dist.c -lm
    // mpirun -np 4 --oversubscribe ./glob
}

