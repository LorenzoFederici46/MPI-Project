#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 1000       // number of point to be generated
#define SPACE_SIZE 100.0 // max axis distance
#define K_MAX 20    

typedef struct {
    double x, y, z; 
    int label;       
} DataPoint;

// function to calculate Euclidean distance
double euclidean_distance(DataPoint p1, DataPoint p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
}

// sorting function for distances
int compare_distances(const void *a, const void *b) {
    return (*(double *)a > *(double *)b) - (*(double *)a < *(double *)b);
}

void generate_random_point(DataPoint *point) {
    point->x = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->y = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->z = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->label = rand() % 2;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start, end;
    DataPoint training_data[N];
    double local_distances[N / 2];
    double global_distances[N];
    int i, j, k;

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // master generate the
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            generate_random_point(&training_data[i]);
        }
    }

    start = MPI_Wtime();

    // broadcasting data to all the processes
    MPI_Bcast(training_data, N * sizeof(DataPoint), MPI_BYTE, 0, MPI_COMM_WORLD);

    // each process calculate distances for his own portion of data
    int num_points_per_process = N / size;
    for (i = 0; i < num_points_per_process; i++) {
        int index = rank * num_points_per_process + i;
        for (j = 0; j < N; j++) {
            local_distances[j] = euclidean_distance(training_data[index], training_data[j]);
        }

        // gathering the calculated distances
        MPI_Gather(local_distances, N / size, MPI_DOUBLE, global_distances, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // master sort the distances and find the k nearest neighbour for each k
        if (rank == 0) {
            qsort(global_distances, N, sizeof(double), compare_distances);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		end = MPI_Wtime();
		printf("Elapsed time: %.2f seconds.\n", (end - start));
	}

    MPI_Finalize();

    return 0;

    // mpicc -o ring knn_ring.c -lm
    // mpirun -np 4 --oversubscribe ./ring

}
