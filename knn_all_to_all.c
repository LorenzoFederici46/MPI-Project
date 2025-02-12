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
} Point;

// Struttura per memorizzare la distanza e l'indice del punto
typedef struct {
    double distance;
    int index;
} DistanceEntry;

double euclidean_distance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                (p1.y - p2.y) * (p1.y - p2.y) +
                (p1.z - p2.z) * (p1.z - p2.z));
}

int compare_distances(const void *a, const void *b) {
    return ((DistanceEntry *)a)->distance > ((DistanceEntry *)b)->distance;
}

void generate_random_point(Point *point) {
    point->x = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->y = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->z = ((double)rand() / RAND_MAX) * SPACE_SIZE;
    point->label = rand() % 100;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start, end;
    
    Point training_data[N];
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points_per_process = N / size;  // each process get a portion of the dataset
    Point local_points[num_points_per_process];

    if (rank == 0) {
        for (i = 0; i < N; i++) {
            generate_random_point(&training_data[i]);
        }
    }

    start = MPI_Wtime();

    MPI_Bcast(training_data, N * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Scatter(training_data, num_points_per_process * sizeof(Point), MPI_BYTE, 
                    local_points, num_points_per_process * sizeof(Point), MPI_BYTE,0, MPI_COMM_WORLD);

    //finding the k nearest neighbours locally
    for (i = 0; i < num_points_per_process; i++) {
        DistanceEntry distances[N];
        for (j = 0; j < N; j++) {
            distances[j].distance = euclidean_distance(local_points[i], training_data[j]);
            distances[j].index = j;
        }
        // local sorting of the distances
        qsort(distances, N, sizeof(DistanceEntry), compare_distances);
        }
    

    MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		end = MPI_Wtime();
		printf("Elapsed time: %.2f seconds.\n", (end - start));
	}

    MPI_Finalize();
    return 0;

    // mpicc -o all knn_all_to_all.c -lm
    // mpirun -np 4 --oversubscribe ./all

}
    

