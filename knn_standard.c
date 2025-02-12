#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NUM_POINTS 100    
#define DIM 3              
#define MAX_K 20            

typedef struct {
    double x, y, z;
} Point;

typedef struct {
    int index;
    double distance;
} Neighbor;

// Function to calculate Euclidean distance between two points
double euclidean_distance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

// Comparator function for sorting distances
int compare_neighbors(const void *a, const void *b) {
    Neighbor *neighbor1 = (Neighbor *)a;
    Neighbor *neighbor2 = (Neighbor *)b;
    //Return positive number if the first one is greater, zero if both are equal, negative if the second one is greater
    return (neighbor1->distance > neighbor2->distance) - (neighbor1->distance < neighbor2->distance);
}

// Function to find the k nearest neighbors for a given point subset
void find_k_nearest_neighbors(Point *points, int num_points, int start_index, int end_index, int k_values[], int num_k_values) {
    for (int i = start_index; i < end_index; i++) {

        Neighbor neighbors[NUM_POINTS - 1]; // avoid computing with itself
        int neighbor_count = 0;

        // Calculate distances from point i to all other points
        for (int j = 0; j < num_points; j++) {
            if (i != j) {
                neighbors[neighbor_count].index = j;
                neighbors[neighbor_count].distance = euclidean_distance(points[i], points[j]);
                neighbor_count++;
            }
        }
        qsort(neighbors, neighbor_count, sizeof(Neighbor), compare_neighbors);

    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double start, end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int k_values[] = {5, 10, 15, 20};
    int num_k_values = sizeof(k_values) / sizeof(k_values[0]);
    Point points[NUM_POINTS];

    // Initialize points with random coordinates
    if (rank == 0) {
        for (int i = 0; i < NUM_POINTS; i++) {
            points[i].x = (double)rand() / RAND_MAX;
            points[i].y = (double)rand() / RAND_MAX;
            points[i].z = (double)rand() / RAND_MAX;
        }
    }

    start = MPI_Wtime();

    // Broadcast points array to all processes
    MPI_Bcast(points, NUM_POINTS * DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine the subset of points for each process
    int points_per_process = NUM_POINTS / size;
    int start_index = rank * points_per_process;
    int end_index = (rank == size - 1) ? NUM_POINTS : start_index + points_per_process;

    // Each process finds the k nearest neighbors for its subset of points
    find_k_nearest_neighbors(points, NUM_POINTS, start_index, end_index, k_values, num_k_values);

    MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		end = MPI_Wtime();
		printf("Elapsed time: %.2f seconds.\n", (end - start));
	}

    MPI_Finalize();
    return 0;

    // mpicc -o standard knn_standard.c -lm
    // mpirun -np 4 --oversubscribe ./standard
}