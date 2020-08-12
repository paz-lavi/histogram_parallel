/*
 ============================================================================
 Name        : hw8.c
 Paz Lavi 208944124
 Daniel Kozachkevich 203690359
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "hw8.h"
#include "cuda.h"

int main(int argc, char *argv[]){
	int numOfProcs, myRank, arr_size = 0;
	MPI_Status status;
	int *merged_array;
	int *received_array;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);

	if (numOfProcs != MAX_PROC) {
		printf("Program requires %d nodes\n", MAX_PROC);
		MPI_Finalize();
		exit(1);
	}

	if (myRank == MASTER) {

		if (argc != 2) {
			printf(argc < 2 ? "Data file is required!" :    //handle input error
					"Only data file is required!");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		int *initial_array;
		//space allocation and read the data from the file
		initArrays(argv[1], &initial_array, &received_array, &arr_size);

		//send size of array to slave
		MPI_Send(&arr_size, 1, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);

		// send cuda task to slave
		MPI_Send(initial_array + arr_size / 2, ((arr_size / 2)+ ((arr_size % 2) == 0 ? 0:1)), MPI_INT, SLAVE, 0,
		MPI_COMM_WORLD);

		//perform openmp task, do merge
		merged_array = OpenMPTask(initial_array, arr_size);

		//receive result from slave
		MPI_Recv(received_array, arr_size, MPI_INT, SLAVE, 0, MPI_COMM_WORLD,
				&status);

		//merge both processes results
		OpenMPFinalMergeTask(merged_array, received_array, RANGE_SIZE);

		// Print array after merge
		PrintHistogram(merged_array);

		// free memory
		free(initial_array);

	} else {
		//receive size of array from master
		MPI_Recv(&arr_size, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

		//space allocation for the receive data
		myIntArrCalloc(&received_array, ((arr_size / 2) + ((arr_size % 2) == 0 ? 0:1)));

		// receive cuda task
		MPI_Recv(received_array, ((arr_size / 2)+ ((arr_size % 2) == 0 ? 0:1)), MPI_INT, MASTER, 0,
		MPI_COMM_WORLD, &status);
		// perform cuda task
		merged_array = CUDATask(received_array, ((arr_size / 2)+ ((arr_size % 2) == 0 ? 0:1)), arr_size);

		//send result to master
		MPI_Send(merged_array, arr_size, MPI_INT, MASTER, 0, MPI_COMM_WORLD);

	}
	// free memory
	free(merged_array);
	free(received_array);
	MPI_Finalize();

	return 0;

}

//space allocation and read the data from the file.
void initArrays(char *filename, int **initial_array, int **received_array,
		int *size) {
	FILE *file = fopen(filename, "r");
	if (!file) {
		fprintf(stderr, "Could not open file: %s\n", filename);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int i;
	if (fscanf(file, "%d", size) == 1) {
		myIntArrCalloc(initial_array, *size); //space allocation set all to 0
		myIntArrCalloc(received_array, *size); //space allocation set all to 0
		for (i = 0; i < *size; i++) {
			if (fscanf(file, "%d", (*initial_array) + i) != 1) { //read each line and insert to the array
				fprintf(stderr, "row %d is not represented correctly!",
				__LINE__);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

		}

	}
	fclose(file);

}

//perform calloc (set all to 0). if allocation failed the program abort
void myIntArrCalloc(int **arr, int size) {
	*arr = (int*) calloc(size, sizeof(int)); // space allocation
	if (arr == NULL) { //check if allocation succeeded. if not the program abort
		fprintf(stderr, "Could not allocate array");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

int* OpenMPTask(int *src_arr, int size) {
	int *dst_arr;
	myIntArrCalloc(&dst_arr, RANGE_SIZE); //space allocation

	int *tmp_hist;
#pragma omp parallel //num_threads(size/2)
	{
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
#pragma omp single
		{
			myIntArrCalloc(&tmp_hist, size * nthreads); //space allocation

		}

#pragma omp for
		for (int i = 0; i < size / 2; i++) // for openmp task we running only on first arr part
			tmp_hist[tid * size + src_arr[i]]++;

		// merge
#pragma omp for
		for (int i = 0; i < RANGE_SIZE; i++)
			for (int j = 0; j < nthreads; j++)
				dst_arr[i] += tmp_hist[j * size + i]; // each thread merges specific cell in tmp_arr to dst_arr

	}
	free(tmp_hist);
	return dst_arr;
}

int* CUDATask(int *arr, int size, int arr_size) {
	return calculateHistogramm(arr, size, arr_size);
}


void OpenMPFinalMergeTask(int *dest_array, int *src_array, int size) {
#pragma omp for
	for (int i = 0; i < RANGE_SIZE; i++)
		dest_array[i] += src_array[i]; // each thread merges specific cell in tmp_arr to dst_arr
}


void PrintHistogram(int *arr) {
	int i, sum = 0;

	printHeadline(1);
	//res = calcHistograma(arr, size);
	for (i = 0; i < RANGE_SIZE; i++) {
		if (i % 10 == 0)
			printf("\n");
		printf("| arr[%-3d] = %-3d |", i, arr[i]);
		sum += arr[i];

	}
	printf("\n\n\nsum of array= %d", sum);
	printRes(arr, RANGE_SIZE);
	fflush(stdout);
}
void printRes(int *res, int size) {
	int i, j;
	printHeadline(2);
	for (i = 0; i < size; i++) {
		printf("%3d |", i);
		for (j = 0; j < res[i]; j++) {
			printf("*");
		}
		printf("\n");
	}

	printf("%s\n\n", LINE);
}

void printHeadline(int msg_num) {
	if (!(msg_num == 1 || msg_num == 2))
		return;
	printf("\n%s", LINE);
	printf("%s", LINE);

	printf("%s", msg_num == 1 ? MSG1 : MSG2);

	printf("%s", LINE);
	printf("%s\n\n", LINE);
}
