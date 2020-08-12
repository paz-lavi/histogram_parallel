build:
	mpicxx -fopenmp -c hw8.c -o hw8.o

	nvcc -I./inc -c cuda.cu -o cuda.o
	mpicxx -fopenmp -o hw8  hw8.o cuda.o  /usr/local/cuda-10.2/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./hw8

run:
	mpiexec -np 2 ./hw8 data.txt

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./hw8 data.txt
