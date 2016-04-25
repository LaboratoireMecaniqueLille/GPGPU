all:executable

executable: CUcorrel.o fields.o kernels.o util.o
	nvcc -lcusolver -o CUcorrel *.o

CUcorrel.o: CUcorrel.cu CUcorrel.h
	nvcc -c CUcorrel.cu

fields.o: fields.cu
	nvcc -c fields.cu

util.o: util.cu util.h
	nvcc -c util.cu

kernels.o: kernels.cu kernels.cuh
	nvcc -c kernels.cu

clean: 
	rm -rf *.o
