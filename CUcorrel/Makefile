all:executable

executable: CUcorrel.o fields.o kernels.o util.o lodepng.o
	nvcc -lcusolver -o CUcorrel *.o

CUcorrel.o: CUcorrel.cu CUcorrel.h util.h kernels.cuh
	nvcc -c CUcorrel.cu

fields.o: fields.cu CUcorrel.h
	nvcc -c fields.cu

util.o: util.cu kernels.cuh CUcorrel.h lodepng/lodepng.h
	nvcc -c util.cu

kernels.o: kernels.cu kernels.cuh util.h
	nvcc -c kernels.cu

lodepng.o: lodepng/lodepng.cpp lodepng/lodepng.h
	nvcc -c lodepng/lodepng.cpp

clean: 
	rm -rf *.o
