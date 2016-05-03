all: CUcorrel

CUcorrel: CUcorrel.o fields.o kernels.o util.o lodepng.o reduction.o
	nvcc -lcusolver -o CUcorrel *.o

CUcorrel.o: CUcorrel.cu CUcorrel.h util.h kernels.cuh reduction.cuh
	nvcc -c CUcorrel.cu

fields.o: fields.cu CUcorrel.h
	nvcc -c fields.cu

util.o: util.cu kernels.cuh CUcorrel.h lodepng/lodepng.h util.h
	nvcc -c util.cu

kernels.o: kernels.cu CUcorrel.h kernels.cuh util.h bicubic.cu
	nvcc -c kernels.cu

lodepng.o: lodepng/lodepng.cpp lodepng/lodepng.h
	nvcc -c lodepng/lodepng.cpp

reduction.o: CUcorrel.h
	nvcc -c reduction.cu

clean: 
	rm -rf *.o

run: CUcorrel
	./CUcorrel
