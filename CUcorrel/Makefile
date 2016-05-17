all: CUcorrel

CUcorrel: CUcorrel.o util.o lodepng.o reduction.o img.o
	nvcc -lcusolver -o CUcorrel *.o

CUcorrel.o: CUcorrel.cu CUcorrel.h util.h img.h
	nvcc -c CUcorrel.cu

util.o: util.cu CUcorrel.h lodepng/lodepng.h util.h
	nvcc -c util.cu

lodepng.o: lodepng/lodepng.cpp lodepng/lodepng.h
	nvcc -c lodepng/lodepng.cpp

reduction.o: CUcorrel.h
	nvcc -c reduction.cu

img.o: img.cu img.h CUcorrel.h reduction.cuh lodepng/lodepng.h
	nvcc -c img.cu

clean: 
	rm -rf CUcorrel *.o

run: CUcorrel
	./CUcorrel
