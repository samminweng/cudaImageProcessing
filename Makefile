IDIR=./lib
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda-12.4/include -lcuda --std c++17 -rdc=true 

.PHONY: clean build run

build: src/gaussian_blur.cu src/gaussian_blur.h
	$(COMPILER) $(COMPILER_FLAGS) src/gaussian_blur.cu -o bin/gaussian_blur.exe

clean:
	rm -f bin/gaussian_blur.exe images/*_blurred.*

run:
	./bin/gaussian_blur.exe images/lena_rgb.png images/lena_rgb_blurred.png 512 512 3

all: clean build run
