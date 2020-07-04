NVCC=nvcc

all: cuckoo

cuckoo: main.cu cuckoo_serial.hpp cuckoo_cuda.cuh
	${NVCC} $< -o $@

clean: 
	rm cuckoo