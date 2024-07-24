NVCC="$(shell which nvcc)"

FLAGS=-O3 -std=c++17 -lcublas

$(shell rm -rf bin)
$(shell mkdir bin)

TARGETS = naive v1 v2 v3
all : ${TARGETS}

naive:
	${NVCC} ${FLAGS} -o bin/sgemm0 sgemm/sgemm0.cu
v1:
	${NVCC} ${FLAGS} -o bin/sgemm1 sgemm/sgemm1.cu
v2:
	${NVCC} ${FLAGS} -o bin/sgemm2 sgemm/sgemm2.cu
v3:
	${NVCC} ${FLAGS} -o bin/sgemm3 sgemm/sgemm3.cu

.PHONY: clean

clean:
	rm -rf bin/