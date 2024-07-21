NVCC="$(shell which nvcc)"

FLAGS=-O3 -std=c++17 -lcublas

$(shell rm -rf bin)
$(shell mkdir bin)

TARGETS = naive v1 v2
all : ${TARGETS}

naive:
	${NVCC} ${FLAGS} -o bin/sgemm_v0 sgemm/sgemm_v0.cu
v1:
	${NVCC} ${FLAGS} -o bin/sgemm_v1 sgemm/sgemm_v1.cu
v2:
	${NVCC} ${FLAGS} -o bin/sgemm_v2 sgemm/sgemm_v2.cu

clean:
	rm -rf bin/