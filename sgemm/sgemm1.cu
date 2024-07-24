#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

// Just tiling and TransA -> As
// Achieve cuBLAS 50~70% performance
template< int bm, int bk, int bn, int rm, int rn > 
__global__ void Sgemm_v1( 
    float* __restrict__ A,
    float* __restrict__ B,
    float *  C, 
    const int M,
    const int K,
    const int N
){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = bn / rn;
    const int THREAD_Y_PER_BLOCK = bm / rm;

    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // Trans A_tile -> As
    __shared__ float As[bk][bm];
    __shared__ float Bs[bk][bn];

    float a_frag[rm];
    float b_frag[rn];
    float c_frag[rm][rn] = {0.0};

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = bk / 4;
    const int B_TILE_THREAD_PER_ROW = bn / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    #pragma unroll
    for(int i = 0; i < K; i += bk){
        // Load tiles of A, B from global mem -> shared mem
        #pragma unroll
        for(int j = 0; j < bm; j += A_TILE_ROW_STRIDE){
            int ldg_index = OFFSET(
                bm * by + A_TILE_ROW_START + j, 
                i + A_TILE_COL, 
                K );
            As[A_TILE_COL    ][A_TILE_ROW_START + j] = A[ldg_index    ];
            As[A_TILE_COL + 1][A_TILE_ROW_START + j] = A[ldg_index + 1];
            As[A_TILE_COL + 2][A_TILE_ROW_START + j] = A[ldg_index + 2];
            As[A_TILE_COL + 3][A_TILE_ROW_START + j] = A[ldg_index + 3];
        }
        #pragma unroll
        for ( int j = 0 ; j < bk; j += B_TILE_ROW_STRIDE) {
            FLOAT4(Bs[B_TILE_ROW_START + j][B_TILE_COL]) = FLOAT4(B[OFFSET(
                    i + B_TILE_ROW_START + j, // row
                    bn * bx + B_TILE_COL, // col
                    N )]);
        }
        __syncthreads();
        // shared mem -> Reg file
        #pragma unroll
        for(int j = 0; j < bk; j++){
            #pragma unroll
            for(int k = 0; k < rm; k += 4){
                FLOAT4(a_frag[k]) = FLOAT4(As[j][ty * rm + k]);
            }
            #pragma unroll
            for(int k = 0; k < rn; k += 4){
                FLOAT4(b_frag[k]) = FLOAT4(Bs[j][rn * tx + k]);
            }
            __syncthreads();
            // Each thread calculate a fragment
            #pragma unroll
            for(int p = 0; p < rm; p++){
                #pragma unroll
                for(int q = 0; q < rn; q++){
                    c_frag[p][q] += a_frag[p] * b_frag[q];
                }
            }
        }
    }
    __syncthreads();
    // Results write back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < rm; thread_y++) {
        #pragma unroll
        for (int thread_x = 0; thread_x < rn; thread_x += 4) {
            FLOAT4(C[OFFSET(
                bm * by + ty * rm + thread_y,
                bn * bx + tx * rn + thread_x,
                N )]) = FLOAT4(c_frag[thread_y][thread_x]);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    const int bm = 128;
    const int bk = 16;
    const int bn = 128;
    const int rn = 8;
    const int rm = 8;

    assert( bm % rm == 0); 
    assert( bn % rn == 0); 
    assert( M % bm == 0); 
    assert( K % bk == 0); 
    assert( N % bn == 0); 

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1);
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = distrib(gen);
    }
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = distrib(gen);
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0.0f;
    int nIter = 10;

    checkCudaErrors(cudaEventRecord(start));
    dim3 dimBlock(bn / rn, bm / rm);
    dim3  dimGrid(N  / bn, M  / bm);
    for (int run = 0 ; run < nIter; run++) {
        Sgemm_v1<bm, bk, bn, rm, rn><<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, K, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cuBLAS
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    // warmup
    for (int run = 0 ; run < 10; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, M
        );
    }
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, M
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);
    
    double eps = 1.e-6;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("Version1 achieve %.2f%% performance of cuBLAS.\n", 100 * gigaFlops[0] / gigaFlops[1]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);

    cublasDestroy(blas_handle); 
}
