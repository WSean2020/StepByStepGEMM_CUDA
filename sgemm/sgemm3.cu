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

#define bm 128
#define bk 8
#define bn 128
#define rn 8
#define rm 8

__global__ void Sgemm_v3( 
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

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][bk][bm];
    __shared__ float Bs[2][bk][bn];
    // registers for C
    float frag_c[rm][rn] = {0.0};

    // registers for A and B
    float frag_a[2][rm];
    float frag_b[2][rn];
    // registers load global memory
    const int ldg_num_a = bm * bk / THREAD_NUM_PER_BLOCK;
    const int ldg_num_b = bk * bn / THREAD_NUM_PER_BLOCK;
    float ldg_a_reg[ldg_num_a];
    float ldg_b_reg[ldg_num_b];

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

    A = &A[(bm * by)* K];
    B = &B[bn * bx];

    // load index of the tile
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index =  warp_id/2*16 + lane_id/8*4; //warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int b_tile_index =  warp_id%2*32 + lane_id%8*4; //(lane_id % 16) * 4; // (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    
    // transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < bm ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FLOAT4(ldg_a_reg[ldg_index]) = FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL  ][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index  ];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < bk; i += B_TILE_ROW_STRIDE) {
        FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    
    // load A from shared memory to register
    FLOAT4(frag_a[0][0]) = FLOAT4(As[0][0][a_tile_index]);
    FLOAT4(frag_a[0][4]) = FLOAT4(As[0][0][a_tile_index + 64]);
    // load B from shared memory to register
    FLOAT4(frag_b[0][0]) = FLOAT4(Bs[0][0][b_tile_index]);
    FLOAT4(frag_b[0][4]) = FLOAT4(Bs[0][0][b_tile_index + 64]);
    
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += bk;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < bm ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FLOAT4(ldg_a_reg[ldg_index]) = FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < bk; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FLOAT4(ldg_b_reg[ldg_index]) = FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j = 0; j < bk - 1; j++){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            FLOAT4(frag_a[(j+1)%2][0]) = FLOAT4(As[load_stage_idx][(j+1)][a_tile_index]);
            FLOAT4(frag_a[(j+1)%2][4]) = FLOAT4(As[load_stage_idx][(j+1)][a_tile_index + 64]);
            // load B from shared memory to register
            FLOAT4(frag_b[(j+1)%2][0]) = FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index]);
            FLOAT4(frag_b[(j+1)%2][4]) = FLOAT4(Bs[load_stage_idx][(j+1)][b_tile_index + 64]);
            // compute C rn x rm
            #pragma unroll
            for (int thread_y = 0; thread_y < rm; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < rn; ++thread_x) {
                    frag_c[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            // load A from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < bm ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL  ][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index  ];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < bk; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        FLOAT4(frag_a[0][0]) = FLOAT4(As[load_stage_idx^1][0][a_tile_index]);
        FLOAT4(frag_a[0][4]) = FLOAT4(As[load_stage_idx^1][0][a_tile_index + 64]);
        // load B from shared memory to register
        FLOAT4(frag_b[0][0]) = FLOAT4(Bs[load_stage_idx^1][0][b_tile_index]);
        FLOAT4(frag_b[0][4]) = FLOAT4(Bs[load_stage_idx^1][0][b_tile_index + 64]);
        // compute C rn x rm
        #pragma unroll
        for (int thread_y = 0; thread_y < rm; thread_y++) {
            #pragma unroll
            for (int thread_x = 0; thread_x < rn; thread_x++) {
                frag_c[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while(tile_idx < K);
    
    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;

    //store C00 block
    for(int i=0; i<4; i++){
      FLOAT4(C[OFFSET(
        bm * by + c_block_row + i,
        bn * bx + c_block_col,
        N)]) = FLOAT4(frag_c[i][0]);
    }
    //store C01 block
    for(int i=0; i<4; i++){
      FLOAT4(C[OFFSET(
        bm * by + c_block_row + i,
        bn * bx + c_block_col + 64,
        N)]) = FLOAT4(frag_c[i][4]);
    }
    //store C10 block
    for(int i=0; i<4; i++){
      FLOAT4(C[OFFSET(
        bm * by + c_block_row + 64 + i,
        bn * bx + c_block_col,
        N)]) = FLOAT4(frag_c[i+4][0]);
    }
    //store C11 block
    for(int i=0; i<4; i++){
      FLOAT4(C[OFFSET(
        bm * by + c_block_row + 64 + i,
        bn * bx + c_block_col + 64,
        N)]) = FLOAT4(frag_c[i+4][4]);
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
        Sgemm_v3<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, K, N);
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
    printf("Version3 achieve %.2f%% performance of cuBLAS.\n", 100 * gigaFlops[0] / gigaFlops[1]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);

    cublasDestroy(blas_handle); 
}
