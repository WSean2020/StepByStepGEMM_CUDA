# StepByStepGEMM_CUDA

optimize GEMM on GPUs step by step

#### Build and test

1. You can run individual test like :
    ```shell
    make
    ```
    or
    ```shell
    make v3
    ```  
    Run as order : \<M> \<K> \<N>
    ```shell
    ./bin/sgemm_v3 1024 2048 4096
    ```

2. Also can run batch tests :

    ```shell
    chmod +x run_test.sh
    ```

    ```shell
    ./run_test.sh
    ```