# CUDA_GEMM

## usage
````
mkdir buid

cd ./build 

cmake ..

make -j4 

# performance only means careless about the correctness. 
# if performance_only == 0, will check the result, may be slow in big matrix test.
./cuda_gemm_test [M] [N] [K] [performance_only]
````
