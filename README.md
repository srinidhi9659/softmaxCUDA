# softmaxCUDA
Implemented softmax functionality in CUDA


softmax: softmax(xi) = exp(xi) / (exp(x0) + exp(x1) + exp(x2) + ........ + exp(xn))
 
Implementation contains 2 CUDA kernels
1. softmaxDenomKernel: calculates the denominator part of the equation using parallel reduction method.
2. softmaxKernel: calculates the final (numerator / denominator)

With 64 classes and total 10000384 data points below are the results

|  Function Name    | Duration(us)| Compute Throughput(%) | Memory Throughput(%) | Registers(register/thread) | Grid Size      | Block Size (block)  | 
|-------------------|-------------|-----------------------|----------------------|----------------------------|----------------|---------------------|
| softmaxDenomKernel|   596.03    |       27.23           |         84.50        |             16             |   156256, 1, 1 |  64, 1, 1           |
| softmaxKernel     |   347.65    |       65.95           |         65.95        |             16             |   156256, 1, 1 |  64, 1, 1           |

Note: This was run on Nvidia GTX 1660 TI
