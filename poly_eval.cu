#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void poly_eval_ref(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch) {
    const T* curr_coeffs = coeffs + idx_in_batch * coeffs_size;
    T* curr_evals = evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
      curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
        curr_evals[eval_idx] =
          curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
      }
    }
  }
}




template <typename T>
__global__ void poly_eval_kernel(
    const T* coeffs,
    const T* domain,
    int coeffs_size,
    int domain_size,
    int batch_size,
    T* evals)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < batch_size * domain_size; idx += stride) {
        const int eval_idx = idx % domain_size;
        const int batch_idx = idx / domain_size;

        const T* curr_coeffs = coeffs + batch_idx * coeffs_size;
        const T x = domain[eval_idx];
        T result = curr_coeffs[coeffs_size - 1];

        // Horner's method
        for (int coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
            result = result * x + curr_coeffs[coeff_idx];
        }

        evals[batch_idx * domain_size + eval_idx] = result;
    }
}

template <typename T>
void poly_eval(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals)
{
    const int total_evals = batch_size * domain_size;
    if (total_evals == 0) return;

    // CUDA grid configuration
    const int threads_per_block = 256;
    const int blocks = (total_evals + threads_per_block - 1) / threads_per_block;

    // Launch kernel with error checking
    poly_eval_kernel<<<blocks, threads_per_block>>>(coeffs, domain, coeffs_size, domain_size, batch_size, evals);
    
    // Verify kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}

// Explicit instantiation for int type
template void poly_eval<int>(const int*, const int*, int, int, int, int*);