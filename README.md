# Polynomial Evaluation Challenge

In many zero-knowledge (ZK) protocols, we need to evaluate polynomials defined over a scalar finite field. Your task is to implement an efficient solution in CUDA that can evaluate multiple polynomials over the same domain.

For example, evaluate the polynomials P₁(x), P₂(x), ..., Pₙ(x) over the domain {4, 7, 9}. Here, `n` is the `batch_size`, `domain_size` is 3, and `coeffs_size` represents the shared degree of the polynomials.

Good luck!
