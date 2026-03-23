// clang++ -std=c++23 -arch arm64 -Wall -Wextra -Wpedantic -O3 -ffast-math -march=native
// -funroll-loops -DNDEBUG -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
// -L/opt/homebrew/opt/libomp/lib -lomp mm.cpp
#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <print>
#include <vector>

static constexpr size_t T = 4;  // TxT block for the kernel

// We pass 'stride' (which is N) so the kernel knows how wide the matrix is
inline void kernel4x4(const float *A, const float *B, float *C, size_t strideA,
                      size_t strideBC) {
  // 1. Load the current 4x4 block of C into local registers
  // Use strideBC (N) because C is M x N
  float c0 = C[0 * strideBC + 0], c1 = C[0 * strideBC + 1], c2 = C[0 * strideBC + 2],
        c3 = C[0 * strideBC + 3];
  float c4 = C[1 * strideBC + 0], c5 = C[1 * strideBC + 1], c6 = C[1 * strideBC + 2],
        c7 = C[1 * strideBC + 3];
  float c8 = C[2 * strideBC + 0], c9 = C[2 * strideBC + 1], cA = C[2 * strideBC + 2],
        cB = C[2 * strideBC + 3];
  float cC = C[3 * strideBC + 0], cD = C[3 * strideBC + 1], cE = C[3 * strideBC + 2],
        cF = C[3 * strideBC + 3];

  // 2. Perform the 4x4 x 4x4 inner product
  for (int k = 0; k < 4; ++k) {
    // A is M x K, so rows are separated by strideA (K)
    float a0 = A[0 * strideA + k];
    float a1 = A[1 * strideA + k];
    float a2 = A[2 * strideA + k];
    float a3 = A[3 * strideA + k];

    // B is K x N, so rows are separated by strideBC (N)
    const float *Brow = &B[k * strideBC];

    c0 = std::fma(a0, Brow[0], c0);
    c1 = std::fma(a0, Brow[1], c1);
    c2 = std::fma(a0, Brow[2], c2);
    c3 = std::fma(a0, Brow[3], c3);

    c4 = std::fma(a1, Brow[0], c4);
    c5 = std::fma(a1, Brow[1], c5);
    c6 = std::fma(a1, Brow[2], c6);
    c7 = std::fma(a1, Brow[3], c7);

    c8 = std::fma(a2, Brow[0], c8);
    c9 = std::fma(a2, Brow[1], c9);
    cA = std::fma(a2, Brow[2], cA);
    cB = std::fma(a2, Brow[3], cB);

    cC = std::fma(a3, Brow[0], cC);
    cD = std::fma(a3, Brow[1], cD);
    cE = std::fma(a3, Brow[2], cE);
    cF = std::fma(a3, Brow[3], cF);
  }

  // 3. Store the results back to C
  C[0 * strideBC + 0] = c0;
  C[0 * strideBC + 1] = c1;
  C[0 * strideBC + 2] = c2;
  C[0 * strideBC + 3] = c3;
  C[1 * strideBC + 0] = c4;
  C[1 * strideBC + 1] = c5;
  C[1 * strideBC + 2] = c6;
  C[1 * strideBC + 3] = c7;
  C[2 * strideBC + 0] = c8;
  C[2 * strideBC + 1] = c9;
  C[2 * strideBC + 2] = cA;
  C[2 * strideBC + 3] = cB;
  C[3 * strideBC + 0] = cC;
  C[3 * strideBC + 1] = cD;
  C[3 * strideBC + 2] = cE;
  C[3 * strideBC + 3] = cF;
}

void kernel_fallback(const float *A, const float *B, float *C, size_t m_block,
                     size_t n_block, size_t k_block, size_t stride_A, size_t stride_B) {
  for (size_t i = 0; i < m_block; ++i) {
    for (size_t k = 0; k < k_block; ++k) {
      for (size_t j = 0; j < n_block; ++j) {
        // stride_A is the global K, stride_B is the global N
        C[i * stride_B + j] += A[i * stride_A + k] * B[k * stride_B + j];
      }
    }
  }
}

int main() {
  static constexpr size_t N = 2048 / 1;  // 20T8/2
  static constexpr size_t M = 2048 / 1;
  static constexpr size_t K = 2048 / 1;
  // static std::array<float, N * N> A_v, B_v, C_v;
  std::vector<float> A_v(M * K), B_v(K * N), C_v(M * N);
  std::srand(0);  // Constant seed for reproducibility
  for (auto &x : A_v) {
    x = std::rand() / (static_cast<float>(RAND_MAX) * 0.3f);
  }
  for (auto &x : B_v) {
    x = std::rand() / (static_cast<float>(RAND_MAX) * 0.3f);
  }

  int ind_m = std::rand() % M;
  int ind_n = std::rand() % N;
  std::fill(C_v.begin(), C_v.end(), 0.0f);

  float *A = A_v.data();
  float *B = B_v.data();
  float *C = C_v.data();

  // --- TEST 3: Kernel TxT ---
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < M; i += T) {
    for (size_t k = 0; k < K; k += T) {
      for (size_t j = 0; j < N; j += T) {
        size_t current_M = std::min(T, M - i);
        size_t current_N = std::min(T, N - j);
        size_t current_K = std::min(T, K - k);

        if (current_M == T && current_N == T && current_K == T) {
          // Pass K for A's stride, N for B and C's stride
          kernel4x4(&A[i * K + k], &B[k * N + j], &C[i * N + j], K, N);
        } else {
          kernel_fallback(&A[i * K + k], &B[k * N + j], &C[i * N + j], current_M,
                          current_N, current_K, K, N);
        }
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration<double>(end - start).count();
  std::println("Kernel Time: {}s | Result: {}", time, C[ind_m * N + ind_n]);
  std::println("GFLOPS: {}", (2.0 * N * M * K) / time / 1e9);

  std::vector<float> D(M * N);
  start = std::chrono::high_resolution_clock::now();
  // Naive

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        D[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (std::abs(D[i * N + j] - C[i * N + j]) >= 1e-5f) {
        std::println("Mismatch at ({}, {}): D = {}, C = {}", i, j, D[i * N + j],
                     C[i * N + j]);
        return 1;
      }
    }
  }

  time = std::chrono::duration<double>(end - start).count();
  std::println("Kernel Time: {}s | Result: {}", time, C[ind_m * N + ind_n]);
  std::println("GFLOPS: {}", (2.0 * N * M * K) / time / 1e9);

  return 0;
}
