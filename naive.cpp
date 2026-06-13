// clang++ -std=c++23 -arch arm64 -Wall -Wextra -Wpedantic -O3 -ffast-math -march=native
// -funroll-loops -DNDEBUG -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
// -L/opt/homebrew/opt/libomp/lib -lomp mm.cpp
// #include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <print>
#include <vector>

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

  // #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
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
