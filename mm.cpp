#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <print>

static constexpr size_t N = 1024;  // 2048/2
alignas(32) static std::array<float, N * N> A, B, C;

// We pass 'stride' (which is N) so the kernel knows how wide the matrix is
void kernel4x4(const float *A, const float *B, float *C, size_t stride) {
  // Load C into 16 local registers using the stride
  float c0 = C[0 * stride + 0], c1 = C[0 * stride + 1], c2 = C[0 * stride + 2],
        c3 = C[0 * stride + 3];
  float c4 = C[1 * stride + 0], c5 = C[1 * stride + 1], c6 = C[1 * stride + 2],
        c7 = C[1 * stride + 3];
  float c8 = C[2 * stride + 0], c9 = C[2 * stride + 1], cA = C[2 * stride + 2],
        cB = C[2 * stride + 3];
  float cC = C[3 * stride + 0], cD = C[3 * stride + 1], cE = C[3 * stride + 2],
        cF = C[3 * stride + 3];

  for (int k = 0; k < 4; ++k) {
    // Rows of B and columns of A are separated by 'stride'
    const float *Brow = &B[k * stride];

    float a0 = A[0 * stride + k];
    float a1 = A[1 * stride + k];
    float a2 = A[2 * stride + k];
    float a3 = A[3 * stride + k];

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

  // Write back to the large C matrix
  C[0 * stride + 0] = c0;
  C[0 * stride + 1] = c1;
  C[0 * stride + 2] = c2;
  C[0 * stride + 3] = c3;
  C[1 * stride + 0] = c4;
  C[1 * stride + 1] = c5;
  C[1 * stride + 2] = c6;
  C[1 * stride + 3] = c7;
  C[2 * stride + 0] = c8;
  C[2 * stride + 1] = c9;
  C[2 * stride + 2] = cA;
  C[2 * stride + 3] = cB;
  C[3 * stride + 0] = cC;
  C[3 * stride + 1] = cD;
  C[3 * stride + 2] = cE;
  C[3 * stride + 3] = cF;
}

int main() {
  std::srand(0);  // Constant seed for reproducibility
  for (size_t i = 0; i < N * N; ++i) {
    A[i] = std::rand() / static_cast<float>(RAND_MAX);
    B[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  std::fill(C.begin(), C.end(), 0.0f);
  int ind = std::rand() % N;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < N; ++k) {
      for (size_t j = 0; j < N; ++j) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::println("Time taken: {} seconds",
               std::chrono::duration<double>(end - start).count());
  std::println("{}", C[ind * N + ind]);

  std::fill(C.begin(), C.end(), 0.0f);

  start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < N; ++k) {
      for (size_t j = 0; j < N; ++j) {
        C[i * N + j] = std::fma(A[i * N + k], B[k * N + j], C[i * N + j]);
      }
    }
  }

  end = std::chrono::high_resolution_clock::now();

  std::println("Time taken: {} seconds",

               std::chrono::duration<double>(end - start).count());

  std::println("{}", C[ind * N + ind]);

  std::fill(C.begin(), C.end(), 0.0f);

  // --- TEST 2: FMA Loop ---
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < N; ++k) {
      for (size_t j = 0; j < N; ++j) {
        C[i * N + j] = std::fma(A[i * N + k], B[k * N + j], C[i * N + j]);
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::println("FMA Loop Time: {}s | Result: {}",
               std::chrono::duration<double>(end - start).count(), C[ind * N + ind]);

  // --- TEST 3: Kernel 4x4 ---
  std::fill(C.begin(), C.end(), 0.0f);
  start = std::chrono::high_resolution_clock::now();
  // Reorder loops to IKJ for better cache locality
  for (size_t i = 0; i < N; i += 4) {
    for (size_t k = 0; k < N; k += 4) {
      for (size_t j = 0; j < N; j += 4) {
        // Pass the pointers AND the stride N
        kernel4x4(&A[i * N + k], &B[k * N + j], &C[i * N + j], N);
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();

  double time = std::chrono::duration<double>(end - start).count();
  std::println("Kernel Time:   {}s | Result: {}", time, C[ind * N + ind]);
  std::println("GFLOPS: {}", (2.0 * N * N * N) / time / 1e9);

  return 0;
}
