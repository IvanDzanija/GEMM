// clang++ -std=c++23 -arch arm64 -Wall -Wextra -Wpedantic -O3 -ffast-math -march=native
// -funroll-loops -DNDEBUG -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
// -L/opt/homebrew/opt/libomp/lib -lomp mm_packed.cpp

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

// ---------------------------------------------------------------------------
// Tile sizes
//   MC x KC  : A-panel fits in L2 (MC*KC*4 bytes)
//   KC x NC  : B-panel fits in L3 (KC*NC*4 bytes)
//   MR x NR  : micro-kernel register tile
//
// For Apple M-series (L2 ~12 MB, L3 ~~12-24 MB) these are reasonable defaults.
// Tune MC/NC/KC for your specific chip and cache sizes.
// ---------------------------------------------------------------------------
static constexpr size_t MR = 4;    // micro-kernel rows  (register tile)
static constexpr size_t NR = 4;    // micro-kernel cols  (register tile)
static constexpr size_t MC = 256;  // macro A-panel rows (fits in L2)
static constexpr size_t NC = 256;  // macro B-panel cols (fits in L3)
static constexpr size_t KC = 256;  // macro panel depth

// ---------------------------------------------------------------------------
// Micro-kernel: C_MR×NR += A_packed[MR×KC] * B_packed[KC×NR]
//   A_packed: row-major, rows are MR wide (no stride needed — packed!)
//   B_packed: row-major, rows are NR wide
// ---------------------------------------------------------------------------
inline void kernel4x4_packed(const float *__restrict__ A_p,
                             const float *__restrict__ B_p, float *__restrict__ C,
                             size_t strideC, size_t kc) {
  float c0 = C[0 * strideC + 0], c1 = C[0 * strideC + 1], c2 = C[0 * strideC + 2],
        c3 = C[0 * strideC + 3];
  float c4 = C[1 * strideC + 0], c5 = C[1 * strideC + 1], c6 = C[1 * strideC + 2],
        c7 = C[1 * strideC + 3];
  float c8 = C[2 * strideC + 0], c9 = C[2 * strideC + 1], cA = C[2 * strideC + 2],
        cB = C[2 * strideC + 3];
  float cC = C[3 * strideC + 0], cD = C[3 * strideC + 1], cE = C[3 * strideC + 2],
        cF = C[3 * strideC + 3];

  for (size_t k = 0; k < kc; ++k) {
    // A_packed is laid out as: [row0_k, row1_k, row2_k, row3_k] at each k
    float a0 = A_p[k * MR + 0];
    float a1 = A_p[k * MR + 1];
    float a2 = A_p[k * MR + 2];
    float a3 = A_p[k * MR + 3];

    // B_packed is laid out as: [col0_k, col1_k, col2_k, col3_k] at each k
    const float *Brow = B_p + k * NR;

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

  C[0 * strideC + 0] = c0;
  C[0 * strideC + 1] = c1;
  C[0 * strideC + 2] = c2;
  C[0 * strideC + 3] = c3;
  C[1 * strideC + 0] = c4;
  C[1 * strideC + 1] = c5;
  C[1 * strideC + 2] = c6;
  C[1 * strideC + 3] = c7;
  C[2 * strideC + 0] = c8;
  C[2 * strideC + 1] = c9;
  C[2 * strideC + 2] = cA;
  C[2 * strideC + 3] = cB;
  C[3 * strideC + 0] = cC;
  C[3 * strideC + 1] = cD;
  C[3 * strideC + 2] = cE;
  C[3 * strideC + 3] = cF;
}

// Fallback for edge tiles (non-multiples of MR/NR)
inline void kernel_fallback(const float *A, const float *B, float *C, size_t m_block,
                            size_t n_block, size_t k_block, size_t stride_A,
                            size_t stride_B) {
  for (size_t i = 0; i < m_block; ++i)
    for (size_t k = 0; k < k_block; ++k)
      for (size_t j = 0; j < n_block; ++j)
        C[i * stride_B + j] += A[i * stride_A + k] * B[k * stride_B + j];
}

// ---------------------------------------------------------------------------
// Pack a column-panel of B: B[k0..k0+kc, j0..j0+nc]  →  B_packed[kc][nc/NR][NR]
// Logical layout: B_packed[nr_block * kc * NR + k * NR + nr]
//   i.e. for each NR-wide column block, KC depth-slices laid out contiguously.
// ---------------------------------------------------------------------------
void pack_B(const float *B, float *B_packed, size_t k0, size_t kc, size_t j0, size_t nc,
            size_t N) {
  // For each NR-column strip within nc:
  size_t j = 0;
  for (; j + NR <= nc; j += NR) {
    float *dst = B_packed + (j / NR) * (kc * NR);
    for (size_t k = 0; k < kc; ++k) {
      const float *src = B + (k0 + k) * N + (j0 + j);
      dst[k * NR + 0] = src[0];
      dst[k * NR + 1] = src[1];
      dst[k * NR + 2] = src[2];
      dst[k * NR + 3] = src[3];
    }
  }
  // Remaining columns (edge) — zero-padded to NR
  if (j < nc) {
    size_t rem = nc - j;
    float *dst = B_packed + (j / NR) * (kc * NR);
    for (size_t k = 0; k < kc; ++k) {
      const float *src = B + (k0 + k) * N + (j0 + j);
      for (size_t r = 0; r < NR; ++r) dst[k * NR + r] = (r < rem) ? src[r] : 0.0f;
    }
  }
}

// ---------------------------------------------------------------------------
// Pack a row-panel of A: A[i0..i0+mc, k0..k0+kc]  →  A_packed[mc/MR][MR][kc]
// Logical layout: A_packed[mr_block * kc * MR + k * MR + mr]
// ---------------------------------------------------------------------------
void pack_A(const float *A, float *A_packed, size_t i0, size_t mc, size_t k0, size_t kc,
            size_t K) {
  size_t i = 0;
  for (; i + MR <= mc; i += MR) {
    float *dst = A_packed + (i / MR) * (kc * MR);
    for (size_t k = 0; k < kc; ++k) {
      dst[k * MR + 0] = A[(i0 + i + 0) * K + (k0 + k)];
      dst[k * MR + 1] = A[(i0 + i + 1) * K + (k0 + k)];
      dst[k * MR + 2] = A[(i0 + i + 2) * K + (k0 + k)];
      dst[k * MR + 3] = A[(i0 + i + 3) * K + (k0 + k)];
    }
  }
  // Remaining rows (edge) — zero-padded to MR
  if (i < mc) {
    size_t rem = mc - i;
    float *dst = A_packed + (i / MR) * (kc * MR);
    for (size_t k = 0; k < kc; ++k) {
      for (size_t r = 0; r < MR; ++r)
        dst[k * MR + r] = (r < rem) ? A[(i0 + i + r) * K + (k0 + k)] : 0.0f;
    }
  }
}

// ---------------------------------------------------------------------------
// GEMM with packing
//   Loop order (outermost → innermost):
//     jc  (NC cols of B/C)
//       kc  (KC depth slice)
//         ic  (MC rows of A/C)
//           jr  (NR micro-cols)
//             ir  (MR micro-rows)   ← micro-kernel
//
//   B is packed once per (jc, kc) block  → fits L3
//   A is packed once per (jc, kc, ic) block → fits L2
// ---------------------------------------------------------------------------
void gemm_packed(const float *A, const float *B, float *C, size_t M, size_t N,
                 size_t K) {
  // Per-thread buffers (each OpenMP thread gets its own A_packed)
  // B_packed is shared within a jc-kc tile (all ic-threads reuse it)
  // We allocate per-thread A_packed and one shared B_packed per jc-kc.

  int nthreads = omp_get_max_threads();

  // B_packed: one per (jc, kc) pair — we'll allocate a single slab and
  // reuse it (safe if jc/kc loop is not parallelized; kc is the outer par dim).
  // Here we parallelize over jc so each thread needs its own B_packed too.
  std::vector<std::vector<float>> B_packed_per_thread(nthreads,
                                                      std::vector<float>(KC * NC));
  std::vector<std::vector<float>> A_packed_per_thread(nthreads,
                                                      std::vector<float>(MC * KC));

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    float *B_packed = B_packed_per_thread[tid].data();
    float *A_packed = A_packed_per_thread[tid].data();

    // Parallelize the outermost j-panel loop across threads
#pragma omp for schedule(static)
    for (size_t jc = 0; jc < N; jc += NC) {
      size_t nc = std::min(NC, N - jc);

      for (size_t kc = 0; kc < K; kc += KC) {
        size_t kc_sz = std::min(KC, K - kc);

        // Pack B panel [kc..kc+kc_sz, jc..jc+nc] → B_packed
        pack_B(B, B_packed, kc, kc_sz, jc, nc, N);

        for (size_t ic = 0; ic < M; ic += MC) {
          size_t mc = std::min(MC, M - ic);

          // Pack A panel [ic..ic+mc, kc..kc+kc_sz] → A_packed
          pack_A(A, A_packed, ic, mc, kc, kc_sz, K);

          // Micro-kernel sweep over the mc × nc output tile
          for (size_t ir = 0; ir < mc; ir += MR) {
            size_t mr = std::min(MR, mc - ir);
            for (size_t jr = 0; jr < nc; jr += NR) {
              size_t nr = std::min(NR, nc - jr);

              float *C_tile = &C[(ic + ir) * N + (jc + jr)];

              if (mr == MR && nr == NR) {
                // Packed A block for this MR strip: A_packed + (ir/MR)*kc_sz*MR
                const float *A_blk = A_packed + (ir / MR) * (kc_sz * MR);
                // Packed B block for this NR strip: B_packed + (jr/NR)*kc_sz*NR
                const float *B_blk = B_packed + (jr / NR) * (kc_sz * NR);
                kernel4x4_packed(A_blk, B_blk, C_tile, N, kc_sz);
              } else {
                // Edge: use unpacked fallback directly on original A/B
                kernel_fallback(&A[(ic + ir) * K + kc], &B[kc * N + (jc + jr)], C_tile,
                                mr, nr, kc_sz, K, N);
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Original (reference) kernel for correctness check
// ---------------------------------------------------------------------------
static constexpr size_t T = 4;

inline void kernel4x4_orig(const float *A, const float *B, float *C, size_t strideA,
                           size_t strideBC) {
  float c0 = C[0 * strideBC + 0], c1 = C[0 * strideBC + 1], c2 = C[0 * strideBC + 2],
        c3 = C[0 * strideBC + 3];
  float c4 = C[1 * strideBC + 0], c5 = C[1 * strideBC + 1], c6 = C[1 * strideBC + 2],
        c7 = C[1 * strideBC + 3];
  float c8 = C[2 * strideBC + 0], c9 = C[2 * strideBC + 1], cA = C[2 * strideBC + 2],
        cB = C[2 * strideBC + 3];
  float cC = C[3 * strideBC + 0], cD = C[3 * strideBC + 1], cE = C[3 * strideBC + 2],
        cF = C[3 * strideBC + 3];
  for (int k = 0; k < 4; ++k) {
    float a0 = A[0 * strideA + k], a1 = A[1 * strideA + k], a2 = A[2 * strideA + k],
          a3 = A[3 * strideA + k];
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

void kernel_fallback_orig(const float *A, const float *B, float *C, size_t m_block,
                          size_t n_block, size_t k_block, size_t stride_A,
                          size_t stride_B) {
  for (size_t i = 0; i < m_block; ++i)
    for (size_t k = 0; k < k_block; ++k)
      for (size_t j = 0; j < n_block; ++j)
        C[i * stride_B + j] += A[i * stride_A + k] * B[k * stride_B + j];
}

// ---------------------------------------------------------------------------
int main() {
  static constexpr size_t N = 2048;
  static constexpr size_t M = 2048;
  static constexpr size_t K = 2048;

  std::vector<float> A_v(M * K), B_v(K * N), C_orig(M * N, 0.0f), C_packed(M * N, 0.0f);

  std::srand(0);
  for (auto &x : A_v) x = std::rand() / (static_cast<float>(RAND_MAX) * 0.3f);
  for (auto &x : B_v) x = std::rand() / (static_cast<float>(RAND_MAX) * 0.3f);

  int ind_m = std::rand() % M;
  int ind_n = std::rand() % N;

  float *A = A_v.data();
  float *B = B_v.data();

  // ---- Original tiled kernel ----
  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < M; i += T) {
    for (size_t k = 0; k < K; k += T) {
      for (size_t j = 0; j < N; j += T) {
        size_t cm = std::min(T, M - i), cn = std::min(T, N - j),
               ck = std::min(T, K - k);
        if (cm == T && cn == T && ck == T)
          kernel4x4_orig(&A[i * K + k], &B[k * N + j], &C_orig[i * N + j], K, N);
        else
          kernel_fallback_orig(&A[i * K + k], &B[k * N + j], &C_orig[i * N + j], cm, cn,
                               ck, K, N);
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double t_orig = std::chrono::duration<double>(end - start).count();
  std::println("=== Original (4x4 tiled, no packing) ===");
  std::println("Time: {:.4f}s | Sample C[{},{}] = {:.4f}", t_orig, ind_m, ind_n,
               C_orig[ind_m * N + ind_n]);
  std::println("GFLOPS: {:.2f}", (2.0 * N * M * K) / t_orig / 1e9);

  // ---- Packed GEMM ----
  start = std::chrono::high_resolution_clock::now();
  gemm_packed(A, B, C_packed.data(), M, N, K);
  end = std::chrono::high_resolution_clock::now();
  double t_packed = std::chrono::duration<double>(end - start).count();

  // Correctness check
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (std::abs(C_packed[i * N + j] - C_orig[i * N + j]) >= 1e-2f) {
        std::println("Mismatch at ({},{}): packed={:.6f}  orig={:.6f}", i, j,
                     C_packed[i * N + j], C_orig[i * N + j]);
        return 1;
      }
    }
  }
  std::println("\n=== Packed GEMM (MC={} NC={} KC={}, MR={} NR={}) ===", MC, NC, KC, MR,
               NR);
  std::println("Time: {:.4f}s | Sample C[{},{}] = {:.4f}", t_packed, ind_m, ind_n,
               C_packed[ind_m * N + ind_n]);
  std::println("GFLOPS: {:.2f}", (2.0 * N * M * K) / t_packed / 1e9);
  std::println("\nSpeedup: {:.2f}x", t_orig / t_packed);

  return 0;
}
