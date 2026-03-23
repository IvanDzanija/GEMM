#include <cblas.h>  // OpenBLAS header

#include <chrono>
#include <iostream>
#include <vector>

extern "C" int openblas_get_num_threads();
extern "C" int openblas_get_num_procs();
extern "C" char *openblas_get_config();
int main() {
  constexpr int N = 1024 * 4;
  std::vector<float> A(N * N, 1.1f), B(N * N, 2.2f), C(N * N, 0.0f);

  // Inside main...
  std::cout << "Config: " << openblas_get_config() << std::endl;
  std::cout << "Max threads OpenBLAS can use: " << openblas_get_num_threads()
            << std::endl;
  std::cout << "CPUs detected: " << openblas_get_num_procs() << std::endl;
  openblas_set_num_threads(8);
  auto start = std::chrono::high_resolution_clock::now();

  // cblas_sgemm is the industry standard for matrix mult
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A.data(), N,
              B.data(), N, 0.0f, C.data(), N);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  double gflops = (2.0 * N * N * N) / (diff.count() * 1e9);
  std::cout << "OpenBLAS Performance: " << gflops << " GFLOPS" << std::endl;

  return 0;
}
