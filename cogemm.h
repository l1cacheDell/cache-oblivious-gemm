#pragma once

#include <cstddef>

using namespace std;

// ============= single-precision function declarations =============
void sgemm_naive(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8>
void sgemm_v2(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8>
void sgemm_v3(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8, const int TILE_BASE = 64>
void sgemm_v4(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8, const int TILE_BASE = 64>
void sgemm_v5(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8, const int TILE_BASE = 64>
void sgemm_v6(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

template <const int NUM_THREADS = 8, const int TILE_BASE = 64>
void sgemm_v7(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K);

void sgemm_micro_v8_m8n24k8_row_major_kernel(
    const float* __restrict A,  // [M,K], col-major, lda=M
    const float* __restrict B,  // [K,N], col-major, ldb=K
    float* __restrict       C,  // [M,N], col-major, ldc=M
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc
);