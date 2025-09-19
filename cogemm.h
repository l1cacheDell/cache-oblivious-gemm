#pragma once

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