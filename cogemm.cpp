// Implement different block-tile tailored GEMM for different datatype.
// This is CPU implementation, check `gemm_5206.cu` for GPU impl.
// 
// Target datatype: FP64 (double), FP32, FP16

#include <cstddef>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cogemm.h"

using namespace std;    // avoid using this in production :)
namespace py = pybind11;

#define OFFSET(i, j, ld) ((i * ld) + j)

// ================================================= v1 ====================================================
// single precision gemm, float32
void sgemm_naive(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                c[OFFSET(i, j, N)] += a[OFFSET(i, k, K)] * b[OFFSET(k, j, N)];
            }
        }
    }
}

// ================================================= v2 ====================================================
template <const int NUM_THREADS /* The device may have more than 8 cores, but to align with matrix size, use 8 cores to compte*/>
void sgemm_v2(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int TILE_SIZE_X = M / (NUM_THREADS / 2);  // ------------ > x, the horizon
    const int TILE_SIZE_Y = N / (2);                // y, the vertical
    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int total_threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();       // current tid, same programming model as cuda
        const int tile_id_x = tid % (NUM_THREADS / 2);  // 0, 1, 2, 3
        const int tile_id_y = tid / (NUM_THREADS / 2);  // 0, 1
        const int block_x = tile_id_x * TILE_SIZE_X;
        const int block_y = tile_id_y * TILE_SIZE_Y;

        // compute
        for (int m = 0; m < TILE_SIZE_Y; m++) {
            int global_idx_m = block_y + m;
            for (int n = 0; n < TILE_SIZE_X; n++) {
                int global_idx_n = block_x + n;
                for (int k = 0; k < M; k++) {
                    c[OFFSET(global_idx_m, global_idx_n, N)] += a[OFFSET(global_idx_m, k, K)] * b[OFFSET(k, global_idx_n, N)];
                }
            }
        }
    }
}

// ================================================= v3 ====================================================
void sgemm_v3_micro_kernel(float* a, float* b, float* c, 
                            size_t C_M, size_t C_N, size_t C_K, 
                            size_t lda, size_t ldb, size_t ldc) {
    for (int i = 0; i < C_M; i++) {
        float* _c_row = c + (i) * ldc;   // we will compute each row
        float* _a_row = a + (i) * lda;
        for (int j = 0; j < C_N; j++) {     // col
            float* _b = b + (j) * ldb;
            for (int k = 0; k < C_K; k++) { // row for b, col for a
                _c_row[j] += _a_row[k] * _b[k * ldb];
            }
        }
    }
}

// the input a,b,c will be added the thread block offset before passing in.
template <const int TILE_SIZE = 64>
void sgemm_v3_recursive(float* a, float* b, float* c, 
                            size_t C_M, size_t C_N, size_t C_K, 
                            size_t lda, size_t ldb, size_t ldc) {
    if (C_M <= TILE_SIZE && C_N <= TILE_SIZE) {
        sgemm_v3_micro_kernel(a, b, c, C_M, C_N, C_K, lda, ldb, ldc);
        return;
    }

    // divide recursively
    // assume we divide the matrix 2x2
    const size_t M_u = C_M / 2;     // `u` stands for up
    const size_t M_d = C_M - M_u;   // `d` stands for down
    const size_t N_l = C_N / 2;     // `l` stands for left
    const size_t N_r = C_N - N_l;

    sgemm_v3_recursive(a,               b,              c,                      M_u, N_l, C_K, lda, ldb, ldc);
    sgemm_v3_recursive(a + M_u * lda, b,            c + M_u * ldc,          M_d, N_l, C_K, lda, ldb, ldc);
    sgemm_v3_recursive(a,               b + N_l,    c + N_l,                M_u, N_r, C_K, lda, ldb, ldc);
    sgemm_v3_recursive(a + M_u * lda, b + N_l,      c + M_u * ldc + N_l, M_d, N_r, C_K, lda, ldb, ldc);
}

template <const int NUM_THREADS>
void sgemm_v3(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int TILE_SIZE_X = M / (NUM_THREADS / 2);  // ------------ > x, the horizon
    const int TILE_SIZE_Y = N / (2);                // y, the vertical

    constexpr int TILE_SIZE = 64;

    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int tid = omp_get_thread_num();       // current tid, same programming model as cuda
        const int tile_id_x = tid % (NUM_THREADS / 2);  // 0, 1, 2, 3
        const int tile_id_y = tid / (NUM_THREADS / 2);  // 0, 1
        const int block_x = tile_id_x * TILE_SIZE_X;
        const int block_y = tile_id_y * TILE_SIZE_Y;

        sgemm_v3_recursive<TILE_SIZE>(a + block_y * K, 
                            b + block_x, 
                            c + block_y * N + block_x, 
                        TILE_SIZE_Y, TILE_SIZE_X, K, 
                        K, N, N);
    }
}

// ================================================= v4 ====================================================
template <int BASE>
static inline void sgemm_micro_v4_blockbuffer_kernel(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict       C,
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    float acc[BASE * BASE];                      // 2D flatten
    for (size_t i = 0; i < C_M; ++i) {
        float* row = acc + i * BASE;
        for (size_t j = 0; j < C_N; ++j) row[j] = 0.0f;
    }

    for (size_t k = 0; k < C_K; ++k) {
        const float* __restrict b_row = B + k * ldb;   // B[k, :]
        for (size_t i = 0; i < C_M; ++i) {
            const float a_ik = A[i * lda + k];
            float* acc_row = acc + i * BASE;
            for (size_t j = 0; j < C_N; ++j) {
                acc_row[j] += a_ik * b_row[j];
            }
        }
    }

    for (size_t i = 0; i < C_M; ++i) {
        const float* acc_row = acc + i * BASE;
        float* __restrict c_row = C + i * ldc;
        for (size_t j = 0; j < C_N; ++j) {
            c_row[j] += acc_row[j];
        }
    }
}

template <const int BASE = 64 /* threshold */>
void sgemm_v4_recursive(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict       C,
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    if (C_M <= (size_t)BASE && C_N <= (size_t)BASE && C_K <= (size_t)BASE) {
        sgemm_micro_v4_blockbuffer_kernel<BASE>(A, B, C, C_M, C_N, C_K, lda, ldb, ldc);
        return;
    }

    // split recursively along the longest dim
    if (C_M >= C_N && C_M >= C_K) {
        // split M
        const size_t M1 = C_M / 2;
        const size_t M2 = C_M - M1;
        sgemm_v4_recursive<BASE>(A,              B, C,              M1, C_N, C_K, lda, ldb, ldc);
        sgemm_v4_recursive<BASE>(A + M1 * lda,   B, C + M1 * ldc,   M2, C_N, C_K, lda, ldb, ldc);
    } else if (C_N >= C_M && C_N >= C_K) {
        // split N
        const size_t N1 = C_N / 2;
        const size_t N2 = C_N - N1;
        sgemm_v4_recursive<BASE>(A, B,             C,             C_M, N1, C_K, lda, ldb, ldc);
        sgemm_v4_recursive<BASE>(A, B + N1,        C + N1,        C_M, N2, C_K, lda, ldb, ldc);
    } else {
        // split K: this is outer product!
        const size_t K1 = C_K / 2;
        const size_t K2 = C_K - K1;
        sgemm_v4_recursive<BASE>(A,           B,             C, C_M, C_N, K1, lda, ldb, ldc);
        sgemm_v4_recursive<BASE>(A + K1,      B + K1 * ldb,  C, C_M, C_N, K2, lda, ldb, ldc);
    }
}

template <const int NUM_THREADS, const int TILE_BASE>
void sgemm_v4(float* a, float* b, float* c, const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int tiles_x = NUM_THREADS / 2;   // 4
    const int tiles_y = 2;                 // 2
    // const int tiles_x = 1;   // 4
    // const int tiles_y = 1;                 // 2

    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int tid = omp_get_thread_num();
        const int tx  = tid % tiles_x;     // 0..3
        const int ty  = tid / tiles_x;     // 0..1

        const size_t x0 = (size_t)tx * N / tiles_x;
        const size_t x1 = (size_t)(tx + 1) * N / tiles_x;
        const size_t y0 = (size_t)ty * M / tiles_y;
        const size_t y1 = (size_t)(ty + 1) * M / tiles_y;

        const size_t C_M = (y1 - y0);
        const size_t C_N = (x1 - x0);

        if (!(C_M == 0 || C_N == 0)) {
            const float* Ablk = a + y0 * K;            // A[y0: , 0:]
            const float* Bblk = b + x0;                // B[0: , x0:]
            float*       Cblk = c + y0 * N + x0;       // C[y0: , x0:]
    
            for (size_t i = 0; i < C_M; ++i) {
                std::fill_n(Cblk + i * N, C_N, 0.0f);  // ldc=N
            }
    
            sgemm_v4_recursive<TILE_BASE>(
                Ablk, Bblk, Cblk,
                C_M, C_N, K,
                K,   N,   N
            );
        } else {
            printf("666\n");
        }

    }
}

// tuning zone: TILE_BASE
template <const int BASE>
static void call_sgemm_v4_8t(float* a, float* b, float* c,
                             std::size_t M, std::size_t N, std::size_t K) {
    sgemm_v4<8, BASE>(a, b, c, M, N, K);
}

using GemmFn = void(*)(float*, float*, float*,
                       std::size_t, std::size_t, std::size_t);

static GemmFn pick_impl_8t(int tile_base) {
    switch (tile_base) {
        case 16:  return &call_sgemm_v4_8t<16>;
        case 32:  return &call_sgemm_v4_8t<32>;
        case 48:  return &call_sgemm_v4_8t<48>;
        case 64:  return &call_sgemm_v4_8t<64>;
        case 96:  return &call_sgemm_v4_8t<96>;
        case 128: return &call_sgemm_v4_8t<128>;
        default:  return nullptr;
    }
}


// ================================================= v5 ====================================================
static inline __m256i avx_tail_mask_8(int n) {
    // n ∈ [0,8]
    alignas(32) int32_t m[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < n; ++i) m[i] = -1;
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(m));
}

template <int BASE>
static inline void sgemm_micro_v5_avx(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict       C,
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    for (size_t i0 = 0; i0 < C_M; i0 += 8) {
        const int im = static_cast<int>(std::min<size_t>(8, C_M - i0));
        for (size_t j0 = 0; j0 < C_N; j0 += 8) {
            const int jm = static_cast<int>(std::min<size_t>(8, C_N - j0));

            __m256 acc[8];
            for (int r = 0; r < 8; ++r) acc[r] = _mm256_setzero_ps();

            const __m256i mask = (jm == 8) ? _mm256_set1_epi32(-1) : avx_tail_mask_8(jm);

            for (size_t k = 0; k < C_K; ++k) {
                const float* __restrict b_ptr = B + k * ldb + j0;
                __m256 bvec = (jm == 8)
                              ? _mm256_loadu_ps(b_ptr)
                              : _mm256_maskload_ps(b_ptr, mask);

                for (int r = 0; r < im; ++r) {
                    const float a_ik = A[(i0 + r) * lda + k];
                    __m256 a_bcast = _mm256_set1_ps(a_ik);
                    acc[r] = _mm256_fmadd_ps(a_bcast, bvec, acc[r]);
                }
            }

            for (int r = 0; r < im; ++r) {
                float* c_ptr = C + (i0 + r) * ldc + j0;
                __m256 cvec = (jm == 8)
                              ? _mm256_loadu_ps(c_ptr)
                              : _mm256_maskload_ps(c_ptr, mask);
                cvec = _mm256_add_ps(cvec, acc[r]);
                if (jm == 8) _mm256_storeu_ps(c_ptr, cvec);
                else         _mm256_maskstore_ps(c_ptr, mask, cvec);
            }
        }
    }
}

template <const int BASE = 64 /* threshold */>
void sgemm_v5_recursive(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict       C,
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    if (C_M <= (size_t)BASE && C_N <= (size_t)BASE && C_K <= (size_t)BASE) {
        sgemm_micro_v5_avx<BASE>(A, B, C, C_M, C_N, C_K, lda, ldb, ldc);
        return;
    }

    // split recursively along the longest dim
    if (C_M >= C_N && C_M >= C_K) {
        // split M
        const size_t M1 = C_M / 2;
        const size_t M2 = C_M - M1;
        sgemm_v5_recursive<BASE>(A,              B, C,              M1, C_N, C_K, lda, ldb, ldc);
        sgemm_v5_recursive<BASE>(A + M1 * lda,   B, C + M1 * ldc,   M2, C_N, C_K, lda, ldb, ldc);
    } else if (C_N >= C_M && C_N >= C_K) {
        // split N
        const size_t N1 = C_N / 2;
        const size_t N2 = C_N - N1;
        sgemm_v5_recursive<BASE>(A, B,             C,             C_M, N1, C_K, lda, ldb, ldc);
        sgemm_v5_recursive<BASE>(A, B + N1,        C + N1,        C_M, N2, C_K, lda, ldb, ldc);
    } else {
        // split K: this is outer product!
        const size_t K1 = C_K / 2;
        const size_t K2 = C_K - K1;
        sgemm_v5_recursive<BASE>(A,           B,             C, C_M, C_N, K1, lda, ldb, ldc);
        sgemm_v5_recursive<BASE>(A + K1,      B + K1 * ldb,  C, C_M, C_N, K2, lda, ldb, ldc);
    }
}

template <const int NUM_THREADS, const int TILE_BASE>
void sgemm_v5(float* a, float* b, float* c,
              const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int tiles_x = NUM_THREADS / 2;   // 4
    const int tiles_y = 2;                 // 2
    // const int tiles_x = 1;   // 4
    // const int tiles_y = 1;                 // 2

    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int tid = omp_get_thread_num();
        const int tx  = tid % tiles_x;     // 0..3
        const int ty  = tid / tiles_x;     // 0..1

        const size_t x0 = (size_t)tx * N / tiles_x;
        const size_t x1 = (size_t)(tx + 1) * N / tiles_x;
        const size_t y0 = (size_t)ty * M / tiles_y;
        const size_t y1 = (size_t)(ty + 1) * M / tiles_y;

        const size_t C_M = (y1 - y0);
        const size_t C_N = (x1 - x0);

        if (!(C_M == 0 || C_N == 0)) {
            const float* Ablk = a + y0 * K;            // A[y0: , 0:]
            const float* Bblk = b + x0;                // B[0: , x0:]
            float*       Cblk = c + y0 * N + x0;       // C[y0: , x0:]
    
            for (size_t i = 0; i < C_M; ++i) {
                std::fill_n(Cblk + i * N, C_N, 0.0f);  // ldc=N
            }
    
            sgemm_v5_recursive<TILE_BASE>(
                Ablk, Bblk, Cblk,
                C_M, C_N, K,
                K,   N,   N
            );
        } else {
            printf("666\n");
        }

    }
}

// ================================================= v6 ====================================================
static inline __m256i mask8(int n) {
    alignas(32) int32_t m[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < n; ++i) m[i] = -1;
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(m));
}

static inline void sgemm_micro_v6_avx_outer_8x6_colmajor(
    const float* __restrict A,  // [M,K], col-major, lda=M
    const float* __restrict B,  // [K,N], col-major, ldb=K
    float* __restrict       C,  // [M,N], col-major, ldc=M
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    for (size_t i0 = 0; i0 < C_M; i0 += 8) {
        const int im = (int)std::min<size_t>(8, C_M - i0);
        const __m256i rmask = (im == 8) ? _mm256_set1_epi32(-1) : mask8(im);

        for (size_t j0 = 0; j0 < C_N; j0 += 6) {
            const int jm = (int)std::min<size_t>(6, C_N - j0);

            // 6 x 8-lane accumulator j0..j0+jm-1
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();

            size_t k = 0;

            for (; k + 3 < C_K; k += 4) {
                // unroll manually: k+0
                {
                    const float* a_ptr = A + (k + 0) * lda + i0;  // A[i0 : i0 + 8, k + 0]
                    __m256 avec = (im == 8) ? _mm256_loadu_ps(a_ptr)
                                            : _mm256_maskload_ps(a_ptr, rmask);

                    const float* b = B + (j0 + 0) * ldb + (k + 0);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 0*ldb), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 1*ldb), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 2*ldb), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 3*ldb), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 4*ldb), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 5*ldb), c5);
                }
                // unroll manually: k+1
                {
                    const float* a_ptr = A + (k + 1) * lda + i0;
                    __m256 avec = (im == 8) ? _mm256_loadu_ps(a_ptr)
                                            : _mm256_maskload_ps(a_ptr, rmask);
                    const float* b = B + (j0 + 0) * ldb + (k + 1);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 0*ldb), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 1*ldb), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 2*ldb), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 3*ldb), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 4*ldb), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 5*ldb), c5);
                }
                // unroll manually: k+2
                {
                    const float* a_ptr = A + (k + 2) * lda + i0;
                    __m256 avec = (im == 8) ? _mm256_loadu_ps(a_ptr)
                                            : _mm256_maskload_ps(a_ptr, rmask);
                    const float* b = B + (j0 + 0) * ldb + (k + 2);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 0*ldb), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 1*ldb), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 2*ldb), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 3*ldb), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 4*ldb), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 5*ldb), c5);
                }
                // unroll manually: k+3
                {
                    const float* a_ptr = A + (k + 3) * lda + i0;
                    __m256 avec = (im == 8) ? _mm256_loadu_ps(a_ptr)
                                            : _mm256_maskload_ps(a_ptr, rmask);
                    const float* b = B + (j0 + 0) * ldb + (k + 3);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 0*ldb), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 1*ldb), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 2*ldb), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 3*ldb), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 4*ldb), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(b + 5*ldb), c5);
                }
            }

            // ---- rest K ----
            for (; k < C_K; ++k) {
                const float* a_ptr = A + k * lda + i0;
                __m256 avec = (im == 8) ? _mm256_loadu_ps(a_ptr)
                                        : _mm256_maskload_ps(a_ptr, rmask);
                // B[k, j0 + t]
                if (jm >= 1) c0 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 0)*ldb + k), c0);
                if (jm >= 2) c1 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 1)*ldb + k), c1);
                if (jm >= 3) c2 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 2)*ldb + k), c2);
                if (jm >= 4) c3 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 3)*ldb + k), c3);
                if (jm >= 5) c4 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 4)*ldb + k), c4);
                if (jm >= 6) c5 = _mm256_fmadd_ps(avec, _mm256_broadcast_ss(B + (j0 + 5)*ldb + k), c5);
            }

            // ---- Write back ----
            float* c0p = C + (j0 + 0) * ldc + i0;
            float* c1p = C + (j0 + 1) * ldc + i0;
            float* c2p = C + (j0 + 2) * ldc + i0;
            float* c3p = C + (j0 + 3) * ldc + i0;
            float* c4p = C + (j0 + 4) * ldc + i0;
            float* c5p = C + (j0 + 5) * ldc + i0;

            if (im == 8) {
                if (jm >= 1) _mm256_storeu_ps(c0p, _mm256_add_ps(_mm256_loadu_ps(c0p), c0));
                if (jm >= 2) _mm256_storeu_ps(c1p, _mm256_add_ps(_mm256_loadu_ps(c1p), c1));
                if (jm >= 3) _mm256_storeu_ps(c2p, _mm256_add_ps(_mm256_loadu_ps(c2p), c2));
                if (jm >= 4) _mm256_storeu_ps(c3p, _mm256_add_ps(_mm256_loadu_ps(c3p), c3));
                if (jm >= 5) _mm256_storeu_ps(c4p, _mm256_add_ps(_mm256_loadu_ps(c4p), c4));
                if (jm >= 6) _mm256_storeu_ps(c5p, _mm256_add_ps(_mm256_loadu_ps(c5p), c5));
            } else {
                if (jm >= 1) {
                    __m256 old = _mm256_maskload_ps(c0p, rmask);
                    _mm256_maskstore_ps(c0p, rmask, _mm256_add_ps(old, c0));
                }
                if (jm >= 2) {
                    __m256 old = _mm256_maskload_ps(c1p, rmask);
                    _mm256_maskstore_ps(c1p, rmask, _mm256_add_ps(old, c1));
                }
                if (jm >= 3) {
                    __m256 old = _mm256_maskload_ps(c2p, rmask);
                    _mm256_maskstore_ps(c2p, rmask, _mm256_add_ps(old, c2));
                }
                if (jm >= 4) {
                    __m256 old = _mm256_maskload_ps(c3p, rmask);
                    _mm256_maskstore_ps(c3p, rmask, _mm256_add_ps(old, c3));
                }
                if (jm >= 5) {
                    __m256 old = _mm256_maskload_ps(c4p, rmask);
                    _mm256_maskstore_ps(c4p, rmask, _mm256_add_ps(old, c4));
                }
                if (jm >= 6) {
                    __m256 old = _mm256_maskload_ps(c5p, rmask);
                    _mm256_maskstore_ps(c5p, rmask, _mm256_add_ps(old, c5));
                }
            }
        }
    }
}

template <const int BASE = 64 /* threshold */>
void sgemm_v6_recursive(
    const float* __restrict A,  // col-major
    const float* __restrict B,  // col-major
    float* __restrict       C,  // col-major
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    if (C_M <= (size_t)BASE && C_N <= (size_t)BASE && C_K <= (size_t)BASE) {
        sgemm_micro_v6_avx_outer_8x6_colmajor(A, B, C, C_M, C_N, C_K, lda, ldb, ldc);
        return;
    }

    // split recursively along the longest dim
    if (C_M >= C_N && C_M >= C_K) {
        // split M
        const size_t M1 = C_M / 2;
        const size_t M2 = C_M - M1;
        sgemm_v6_recursive<BASE>(A,              B,     C,              M1, C_N, C_K, lda, ldb, ldc);
        sgemm_v6_recursive<BASE>(A + M1,        B, C + M1,   M2, C_N, C_K, lda, ldb, ldc);      // now we are at col-major!!!!!!!!!
    } else if (C_N >= C_M && C_N >= C_K) {
        // split N
        const size_t N1 = C_N / 2;
        const size_t N2 = C_N - N1;
        sgemm_v6_recursive<BASE>(A, B,                          C,             C_M, N1, C_K, lda, ldb, ldc);
        sgemm_v6_recursive<BASE>(A, B + N1 * ldb,           C + N1 * ldc,        C_M, N2, C_K, lda, ldb, ldc); // now we are at col-major!!!!!!!!!
    } else {
        // split K: this is outer product!
        const size_t K1 = C_K / 2;
        const size_t K2 = C_K - K1;
        sgemm_v6_recursive<BASE>(A,           B,             C, C_M, C_N, K1, lda, ldb, ldc);
        sgemm_v6_recursive<BASE>(A + K1 * lda,      B + K1,  C, C_M, C_N, K2, lda, ldb, ldc);           // now we are at col-major!!!!!!!!!
    }
}

template <const int NUM_THREADS, const int TILE_BASE>
void sgemm_v6(float* a, float* b, float* c,
              const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int tiles_x = NUM_THREADS / 2;   // 4
    const int tiles_y = 2;                 // 2
    // const int tiles_x = 1;   // 4
    // const int tiles_y = 1;                 // 2

    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int tid = omp_get_thread_num();
        const int tx  = tid % tiles_x;     // 0..3
        const int ty  = tid / tiles_x;     // 0..1

        const size_t x0 = (size_t)tx * N / tiles_x;
        const size_t x1 = (size_t)(tx + 1) * N / tiles_x;
        const size_t y0 = (size_t)ty * M / tiles_y;
        const size_t y1 = (size_t)(ty + 1) * M / tiles_y;

        const size_t C_M = (y1 - y0);
        const size_t C_N = (x1 - x0);

        if (!(C_M == 0 || C_N == 0)) {
            const float* Ablk = a + y0;                         // A[y0: , 0:]
            const float* Bblk = b + x0 * K;                     // B[0: , x0:], ldb = K
            float*       Cblk = c + y0 + x0 * M;                // C[y0: , x0:], ldc = M
    
            for (size_t j = 0; j < C_N; ++j) {
                std::fill_n(Cblk + j * M, C_M, 0.0f);       // now we are at col-major!!!!!!!!!
            }
    
            sgemm_v6_recursive<TILE_BASE>(
                Ablk, Bblk, Cblk,
                C_M, C_N, K,
                M,   K,   M
            );
        } else {
            printf("666\n");
        }

    }
}

// ================================================= v7 ====================================================
template<int BASE>
static inline void sgemm_micro_v7_pack_avx_outer_8x6_colmajor_v1(
    const float* __restrict A,  // [M,K], col-major, lda=M
    const float* __restrict B,  // [K,N], col-major, ldb=K
    float* __restrict       C,  // [M,N], col-major, ldc=M
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    for (size_t i0 = 0; i0 < C_M; i0 += 8) {
        const int im = (int)std::min<size_t>(8, C_M - i0);
        const __m256i rmask = (im == 8) ? _mm256_set1_epi32(-1) : mask8(im);

        // packing A
        alignas(32) float Apack[BASE * 8];
        for (int k = 0; k < (int)C_K; ++k) {
            const float* ap = A + (size_t)k * lda + i0; // A[i0:i0+8, k]
            __m256 avec = (im == 8) ? _mm256_loadu_ps(ap)
                                    : _mm256_maskload_ps(ap, rmask);
            _mm256_store_ps(Apack + k * 8, avec);
        }

        for (size_t j0 = 0; j0 < C_N; j0 += 6) {
            const int jm = (int)std::min<size_t>(6, C_N - j0);

            // packing B: transpose
            alignas(32) float Bpack[BASE * 8];
            int bp_ofs = 0;
            for (int kb = 0; kb < (int)C_K; kb += 8) {
                const int u = std::min(8, (int)C_K - kb);
                const __m256i kmask = (u == 8) ? _mm256_set1_epi32(-1) : mask8(u);
                for (int t = 0; t < jm; ++t) {
                    const float* src = B + (size_t)(j0 + t) * ldb + kb; // B[kb:kb+u, col]
                    __m256 v = (u == 8) ? _mm256_loadu_ps(src)
                                        : _mm256_maskload_ps(src, kmask);
                    _mm256_store_ps(Bpack + bp_ofs + t*8, v);
                }
                bp_ofs += 6 * 8; // next 6*8 tile
            }

            // outer product
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();

            bp_ofs = 0;
            for (int kb = 0; kb < (int)C_K; kb += 8) {
                const int u = std::min(8, (int)C_K - kb);
                const float* bp = Bpack + bp_ofs;
                for (int kk = 0; kk < u; ++kk) {
                    __m256 a = _mm256_load_ps(Apack + (kb + kk) * 8);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[0*8 + kk]), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[1*8 + kk]), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[2*8 + kk]), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[3*8 + kk]), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[4*8 + kk]), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[5*8 + kk]), c5);
                }
                bp_ofs += 6 * 8;
            }

            float* c0p = C + (j0 + 0) * ldc + i0;
            float* c1p = C + (j0 + 1) * ldc + i0;
            float* c2p = C + (j0 + 2) * ldc + i0;
            float* c3p = C + (j0 + 3) * ldc + i0;
            float* c4p = C + (j0 + 4) * ldc + i0;
            float* c5p = C + (j0 + 5) * ldc + i0;

            if (im == 8) {
                if (jm >= 1) _mm256_storeu_ps(c0p, _mm256_add_ps(_mm256_loadu_ps(c0p), c0));
                if (jm >= 2) _mm256_storeu_ps(c1p, _mm256_add_ps(_mm256_loadu_ps(c1p), c1));
                if (jm >= 3) _mm256_storeu_ps(c2p, _mm256_add_ps(_mm256_loadu_ps(c2p), c2));
                if (jm >= 4) _mm256_storeu_ps(c3p, _mm256_add_ps(_mm256_loadu_ps(c3p), c3));
                if (jm >= 5) _mm256_storeu_ps(c4p, _mm256_add_ps(_mm256_loadu_ps(c4p), c4));
                if (jm >= 6) _mm256_storeu_ps(c5p, _mm256_add_ps(_mm256_loadu_ps(c5p), c5));
            } else {
                if (jm >= 1) { __m256 old = _mm256_maskload_ps(c0p, rmask);
                               _mm256_maskstore_ps(c0p, rmask, _mm256_add_ps(old, c0)); }
                if (jm >= 2) { __m256 old = _mm256_maskload_ps(c1p, rmask);
                               _mm256_maskstore_ps(c1p, rmask, _mm256_add_ps(old, c1)); }
                if (jm >= 3) { __m256 old = _mm256_maskload_ps(c2p, rmask);
                               _mm256_maskstore_ps(c2p, rmask, _mm256_add_ps(old, c2)); }
                if (jm >= 4) { __m256 old = _mm256_maskload_ps(c3p, rmask);
                               _mm256_maskstore_ps(c3p, rmask, _mm256_add_ps(old, c3)); }
                if (jm >= 5) { __m256 old = _mm256_maskload_ps(c4p, rmask);
                               _mm256_maskstore_ps(c4p, rmask, _mm256_add_ps(old, c4)); }
                if (jm >= 6) { __m256 old = _mm256_maskload_ps(c5p, rmask);
                               _mm256_maskstore_ps(c5p, rmask, _mm256_add_ps(old, c5)); }
            }
        }
    }
}

template<int BASE /*=64*/>
static inline void sgemm_micro_v7_pack_avx_outer_8x6_colmajor_v2(
    const float* __restrict A,  // [M,K], col-major, lda=M
    const float* __restrict B,  // [K,N], col-major, ldb=K
    float* __restrict       C,  // [M,N], col-major, ldc=M
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    for (size_t i0 = 0; i0 < C_M; i0 += 8) {
        const int im = (int)std::min<size_t>(8, C_M - i0);
        const __m256i rmask = (im == 8) ? _mm256_set1_epi32(-1) : mask8(im);

        // 预打包 A: Apack[k] 是一个 8-lane 向量（尾行 mask）
        alignas(32) float Apack[BASE * 8];
        for (int k = 0; k < (int)C_K; ++k) {
            const float* ap = A + (size_t)k * lda + i0; // A[i0:i0+8, k]
            __m256 avec = (im == 8) ? _mm256_loadu_ps(ap)
                                    : _mm256_maskload_ps(ap, rmask);
            _mm256_store_ps(Apack + k * 8, avec);
        }

        for (size_t j0 = 0; j0 < C_N; j0 += 6) {
            const int jm = (int)std::min<size_t>(6, C_N - j0);

            // 预打包 B: 以 8 为块，按列打包到 Bpack（列 stride 固定为 8）
            alignas(32) float Bpack[BASE * 8]; // 足够容纳 jm 列 × ceil(kc/8)*8
            int bp_ofs = 0; // 以 8 为单元的块偏移（单位：float）
            for (int kb = 0; kb < (int)C_K; kb += 8) {
                const int u = std::min(8, (int)C_K - kb);
                const __m256i kmask = (u == 8) ? _mm256_set1_epi32(-1) : mask8(u);
                
                // 列主序下，6 列指针
                const float* a0 = (jm >= 1) ? (B + (size_t)(j0 + 0) * ldb + kb) : nullptr;
                const float* a1 = (jm >= 2) ? (B + (size_t)(j0 + 1) * ldb + kb) : nullptr;
                const float* a2 = (jm >= 3) ? (B + (size_t)(j0 + 2) * ldb + kb) : nullptr;
                const float* a3 = (jm >= 4) ? (B + (size_t)(j0 + 3) * ldb + kb) : nullptr;
                const float* a4 = (jm >= 5) ? (B + (size_t)(j0 + 4) * ldb + kb) : nullptr;
                const float* a5 = (jm >= 6) ? (B + (size_t)(j0 + 5) * ldb + kb) : nullptr;

                // 读入 6 个 8-lane 向量；缺列用 0，尾块用 maskload 补 0
                const __m256 z = _mm256_setzero_ps();
                __m256 v0 = (jm >= 1) ? ((u == 8) ? _mm256_loadu_ps(a0) : _mm256_maskload_ps(a0, kmask)) : z;
                __m256 v1 = (jm >= 2) ? ((u == 8) ? _mm256_loadu_ps(a1) : _mm256_maskload_ps(a1, kmask)) : z;
                __m256 v2 = (jm >= 3) ? ((u == 8) ? _mm256_loadu_ps(a2) : _mm256_maskload_ps(a2, kmask)) : z;
                __m256 v3 = (jm >= 4) ? ((u == 8) ? _mm256_loadu_ps(a3) : _mm256_maskload_ps(a3, kmask)) : z;
                __m256 v4 = (jm >= 5) ? ((u == 8) ? _mm256_loadu_ps(a4) : _mm256_maskload_ps(a4, kmask)) : z;
                __m256 v5 = (jm >= 6) ? ((u == 8) ? _mm256_loadu_ps(a5) : _mm256_maskload_ps(a5, kmask)) : z;

                // 6x8 -> 8x6 转置网络（与你提供的指令序列等价）
                __m256 unpack0 = _mm256_unpacklo_ps(v0, v1);
                __m256 unpack1 = _mm256_unpackhi_ps(v0, v1);
                __m256 unpack2 = _mm256_unpacklo_ps(v2, v3);
                __m256 unpack3 = _mm256_unpackhi_ps(v2, v3);
                __m256 unpack4 = _mm256_unpacklo_ps(v4, v5);
                __m256 unpack5 = _mm256_unpackhi_ps(v4, v5);

                __m256 shf0 = _mm256_shuffle_ps(unpack0, unpack2, 0x44);
                __m256 shf1 = _mm256_shuffle_ps(unpack4, unpack0, 0xE4);
                __m256 shf2 = _mm256_shuffle_ps(unpack2, unpack4, 0xEE);
                __m256 shf3 = _mm256_shuffle_ps(unpack5, unpack1, 0xE4);
                __m256 shf4 = _mm256_shuffle_ps(unpack3, unpack5, 0xEE);
                __m256 shf5 = _mm256_shuffle_ps(unpack1, unpack3, 0x44);

                __m128 low_shf1 = _mm256_castps256_ps128(shf1);
                __m256 res0 = _mm256_insertf128_ps(shf0, low_shf1, 0x1);
                __m256 res1 = _mm256_permute2f128_ps(shf0, shf1, 0x31);

                __m128 low_shf5 = _mm256_castps256_ps128(shf5);
                __m256 res2 = _mm256_insertf128_ps(shf2, low_shf5, 0x1);
                __m256 res3 = _mm256_permute2f128_ps(shf2, shf5, 0x31);

                __m128 low_shf4 = _mm256_castps256_ps128(shf4);
                __m256 res4 = _mm256_insertf128_ps(shf3, low_shf4, 0x1);
                __m256 res5 = _mm256_permute2f128_ps(shf3, shf4, 0x31);

                // 直接 store 到 Bpack：每列 stride=8（一个块写 6 次，每次 8 个 float）
                float* dst = Bpack + bp_ofs;
                if (jm >= 1) _mm256_storeu_ps(dst + 0 * 8, res0);
                if (jm >= 2) _mm256_storeu_ps(dst + 1 * 8, res2);
                if (jm >= 3) _mm256_storeu_ps(dst + 2 * 8, res4);
                if (jm >= 4) _mm256_storeu_ps(dst + 3 * 8, res1);
                if (jm >= 5) _mm256_storeu_ps(dst + 4 * 8, res3);
                if (jm >= 6) _mm256_storeu_ps(dst + 5 * 8, res5);

                bp_ofs += 6 * 8; // 下一 8 行 k-block
            }

            // 计算：8x6 外积，遍历 k 的 8 块；B 从 Bpack 按 stride=8 取标量
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();

            bp_ofs = 0;
            for (int kb = 0; kb < (int)C_K; kb += 8) {
                const int u = std::min(8, (int)C_K - kb);
                const float* bp = Bpack + bp_ofs;
                for (int kk = 0; kk < u; ++kk) {
                    __m256 a = _mm256_load_ps(Apack + (kb + kk) * 8);
                    if (jm >= 1) c0 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[0*8 + kk]), c0);
                    if (jm >= 2) c1 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[1*8 + kk]), c1);
                    if (jm >= 3) c2 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[2*8 + kk]), c2);
                    if (jm >= 4) c3 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[3*8 + kk]), c3);
                    if (jm >= 5) c4 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[4*8 + kk]), c4);
                    if (jm >= 6) c5 = _mm256_fmadd_ps(a, _mm256_set1_ps(bp[5*8 + kk]), c5);
                }
                bp_ofs += 6 * 8; // 跳到下一个 8 行块
            }

            // 回写到 C（列主序），尾行用 mask
            float* c0p = C + (j0 + 0) * ldc + i0;
            float* c1p = C + (j0 + 1) * ldc + i0;
            float* c2p = C + (j0 + 2) * ldc + i0;
            float* c3p = C + (j0 + 3) * ldc + i0;
            float* c4p = C + (j0 + 4) * ldc + i0;
            float* c5p = C + (j0 + 5) * ldc + i0;

            if (im == 8) {
                if (jm >= 1) _mm256_storeu_ps(c0p, _mm256_add_ps(_mm256_loadu_ps(c0p), c0));
                if (jm >= 2) _mm256_storeu_ps(c1p, _mm256_add_ps(_mm256_loadu_ps(c1p), c1));
                if (jm >= 3) _mm256_storeu_ps(c2p, _mm256_add_ps(_mm256_loadu_ps(c2p), c2));
                if (jm >= 4) _mm256_storeu_ps(c3p, _mm256_add_ps(_mm256_loadu_ps(c3p), c3));
                if (jm >= 5) _mm256_storeu_ps(c4p, _mm256_add_ps(_mm256_loadu_ps(c4p), c4));
                if (jm >= 6) _mm256_storeu_ps(c5p, _mm256_add_ps(_mm256_loadu_ps(c5p), c5));
            } else {
                if (jm >= 1) { __m256 old = _mm256_maskload_ps(c0p, rmask);
                               _mm256_maskstore_ps(c0p, rmask, _mm256_add_ps(old, c0)); }
                if (jm >= 2) { __m256 old = _mm256_maskload_ps(c1p, rmask);
                               _mm256_maskstore_ps(c1p, rmask, _mm256_add_ps(old, c1)); }
                if (jm >= 3) { __m256 old = _mm256_maskload_ps(c2p, rmask);
                               _mm256_maskstore_ps(c2p, rmask, _mm256_add_ps(old, c2)); }
                if (jm >= 4) { __m256 old = _mm256_maskload_ps(c3p, rmask);
                               _mm256_maskstore_ps(c3p, rmask, _mm256_add_ps(old, c3)); }
                if (jm >= 5) { __m256 old = _mm256_maskload_ps(c4p, rmask);
                               _mm256_maskstore_ps(c4p, rmask, _mm256_add_ps(old, c4)); }
                if (jm >= 6) { __m256 old = _mm256_maskload_ps(c5p, rmask);
                               _mm256_maskstore_ps(c5p, rmask, _mm256_add_ps(old, c5)); }
            }
        }
    }
}

template <const int BASE = 64 /* threshold */>
void sgemm_v7_recursive(
    const float* __restrict A,  // col-major
    const float* __restrict B,  // col-major
    float* __restrict       C,  // col-major
    size_t C_M, size_t C_N, size_t C_K,
    size_t lda, size_t ldb, size_t ldc)
{
    if (C_M <= (size_t)BASE && C_N <= (size_t)BASE && C_K <= (size_t)BASE) {
        sgemm_micro_v7_pack_avx_outer_8x6_colmajor_v1<BASE>(A, B, C, C_M, C_N, C_K, lda, ldb, ldc);
        return;
    }

    // split recursively along the longest dim
    if (C_M >= C_N && C_M >= C_K) {
        // split M
        const size_t M1 = C_M / 2;
        const size_t M2 = C_M - M1;
        sgemm_v7_recursive<BASE>(A,              B,     C,              M1, C_N, C_K, lda, ldb, ldc);
        sgemm_v7_recursive<BASE>(A + M1,        B, C + M1,   M2, C_N, C_K, lda, ldb, ldc);      // now we are at col-major!!!!!!!!!
    } else if (C_N >= C_M && C_N >= C_K) {
        // split N
        const size_t N1 = C_N / 2;
        const size_t N2 = C_N - N1;
        sgemm_v7_recursive<BASE>(A, B,                          C,             C_M, N1, C_K, lda, ldb, ldc);
        sgemm_v7_recursive<BASE>(A, B + N1 * ldb,           C + N1 * ldc,        C_M, N2, C_K, lda, ldb, ldc); // now we are at col-major!!!!!!!!!
    } else {
        // split K: this is outer product!
        const size_t K1 = C_K / 2;
        const size_t K2 = C_K - K1;
        sgemm_v7_recursive<BASE>(A,           B,             C, C_M, C_N, K1, lda, ldb, ldc);
        sgemm_v7_recursive<BASE>(A + K1 * lda,      B + K1,  C, C_M, C_N, K2, lda, ldb, ldc);           // now we are at col-major!!!!!!!!!
    }
}

template <const int NUM_THREADS, const int TILE_BASE>
void sgemm_v7(float* a, float* b, float* c,
              const size_t M, const size_t N, const size_t K) {
    // use multi thread to handle this GEMM process
    const int tiles_x = NUM_THREADS / 2;   // 4
    const int tiles_y = 2;                 // 2
    // const int tiles_x = 1;   // 4
    // const int tiles_y = 1;                 // 2

    #pragma omp parallel num_threads(NUM_THREADS) proc_bind(spread)
    {
        const int tid = omp_get_thread_num();
        const int tx  = tid % tiles_x;     // 0..3
        const int ty  = tid / tiles_x;     // 0..1

        const size_t x0 = (size_t)tx * N / tiles_x;
        const size_t x1 = (size_t)(tx + 1) * N / tiles_x;
        const size_t y0 = (size_t)ty * M / tiles_y;
        const size_t y1 = (size_t)(ty + 1) * M / tiles_y;

        const size_t C_M = (y1 - y0);
        const size_t C_N = (x1 - x0);

        if (!(C_M == 0 || C_N == 0)) {
            const float* Ablk = a + y0;                         // A[y0: , 0:]
            const float* Bblk = b + x0 * K;                     // B[0: , x0:], ldb = K
            float*       Cblk = c + y0 + x0 * M;                // C[y0: , x0:], ldc = M
    
            for (size_t j = 0; j < C_N; ++j) {
                std::fill_n(Cblk + j * M, C_M, 0.0f);       // now we are at col-major!!!!!!!!!
            }
    
            sgemm_v7_recursive<TILE_BASE>(
                Ablk, Bblk, Cblk,
                C_M, C_N, K,
                M,   K,   M
            );
        } else {
            printf("666\n");
        }

    }
}

// ============== Binding Zone ==============
// These are boring functions, just jump it
// ==========================================

py::array_t<float> gemm_f32_naive(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_naive(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}

py::array_t<float> gemm_f32_v2(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_v2<8>(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}

py::array_t<float> gemm_f32_v3(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_v3<8>(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}

py::array_t<float> gemm_f32_v4(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B,
    int tile_base = 64
) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    GemmFn fn = pick_impl_8t(tile_base);
    if (!fn)
        throw std::runtime_error("tile_base must be one of {16,32,48,64,96,128}");

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        fn(Ap, Bp, Cp, static_cast<std::size_t>(M),
                       static_cast<std::size_t>(N),
                       static_cast<std::size_t>(K));
    }

    return C;
}

py::array_t<float> gemm_f32_v5(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_v5<8, 64>(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}

py::array_t<float> gemm_f32_v6(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_v6<8, 64>(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}

py::array_t<float> gemm_f32_v7(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B) {

    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    const py::ssize_t M = A.shape(0);
    const py::ssize_t K = A.shape(1);
    const py::ssize_t K2 = B.shape(0);
    const py::ssize_t N = B.shape(1);
    if (K != K2) {
        throw std::runtime_error("A.shape[1] must equal B.shape[0]");
    }

    auto C = py::array_t<float>({M, N});
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* Ap = static_cast<float*>(bufA.ptr);
    float* Bp = static_cast<float*>(bufB.ptr);
    float*       Cp = static_cast<float*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        sgemm_v7<8, 64>(Ap, Bp, Cp, (std::size_t)M, (std::size_t)K, (std::size_t)N);
    }

    return C;
}



PYBIND11_MODULE(cogemm, m) {
    m.doc() = "Minimal GEMM float32 via pybind11";
    m.def("gemm_f32_naive", &gemm_f32_naive, "C = A @ B (float32, C-order)");
    m.def("gemm_f32_v2", &gemm_f32_v2, "C = A @ B (float32, C-order)");
    m.def("gemm_f32_v3", &gemm_f32_v3, "C = A @ B (float32, C-order)");
    m.def("gemm_f32_v4", &gemm_f32_v4, py::arg("A"), py::arg("B"), py::arg("tile_base") = 64, "C = A @ B (float32, C-order)");
    m.def("gemm_f32_v5", &gemm_f32_v5, "C = A @ B (float32, C-order)");
    m.def("gemm_f32_v6", &gemm_f32_v6, "C = A @ B (float32, col-major)");
    m.def("gemm_f32_v7", &gemm_f32_v7, "C = A @ B (float32, col-major, packing)");
}