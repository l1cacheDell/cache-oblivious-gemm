import time
import matplotlib.pyplot as plt
import numpy as np
import cogemm

def check_correctness(a: np.ndarray, b: np.ndarray, version=1, tile_base=64):
    C_ref = a @ b
    a_col = np.transpose(a).copy()  # make it col-major
    b_col = np.transpose(b).copy()  # make it col-major
    match version:
        case 1:
            C = cogemm.gemm_f32_naive(a, b)
        case 2:
            C = cogemm.gemm_f32_v2(a, b)
        case 3:
            C = cogemm.gemm_f32_v3(a, b)
        case 4:
            C = cogemm.gemm_f32_v4(a, b, tile_base=tile_base)
        case 5:
            C = cogemm.gemm_f32_v5(a, b)
        case 6:
            C = cogemm.gemm_f32_v6(a_col, b_col)
            C = np.transpose(C)  # back to row-major
        case 7:
            C = cogemm.gemm_f32_v7(a_col, b_col)
            C = np.transpose(C)  # back to row-major
        case 8:
            C = cogemm.gemm_f32_v8(a_col, b_col)
            C = np.transpose(C)  # back to row-major
        case _:
            raise ValueError("Unsupported version")

    if np.max(np.abs(C - C_ref)) > 1e-4:
        print("Error: max diff |numpy - cpp|:", np.max(np.abs(C - C_ref)))
        print(np.abs(C - C_ref))
        exit(1)


    
def profile_gemm(a: np.ndarray, b: np.ndarray, repeat=10, version=1, tile_base=64):
    a_col = np.transpose(a).copy()  # make it col-major
    b_col = np.transpose(b).copy()  # make it col-major
    t0 = time.perf_counter()
    for _ in range(repeat):
        match version:
            case 1:
                c = cogemm.gemm_f32_naive(a, b)
            case 2:
                c = cogemm.gemm_f32_v2(a, b)
            case 3:
                c = cogemm.gemm_f32_v3(a, b)
            case 4:
                c = cogemm.gemm_f32_v4(a, b, tile_base=tile_base)
            case 5:
                c = cogemm.gemm_f32_v5(a, b)
            case 6:
                c = cogemm.gemm_f32_v6(a_col, b_col)
            case 7:
                c = cogemm.gemm_f32_v7(a_col, b_col)
            case 8:
                c = cogemm.gemm_f32_v8(a_col, b_col)
            case _:
                raise ValueError("Unsupported version")
    t1 = time.perf_counter()

    M, N, K = a.shape[0], b.shape[1], a.shape[1]

    # profile potential float operations
    flops = 2.0 * M * K * N
    gflops = flops / ((t1 - t0) / repeat) / 1e9
    print(f"v{version}, size: {M}, tile_base: {tile_base}, time: {t1 - t0:.4f}s, REPEAT: {repeat}, {gflops:.2f} GFLOP/s")
    return gflops


def check_and_bench_only_v4(M=512, K=512, N=512, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)

    bench_data = []

    total_version = 4

    for version_id in range(1, total_version + 1):     # here, to add more versions, increase the range
        if version_id >= 4:
            check_correctness(A, B, version=version_id)
        if version_id == 4:
            for tile_base in [16, 32, 48, 64, 96, 128]:
                check_correctness(A, B, version=version_id, tile_base=tile_base)
                gflops = profile_gemm(A, B, repeat=50, version=version_id, tile_base=tile_base)
                bench_data.append((M, f'v{version_id}-t{tile_base}', gflops))
        else:
            gflops = profile_gemm(A, B, repeat=10, version=version_id)
            bench_data.append((M, f'v{version_id}', gflops))

    return bench_data

def check_and_bench(M=512, K=512, N=512, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)

    bench_data = []

    total_version = 8

    for version_id in range(1, total_version + 1):     # here, to add more versions, increase the range
        if version_id >= 4:
            check_correctness(A, B, version=version_id)
        gflops = profile_gemm(A, B, repeat=10, version=version_id)
        bench_data.append((M, f'v{version_id}', gflops))

    return bench_data


if __name__ == "__main__":
    plot_data = []
    for size in [32, 64, 128, 256, 512, 1024]:
        bench_data = check_and_bench(M=size, K=size, N=size)
        plot_data.append(bench_data)

    # plot the result
    series = {}  # e.g. {'v1': [(32,x1), (64,x2), ...], 'v2': [...]}
    for row in plot_data:
        for (size, ver, gflops) in row:
            series.setdefault(ver, []).append((size, gflops))

    plt.figure(figsize=(7, 5))
    for ver, pairs in sorted(series.items()):  # ver: 'v1','v2',...
        pairs.sort(key=lambda t: t[0])  # acending order by size
        xs = [s for (s, _) in pairs]
        ys = [g for (_, g) in pairs]
        plt.plot(xs, ys, marker='o', linewidth=2, label=ver)

    # set to (0, 60), because the max gflops oneAPI MKL can reach is about 55 GFLOPS on my machine
    plt.ylim(0, 200)

    plt.title("GEMM GFLOPS vs Matrix Size")
    plt.xlabel("Matrix size (n in MxNxK)")
    plt.ylabel("GFLOPS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Version")
    plt.tight_layout()
    plt.show()