import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
from scipy.integrate import odeint
import os


vc_path = r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64'
if vc_path not in os.environ['PATH']:
    os.environ['PATH'] = vc_path + os.pathsep + os.environ['PATH']


from cy_verlet import run_cython
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

G = 39.478  # 物理常数

# =========================================================
# 【这里是核心】：老老实实地把五种方法列出来，加好注释
# =========================================================

# --- 方法 1: odeint (官方标准答案，只用来测误差，不参与测速) ---
def nbody_deriv(y, t, m, n, dim):
    r = y[:n*dim].reshape((n, dim))
    v = y[n*dim:].reshape((n, dim))
    # 借用基础版的加速度计算
    a = get_a_py(r, m)
    return np.concatenate((v.flatten(), a.flatten()))

def run_odeint(r0, v0, m, t_span, dt):
    n, dim = r0.shape
    y0 = np.concatenate((r0.flatten(), v0.flatten()))
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    sol = odeint(nbody_deriv, y0, t_eval, args=(m, n, dim))
    return sol[:, :n*dim].reshape(len(t_eval), n, dim)

# --- 方法 2: Python 串行 Verlet (基础逻辑，作为测速基准) ---
def get_a_py(r, m):
    n = len(m)
    a = np.zeros_like(r)
    for i in range(n):
        diff = r - r[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = 1.0 
        a[i] = G * np.sum((m[:, None] * diff) / dist[:, None]**3, axis=0)
    return a

def run_py(r0, v0, m, t_span, dt, return_history=False):
    steps = int((t_span[1] - t_span[0]) / dt)
    r, v = r0.copy(), v0.copy()
    a = get_a_py(r, m)
    history = [r.copy()] if return_history else None
    
    for _ in range(steps):
        r_new = r + v * dt + 0.5 * a * dt**2
        a_new = get_a_py(r_new, m)
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        if return_history: history.append(r.copy())
    return np.array(history) if return_history else r

# --- 方法 3: Multiprocessing 多进程 Verlet (CPU加速) ---
def acc_parallel(args):
    i, r, m = args
    diff = r - r[i]
    dist = np.linalg.norm(diff, axis=1)
    dist[i] = 1.0
    ai = G * np.sum((m[:, None] * diff) / dist[:, None]**3, axis=0)
    return i, ai

def run_mp(r0, v0, m, t_span, dt, return_history=False):
    steps = int((t_span[1] - t_span[0]) / dt)
    r, v = r0.copy(), v0.copy()
    history = [r.copy()] if return_history else None
    
    with Pool(processes=4) as pool: 
        args = [(i, r, m) for i in range(len(m))]
        results = pool.map(acc_parallel, args)
        a = np.zeros_like(r)
        for i, ai in results: a[i] = ai
            
        for _ in range(steps):
            r_new = r + v * dt + 0.5 * a * dt**2
            args_new = [(i, r_new, m) for i in range(len(m))]
            results_new = pool.map(acc_parallel, args_new)
            a_new = np.zeros_like(r)
            for i, ai in results_new: a_new[i] = ai
            v_new = v + 0.5 * (a + a_new) * dt
            r, v, a = r_new, v_new, a_new
            if return_history: history.append(r.copy())
    return np.array(history) if return_history else r

# --- 方法 4: Cython (编译加速，代码在 cy_verlet.pyx 里) ---
# 这里不需要写逻辑，已经在开头 import run_cython 了

# --- 方法 5: CUDA (GPU 显卡加速) ---
cuda_code = """
__global__ void get_a_cuda(double *r, double *m, double *a, int n, int dim, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double ax = 0.0, ay = 0.0;
    double rx = r[i*dim], ry = r[i*dim+1];
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        double dx = r[j*dim] - rx;
        double dy = r[j*dim+1] - ry;
        double dist_sq = dx*dx + dy*dy + 1e-10;
        double dist = sqrt(dist_sq);
        double f = G * m[j] / (dist_sq * dist);
        ax += f * dx;
        ay += f * dy;
    }
    a[i*dim] = ax;
    a[i*dim+1] = ay;
}
"""
mod = SourceModule(cuda_code, arch="sm_86")
get_a_cuda_func = mod.get_function("get_a_cuda")

def run_cuda(r0, v0, m, t_span, dt, return_history=False):
    steps = int((t_span[1] - t_span[0]) / dt)
    n, dim = r0.shape
    r, v, m_arr = r0.astype(np.float64), v0.astype(np.float64), m.astype(np.float64)
    a = np.zeros_like(r)
    history = [r.copy()] if return_history else None
    
    block_s = 256
    grid_s = (n + block_s - 1) // block_s
    
    r_g, m_g, a_g = gpuarray.to_gpu(r), gpuarray.to_gpu(m_arr), gpuarray.to_gpu(a)
    get_a_cuda_func(r_g, m_g, a_g, np.int32(n), np.int32(dim), np.float64(G), block=(block_s,1,1), grid=(grid_s,1))
    a = a_g.get()
    
    for _ in range(steps):
        r_new = r + v * dt + 0.5 * a * dt**2
        r_g.set(r_new)
        get_a_cuda_func(r_g, m_g, a_g, np.int32(n), np.int32(dim), np.float64(G), block=(block_s,1,1), grid=(grid_s,1))
        a_new = a_g.get()
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        if return_history: history.append(r.copy())
    return np.array(history) if return_history else r


def make_fake_data(n):
    # 生成随机假数据用来做压力测试
    np.random.seed(123)
    m = np.random.rand(n) * 1e-4
    r = np.random.rand(n, 2) * 2 - 1
    v = np.random.rand(n, 2) * 0.1 - 0.05
    return m, r, v

def benchmark():
    print("\n--- ЗАДАЧА 3: СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ---")
    t_span = (0, 0.05) # 测速时间不用设太长
    dt = 0.001
    ns = [100, 200, 300, 400, 500]
    
    # 准备四个空本子，记录除 odeint 外的四个方法的时间
    time_py = []
    time_mp = []
    time_cy = []
    time_cu = []

    for n in ns:
        print(f"正在测试 N = {n} ...")
        m, r, v = make_fake_data(n)

        # 测方法 2：Python 串行
        start = time.perf_counter()
        run_py(r, v, m, t_span, dt)
        time_py.append(time.perf_counter() - start)

        # 测方法 3：多进程 MP
        start = time.perf_counter()
        run_mp(r, v, m, t_span, dt)
        time_mp.append(time.perf_counter() - start)

        # 测方法 4：Cython
        start = time.perf_counter()
        run_cython(r.copy(), v.copy(), m.copy(), t_span[0], t_span[1], dt)
        time_cy.append(time.perf_counter() - start)

        # 测方法 5：CUDA GPU
        start = time.perf_counter()
        run_cuda(r, v, m, t_span, dt)
        time_cu.append(time.perf_counter() - start)

    print("action...")

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ns, time_py, 'o-', color='red', label='Python (Baseline)')
    ax1.plot(ns, time_mp, 's-', color='blue', label='Multiprocessing')
    ax1.plot(ns, time_cy, '^-', color='green', label='Cython')
    ax1.plot(ns, time_cu, 'd-', color='purple', label='CUDA')
    ax1.set_xlabel('Количество частиц N')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('3.1.1 Время работы методов')
    ax1.grid(True)
    ax1.legend()

    speedup_mp = [time_py[i] / time_mp[i] for i in range(len(ns))]
    speedup_cy = [time_py[i] / time_cy[i] for i in range(len(ns))]
    speedup_cu = [time_py[i] / time_cu[i] for i in range(len(ns))]

    ax2.plot(ns, speedup_mp, 's-', color='blue', label='MP / Py')
    ax2.plot(ns, speedup_cy, '^-', color='green', label='Cython / Py')
    ax2.plot(ns, speedup_cu, 'd-', color='purple', label='CUDA / Py')
    ax2.axhline(y=1, color='gray', linestyle='--') # 画一条1倍基准线
    ax2.set_xlabel('Количество частиц N')
    ax2.set_ylabel('Ускорение (T_py / T_method)')
    ax2.set_title('3.1.2 График ускорения')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('performance_all.png')
    print("✅ Графики сохранены как 'performance_all.png'")
    plt.show() # 直接弹出来给你看

if __name__ == '__main__':
    # 你这里只要求执行性能测试任务，我就只调这个函数了
    benchmark()