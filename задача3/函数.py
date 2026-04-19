import os
import numpy as np
from multiprocessing import Pool

# 强行给 PyCUDA 指路，找到微软的 cl.exe 编译器
vc_path = r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64'
if vc_path not in os.environ['PATH']:
    os.environ['PATH'] = vc_path + os.pathsep + os.environ['PATH']

# 导入 Cython
from cy_verlet import run_cython

# 导入 CUDA (去掉花哨的提示，直接硬连)
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

G = 39.478  # Гравитационная постоянная 

# CUDA ядро
cuda_code = """
__global__ void get_a_cuda(double *r, double *m, double *a, int n, int dim, double G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double ax = 0.0; double ay = 0.0;
    double rx = r[i*dim]; double ry = r[i*dim+1];
    
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
mod = SourceModule(cuda_code, arch="sm_86", options=["-Xcompiler", "/wd4819"])
get_a_cuda_func = mod.get_function("get_a_cuda")

# --- 2.Verlet ---
def get_a_py(r, m):
    n = len(m)
    a = np.zeros_like(r)
    for i in range(n):
        diff = r - r[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = 1.0 
        a[i] = G * np.sum((m[:, None] * diff) / dist[:, None]**3, axis=0)
    return a

def run_py(r0, v0, m, t_span, dt, save_hist=False):
    t0, t_end = t_span
    steps = int((t_end - t0) / dt)
    r = r0.copy()
    v = v0.copy()
    a = get_a_py(r, m)
    
    hist = [r.copy()] if save_hist else None
    
    for _ in range(steps):
        r_new = r + v * dt + 0.5 * a * dt**2
        a_new = get_a_py(r_new, m)
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        if save_hist: hist.append(r.copy())
        
    return np.array(hist) if save_hist else r

# --- 3. Multiprocessing ---
def acc_parallel(args):
    
    i, r, m = args
    diff = r - r[i]
    dist = np.linalg.norm(diff, axis=1)
    dist[i] = 1.0
    ai = G * np.sum((m[:, None] * diff) / dist[:, None]**3, axis=0)
    return i, ai

def run_mp(r0, v0, m, t_span, dt, procs=4, save_hist=False):
    steps = int((t_span[1] - t_span[0]) / dt)
    r = r0.copy()
    v = v0.copy()
    hist = [r.copy()] if save_hist else None
    
    with Pool(processes=procs) as pool:
        args = [(i, r, m) for i in range(len(m))]
        res = pool.map(acc_parallel, args)
        a = np.zeros_like(r)
        for i, ai in res: a[i] = ai
            
        for _ in range(steps):
            r_new = r + v * dt + 0.5 * a * dt**2
            args_new = [(i, r_new, m) for i in range(len(m))]
            res_new = pool.map(acc_parallel, args_new)
            a_new = np.zeros_like(r)
            for i, ai in res_new: a_new[i] = ai
            v_new = v + 0.5 * (a + a_new) * dt
            r, v, a = r_new, v_new, a_new
            if save_hist: hist.append(r.copy())
            
    return np.array(hist) if save_hist else r

# --- 5. CUDA ---
def run_cuda(r0, v0, m, t_span, dt, save_hist=False):
    steps = int((t_span[1] - t_span[0]) / dt)
    n, dim = r0.shape
    
    r = r0.astype(np.float64)
    v = v0.astype(np.float64)
    m_arr = m.astype(np.float64)
    a = np.zeros_like(r)
    hist = [r.copy()] if save_hist else None
    
    block_s = 256
    grid_s = (n + block_s - 1) // block_s
    
    r_g = gpuarray.to_gpu(r)
    m_g = gpuarray.to_gpu(m_arr)
    a_g = gpuarray.to_gpu(a)
    
    get_a_cuda_func(r_g, m_g, a_g, np.int32(n), np.int32(dim), np.float64(G), block=(block_s,1,1), grid=(grid_s,1))
    a = a_g.get()
    
    for _ in range(steps):
        r_new = r + v * dt + 0.5 * a * dt**2
        r_g.set(r_new)
        get_a_cuda_func(r_g, m_g, a_g, np.int32(n), np.int32(dim), np.float64(G), block=(block_s,1,1), grid=(grid_s,1))
        a_new = a_g.get()
        v_new = v + 0.5 * (a + a_new) * dt
        r, v, a = r_new, v_new, a_new
        if save_hist: hist.append(r.copy())
        
    return np.array(hist) if save_hist else r