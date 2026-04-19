import time
import numpy as np
import matplotlib.pyplot as plt

from 函数 import run_py, run_mp, run_cuda, run_cython

def generate_random(n):
    np.random.seed(42)
    m = np.random.rand(n) * 1e-4
    r = np.random.rand(n, 2) * 2 - 1
    v = np.random.rand(n, 2) * 0.1 - 0.05
    return m, r, v

def run_extreme_benchmark():
    print("🚀 开始极限压测：N 飙升至 1000...")
    # 我们设置 5 个梯度，看看显卡在哪个节点开始反杀 CPU
    ns = [100, 300, 500, 750, 1000] 
    t_span = (0, 0.05)
    dt = 0.001
    repeats = 3  # 为了求稳，每次测 3 遍取平均
    
    t_py, t_mp, t_cy, t_cu = [], [], [], []
    
    for n in ns:
        m, r, v = generate_random(n)
        
        # 1. 纯 Python (N=1000 时会比较慢，耐心等几秒)
        st = time.time()
        for _ in range(repeats): run_py(r, v, m, t_span, dt)
        t_py.append((time.time() - st) / repeats)
        
        # 2. 多进程 MP
        st = time.time()
        for _ in range(repeats): run_mp(r, v, m, t_span, dt, procs=4)
        t_mp.append((time.time() - st) / repeats)
        
        # 3. Cython
        st = time.time()
        for _ in range(repeats): run_cython(r.copy(), v.copy(), m.copy(), t_span[0], t_span[1], dt)
        t_cy.append((time.time() - st) / repeats)
        
        # 4. CUDA (显卡主场)
        st = time.time()
        for _ in range(repeats): run_cuda(r, v, m, t_span, dt)
        t_cu.append((time.time() - st) / repeats)
        
        print(f"N={n:4d} 计算完毕 | Py:{t_py[-1]:.3f}s | Cy:{t_cy[-1]:.4f}s | Cu:{t_cu[-1]:.4f}s")

    # 绘制高阶性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(ns, t_py, 'o-', label='Python')
    ax1.plot(ns, t_mp, 's-', label='MP')
    ax1.plot(ns, t_cy, '^-', label='Cython')
    ax1.plot(ns, t_cu, 'd-', label='CUDA')
    ax1.set_xlabel('Количество частиц N')
    ax1.set_ylabel('Время (с)')
    ax1.set_title('Время выполнения (N до 1000)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(ns, [t_py[i]/t_mp[i] for i in range(len(ns))], 's-', label='MP Speedup')
    ax2.plot(ns, [t_py[i]/t_cy[i] for i in range(len(ns))], '^-', label='Cython Speedup')
    ax2.plot(ns, [t_py[i]/t_cu[i] for i in range(len(ns))], 'd-', label='CUDA Speedup')
    ax2.set_xlabel('Количество частиц N')
    ax2.set_ylabel('Ускорение')
    ax2.set_title('Фактор ускорения (N до 1000)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('task3_perf_1000.png', dpi=300)
    print("\n✅ 极限压测图表已生成：task3_perf_1000.png")
if __name__ == '__main__':
    run_extreme_benchmark()