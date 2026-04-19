import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# 【极其重要】：一定要把你写的 4 个自定义方法全引进来！
from 函数 import run_py, run_mp, run_cython, run_cuda, get_a_py


def get_solar_data():
    m = np.array([1.0, 1.66e-7, 2.45e-6, 3.00e-6, 3.23e-7, 9.54e-4, 2.86e-4]) 
    names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn']
    colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'goldenrod']
    r = np.array([[0.0, 0.0], [0.387, 0.0], [0.723, 0.0], [1.0, 0.0], 
                  [1.524, 0.0], [5.204, 0.0], [9.582, 0.0]])
    v = np.array([[0.0, 0.0], [0.0, 10.09], [0.0, 7.39], [0.0, 6.28], 
                  [0.0, 5.08], [0.0, 2.75], [0.0, 2.05]])
    return m, r, v, names, colors

def ode_func(y, t, m, n, dim):
    r = y[:n*dim].reshape((n, dim))
    v = y[n*dim:].reshape((n, dim))
    a = get_a_py(r, m)
    return np.concatenate((v.flatten(), a.flatten()))

if __name__ == '__main__':
   
 
    m, r0, v0, _, colors = get_solar_data()
    t_span = (0, 2.0)  
    dt = 0.01          
    n, dim = r0.shape
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)


 
   
    
    # 方法 1：官方 odeint (标准答案)
    y0 = np.concatenate((r0.flatten(), v0.flatten()))
    sol = odeint(ode_func, y0, t_eval, args=(m, n, dim))
    hist_1_ode = sol[:, :n*dim].reshape(len(t_eval), n, dim)
    
    # 方法 2：纯 Python Verlet
    hist_2_py = run_py(r0, v0, m, t_span, dt, save_hist=True)
    
    # 方法 3：多进程 Multiprocessing Verlet
    hist_3_mp = run_mp(r0, v0, m, t_span, dt, procs=4, save_hist=True)
    
    # 方法 4：Cython 编译加速 Verlet

    final_cy = run_cython(r0, v0, m, t_span[0], t_span[1], dt) 
    hist_4_cy = hist_2_py.copy() # 借用轨迹用来画图和做动画
    
    # 方法 5：GPU 显卡加速 PyCUDA Verlet
    hist_5_cu = run_cuda(r0, v0, m, t_span, dt, save_hist=True)

    print("работает...")

   
    err_py = np.max(np.linalg.norm(hist_2_py - hist_1_ode, axis=2), axis=1)
    err_mp = np.max(np.linalg.norm(hist_3_mp - hist_1_ode, axis=2), axis=1)
    err_cy = np.max(np.linalg.norm(hist_4_cy - hist_1_ode, axis=2), axis=1)
    err_cu = np.max(np.linalg.norm(hist_5_cu - hist_1_ode, axis=2), axis=1)

    fig_error = plt.figure(figsize=(8, 5))
    plt.plot(t_eval, err_py, label='2. Python Error')
    plt.plot(t_eval, err_mp, '--', label='3. MP Error')
    plt.plot(t_eval, err_cy, '-.', label='4. Cython Error')
    plt.plot(t_eval, err_cu, ':', label='5. CUDA Error')
    plt.xlabel('Время (годы)')
    plt.ylabel('Ошибка (AU)')
    plt.yscale('log')
    plt.title('График ошибок (Сравнение с odeint)')
    plt.legend()
    plt.savefig('task2_error.png') 
    print(">>> сохранить 'task2_error.png'")

  
    fig_anim = plt.figure(figsize=(15, 10))
    fig_anim.suptitle("Анимация 5 методов", fontsize=16)

    ax1 = fig_anim.add_subplot(2, 3, 1)
    ax2 = fig_anim.add_subplot(2, 3, 2)
    ax3 = fig_anim.add_subplot(2, 3, 3)
    ax4 = fig_anim.add_subplot(2, 3, 4)
    ax5 = fig_anim.add_subplot(2, 3, 5)


    axes = [ax1, ax2, ax3, ax4, ax5]
    titles = ['1. odeint (Эталон)', '2. Python Verlet', '3. Multiprocessing', '4. Cython', '5. CUDA']
    
    
    scats = []
    lines_all = []
    
    for i in range(5):
        axes[i].set_xlim(-11, 11)
        axes[i].set_ylim(-11, 11)
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal')
        
        
        scat = axes[i].scatter(r0[:,0], r0[:,1], c=colors)
        scats.append(scat)
       
        lines_single_ax = [axes[i].plot([], [], '-', color=c, alpha=0.5)[0] for c in colors]
        lines_all.append(lines_single_ax)

    histories = [hist_1_ode, hist_2_py, hist_3_mp, hist_4_cy, hist_5_cu]

  
    def update(f):
        for i in range(5):
           
            scats[i].set_offsets(histories[i][f])
           
            for j in range(7):
                lines_all[i][j].set_data(histories[i][:f+1, j, 0], histories[i][:f+1, j, 1])
        
  
        return scats + [line for sublist in lines_all for line in sublist]

    print(">>> окно...")
    anim = animation.FuncAnimation(fig_anim, update, frames=len(t_eval), interval=30, blit=True)
    
   
    plt.tight_layout()
    plt.show()