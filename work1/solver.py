import json
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
class solver:
    def __init__(self, geo):
        self.geo = geo
        self.conf = {}
        self.C0 = 5.67

    def load(self, path):
        with open(path, 'r') as f:
            self.conf = json.load(f) 

    def get_Qr(self, name, t):
        if name not in self.conf:
            return 0.0
        expr = self.conf[name].get("Q_R_func", "0")
        try:
            return eval(str(expr), {"sin": math.sin, "cos": math.cos, "pi": math.pi, "t": t})
        except:
            return 0.0

    def ode(self, t, T):
        dT = np.zeros(len(T))
        idx = {}
        i = 0
        for name in self.geo.names:
            idx[name] = i
            i = i + 1
            
        cts = self.geo.contacts.copy()
        manual = self.conf.get("manual_areas", {})
        for k, v in manual.items():
            p1, p2 = k.split('-')
            pair = tuple(sorted((p1, p2)))
            cts[pair] = v
            
        for i, name in enumerate(self.geo.names):
            data = self.conf[name]
            Si = self.geo.parts[name]['S']

            Q_rad = -data['epsilon'] * Si * self.C0 * ((T[i]/100.0)**4)
            Q_in = self.get_Qr(name, t)
            Q_cond = 0.0
            
            lam_dict = self.conf.get("lambdas", {})

            for (p1, p2), sij in cts.items():

                if name in (p1, p2):
                    neighbor = p2 if p1 == name else p1
                    lam = lam_dict.get(f"{p1}-{p2}", 0)
                    Q_cond += lam * sij * (T[idx[neighbor]] - T[i])

            dT[i] = (Q_cond + Q_rad + Q_in) / data['c']#构造出书上的公式           
        return dT

  
    def run(self, t_start, t_end, mode, T_curr=None):
        N = len(self.geo.names)#储存件的总数
        
        if T_curr is not None:
            T0 = T_curr 
        elif mode == "Steady":
            T0 = fsolve(lambda x: self.ode(0, x), [300]*N) 
        else:
            v = self.conf.get('global', {}).get('T_start_fixed', 300)
            T0 = [v] * N#统一起始温度

        t_span = np.linspace(t_start, t_end, 200)
        res = solve_ivp(self.ode, [t_start, t_end], T0, t_eval=t_span, method='RK45')
        
        return res.t, res.y
