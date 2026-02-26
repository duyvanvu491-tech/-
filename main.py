import sys
import json
import math
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, 
                             QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class Geometry:
  def __init__(self):
        self.parts = {}      
        self.contacts = {}   
        self.names = []
  def load(self,path):
      self.parts={}
      verts=[]
      curr=None
      with open(path,'r') as f:
          for line in f:
              row=line.split()
              if len(row)==0:
                  continue
              if row[0] == 'v':
                   x_text = row[1] 
                   y_text = row[2]
                   z_text = row[3]
                   x_num = float(x_text)
                   y_num = float(y_text)
                   z_num = float(z_text)
                   point = [x_num, y_num, z_num]
                   verts.append(point)
              elif row[0] == 'g':
                  curr = row[1]
                  new_part_info = {}
                  new_part_info['faces']=[]
                  new_part_info['s']=0
                  self.parts[curr] = new_part_info
              elif row[0] == 'f' and curr:
                   face = [] 
                   for i in row[1:]: 
                    index = int(i) - 1   
                    point = verts[index] 
                    face.append(point)   
                   self.parts[curr]['faces'].append(face)
          self.names = sorted(list(self.parts.keys()))
          self.calc_S()
          self.calc_Sij()
  def get_area(self, pts):
        if len(pts) < 3: 
            return 0.0
        v0 = np.array(pts[0])
        s = 0.0
        for i in range(1, len(pts)-1):
            v1 = np.array(pts[i])
            v2 = np.array(pts[i+1])
            s += 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0))
        return s
  def calc_S(self):
    for data in self.parts.values():
        data['S'] = sum(self.get_area(face) for face in data['faces'])
  def calc_Sij(self):
        self.contacts = {}
        face_map = {}
        
        for name, data in self.parts.items():
            for face in data['faces']:
                rounded_face = []
                for v in face:
                    rounded_face.append(tuple(np.round(v, 4)))
                key = tuple(sorted(rounded_face))
                
                if key not in face_map:
                    face_map[key] = []
                face_map[key].append(name)

        for key, names in face_map.items():
            if len(names) > 1:
                area = self.get_area(key)
                unique_names = sorted(list(set(names)))
                for i in range(len(unique_names)):
                    for j in range(i+1, len(unique_names)):
                        pair = (unique_names[i], unique_names[j])
                        if pair not in self.contacts:
                            self.contacts[pair] = 0.0
                        self.contacts[pair] += area
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

            dT[i] = (Q_cond + Q_rad + Q_in) / data['c']
            
        return dT

  
    def run(self, t_start, t_end, mode, T_curr=None):
        N = len(self.geo.names)
        
        if T_curr is not None:
            T0 = T_curr 
        elif mode == "Steady":
            T0 = fsolve(lambda x: self.ode(0, x), [300]*N) 
        else:
            v = self.conf.get('global', {}).get('T_start_fixed', 300)
            T0 = [v] * N

        t_span = np.linspace(t_start, t_end, 200)
        res = solve_ivp(self.ode, [t_start, t_end], T0, t_eval=t_span, method='RK45')
        
        return res.t, res.y
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spacecraft Thermal Simulator")
        self.resize(1000, 600)

      
        self.geo = Geometry()
        self.sol = solver(self.geo)

       
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.reset_history() 

        self.init_ui()

    def reset_history(self):
       
        self.t_offset = 0
        self.last_y = None
        self.hist_t, self.hist_y = [], []

    def init_ui(self):
       
        w = QWidget()
        self.setCentralWidget(w)
        layout = QHBoxLayout(w)
        panel = QVBoxLayout()

    
        self.btn_obj = QPushButton("1. Загрузить .obj")
        self.btn_obj.clicked.connect(lambda: self.load_file("obj"))

        self.btn_conf = QPushButton("2. Загрузить .json")
        self.btn_conf.clicked.connect(lambda: self.load_file("json"))

        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(10, 1e6)
        self.spin_t.setValue(2000)

        self.box_mode = QComboBox()
        self.box_mode.addItems(["Fixed", "Steady"])

        self.btn_run = QPushButton("3. Рассчитать")
        self.btn_run.clicked.connect(self.run_sim)

        self.chk_inf = QCheckBox("Бесконечный режим (Динамика)")
        self.chk_inf.toggled.connect(self.toggle_inf)

       
        widgets = [self.btn_obj, self.btn_conf, QLabel("Время расчета (с):"), self.spin_t, 
                   QLabel("Начальные условия:"), self.box_mode, self.btn_run, self.chk_inf]
        for widget in widgets:
            panel.addWidget(widget)
        panel.addStretch()

      
        self.fig, self.ax = plt.subplots() 
        self.canvas = FigureCanvas(self.fig)
        
        layout.addLayout(panel, 1)
        layout.addWidget(self.canvas, 3)

    def load_file(self, file_type):
        
        ext = f"*.{file_type}"
        f, _ = QFileDialog.getOpenFileName(self, f"Open {file_type.upper()}", "", ext)
        if not f: return
        
        if file_type == "obj":
            self.geo.load(f)
            self.btn_obj.setText("1. Модель ✓")
        else:
            self.sol.load(f)
            self.btn_conf.setText("2. Конфиг ✓")

    def run_sim(self):
  
        if not self.geo.names or not self.sol.conf: return
            
        t_end = self.spin_t.value()
        t, y = self.sol.run(0, t_end, self.box_mode.currentText())
        
        self.draw(t, y)
        
        
        pd.DataFrame(y.T, columns=self.geo.names).assign(Time=t).to_csv("results.csv", index=False)

    def toggle_inf(self, on):
        
        if on and self.geo.names and self.sol.conf:
            self.reset_history()
            self.btn_run.setEnabled(False)
            self.draw([], []) 
            self.timer.start(200) 
        else:
            self.chk_inf.setChecked(False)
            self.timer.stop()
            self.btn_run.setEnabled(True)

    def on_timer(self):
       
        step = self.spin_t.value() / 50.0
        t_end = self.t_offset + step

        t, y = self.sol.run(self.t_offset, t_end, self.box_mode.currentText(), self.last_y)

        
        self.t_offset = t_end
        self.last_y = y[:, -1]

     
        self.hist_t.extend(t.tolist())
        if not self.hist_y:
            self.hist_y = y.tolist()
        else:
            for i in range(len(y)):
                self.hist_y[i].extend(y[i].tolist())

       
        limit = 500
        self.draw(self.hist_t[-limit:], np.array([row[-limit:] for row in self.hist_y]))

    def draw(self, t, y):
       
        self.ax.clear()
        if len(t) == 0: 
            self.canvas.draw()
            return

        y = y.T if y.shape[0] != len(self.geo.names) else y

        for i, name in enumerate(self.geo.names):
            self.ax.plot(t, y[i], label=name)

        self.ax.set(xlabel="Time (s)", ylabel="Temperature (K)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.5)
        self.canvas.draw()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())  