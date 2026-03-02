import sys
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QMenu
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


from ui_design import Ui_MainWindow
from geometry import Geometry
from solver import solver

class App(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        
        self.setupUi(self)
        self.setWindowTitle("Spacecraft Thermal Simulator")
        
        self.geo = Geometry()
        self.sol = solver(self.geo)

        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.reset_history() 

        
        self.init_menu()

        self.init_logic()

  
    def init_menu(self):
        # 获取 QMainWindow 自带的顶部菜单栏
        menubar = self.menuBar()

        # --- 1. 使用 QMenu 创建“文件”主菜单 ---
        file_menu = QMenu("Файл (文件)", self)
        menubar.addMenu(file_menu)

        # --- 2. 使用 QAction 创建具体的下拉动作 ---
        # 动作 1: 加载 OBJ
        action_load_obj = QAction("Загрузить .obj (加载模型)", self)
        action_load_obj.setShortcut("Ctrl+O") 
        action_load_obj.triggered.connect(lambda: self.load_file("obj"))
        
        # 动作 2: 加载 JSON
        action_load_json = QAction("Загрузить .json (加载配置)", self)
        action_load_json.setShortcut("Ctrl+J")
        action_load_json.triggered.connect(lambda: self.load_file("json"))

        # 动作 3: 退出程序
        action_exit = QAction("Выход (退出)", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close) 

        # --- 3. 把动作 (QAction) 塞进菜单 (QMenu) 里 ---
        file_menu.addAction(action_load_obj)
        file_menu.addAction(action_load_json)
        file_menu.addSeparator() # 添加分割线
        file_menu.addAction(action_exit)

        # --- 4. 创建“运行”主菜单 ---
        run_menu = QMenu("Запуск (运行)", self)
        menubar.addMenu(run_menu)
        
        action_run = QAction("Рассчитать (开始计算)", self)
        action_run.setShortcut("F5")
        action_run.triggered.connect(self.run_sim)
        run_menu.addAction(action_run)

    def reset_history(self):
        self.t_offset = 0
        self.last_y = None
        self.hist_t, self.hist_y = [], []

    def init_logic(self):
        # 绑定的逻辑不变
        self.btn_obj.clicked.connect(lambda: self.load_file("obj"))
        self.btn_conf.clicked.connect(lambda: self.load_file("json"))
        self.btn_run.clicked.connect(self.run_sim)
        self.chk_inf.toggled.connect(self.toggle_inf)

        # 把 Matplotlib 塞进你的占位框里
        self.fig, self.ax = plt.subplots() 
        self.canvas = FigureCanvas(self.fig)
        
        layout = QVBoxLayout(self.plot_widget)
        layout.addWidget(self.canvas)

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
        #self.box_mode.currentText()：看用户选的是 "Fixed" 还是 "Steady"。
        self.draw(t, y)
        
        pd.DataFrame(y.T, columns=self.geo.names).assign(Time=t).to_csv("results.csv", index=False)

    def toggle_inf(self, on):
        if on and self.geo.names and self.sol.conf:
          
            self.reset_history()
            self.btn_run.setEnabled(False)# 禁用运行按钮，防止冲突
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