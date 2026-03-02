from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

       
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        
        self.btn_obj = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_obj.setObjectName("btn_obj")
        self.verticalLayout.addWidget(self.btn_obj)

       
        self.btn_conf = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_conf.setObjectName("btn_conf")
        self.verticalLayout.addWidget(self.btn_conf)

        
        self.label_time = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_time.setObjectName("label_time")
        self.verticalLayout.addWidget(self.label_time)

        
        self.spin_t = QtWidgets.QDoubleSpinBox(parent=self.centralwidget)
        self.spin_t.setMinimum(10.0)
        self.spin_t.setMaximum(1000000.0)
        self.spin_t.setProperty("value", 2000.0)
        self.spin_t.setObjectName("spin_t")
        self.verticalLayout.addWidget(self.spin_t)

       
        self.label_mode = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_mode.setObjectName("label_mode")
        self.verticalLayout.addWidget(self.label_mode)

    
        self.box_mode = QtWidgets.QComboBox(parent=self.centralwidget)
        self.box_mode.setObjectName("box_mode")
        self.box_mode.addItem("")
        self.box_mode.addItem("")
        self.verticalLayout.addWidget(self.box_mode)

      
        self.btn_run = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_run.setObjectName("btn_run")
        self.verticalLayout.addWidget(self.btn_run)

        self.chk_inf = QtWidgets.QCheckBox(parent=self.centralwidget)
        self.chk_inf.setObjectName("chk_inf")
        self.verticalLayout.addWidget(self.chk_inf)

       
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        
        self.horizontalLayout.addLayout(self.verticalLayout, 1)

         
        self.plot_widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.plot_widget.setObjectName("plot_widget")
        
        
        self.horizontalLayout.addWidget(self.plot_widget, 3)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Spacecraft Thermal Simulator"))
        self.btn_obj.setText(_translate("MainWindow", "1. Загрузить .obj"))
        self.btn_conf.setText(_translate("MainWindow", "2. Загрузить .json"))
        self.label_time.setText(_translate("MainWindow", "Время расчета (с):"))
        self.label_mode.setText(_translate("MainWindow", "Начальные условия:"))
        self.box_mode.setItemText(0, _translate("MainWindow", "Fixed"))
        self.box_mode.setItemText(1, _translate("MainWindow", "Steady"))
        self.btn_run.setText(_translate("MainWindow", "3. Рассчитать"))
        self.chk_inf.setText(_translate("MainWindow", "Бесконечный режим (Динамика)"))

