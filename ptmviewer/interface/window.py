# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1278, 893)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.viewerLayout = QtWidgets.QVBoxLayout()
        self.viewerLayout.setObjectName("viewerLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.viewerLayout.addWidget(self.label_2)
        self.viewerWidget = QtWidgets.QOpenGLWidget(self.centralwidget)
        self.viewerWidget.setObjectName("viewerWidget")
        self.viewerLayout.addWidget(self.viewerWidget)
        self.viewerLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.viewerLayout)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1278, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.fileListDock = QtWidgets.QDockWidget(MainWindow)
        self.fileListDock.setObjectName("fileListDock")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_3 = QtWidgets.QLabel(self.dockWidgetContents)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_8.addWidget(self.label_3)
        self.fileList = QtWidgets.QListWidget(self.dockWidgetContents)
        self.fileList.setObjectName("fileList")
        self.verticalLayout_8.addWidget(self.fileList)
        self.loadFile = QtWidgets.QPushButton(self.dockWidgetContents)
        self.loadFile.setObjectName("loadFile")
        self.verticalLayout_8.addWidget(self.loadFile)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.addFile = QtWidgets.QPushButton(self.dockWidgetContents)
        self.addFile.setObjectName("addFile")
        self.horizontalLayout_13.addWidget(self.addFile)
        self.removeFile = QtWidgets.QPushButton(self.dockWidgetContents)
        self.removeFile.setObjectName("removeFile")
        self.horizontalLayout_13.addWidget(self.removeFile)
        self.verticalLayout_8.addLayout(self.horizontalLayout_13)
        self.fileListDock.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.fileListDock)
        self.colorDock = QtWidgets.QDockWidget(MainWindow)
        self.colorDock.setObjectName("colorDock")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.dockWidgetContents_2)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_17 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_16.addWidget(self.label_17)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_6 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_12.addWidget(self.label_6)
        self.colorR = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.colorR.setMaximum(1.0)
        self.colorR.setSingleStep(0.01)
        self.colorR.setProperty("value", 1.0)
        self.colorR.setObjectName("colorR")
        self.verticalLayout_12.addWidget(self.colorR)
        self.verticalLayout_12.setStretch(1, 1)
        self.horizontalLayout_10.addLayout(self.verticalLayout_12)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_14 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_13.addWidget(self.label_14)
        self.colorG = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.colorG.setMaximum(1.0)
        self.colorG.setSingleStep(0.01)
        self.colorG.setProperty("value", 1.0)
        self.colorG.setObjectName("colorG")
        self.verticalLayout_13.addWidget(self.colorG)
        self.horizontalLayout_10.addLayout(self.verticalLayout_13)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_15 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_14.addWidget(self.label_15)
        self.colorB = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.colorB.setMaximum(1.0)
        self.colorB.setSingleStep(0.01)
        self.colorB.setProperty("value", 1.0)
        self.colorB.setObjectName("colorB")
        self.verticalLayout_14.addWidget(self.colorB)
        self.horizontalLayout_10.addLayout(self.verticalLayout_14)
        self.verticalLayout_16.addLayout(self.horizontalLayout_10)
        self.verticalLayout_18.addLayout(self.verticalLayout_16)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_13 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_2.addWidget(self.label_13)
        self.shaderCombo = QtWidgets.QComboBox(self.dockWidgetContents_2)
        self.shaderCombo.setCurrentText("")
        self.shaderCombo.setObjectName("shaderCombo")
        self.verticalLayout_2.addWidget(self.shaderCombo)
        self.horizontalLayout_11.addLayout(self.verticalLayout_2)
        self.verticalLayout_23 = QtWidgets.QVBoxLayout()
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.label_18 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_23.addWidget(self.label_18)
        self.rotAngle = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.rotAngle.setMinimum(1.0)
        self.rotAngle.setMaximum(360.0)
        self.rotAngle.setSingleStep(0.1)
        self.rotAngle.setProperty("value", 45.0)
        self.rotAngle.setObjectName("rotAngle")
        self.verticalLayout_23.addWidget(self.rotAngle)
        self.horizontalLayout_11.addLayout(self.verticalLayout_23)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.shinSpin = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.shinSpin.setMinimum(1.0)
        self.shinSpin.setMaximum(300.0)
        self.shinSpin.setSingleStep(0.1)
        self.shinSpin.setObjectName("shinSpin")
        self.verticalLayout.addWidget(self.shinSpin)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout)
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.label_16 = QtWidgets.QLabel(self.dockWidgetContents_2)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_17.addWidget(self.label_16)
        self.ambientCoeff = QtWidgets.QDoubleSpinBox(self.dockWidgetContents_2)
        self.ambientCoeff.setDecimals(2)
        self.ambientCoeff.setMaximum(1.0)
        self.ambientCoeff.setSingleStep(0.01)
        self.ambientCoeff.setProperty("value", 0.2)
        self.ambientCoeff.setObjectName("ambientCoeff")
        self.verticalLayout_17.addWidget(self.ambientCoeff)
        self.horizontalLayout_11.addLayout(self.verticalLayout_17)
        self.verticalLayout_18.addLayout(self.horizontalLayout_11)
        self.verticalLayout_11.addLayout(self.verticalLayout_18)
        self.colorDock.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.colorDock)
        self.rotateDock = QtWidgets.QDockWidget(MainWindow)
        self.rotateDock.setObjectName("rotateDock")
        self.dockWidgetContents_3 = QtWidgets.QWidget()
        self.dockWidgetContents_3.setObjectName("dockWidgetContents_3")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.dockWidgetContents_3)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.groupBox_2 = QtWidgets.QGroupBox(self.dockWidgetContents_3)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.rotCamRBtn = QtWidgets.QRadioButton(self.groupBox_2)
        self.rotCamRBtn.setObjectName("rotCamRBtn")
        self.horizontalLayout_14.addWidget(self.rotCamRBtn)
        self.rotLightRBtn = QtWidgets.QRadioButton(self.groupBox_2)
        self.rotLightRBtn.setObjectName("rotLightRBtn")
        self.horizontalLayout_14.addWidget(self.rotLightRBtn)
        self.verticalLayout_9.addWidget(self.groupBox_2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_11 = QtWidgets.QLabel(self.dockWidgetContents_3)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_4.addWidget(self.label_11)
        self.rotX = QtWidgets.QSpinBox(self.dockWidgetContents_3)
        self.rotX.setMinimum(-180)
        self.rotX.setMaximum(180)
        self.rotX.setObjectName("rotX")
        self.verticalLayout_4.addWidget(self.rotX)
        self.horizontalLayout_7.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_12 = QtWidgets.QLabel(self.dockWidgetContents_3)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_5.addWidget(self.label_12)
        self.rotY = QtWidgets.QSpinBox(self.dockWidgetContents_3)
        self.rotY.setMinimum(-180)
        self.rotY.setMaximum(180)
        self.rotY.setObjectName("rotY")
        self.verticalLayout_5.addWidget(self.rotY)
        self.horizontalLayout_7.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.dockWidgetContents_3)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)
        self.rotZ = QtWidgets.QSpinBox(self.dockWidgetContents_3)
        self.rotZ.setMinimum(-180)
        self.rotZ.setMaximum(180)
        self.rotZ.setObjectName("rotZ")
        self.verticalLayout_6.addWidget(self.rotZ)
        self.horizontalLayout_7.addLayout(self.verticalLayout_6)
        self.verticalLayout_9.addLayout(self.horizontalLayout_7)
        self.verticalLayout_19.addLayout(self.verticalLayout_9)
        self.rotateDock.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.rotateDock)
        self.moveDock = QtWidgets.QDockWidget(MainWindow)
        self.moveDock.setObjectName("moveDock")
        self.dockWidgetContents_4 = QtWidgets.QWidget()
        self.dockWidgetContents_4.setObjectName("dockWidgetContents_4")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.dockWidgetContents_4)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.groupBox = QtWidgets.QGroupBox(self.dockWidgetContents_4)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.moveLightRBtn = QtWidgets.QRadioButton(self.groupBox)
        self.moveLightRBtn.setObjectName("moveLightRBtn")
        self.horizontalLayout_12.addWidget(self.moveLightRBtn)
        self.moveCameraRBtn = QtWidgets.QRadioButton(self.groupBox)
        self.moveCameraRBtn.setObjectName("moveCameraRBtn")
        self.horizontalLayout_12.addWidget(self.moveCameraRBtn)
        self.horizontalLayout_9.addWidget(self.groupBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_9)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.moveXCBox = QtWidgets.QCheckBox(self.dockWidgetContents_4)
        self.moveXCBox.setObjectName("moveXCBox")
        self.horizontalLayout_3.addWidget(self.moveXCBox)
        self.moveYCBox = QtWidgets.QCheckBox(self.dockWidgetContents_4)
        self.moveYCBox.setObjectName("moveYCBox")
        self.horizontalLayout_3.addWidget(self.moveYCBox)
        self.moveZCBox = QtWidgets.QCheckBox(self.dockWidgetContents_4)
        self.moveZCBox.setObjectName("moveZCBox")
        self.horizontalLayout_3.addWidget(self.moveZCBox)
        self.horizontalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.moveUp = QtWidgets.QPushButton(self.dockWidgetContents_4)
        self.moveUp.setAutoRepeat(True)
        self.moveUp.setObjectName("moveUp")
        self.verticalLayout_3.addWidget(self.moveUp)
        self.moveDown = QtWidgets.QPushButton(self.dockWidgetContents_4)
        self.moveDown.setAutoRepeat(True)
        self.moveDown.setObjectName("moveDown")
        self.verticalLayout_3.addWidget(self.moveDown)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.verticalLayout_22.addLayout(self.verticalLayout_7)
        self.moveDock.setWidget(self.dockWidgetContents_4)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.moveDock)
        self.paramsDock = QtWidgets.QDockWidget(MainWindow)
        self.paramsDock.setObjectName("paramsDock")
        self.dockWidgetContents_7 = QtWidgets.QWidget()
        self.dockWidgetContents_7.setObjectName("dockWidgetContents_7")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.dockWidgetContents_7)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_4 = QtWidgets.QLabel(self.dockWidgetContents_7)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_10.addWidget(self.label_4)
        self.paramBrowser = QtWidgets.QTextBrowser(self.dockWidgetContents_7)
        self.paramBrowser.setObjectName("paramBrowser")
        self.verticalLayout_10.addWidget(self.paramBrowser)
        self.captureBtn = QtWidgets.QPushButton(self.dockWidgetContents_7)
        self.captureBtn.setObjectName("captureBtn")
        self.verticalLayout_10.addWidget(self.captureBtn)
        self.paramsDock.setWidget(self.dockWidgetContents_7)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.paramsDock)
        self.transcribeDock = QtWidgets.QDockWidget(MainWindow)
        self.transcribeDock.setObjectName("transcribeDock")
        self.dockWidgetContents_8 = QtWidgets.QWidget()
        self.dockWidgetContents_8.setObjectName("dockWidgetContents_8")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.dockWidgetContents_8)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.label_5 = QtWidgets.QLabel(self.dockWidgetContents_8)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_15.addWidget(self.label_5)
        self.transcribeEdit = QtWidgets.QPlainTextEdit(self.dockWidgetContents_8)
        self.transcribeEdit.setObjectName("transcribeEdit")
        self.verticalLayout_15.addWidget(self.transcribeEdit)
        self.saveBtn = QtWidgets.QPushButton(self.dockWidgetContents_8)
        self.saveBtn.setObjectName("saveBtn")
        self.verticalLayout_15.addWidget(self.saveBtn)
        self.transcribeDock.setWidget(self.dockWidgetContents_8)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.transcribeDock)
        self.glInfoDock = QtWidgets.QDockWidget(MainWindow)
        self.glInfoDock.setObjectName("glInfoDock")
        self.dockWidgetContents_5 = QtWidgets.QWidget()
        self.dockWidgetContents_5.setObjectName("dockWidgetContents_5")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout(self.dockWidgetContents_5)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.label_8 = QtWidgets.QLabel(self.dockWidgetContents_5)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_20.addWidget(self.label_8)
        self.glInfoBrowser = QtWidgets.QTextBrowser(self.dockWidgetContents_5)
        self.glInfoBrowser.setObjectName("glInfoBrowser")
        self.verticalLayout_20.addWidget(self.glInfoBrowser)
        self.glInfoDock.setWidget(self.dockWidgetContents_5)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.glInfoDock)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "GL Viewer"))
        self.label_3.setText(_translate("MainWindow", "File List"))
        self.loadFile.setText(_translate("MainWindow", "Load"))
        self.addFile.setText(_translate("MainWindow", "Add File(s)"))
        self.removeFile.setText(_translate("MainWindow", "Remove File(s)"))
        self.label_17.setText(_translate("MainWindow", "Light Color"))
        self.label_6.setText(_translate("MainWindow", "red"))
        self.label_14.setText(_translate("MainWindow", "green"))
        self.label_15.setText(_translate("MainWindow", "blue"))
        self.label_13.setText(_translate("MainWindow", "Shader"))
        self.label_18.setText(_translate("MainWindow", "Angle"))
        self.label.setText(_translate("MainWindow", "Shininess"))
        self.label_16.setText(_translate("MainWindow", "Ambient Coefficient"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Rotate"))
        self.rotCamRBtn.setText(_translate("MainWindow", "Rotate Camera"))
        self.rotLightRBtn.setText(_translate("MainWindow", "Rotate Light"))
        self.label_11.setText(_translate("MainWindow", "pitch (X)"))
        self.label_12.setText(_translate("MainWindow", "yaw (Y)"))
        self.label_7.setText(_translate("MainWindow", "roll (Z)"))
        self.groupBox.setTitle(_translate("MainWindow", "Move"))
        self.moveLightRBtn.setText(_translate("MainWindow", "Move Light"))
        self.moveCameraRBtn.setText(_translate("MainWindow", "Move Camera"))
        self.moveXCBox.setText(_translate("MainWindow", "X axis"))
        self.moveYCBox.setText(_translate("MainWindow", "Y axis"))
        self.moveZCBox.setText(_translate("MainWindow", "Z axis"))
        self.moveUp.setText(_translate("MainWindow", "+"))
        self.moveDown.setText(_translate("MainWindow", "-"))
        self.label_4.setText(_translate("MainWindow", "Parameters"))
        self.captureBtn.setText(_translate("MainWindow", "capture"))
        self.label_5.setText(_translate("MainWindow", "Transcribe"))
        self.saveBtn.setText(_translate("MainWindow", "save"))
        self.label_8.setText(_translate("MainWindow", "Open GL Info"))

