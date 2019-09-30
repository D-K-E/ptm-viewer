# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL

from PySide2.QtCore import QCoreApplication
from PySide2.QtWidgets import QOpenGLWidget

from PySide2.shiboken2 import VoidPtr

import numpy as np
import sys
import os
from PIL import Image, ImageQt

from ptmviewer.interface.window import Ui_MainWindow
from ptmviewer.glwidget import PtmLambertianGLWidget, PtmNormalMapGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapDirGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapPointGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapSpotGLWidget
from ptmviewer.glwidget import PtmCoefficientShader
from ptmviewer.rgbptm import RGBPTM


class AppWindowInit(Ui_MainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtWidgets.QMainWindow()
        super().setupUi(self.main_window)
        pass


class AppWindowFinal(AppWindowInit):
    "Final window"

    def __init__(self):
        super().__init__()
        self.ptmfiles = {}

        # Main Window Events
        self.main_window.setWindowTitle("Python PTM Viewer")
        self.main_window.closeEvent = self.closeApp
        self.closeShort = QtWidgets.QShortcut(
            QtGui.QKeySequence("ctrl+w"), self.main_window
        )
        self.closeShort.activated.connect(self.closeKey)
        # self.main_window.setShortcut("ctrl+w")

        # Button related events
        self.addFile.clicked.connect(self.browseFolder)
        self.addFile.setShortcut("ctrl+o")

        self.loadFile.clicked.connect(self.loadPtm)
        self.loadFile.setShortcut("ctrl+l")

        # lights
        self.lightRed.valueChanged.connect(self.changeLightColor)
        self.lightGreen.valueChanged.connect(self.changeLightColor)
        self.lightBlue.valueChanged.connect(self.changeLightColor)
        self.rotLightX.valueChanged.connect(self.rotateLights)
        self.rotLightY.valueChanged.connect(self.rotateLights)
        self.rotLightZ.valueChanged.connect(self.rotateLights)
        self.lightMoveUp.clicked.connect(self.moveLightPosForward)
        self.lightMoveDown.clicked.connect(self.moveLightPosBackward)
        self.lightMoveLeft.clicked.connect(self.moveLightPosLeft)
        self.lightMoveRight.clicked.connect(self.moveLightPosRight)

        # camera
        self.rotCamX.valueChanged.connect(self.turnCameraX)
        self.rotCamY.valueChanged.connect(self.turnCameraY)
        self.camMoveUp.clicked.connect(self.moveCameraForward)
        self.camMoveDown.clicked.connect(self.moveCameraBackward)
        self.camMoveLeft.clicked.connect(self.moveCameraLeft)
        self.camMoveRight.clicked.connect(self.moveCameraRight)

        self.lastCamXVal = self.rotCamX.value()
        self.lastCamYVal = self.rotCamY.value()

        # angle shininess, ambient coeff
        self.rotAngle.valueChanged.connect(self.setAngle)
        self.shinSpin.valueChanged.connect(self.setShininess)

        # viewer widget, opengl widgets with different shaders
        self.availableGlWidgets = {
            "Lambertian": PtmLambertianGLWidget,
            "SingleNormalMap": PtmNormalMapGLWidget,
            "PerChannelNormalMapDir": PtmPerChannelNormalMapDirGLWidget,
            "PerChannelNormalMapPoint": PtmPerChannelNormalMapPointGLWidget,
            "PerChannelNormalMapSpot": PtmPerChannelNormalMapSpotGLWidget,
        }
        wnames = [k for k in self.availableGlWidgets.keys()]
        self.shaderCombo.addItems(wnames)
        self.shaderCombo.setEditable(False)
        self.shaderCombo.setCurrentText(wnames[0])
        # self.shaderCombo.currentTextChanged.connect(self.loadPtm)

    # Ptm related stuff
    def loadPtm(self):
        "load ptm file into gl widget"
        self.viewerWidget.close()
        if isinstance(self.viewerWidget, PtmNormalMapGLWidget):
            self.viewerWidget.cleanUpGL()
            #
        citem = self.fileList.currentItem()
        cindex = self.fileList.indexFromItem(citem)
        ptmobj = self.ptmfiles[cindex]
        ptm = RGBPTM(ptmobj["path"])
        # vertices, indices = ptm.getVerticesAndSizeArr()
        glchoice = self.shaderCombo.currentText()
        #
        self.runGlPipeline(glchoice, ptm)
        info = self.viewerWidget.getGLInfo()
        self.statusbar.showMessage(info, 5000)
        # info = self.viewerWidget.getGlInfo()
        self.statusbar.showMessage(info, 5000)
        # self.viewerWidget.update()
        print("gl initialized in app")
        # self.viewerWidget.show()

    def replaceViewerWidget(self, glwidget):
        "replace viewer widget with the given instantiated glwidget"
        self.viewerLayout.replaceWidget(self.viewerWidget, glwidget)
        self.viewerWidget = glwidget

    def runGlPipeline(self, glchoice: str, ptm):
        "run gl pipeline using given gl choice"
        if glchoice == "Lambertian":
            self.runLambertianPipeLine(glchoice, ptm)
        elif glchoice == "SingleNormalMap":
            self.runSingleNormalMapPipeLine(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapDir":
            self.runPerChannelNormalMapDirPipeLine(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapPoint":
            self.runPerChannelNormalMapPointPipeLine(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapSpot":
            self.runPerChannelNormalMapSpotPipeLine(glchoice, ptm)
        elif glchoice == "CoefficientShader":
            self.runRGBCoeffShaderPipeline(glchoice, ptm)

    def runLambertianPipeLine(self, glchoice: str, ptm):
        "run lambertian pipeline"
        image = ptm.getImage()
        imqt = ImageQt.ImageQt(image)
        glwidget = self.availableGlWidgets[glchoice](imqt)
        self.replaceViewerWidget(glwidget)

    def runSingleNormalMapPipeLine(self, glchoice: str, ptm):
        "run single normal map pipeline"
        image = ptm.getImage()
        imqt = ImageQt.ImageQt(image)
        nmaps = ptm.getNormalMaps()
        nmap = nmaps[0]
        nmapqt = ImageQt.ImageQt(nmap)
        glwidget = self.availableGlWidgets[glchoice](imqt, nmapqt)
        self.replaceViewerWidget(glwidget)

    def runPerChannelNormalMapDirPipeLine(self, glchoice: str, ptm):
        "run per channel normal map pipeline"
        image = ptm.getImage()
        imqt = ImageQt.ImageQt(image)
        nmaps = ptm.getNormalMaps()
        nmaps = [ImageQt.ImageQt(nmap) for nmap in nmaps]
        glwidget = self.availableGlWidgets[glchoice](imqt, nmaps)
        self.replaceViewerWidget(glwidget)

    def runPerChannelNormalMapPointPipeLine(self, glchoice: str, ptm):
        "run per channel normal map pipeline"
        image = ptm.getImage()
        imqt = ImageQt.ImageQt(image)
        nmaps = ptm.getNormalMaps()
        nmaps = [ImageQt.ImageQt(nmap) for nmap in nmaps]
        glwidget = self.availableGlWidgets[glchoice](imqt, nmaps)
        self.replaceViewerWidget(glwidget)

    def runPerChannelNormalMapSpotPipeLine(self, glchoice: str, ptm):
        "run per channel normal map pipeline"
        image = ptm.getImage()
        imqt = ImageQt.ImageQt(image)
        nmaps = ptm.getNormalMaps()
        nmaps = [ImageQt.ImageQt(nmap) for nmap in nmaps]
        glwidget = self.availableGlWidgets[glchoice](imqt, nmaps)
        self.replaceViewerWidget(glwidget)

    def runRGBCoeffShaderPipeline(self, glchoice: str, ptm):
        "pipeline for rgb coefficient shader"
        vertices, vertexNb = ptm.getNbVertices()
        glwidget = self.availableGlWidgets[glchoice](vertices, vertexNb)
        self.replaceViewerWidget(glwidget)

    def moveGLCamera(self, direction: str):
        self.viewerWidget.moveCamera(direction)

    def moveCameraForward(self):
        self.moveGLCamera("forward")

    def moveCameraBackward(self):
        self.moveGLCamera("backward")

    def moveCameraLeft(self):
        self.moveGLCamera("left")

    def moveCameraRight(self):
        self.moveGLCamera("right")

    def turnCameraX(self, newVal: int):
        "Turn camera around"
        offsetx = newVal - self.lastCamXVal
        valy = self.rotCamY.value() - self.lastCamYVal
        self.viewerWidget.turnAround(x=float(offsetx), y=float(valy))
        self.lastCamXVal = newVal

    def turnCameraY(self, newVal: int):
        "Turn camera around"
        offsety = newVal - self.lastCamYVal
        valx = self.rotCamX.value() - self.lastCamXVal
        self.viewerWidget.turnAround(x=float(valx), y=float(offsety))
        self.lastCamYVal = newVal

    def setAngle(self):
        angl = self.rotAngle.value()
        self.viewerWidget.setRotationAngle(angl)

    def setShininess(self):
        shin = self.shinSpin.value()
        self.viewerWidget.changeShininess(shin)

    def setAmbientCoeff(self):
        "set ambient coefficient to gl widget"
        val = self.ambientCoeff.value()
        self.viewerWidget.changeAmbientCoeffs(val)

    def moveLightPosForward(self):
        ""
        offsetx = 0.0
        offsety = 0.0
        offsetz = -0.5
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def moveLightPosBackward(self):
        ""
        offsetx = 0.0
        offsety = 0.0
        offsetz = 0.5
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def moveLightPosLeft(self):
        ""
        offsetx = -1.0
        offsety = 0.0
        offsetz = 0.0
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def moveLightPosRight(self):
        ""
        offsetx = 0.5
        offsety = 0.0
        offsetz = 0.0
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def moveLightPosUp(self):
        ""
        offsetx = 0.0
        offsety = 0.5
        offsetz = 0.0
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def moveLightPosDown(self):
        ""
        offsetx = 0.0
        offsety = -0.5
        offsetz = 0.0
        self.viewerWidget.moveLight(
            xoffset=offsetx, yoffset=offsety, zoffset=offsetz
        )

    def rotateLights(self):
        rx = self.rotLightX.value()
        ry = self.rotLightY.value()
        rz = self.rotLightZ.value()
        self.viewerWidget.rotateLight(rx, ry, rz)

    def changeLightColor(self):
        diffr = self.lightRed.value()
        diffg = self.lightGreen.value()
        diffb = self.lightBlue.value()
        self.viewerWidget.changeLampIntensity(channel="red", val=diffr)
        self.viewerWidget.changeLampIntensity(channel="green", val=diffg)
        self.viewerWidget.changeLampIntensity(channel="blue", val=diffb)

    ### Standard Gui Elements ###

    def showInterface(self):
        "Show the interface"
        self.main_window.show()

    def browseFolder(self):
        "Import ptm files from folder using file dialog"
        self.fileList.clear()
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "Select PTM files",
            "ptmviewer/assets/ptms",
            "PTMs (*.ptm)",
        )
        if fdir:
            for fname in fdir[0]:
                ptmitem = QtWidgets.QListWidgetItem(self.fileList)
                itemname = os.path.basename(fname)
                ptmitem.setText(itemname)
                ptmobj = {}
                ptmobj["path"] = fname
                ptmobj["name"] = itemname
                ptmobj["index"] = self.fileList.indexFromItem(ptmitem)
                self.ptmfiles[ptmobj["index"]] = ptmobj
                self.fileList.sortItems()

    def closeApp(self, event):
        "Close application"
        reply = QtWidgets.QMessageBox.question(
            self.centralwidget,
            "Message",
            "Are you sure to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit(0)
        else:
            event.ignore()
            #
        return

    def closeKey(self):
        sys.exit(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindowFinal()
    window.showInterface()
    sys.exit(app.exec_())
