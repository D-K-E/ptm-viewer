# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL

from PySide2.QtCore import QCoreApplication
from PySide2.QtWidgets import QOpenGLWidget

from PySide2.shiboken2 import VoidPtr

import numpy as np
import sys
import os
import json
import pdb
from PIL import Image, ImageQt

from ptmviewer.interface.window import Ui_MainWindow
from ptmviewer.glwidget import PtmLambertianGLWidget, PtmNormalMapGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapDirGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapPointGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapSpotGLWidget
from ptmviewer.glwidget import PtmPerChannelNormalMapPhongGLWidget
from ptmviewer.glwidget import PtmCoefficientShader
from ptmviewer.rgbptm import RGBPTM
from ptmviewer.utils.shaders import shaders


class AppWindowInit(Ui_MainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtWidgets.QMainWindow()
        super().setupUi(self.main_window)

        # adding basic defaults to here
        # Main Window Events
        self.main_window.setWindowTitle("Python PTM Viewer")
        self.main_window.closeEvent = self.closeApp
        self.closeShort = QtWidgets.QShortcut(
            QtGui.QKeySequence("ctrl+w"), self.main_window
        )
        self.closeShort.activated.connect(self.closeKey)
        # make menu
        self.toolbar = self.main_window.addToolBar("Widgets")
        self.createToolbar()

    def prepDockWidget(self, txt: str, keyseq: str, dockWidget):
        "prepare docker widget"
        dockAct = dockWidget.toggleViewAction()
        dockAct.setShortcut(QtGui.QKeySequence(keyseq))
        dockAct.setText(txt)
        dockAct.setToolTip(keyseq)
        self.toolbar.addAction(dockAct)
        return dockAct

    def createToolbar(self):
        "create tool bar"
        self.fact = self.prepDockWidget(
            txt="file list", keyseq="ctrl+f", dockWidget=self.fileListDock
        )
        # self.toolbar.addAction(self.fact)
        self.cact = self.prepDockWidget(
            txt="color controller", keyseq="ctrl+z", dockWidget=self.colorDock
        )
        # self.toolbar.addAction(self.cact)
        self.ract = self.prepDockWidget(
            txt="rotation controller", keyseq="ctrl+r", dockWidget=self.rotateDock
        )
        # self.toolbar.addAction(self.ract)
        self.mact = self.prepDockWidget(
            txt="move controller", keyseq="ctrl+m", dockWidget=self.moveDock
        )
        # self.toolbar.addAction(self.mact)
        self.pact = self.prepDockWidget(
            txt="parameter viewer", keyseq="ctrl+p", dockWidget=self.paramsDock
        )

        # self.toolbar.addAction(self.pact)
        self.tact = self.prepDockWidget(
            txt="transcriber", keyseq="ctrl+t", dockWidget=self.transcribeDock
        )

        # self.toolbar.addAction(self.tact)
        self.gact = self.prepDockWidget(
            txt="opengl info viewer", keyseq="ctrl+g", dockWidget=self.glInfoDock
        )
        # self.toolbar.addAction(self.gact)


class AppWindowFinal(AppWindowInit):
    "Final window"

    def __init__(self):
        super().__init__()
        self.ptmfiles = {}

        # Button related events
        self.addFile.clicked.connect(self.browseFolder)
        self.addFile.setShortcut("ctrl+o")

        self.loadFile.clicked.connect(self.loadPtm)
        self.loadFile.setShortcut("ctrl+l")

        # dock widgets

        ## params widget
        ### Available buttons
        self.captureBtn.clicked.connect(self.captureParams)

        ## transcribe widget
        self.saveBtn.clicked.connect(lambda x: x)

        ## move widget
        ### Available buttons
        self.moveCameraRBtn.toggled.connect(lambda x: x)
        self.moveLightRBtn.toggled.connect(lambda x: x)
        self.moveXCBox.stateChanged.connect(lambda x: x)
        self.moveYCBox.stateChanged.connect(lambda x: x)
        self.moveZCBox.stateChanged.connect(lambda x: x)
        self.moveUp.clicked.connect(lambda x: x)
        self.moveDown.clicked.connect(lambda x: x)
        ## rotate widget
        ### Available buttons
        self.rotCamRBtn.toggled.connect(lambda x: x)
        self.rotLightRBtn.toggled.connect(lambda x: x)
        self.rotX.valueChanged.connect(lambda x: x)
        self.rotY.valueChanged.connect(lambda x: x)
        self.rotZ.valueChanged.connect(lambda x: x)
        ## color widget
        ### Available buttons
        self.colorR.valueChanged.connect(lambda x: x)
        self.colorG.valueChanged.connect(lambda x: x)
        self.colorB.valueChanged.connect(lambda x: x)
        self.rotAngle.valueChanged.connect(lambda x: x)
        self.shinSpin.valueChanged.connect(lambda x: x)
        self.ambientCoeff.valueChanged.connect(lambda x: x)
        ## file list widget
        ### Available buttons
        self.addFile.clicked.connect(lambda x: x)
        self.loadFile.clicked.connect(lambda x: x)
        self.removeFile.clicked.connect(lambda x: x)
        ## gl info widget

        # viewer widget, opengl widgets with different shaders
        self.availableGlWidgets = {
            "Lambertian": PtmLambertianGLWidget,
            "SingleNormalMap": PtmNormalMapGLWidget,
            "PerChannelPhong": PtmPerChannelNormalMapPhongGLWidget,
            "PerChannelNormalMapDir": PtmPerChannelNormalMapDirGLWidget,
            "PerChannelNormalMapPoint": PtmPerChannelNormalMapPointGLWidget,
            "PerChannelNormalMapSpot": PtmPerChannelNormalMapSpotGLWidget,
        }
        wnames = [k for k in self.availableGlWidgets.keys()]
        self.shaderCombo.addItems(wnames)
        self.shaderCombo.setEditable(False)
        self.shaderCombo.setCurrentText(wnames[0])
        # self.shaderCombo.currentTextChanged.connect(self.loadPtm)
        self.cleanViewWidget = QtWidgets.QWidget(self.main_window)
        width = self.viewerWidget.width()
        height = self.viewerWidget.height()
        self.cleanViewWidget.resize(width, height)
        #
        maindir = os.curdir
        ptmdir = os.path.join(maindir, "ptmviewer")
        assetdir = os.path.join(ptmdir, "assets")
        self.jsondir = os.path.join(assetdir, "jsons")

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
        self.glInfoBrowser.clear()
        self.glInfoBrowser.setPlainText(info)

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
        elif glchoice == "PerChannelPhong":
            self.runPerChannelNormalMapsPipeline(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapDir":
            self.runPerChannelNormalMapsPipeline(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapPoint":
            self.runPerChannelNormalMapsPipeline(glchoice, ptm)
        elif glchoice == "PerChannelNormalMapSpot":
            self.runPerChannelNormalMapsPipeline(glchoice, ptm)
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

    def runPerChannelNormalMapsPipeline(self, glchoice: str, ptm):
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

    def getParams(self) -> dict:
        "Get parameters from widgets"
        red = self.colorR.value()
        green = self.colorG.value()
        blue = self.colorB.value()
        rotation_angle = self.rotAngle.value()
        shininess = self.shinSpin.value()
        ambient_coeff = self.ambientCoeff.value()
        light_position = self.viewerWidget.lamp.position
        light_direction = self.viewerWidget.lamp.direction
        light_attenuation = self.viewerWidget.lamp.attenuation
        light_cutoff = self.viewerWidget.lamp.cutOff
        camera_position = self.viewerWidget.camera.position
        camera_matrix = self.viewerWidget.camera.getViewMatrix()
        projectionMatrix = QtGui.QMatrix4x4()
        projectionMatrix.perspective(
            self.viewerWidget.camera.zoom,
            self.viewerWidget.width() / self.viewerWidget.height(),
            0.2,
            100.0,
        )
        model = QtGui.QMatrix4x4()
        parameters = {}
        vals = projectionMatrix.copyDataTo()
        parameters["projection_matrix"] = vals
        vals = camera_matrix.copyDataTo()
        parameters["camera_view_matrix"] = vals
        parameters["camera_position"] = dict(
            x=camera_position.x(), y=camera_position.y(), z=camera_position.z()
        )
        parameters["light_position"] = dict(
            x=light_position.x(), y=light_position.y(), z=light_position.z()
        )
        parameters["light_direction"] = dict(
            x=light_direction.x(), y=light_direction.y(), z=light_direction.z()
        )
        parameters["light_attenuation"] = dict(
            x=light_attenuation.x(), y=light_attenuation.y(), z=light_attenuation.z()
        )
        parameters["light_cutoff"] = str(light_cutoff)
        parameters["ambient_coefficient"] = str(ambient_coeff)
        parameters["intensities"] = {
            "red_channel_light_intensity": str(red),
            "green_channel_light_intensity": str(green),
            "blue_channel_light_intensity": str(blue),
        }
        parameters["lamp_rotation_angle"] = str(rotation_angle)
        return parameters

    def updateParamBrowser(self):
        params = self.getParams()
        self.paramBrowser.clear()
        self.paramBrowser.setPlainText(str(params))

    def captureParams(self):
        params = self.getParams()
        params["image-format"] = "PNG"
        params["image-encoding"] = "hex"
        lampShader = self.viewerWidget.lampShaderName
        objectShader = self.viewerWidget.objectShaderName
        params["light-source-shader"] = shaders[lampShader]
        params["object-shader"] = shaders[objectShader]
        screen = self.viewerWidget.grabFramebuffer()
        barr = QtCore.QByteArray()
        qbfr = QtCore.QBuffer(barr)
        qbfr.open(QtCore.QIODevice.WriteOnly)
        screen.save(qbfr, params["image-format"])
        params["image"] = barr.data().hex()
        fileName = QtWidgets.QFileDialog.getSaveFileName(
            self.centralwidget, "Save Parameters", self.jsondir, "Json Files (*.json)"
        )
        fpath = fileName[0]
        if fpath:
            with open(fpath, "w", encoding="utf-8", newline="\n") as fd:
                pdb.set_trace()
                json.dump(params, fd, ensure_ascii=False, indent=2)

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
