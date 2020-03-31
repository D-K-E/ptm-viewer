# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL

from PySide2.QtCore import QCoreApplication
from PySide2.QtWidgets import QOpenGLWidget
from PySide2.QtGui import QVector3D

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
        self.sact = self.prepDockWidget(
            txt="shader controller", keyseq="ctrl+s", dockWidget=self.shaderDock
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


class AppWindowGLEvents(AppWindowInit):
    def get_intensity_values(self) -> dict:
        "get intensity values from spin boxes"
        ambient = self.viewerWidget.lamp.ambient.intensity
        diffuse = self.viewerWidget.lamp.diffuse.intensity
        specular = self.viewerWidget.lamp.specular.intensity
        return {
            "intensities": {
                "ambient": ambient,
                "diffuse": diffuse,
                "specular": specular,
            }
        }

    def get_coefficient_values(self):
        "get coefficient values"
        ambient = self.viewerWidget.lamp.ambient.coeffs
        diffuse = self.viewerWidget.lamp.diffuse.coeffs
        specular = self.viewerWidget.lamp.specular.coeffs
        return {
            "coefficients": {
                "ambient": ambient,
                "diffuse": diffuse,
                "specular": specular,
            }
        }

    def get_lamp_parameters(self) -> dict:
        "get lamp parameters"
        pos = self.viewerWidget.lamp.position
        position = {"position": {"x": pos.x(), "y": pos.y(), "z": pos.z()}}
        frt = self.viewerWidget.lamp.front
        front = {"front": {"x": frt.x(), "y": frt.y(), "z": frt.z()}}
        upl = self.viewerWidget.lamp.up
        up = {"up": {"x": upl.x(), "y": upl.y(), "z": upl.z()}}
        rgt = self.viewerWidget.lamp.right
        right = {"right": {"x": rgt.x(), "y": rgt.y(), "z": rgt.z()}}
        yaw = self.viewerWidget.lamp.yaw
        pitch = self.viewerWidget.lamp.pitch
        roll = self.viewerWidget.lamp.roll
        angles = {"angles": {"yaw": yaw, "pitch": pitch, "roll": roll}}
        atten = self.viewerWidget.lamp.attenuation
        attenuation = {
            "attenuation": {
                "constant": atten.x(),
                "linear": atten.y(),
                "quadratic": atten.z(),
            }
        }
        params = {}
        params.update(position)
        params.update(front)
        params.update(up)
        params.update(right)
        params.update(angles)
        params.update(attenuation)
        params["lightCutOff"] = self.viewerWidget.lamp.cutOff
        params["lightOuterCutOff"] = self.viewerWidget.lamp.outerCutOff
        return params

    def get_camera_parameters(self) -> dict:
        "get camera parameters"
        pos = self.viewerWidget.camera.position
        position = {"position": {"x": pos.x(), "y": pos.y(), "z": pos.z()}}
        frt = self.viewerWidget.camera.front
        front = {"front": {"x": frt.x(), "y": frt.y(), "z": frt.z()}}
        upl = self.viewerWidget.camera.up
        up = {"up": {"x": upl.x(), "y": upl.y(), "z": upl.z()}}
        rgt = self.viewerWidget.camera.right
        right = {"right": {"x": rgt.x(), "y": rgt.y(), "z": rgt.z()}}
        yaw = self.viewerWidget.camera.yaw
        pitch = self.viewerWidget.camera.pitch
        roll = self.viewerWidget.camera.roll
        viewMat = self.viewerWidget.camera.getViewMatrix()
        angles = {"angles": {"yaw": yaw, "pitch": pitch, "roll": roll}}
        params = {}
        params["zoom"] = self.viewerWidget.camera.zoom
        params["view_matrix"] = repr(viewMat)
        params.update(position)
        params.update(front)
        params.update(up)
        params.update(right)
        params.update(angles)
        return params

    def get_shader_parameters(self) -> dict:
        "get shader parameters"
        params = {}
        shininess = self.shinSpin.value()
        lampShader = self.viewerWidget.lampShaderName
        objectShader = self.viewerWidget.objectShaderName
        params["light-source-shader"] = shaders[lampShader]
        params["object-shader"] = shaders[objectShader]
        params["shininess"] = shininess
        return params


class AppWindowLightControlEvents(AppWindowGLEvents):
    "Light Control events"

    def set_intensity_coefficient(self, val: float, channel: str, isIntensity=False):
        "set intensity"
        if self.ambientRBtn.isChecked():
            if isIntensity:
                self.viewerWidget.change_lamp_ambient_intensity(
                    channel=channel, val=val
                )
            else:
                self.viewerWidget.change_lamp_ambient_coefficient(
                    channel=channel, val=val
                )

        elif self.diffuseRBtn.isChecked():
            if isIntensity:
                self.viewerWidget.change_lamp_diffuse_intensity(
                    channel=channel, val=val
                )
            else:
                self.viewerWidget.change_lamp_diffuse_coefficient(
                    channel=channel, val=val
                )

        elif self.specularRBtn.isChecked():
            if isIntensity:
                self.viewerWidget.change_lamp_specular_intensity(
                    channel=channel, val=val
                )
            else:
                self.viewerWidget.change_lamp_specular_coefficient(
                    channel=channel, val=val
                )

        else:
            return

    def set_red_intensity(self):
        redval = self.intensityR.value()
        self.set_intensity_coefficient(val=redval, channel="red", isIntensity=True)

    def set_green_intensity(self):
        greenval = self.intensityG.value()
        self.set_intensity_coefficient(val=greenval, channel="green", isIntensity=True)

    def set_blue_intensity(self):
        blueval = self.intensityB.value()
        self.set_intensity_coefficient(val=blueval, channel="blue", isIntensity=True)

    def set_red_coefficient(self):
        redval = self.coefficientR.value()
        self.set_intensity_coefficient(val=redval, channel="red", isIntensity=False)

    def set_green_coefficient(self):
        greenval = self.coefficientG.value()
        self.set_intensity_coefficient(val=greenval, channel="green", isIntensity=False)

    def set_blue_coefficient(self):
        blueval = self.coefficientB.value()
        self.set_intensity_coefficient(val=blueval, channel="blue", isIntensity=False)


class AppWindowRotateControl(AppWindowLightControlEvents):
    "Rotate Control"

    def get_axes(self) -> list:
        zaxis = self.rotZCbox.isChecked()
        yaxis = self.rotYCbox.isChecked()
        xaxis = self.rotXCbox.isChecked()
        rotaxes = []
        if zaxis:
            rotaxes.append("z")
        if yaxis:
            rotaxes.append("y")
        if xaxis:
            rotaxes.append("x")
        return rotaxes

    def set_angles(self):
        "Set euler angles either to yaw, pitch and roll"
        rotation_axes = self.get_axes()
        angle = self.angleSpin.value()
        self.viewerWidget.set_rotate_axes(rotation_axes)
        if self.rotCamRBtn.isChecked():
            self.viewerWidget.set_euler_angles_to_camera(angle)
        elif self.rotLightRBtn.isChecked():
            self.viewerWidget.set_euler_angle_to_lamp(angle)


class AppWindowMoveControl(AppWindowRotateControl):
    "Move window"

    def move_light_camera(self, isUp=False):
        ""
        zaxis = self.moveZCBox.isChecked()
        yaxis = self.moveYCBox.isChecked()
        xaxis = self.moveXCBox.isChecked()
        zdir = "+z" if isUp else "-z"
        ydir = "+y" if isUp else "-y"
        xdir = "+x" if isUp else "-x"
        if self.moveLightRBtn.isChecked():
            if zaxis:
                self.viewerWidget.move_light(zdir)
            if yaxis:
                self.viewerWidget.move_light(ydir)
            if xaxis:
                self.viewerWidget.move_light(xdir)
        elif self.moveCameraRBtn.isChecked():
            if zaxis:
                self.viewerWidget.move_camera(zdir)
            if yaxis:
                self.viewerWidget.move_camera(ydir)
            if xaxis:
                self.viewerWidget.move_camera(xdir)

    def move_up_light_camera(self):
        self.move_light_camera(isUp=True)

    def move_down_light_camera(self):
        self.move_light_camera(isUp=False)


class AppWindowFinal(AppWindowMoveControl):
    "Final window"

    def __init__(self):
        super().__init__()
        self.ptmfiles = {}

        # dock widgets

        ## params widget
        ### Available buttons
        self.captureBtn.clicked.connect(self.captureParams)

        ## transcribe widget
        self.saveBtn.clicked.connect(self.saveNotes)

        ## move widget
        ### Available buttons
        # self.moveCameraRBtn.toggled.connect(lambda x: x)
        # self.moveLightRBtn.toggled.connect(lambda x: x)
        # self.moveXCBox.stateChanged.connect(lambda x: x)
        # self.moveYCBox.stateChanged.connect(lambda x: x)
        # self.moveZCBox.stateChanged.connect(lambda x: x)
        self.moveUp.clicked.connect(self.move_up_light_camera)
        self.moveDown.clicked.connect(self.move_down_light_camera)
        ## rotate widget
        ### Available buttons
        # self.rotCamRBtn.toggled.connect(lambda x: x)
        # self.rotLightRBtn.toggled.connect(lambda x: x)
        self.angleSpin.valueChanged.connect(self.set_angles)
        ## color widget
        ### Available buttons
        # self.ambientRBtn.toggled.connect(lambda x: x)
        # self.diffuseRBtn.toggled.connect(lambda x: x)
        # self.specularRBtn.toggled.connect(lambda x: x)
        self.intensityR.valueChanged.connect(self.set_red_intensity)
        self.intensityG.valueChanged.connect(self.set_green_intensity)
        self.intensityB.valueChanged.connect(self.set_blue_intensity)
        self.coefficientR.valueChanged.connect(self.set_red_coefficient)
        self.coefficientG.valueChanged.connect(self.set_green_coefficient)
        self.coefficientB.valueChanged.connect(self.set_blue_coefficient)

        ## file list widget
        ### Available buttons
        self.addFile.clicked.connect(self.browseFolder)
        self.addFile.setShortcut("ctrl+o")

        self.loadFile.clicked.connect(self.loadPtm)
        self.loadFile.setShortcut("ctrl+l")

        self.removeFile.clicked.connect(self.removeItems)
        ## shader widget

        # viewer widget, opengl widgets with different shaders
        self.availableGlWidgets = {
            "Lambertian": PtmLambertianGLWidget,
            "SingleNormalMap": PtmNormalMapGLWidget,
            "PerChannelPhong": PtmPerChannelNormalMapPhongGLWidget,
            "PerChannelNormalMapDir": PtmPerChannelNormalMapDirGLWidget,
            "PerChannelNormalMapPoint": PtmPerChannelNormalMapPointGLWidget,
            # "PerChannelNormalMapSpot": PtmPerChannelNormalMapSpotGLWidget,
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
        self.notedir = os.path.join(assetdir, "notes")

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
        pdb.set_trace()
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
        light_params = self.get_lamp_parameters()
        camera_params = self.get_camera_parameters()
        parameters = {}
        parameters["light"] = self.get_lamp_parameters()
        parameters["camera"] = self.get_camera_parameters()
        parameters["shader"] = self.get_shader_parameters()
        return parameters

    def updateParamBrowser(self):
        params = self.getParams()
        self.paramBrowser.clear()
        self.paramBrowser.setPlainText(str(params))

    def captureParams(self):
        params = self.getParams()
        params["image-format"] = "PNG"
        params["image-encoding"] = "hex"
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
                json.dump(params, fd, ensure_ascii=False, indent=2)

    def saveNotes(self):
        "Save notes in the transcription dock widget"
        text = self.transcribeEdit.toPlainText()
        fileName = QtWidgets.QFileDialog.getSaveFileName(
            self.centralwidget, "Save Notes", self.notedir, "Text Files (*.txt)"
        )
        fpath = fileName[0]
        if fpath:
            with open(fpath, "w", encoding="utf-8", newline="\n") as fd:
                fd.write(text)

    def set_shininess(self):
        shininess = self.shinSpin.value()
        self.viewerWidget.change_shininess(val=shininess)

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

    def removeItems(self):
        "remove items from list"
        items = self.fileList.selectedItems()
        if not items:
            return
        for item in items:
            index = self.fileList.indexFromItem(item)
            itemRow = self.fileList.row(item)
            self.ptmfiles.pop(index)
            self.fileList.takeItem(itemRow)

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
