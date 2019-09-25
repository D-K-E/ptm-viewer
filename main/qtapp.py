# author: Kaan Eraslan
# Purpose: Application wrapper for ptm viewer

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import core
import numpy as np
import sys
import os
import OpenGL.GL as gl

from interface.brutInterface import Ui_MainWindow


class AppWindowInit(Ui_MainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = Qt.QMainWindow()
        super().setupUi(self.main_window)
        pass


class GLWidget(QtWidgets.QOpenGLWidget):
    "Extends qopengl widget to take light source etc as arguments"
    lightXPosChanged = QtCore.pyqtSignal(int)
    lightYPosChanged = QtCore.pyqtSignal(int)
    lightZPosChanged = QtCore.pyqtSignal(int)
    lightIntensityChanged = QtCore.pyqtSignal(float)
    ambientTermChanged = QtCore.pyqtSignal(float)
    diffuseCoeffChanged = QtCore.pyqtSignal(float)
    shininessChanged = QtCore.pyqtSignal(float)
    zoomChanged = QtCore.pyqtSignal(float)
    specColorChanged = QtCore.pyqtSignal(float)
    vert_shader = """
attribute vec3 inputPosition;
attribute vec2 inputTexCoord;
attribute vec3 inputNormal;

uniform mat4 projection, modelview, normalMat;

varying vec3 normalInterp;
varying vec3 vertPos;

void main(){
    gl_Position = projection * modelview * vec4(inputPosition, 1.0);
    vec4 vertPos4 = modelview * vec4(inputPosition, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = vec3(normalMat * vec4(inputNormal, 0.0));
}
"""
    frag_shader = """
precision mediump float; 

varying vec3 normalInterp;
varying vec3 vertPos;

uniform int mode;

const vec3 lightPos = vec3(1.0,1.0,1.0);
const vec3 ambientColor = vec3(0.1, 0.0, 0.0);
const vec3 diffuseColor = vec3(0.5, 0.0, 0.0);
const vec3 specColor = vec3(1.0, 1.0, 1.0);

void main() {

  vec3 normal = normalize(normalInterp);
  vec3 lightDir = normalize(lightPos - vertPos);

  float lambertian = max(dot(lightDir,normal), 0.0);
  float specular = 0.0;

  if(lambertian > 0.0) {

    vec3 viewDir = normalize(-vertPos);

    // this is blinn phong
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(halfDir, normal), 0.0);
    specular = pow(specAngle, 16.0);
    }
    gl_FragColor = vec4(ambientColor +
                      lambertian * diffuseColor +
                      specular * specColor, 1.0);
}
"""

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.object = 0
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.diffuse_coeff = 0.08
        self.specColor = 0.1
        self.ambientTerm = 0.1
        self.lastPos = QtCore.QPoint()
        self.program = None

    def getGlInfo(self):
        "Get opengl info"
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )
        return info

    def resizeGL(self, width, height):
        "resize gl widget"
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side,
                      side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXPos(self.xpos + 8 * dy)
            self.setYPos(self.ypos + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXPos(self.xpos + 8 * dy)
            self.setZPos(self.zpos + 8 * dx)

        self.lastPos = event.pos()

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def setXPos(self, angle):
        "Set x angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.xpos:
            self.xpos = angle
            self.lightXPosChanged.emit(angle)
            self.update()

    def setYPos(self, angle):
        "Set y angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.ypos:
            self.ypos = angle
            self.lightYPosChanged.emit(angle)
            self.update()

    def setZPos(self, angle):
        "Set z angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.zpos:
            self.zpos = angle
            self.lightZPosChanged.emit(angle)
            self.update()

    def setSpecColor(self, val: float):
        "Set spec color"
        if val != self.specColor:
            self.specColor = val
            self.specColorChanged.emit(val)
            self.update()

    def setDiffuseCoeff(self, val: float):
        "Set diffuse coeff color"
        if val != self.diffuse_coeff:
            self.diffuse_coeff = val
            self.diffuseCoeffChanged.emit(val)
            self.update()

    def setShininess(self, val: float):
        "Set diffuse coeff color"
        if val != self.shininess:
            self.shininess = val
            self.shininessChanged.emit(val)
            self.update()

    def setAmbientTerm(self, val: float):
        "Set ambient term color"
        if val != self.ambientTerm:
            self.ambientTerm = val
            self.ambientTermChanged.emit(val)
            self.update()

    def paintGL(self):
        pass

    def initializeGL(self):
        pass

    def makeObject(self):
        pass


class AppWindowFinal(AppWindowInit):
    "Final window"

    def __init__(self):
        super().__init__()
        self.ptmfiles = {}
        self.ptm = None
        self.image = None
        self.pilimg = None

        # Main Window Events
        self.main_window.setWindowTitle("Python PTM Viewer")
        self.main_window.closeEvent = self.closeApp

    ### Standard Gui Elements ###

    def showInterface(self):
        "Show the interface"
        self.main_window.show()

    def browseFolder(self):
        "Import ptm files from folder using file dialog"
        self.listWidget.clear()
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "Select PTM files", "", "PTMs (*.ptm)")
        if fdir:
            for fname in fdir[0]:
                ptmitem = QtWidgets.QListWidgetItem(self.listWidget)
                itemname = os.path.basename(fname)
                ptmitem.setText(itemname)
                ptmobj = {}
                ptmobj['path'] = fname
                ptmobj['name'] = itemname
                ptmobj['index'] = self.listWidget.indexFromItem(ptmitem)
                self.ptmfiles[ptmobj['index']] = ptmobj
                self.listWidget.sortItems()

    def closeApp(self, event):
        "Close application"
        reply = QtWidgets.QMessageBox.question(self.centralwidget, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit(0)
        else:
            event.ignore()
            #
        return


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = AppWindowFinal()
    window.showInterface()
    sys.exit(app.exec_())
