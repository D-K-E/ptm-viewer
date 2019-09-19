# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL

from PySide2.QtCore import QCoreApplication

from shiboken2 import VoidPtr

import numpy as np
import sys
import os
from PIL import Image, ImageQt

from ptmviewer.interface.window import Ui_MainWindow
from ptmviewer.glwidget import PtmGLWidget
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
        # gl widget
        self.glwidget = None

        # Main Window Events
        self.main_window.setWindowTitle("Python PTM Viewer")
        self.main_window.closeEvent = self.closeApp
        self.closeShort = QtWidgets.QShortcut(QtGui.QKeySequence("ctrl+w"),
                self.main_window)
        self.closeShort.activated.connect(self.closeKey)
        # self.main_window.setShortcut("ctrl+w")

        # Button related events
        self.addFile.clicked.connect(self.browseFolder)
        self.addFile.setShortcut("ctrl+o")

        self.loadFile.clicked.connect(self.loadPtm)

    # Ptm related stuff
    def loadPtm(self):
        "load ptm file into gl widget"
        citem = self.fileList.currentItem()
        cindex = self.fileList.indexFromItem(citem)
        ptmobj = self.ptmfiles[cindex]
        ptm = RGBPTM(ptmobj["path"])
        nimg = ptm.getNormalMap()
        nimgQt = ImageQt.ImageQt(nimg)
        texture = ptm.getImage()
        textureQt = ImageQt.ImageQt(texture)
        self.glwidget = PtmGLWidget(surfaceNormals=nimgQt, texture=textureQt)

    ### Standard Gui Elements ###

    def showInterface(self):
        "Show the interface"
        self.main_window.show()

    def browseFolder(self):
        "Import ptm files from folder using file dialog"
        self.fileList.clear()
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget, "Select PTM files", "", "PTMs (*.ptm)"
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
