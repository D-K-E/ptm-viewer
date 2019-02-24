# PTM viewer for PTM_FORMAT_RGB
# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

from core import PTMFileParse, PTMHandler
from core import setUpHandler

import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import os
import numpy as np
from PIL import Image
from PIL import ImageTk


class Viewer:
    "Viewer application"

    def __init__(self):
        "Viewer gui app"

        self.ptmfiles = {}
        self.handler = None
        self.image_id = None
        self.image = None

        # Widgets
        self.mainWindow = None
        self.scrolly = None
        self.listBox = None
        self.importBtn = None
        self.loadBtn = None
        self.saveBtn = None
        self.canvas = None
        self.canvasScrollx = None
        self.canvasScrolly = None
        self.renderBtn = None
        self.quitBtn = None
        self.controlFrame = None
        self.diffFrame = None
        self.diffGainSpin = None
        self.cmapFrame = None
        self.cmapComboBox = None
        self.lightAngleFrame = None
        self.lightAngleSpin = None
        self.lightPosFrame = None
        self.lightPosSpin = None

    def createMainWindow(self):
        "Create the main application window"
        self.mainWindow = tkinter.Tk()
        self.mainWindow.title('Python PTM Viewer')
        self.mainWindow.geometry('1024x768')
        self.mainWindow.resizable(width=True,
                                  height=True)
        self.mainWindow.grid_propagate(True)
        self.mainWindow.grid()

    def createListBox(self):
        "Create the listbox that would hold the ptm file paths"
        self.scrolly = tkinter.Scrollbar(
            master=self.mainWindow,
            orient=tkinter.VERTICAL)
        self.scrolly.grid(row=0, column=1,
                          sticky=tkinter.E + tkinter.S)
        self.listBox = tkinter.Listbox(
            master=self.mainWindow,
            selectmode='single',
            yscrollcommand=self.scrolly.set,
            height=60)
        self.listBox.grid(
            column=0,
            columnspan=2,
            row=0,
            sticky=tkinter.N+tkinter.S+tkinter.E+tkinter.W)
        self.scrolly.config(command=self.listBox.yview)

    def createLoadBtn(self):
        "Create loadBtn"
        self.loadBtn = tkinter.Button(
            master=self.mainWindow,
            command=self.load2Canvas,
            text='Load')
        self.loadBtn.grid(column=0,
                          row=2)

    def createSaveBtn(self):
        "Create save button"
        self.saveBtn = tkinter.Button(
            master=self.mainWindow,
            text='Save')
        self.saveBtn.grid(column=1,
                          row=2)

    def createCanvas(self):
        "Create canvas"
        self.canvasScrollx = tkinter.Scrollbar(
            master=self.mainWindow,
            orient=tkinter.HORIZONTAL)
        self.canvasScrollx.grid(row=0, 
                                column=2,
                                sticky=tkinter.S)
        self.canvasScrolly = tkinter.Scrollbar(
            master=self.mainWindow,
            orient=tkinter.VERTICAL)
        self.canvasScrolly.grid(row=0,
                                column=3,
                                sticky=tkinter.E)

        self.canvas = tkinter.Canvas(
            master=self.mainWindow,
            bd=2,
            width=800,
            height=624,
            scrollregion=(
                0, 0,
                800,
                624),
            yscrollcommand=self.canvasScrolly.set,
            xscrollcommand=self.canvasScrollx.set,
        )
        self.canvas.grid(column=2, row=0,
                         sticky=tkinter.NW)
        self.canvas.columnconfigure(2, weight=2)
        self.canvas.rowconfigure(0, weight=2)
        self.canvasScrollx.config(
            command=self.canvas.xview)
        self.canvasScrolly.config(
            command=self.canvas.yview)


    def createRenderBtn(self):
        "create render button"
        self.renderBtn = tkinter.Button(
            master=self.mainWindow,
            text='Render'
        )
        self.renderBtn.grid(column=0, row=4)

    def createQuitBtn(self):
        "create quit button"
        self.quitBtn = tkinter.Button(
            master=self.mainWindow,
            text='Quit'
        )
        self.quitBtn.grid(column=1, row=4)

    def createImportBtn(self):
        self.importBtn = tkinter.Button(
            master=self.mainWindow,
            command=self.getPtmFiles,
            text='Import Files')
        self.importBtn.grid(row=1,
                            column=0,
                            columnspan=2)

    def createControlFrame(self):
        "create control frame"
        self.controlFrame = tkinter.LabelFrame(
            master=self.mainWindow,
            text='Controls')
        self.controlFrame.grid(column=2,
                               columnspan=1,
                               row=1,
                               rowspan=3)

    def createDiffFrame(self):
        self.diffFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            height=100,
            width=100,
            text='Diffusion Gain Value'
        )
        self.diffFrame.grid(column=2,
                            row=1,
                            in_=self.controlFrame,
                            ipadx=2,
                            )

    def createDiffGainSpin(self):
        "create diff gain scale"
        self.diffGainSpin = tkinter.Spinbox(
            master=self.diffFrame,
            from_=0.01,
            to=10.0,
            increment=0.03)
        self.diffGainSpin.grid(
            column=2,
            in_=self.diffFrame,
            row=1)

    def createCmapFrame(self):
        "create colormap frame"
        self.cmapFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text='Colormaps')
        self.cmapFrame.grid(
            column=3,
            row=1,
            in_=self.controlFrame,
            ipadx=1,
            padx=1)

    def createCmapComboBox(self):
        "create colormap combobox"
        self.cmapComboBox = ttk.Combobox(
            master=self.cmapFrame,
            values=['hsv',
                    'red-yellow-blue',
                    'red-yellow-green',
                    'spectral',
                    'viridis',
                    'Greys',
                    'rainbow']
        )
        self.cmapComboBox.grid(column=2,
                               in_=self.cmapFrame,
                               row=2)

    def createLightAngleFrame(self):
        "Create light angle frame"
        self.lightAngleFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text='Angle of Altitude'
        )
        self.lightAngleFrame.grid(column=0,
                                  row=1,
                                  ipadx=15,
                                  )

    def createLightAngleSpin(self):
        "Create altitude angle spinbox"
        self.lightAngleSpin = tkinter.Spinbox(
            master=self.lightAngleFrame,
            from_=1,
            increment=1,
            to=90)
        self.lightAngleSpin.grid(column=0,
                                 row=0)

    def createLightPosFrame(self):
        "Create light position frame"
        self.lightPosFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text='Light Position'
        )
        self.lightPosFrame.grid(column=1,
                                row=1,
                                in_=self.controlFrame,
                                ipadx=19)

    def createLightPosSpin(self):
        "Create light pos spin"
        self.lightPosSpin = tkinter.Spinbox(
            master=self.lightPosFrame,
            from_=1,
            to=360,
            increment=1)
        self.lightPosSpin.grid(column=0,
                               in_=self.lightPosFrame,
                               row=0
                               )

    def createWidgets(self):
        "Create widgets in their proper layout"
        self.createMainWindow()
        self.createListBox()
        self.createImportBtn()
        self.createLoadBtn()
        self.createSaveBtn()
        self.createCanvas()
        self.createRenderBtn()
        self.createQuitBtn()
        self.createControlFrame()
        self.createDiffFrame()
        self.createDiffGainSpin()
        self.createCmapFrame()
        self.createCmapComboBox()
        self.createLightAngleFrame()
        self.createLightAngleSpin()
        self.createLightPosFrame()
        self.createLightPosSpin()

    # App logic

    def getPtmFiles(self):
        "Get ptm files from user"
        self.ptmfiles = {}
        files = filedialog.askopenfilenames(
            filetypes=(("PTM files", "*.ptm"),
                       ("all files", "*.*")),
            initialdir=os.path.dirname(__file__)
        )
        for i, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            self.ptmfiles[fname] = {'path': fpath,
                                    'name': fname,
                                    'index': i}
            self.listBox.insert(i, fname)

    def set_handler(self, ptmname: str):
        "Get ptmpath and set up ptm handler"
        print(ptmname)
        path = self.ptmfiles[ptmname]['path']
        self.handler = setUpHandler(path)

    def loadImage2Canvas(self):
        "Load handler image to canvas"
        if self.image_id is not None:
            self.canvas.delete(self.image_id)
        pilimg = Image.fromarray(
            np.copy(self.handler.image)
        )
        self.image = ImageTk.PhotoImage(image=pilimg)
        self.canvas.config(width=pilimg.width,
                           height=pilimg.height,
                           )
        self.image_id = self.canvas.create_image(
            0,
            0,
            image=self.image,
            anchor=tkinter.N+tkinter.W
        )
        print('loading done')

    def load2Canvas(self):
        "Handle image loading event"
        cur_ptm = self.listBox.curselection()
        if not cur_ptm:
            messagebox.showerror(
                title='Selection Error',
                message='Please select a ptm file from list')
            return
        #
        index = cur_ptm[0]
        ptmname = self.listBox.get(index)
        self.set_handler(ptmname)
        self.loadImage2Canvas()


if __name__ == '__main__':
    curdir = os.getcwd()
    assetdir = os.path.join(curdir, 'assets')
    uipath = os.path.join(assetdir, 'interface.ui')
    viewapp = Viewer()
    viewapp.createWidgets()
    viewapp.mainWindow.mainloop()
    #
