# PTM viewer for PTM_FORMAT_RGB
# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

from core import PTMFileParse, PTMHandler
from core import setUpHandler

import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageTk


class Viewer:
    "Viewer application"

    def __init__(self):
        "Viewer gui app"

        self.cmaps = {
            'hsv': plt.cm.hsv,
            'red-yellow-blue': plt.cm.RdYlBu,
            'red-yellow-green': plt.cm.RdYlGn,
            'spectral': plt.cm.Spectral,
            'viridis': plt.cm.viridis,
            'greys': plt.cm.Greys,
            'rainbow': plt.cm.rainbow}

        self.ptmfiles = {}
        self.handler = None
        self.image_id = None
        self.image = None
        self.pilimg = None

        # Widgets
        self.mainWindow = None
        self.scrolly = None
        self.listBox = None
        # button font
        # self.buttonFont = tkFont.Font(family='Times',
        #                               size=12)
        self.buttonFont = ('Helvetica', 11)
        # self.labelFont = tkFont.Font(family='Times',
        #                              size=12,
        #                              weight='bold')
        self.labelFont = ('Helvetica', 11, 'bold')

        # button container
        self.buttonFrame = None
        # buttons
        self.importBtn = None
        self.saveBtn = None
        self.loadBtn = None
        self.renderBtn = None
        self.quitBtn = None
        self.resetRenderBtn = None
        #
        self.canvas = None
        self.canvasScrollx = None
        self.canvasScrolly = None
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

        self.mainWindow.grid_rowconfigure(0, weight=1)

    # button related

    def createButtonFrame(self):
        "Create the frame that would contain the buttons"
        self.buttonFrame = tkinter.Frame(
            master=self.mainWindow)
        self.buttonFrame.grid(row=1,
                              column=0,
                              rowspan=4,
                              columnspan=2)
        self.mainWindow.grid_columnconfigure(0, weight=1)

    def createRenderBtn(self):
        "create render button"
        self.renderBtn = tkinter.Button(
            master=self.buttonFrame,
            command=self.renderImage,
            text='Render',
            font=self.buttonFont
        )
        self.renderBtn.grid(column=0, row=4)

    def createQuitBtn(self):
        "create quit button"
        self.quitBtn = tkinter.Button(
            master=self.buttonFrame,
            command=self.quit,
            text='Quit',
            font=self.buttonFont
        )
        self.quitBtn.grid(column=1, row=4)

    def createResetRenderBtn(self):
        "create reset rendering button"
        self.resetRenderBtn = tkinter.Button(
            master=self.buttonFrame,
            command=self.resetRendering,
            text='Reset Rendering',
            font=self.buttonFont)
        self.resetRenderBtn.grid(column=0,
                                 columnspan=2,
                                 row=5)

    def createImportBtn(self):
        self.importBtn = tkinter.Button(
            master=self.buttonFrame,
            command=self.getPtmFiles,
            text='Import Files',
            font=self.buttonFont
        )
        self.importBtn.grid(row=1,
                            column=0,
                            columnspan=2)

    def createLoadBtn(self):
        "Create loadBtn"
        self.loadBtn = tkinter.Button(
            master=self.buttonFrame,
            command=self.load2Canvas,
            text='Load',
            font=self.buttonFont
        )
        self.loadBtn.grid(column=0,
                          row=2)

    def createSaveBtn(self):
        "Create save button"
        self.saveBtn = tkinter.Button(
            master=self.buttonFrame,
            text='Save',
            font=self.buttonFont
        )
        self.saveBtn.grid(column=1,
                          row=2)

    # canvas related

    def createCanvasScrollx(self):
        "Create canvas scroll x"
        self.canvasScrollx = tkinter.Scrollbar(
            master=self.mainWindow,
            orient=tkinter.HORIZONTAL)

    def createCanvasScrolly(self):
        "Create canvas scroll bar y direction"
        self.canvasScrolly = tkinter.Scrollbar(
            master=self.mainWindow,
            orient=tkinter.VERTICAL)

    def startCanvasMouseScroll(self, event):
        "Start scrolling with mouse button"
        self.canvas.scan_mark(event.y, event.y)

    def endCanvasMouseScroll(self, event):
        "End scrolling with mouse button"
        self.canvas.scan_dragto(event.x, event.y, gain=2)

    def createCanvasZone(self):
        "Create canvas zone"
        self.canvas = tkinter.Canvas(
            master=self.mainWindow,
            bd=2,
            width=800,
            height=624,
            relief=tkinter.SUNKEN,
            highlightthickness=0
        )
        self.canvas.bind("<ButtonPress-1>", self.startCanvasMouseScroll)
        self.canvas.bind("<B1-Motion>", self.endCanvasMouseScroll)

    def createCanvas(self):
        "Create canvas"
        self.createCanvasScrollx()
        self.createCanvasScrolly()
        self.createCanvasZone()
        self.canvasScrollx.config(
            command=self.canvas.xview)
        self.canvasScrolly.config(
            command=self.canvas.yview)

        self.canvas.config(xscrollcommand=self.canvasScrollx.set,
                           yscrollcommand=self.canvasScrolly.set)

        self.canvasScrolly.grid(row=0,
                                column=3,
                                sticky=tkinter.E)

        self.canvasScrollx.grid(row=1,
                                column=2,
                                sticky=tkinter.S)

        self.canvas.grid(column=2, row=0,
                         sticky=tkinter.NSEW)
        self.mainWindow.columnconfigure(2, weight=2)
        self.mainWindow.rowconfigure(0, weight=1)

    def createControlFrame(self):
        "create control frame"
        self.controlFrame = tkinter.LabelFrame(
            master=self.mainWindow,
            text='Controls')
        self.controlFrame.grid(column=2,
                               columnspan=1,
                               row=2,
                               rowspan=4)
        self.mainWindow.grid_columnconfigure(2, weight=1)

    def createDiffFrame(self):
        self.diffFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            height=100,
            width=100,
            text='Diffusion Gain Value',
            font=self.labelFont
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
            text='Colormaps',
            font=self.labelFont
        )
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
            values=list(self.cmaps.keys())
        )
        self.cmapComboBox.grid(column=2,
                               in_=self.cmapFrame,
                               row=2)

    def createLightAngleFrame(self):
        "Create light angle frame"
        self.lightAngleFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text='Angle of Altitude',
            font=self.labelFont
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
            text='Light Position',
            font=self.labelFont
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
        self.createResetRenderBtn()
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

    def quit(self):
        "Quit application"
        res = messagebox.askyesno(
            'Quit Application',
            'Are you sure you want to quit the application')
        if res:
            saveres = messagebox.askyesno(
                'Save Image',
                'Would like to save currently rendered image before you quit ?'
            )
            if saveres:
                self.saveImage()
            #
            self.mainWindow.destroy()
            print('Widgets are destroyed, bye bye ...')
            sys.exit(0)

    def saveImage(self):
        "Save the image"
        savefile = filedialog.asksaveasfile(mode="wb",
                                            defaultextension='.png')
        self.pilimg.save(savefile, 'png')

    def getPtmFiles(self):
        "Get ptm files from user"
        self.listBox.delete(0, tkinter.END)
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
        path = self.ptmfiles[ptmname]['path']
        self.handler = setUpHandler(path)

    def loadImage2Canvas(self, imarray):
        "Load handler image to canvas"
        if self.image_id is not None:
            self.canvas.delete(self.image_id)
        self.pilimg = Image.fromarray(
            np.copy(imarray)
        )
        self.image = ImageTk.PhotoImage(image=self.pilimg)
        self.canvas.config(scrollregion=(0, 0,
                                         self.pilimg.width,
                                         self.pilimg.height))
        self.image_id = self.canvas.create_image(
            0,
            0,
            image=self.image,
            anchor='nw'
        )

    def getRenderValues(self):
        "Get values to render the image"
        light_position = self.lightPosSpin.get()
        light_angle = self.lightAngleSpin.get()
        diffusion_gain = self.diffGainSpin.get()
        cmap_index = self.cmapComboBox.get()
        cmap_fn = self.cmaps[cmap_index]
        light_position = int(light_position)
        light_angle = int(light_angle)
        diffusion_gain = float(diffusion_gain)
        coeffarr = np.copy(self.handler.arr)
        self.handler.g = diffusion_gain
        newim = self.handler.render_diffuse_gain(coeffarr)
        newim = self.handler.shade_with_light_source(
            rgb_image=newim,
            angle=light_position,
            elevation=light_angle,
            cmap_fn=cmap_fn)
        return newim

    def renderImage(self):
        "render image with controller values"
        newimarr = self.getRenderValues()
        self.loadImage2Canvas(newimarr)

    def resetRendering(self):
        "reset rendering values"
        coeffarr = np.copy(self.handler.arr)
        imarr = self.handler.computeImage(coeffarr)
        self.loadImage2Canvas(imarr)

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
        self.loadImage2Canvas(self.handler.image)


if __name__ == '__main__':
    curdir = os.getcwd()
    assetdir = os.path.join(curdir, 'assets')
    uipath = os.path.join(assetdir, 'interface.ui')
    viewapp = Viewer()
    viewapp.createWidgets()
    viewapp.mainWindow.mainloop()
    #
