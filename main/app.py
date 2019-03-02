# PTM viewer for PTM_FORMAT_RGB
# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

from core import PTMFileParse, PTMHandler, LightSource
from core import setUpHandler

import tkinter
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
        #
        self.controlFrame = None
        #
        self.diffFrame = None
        self.diffGainSpin = None
        self.shaderFrame = None
        self.shaderComboBox = None
        #
        self.ambientTermFrame = None
        self.ambientTermSpin = None
        #
        self.lightOptionsFrame = None
        #
        self.lightPosXFrame = None
        self.lightPosXSpin = None
        self.lightPosYFrame = None
        self.lightPosYSpin = None
        self.lightRadiusFrame = None
        self.lightRadiusSpin = None
        self.lightIntensityFrame = None
        self.lightIntensitySpin = None

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
                                column=4,
                                sticky=tkinter.E)

        self.canvasScrollx.grid(row=1,
                                column=2,
                                sticky=tkinter.S)

        self.canvas.grid(column=2, row=0,
                         sticky=tkinter.NSEW)
        self.mainWindow.columnconfigure(2, weight=2)
        self.mainWindow.rowconfigure(0, weight=1)

    # Options related
    def createControlFrame(self):
        "create control frame"
        self.controlFrame = tkinter.LabelFrame(
            master=self.mainWindow,
            text='Controls')
        self.controlFrame.grid(column=2,
                               columnspan=4,
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
        self.diffFrame.grid(column=4,
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
            column=4,
            in_=self.diffFrame,
            row=1)

    def createShaderFrame(self):
        "create shader frame"
        self.shaderFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text='Shaders',
            font=self.labelFont
        )
        self.shaderFrame.grid(
            column=3,
            row=1,
            ipadx=1,
            padx=1)

    def createShaderComboBox(self):
        "create colormap combobox"
        self.shaderComboBox = ttk.Combobox(
            master=self.shaderFrame,
            values=['cell', 'phong']
        )
        self.shaderComboBox.grid(column=2,
                                 in_=self.shaderFrame,
                                 row=2)

    def createAmbientTermFrame(self):
        "Ambient term container"
        self.ambientTermFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text="Ambient Term",
            font=self.labelFont)
        self.ambientTermFrame.grid(column=3,
                                   row=0)

    def createAmbientTermSpin(self):
        "Ambient term"
        self.ambientTermSpin = tkinter.Spinbox(
            master=self.ambientTermFrame,
            from_=0.00,
            to=0.99,
            increment=0.01)
        self.ambientTermSpin.grid(column=3,
                                  row=0)

    def createLightOptionsFrame(self):
        "Regroups the light options"
        self.lightOptionsFrame = tkinter.LabelFrame(
            master=self.controlFrame,
            text="Light Controls",
            font=self.labelFont)
        self.lightOptionsFrame.grid(column=2,
                                    rowspan=2,
                                    row=0)

    def createLightPosXFrame(self):
        "Create light angle frame"
        self.lightPosXFrame = tkinter.LabelFrame(
            master=self.lightOptionsFrame,
            text='Light Position X val',
            font=self.labelFont
        )
        self.lightPosXFrame.grid(column=0,
                                 row=0,
                                 )

    def createLightPosXSpin(self):
        "Create altitude angle spinbox"
        self.lightPosXSpin = tkinter.Spinbox(
            master=self.lightPosXFrame,
            increment=1,
        )
        self.lightPosXSpin.grid(column=0,
                                row=0)

    def createLightPosYFrame(self):
        "Create light position frame"
        self.lightPosYFrame = tkinter.LabelFrame(
            master=self.lightOptionsFrame,
            text='Light Position Y val',
            font=self.labelFont
        )
        self.lightPosYFrame.grid(column=1,
                                 row=0,
                                 )

    def createLightPosYSpin(self):
        "Create light pos spin"
        self.lightPosYSpin = tkinter.Spinbox(
            master=self.lightPosYFrame,
            increment=1)
        self.lightPosYSpin.grid(column=1,
                                row=0
                                )

    def createLightRadiusFrame(self):
        "Create light radius frame"
        self.lightRadiusFrame = tkinter.LabelFrame(
            master=self.lightOptionsFrame,
            text="Light Radius",
            font=self.labelFont)
        self.lightRadiusFrame.grid(column=0,
                                   row=1)

    def createLightRadiusSpin(self):
        "light radius spin inside light radius frame"
        self.lightRadiusSpin = tkinter.Spinbox(
            master=self.lightRadiusFrame,
            from_=1.0,
            to=100.0,
            increment=1)
        self.lightRadiusSpin.grid(column=0,
                                  row=1)

    def createLightIntensityFrame(self):
        "Light intensity that goes inside the illumination equation container"
        self.lightIntensityFrame = tkinter.LabelFrame(
            master=self.lightOptionsFrame,
            text="Light Intensity",
            font=self.labelFont)
        self.lightIntensityFrame.grid(column=1,
                                      row=1)

    def createLightIntensitySpin(self):
        "Light intensity spin box"
        self.lightIntensitySpin = tkinter.Spinbox(
            master=self.lightIntensityFrame,
            from_=0.1,
            to=20.0,
            increment=0.1)
        self.lightIntensitySpin.grid(column=1,
                                     row=1)

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
        # Control
        self.createControlFrame()
        # Ambient term
        self.createAmbientTermFrame()
        self.createAmbientTermSpin()
        # Diffusion gain
        self.createDiffFrame()
        self.createDiffGainSpin()
        # Shader
        self.createShaderFrame()
        self.createShaderComboBox()
        # light options
        self.createLightOptionsFrame()
        self.createLightPosXFrame()
        self.createLightPosXSpin()
        self.createLightPosYFrame()
        self.createLightPosYSpin()
        self.createLightRadiusFrame()
        self.createLightRadiusSpin()
        self.createLightIntensityFrame()
        self.createLightIntensitySpin()

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
        self.handler_original = setUpHandler(path)
        self.handler = self.handler_original.copy()

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
        self.lightPosYSpin.config(from_=1, to=self.pilimg.width - 1)
        self.lightPosXSpin.config(from_=1, to=self.pilimg.height - 1)
        self.image_id = self.canvas.create_image(
            0,
            0,
            image=self.image,
            anchor='nw'
        )

    def getRenderValues(self):
        "Get values to render the image"
        self.handler = self.handler_original.copy()
        light_source_x = self.lightPosXSpin.get()
        light_source_y = self.lightPosYSpin.get()
        light_source_z = self.lightRadiusSpin.get()
        # diffusion_gain = self.diffGainSpin.get()
        light_source_x = int(light_source_x)
        light_source_y = int(light_source_y)
        light_source_z = float(light_source_z)
        lsource = LightSource(x=light_source_x,
                              y=light_source_y,
                              z=light_source_z)
        ambient_term = float(self.ambientTermSpin.get())
        light_intensity = float(self.lightIntensitySpin.get())
        image = self.handler.shade_image(light_source=lsource,
                                         ambient_term=ambient_term,
                                         light_intensity=light_intensity)
        return image

    def renderImage(self):
        "render image with controller values"
        newimarr = self.getRenderValues()
        self.loadImage2Canvas(newimarr)

    def resetRendering(self):
        "reset rendering values"
        self.loadImage2Canvas(self.handler_original.image)

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
