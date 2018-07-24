"""
    Reference:
    http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/
    https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
"""

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import os
import PyPDF2
import numpy as np
import cv2
from math import *
import time
from skimage.feature import hog
from skimage import exposure
from pdf2image import convert_from_path, convert_from_bytes
from flask import Flask, render_template
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# app.run(debug=True)



class Interface:

    def __init__(self, tk):
        self.tk = tk

        self.WINDOW_WIDTH = self.tk.winfo_screenwidth()
        self.WINDOW_HEIGHT = self.tk.winfo_screenheight()

        self.configWindow()
        self.configComponents()

    def configWindow(self):
        self.tk.configure(background='dark orange')

        w = 550
        h = 300

        geometry = self.get_center(w, h)
        self.tk.geometry('{}x{}+{}+{}'.format(*geometry))
        self.tk.title('Eye Tracking System')

    def configComponents(self):

        self.numerosPags = StringVar()
        self.nombreImagen = StringVar()
        self.message = StringVar()

        # self.nombreImagen.set("images/img1.jpg")
        self.nombreImagen.set("images/img2.jpg")

        Button(self.tk, text="Select PIC", command=self.seleccionarImagen).grid(row=0, column=1, padx=20, pady=10)
        Button(self.tk, text="Select PDF", command=self.seleccionarPdfs).grid(row=1, column=1, padx=20, pady=10)
        Button(self.tk, text="Buscar Imagen",command=self.algoritmoBuscador).grid(row=5, column=1, columnspan=2, pady=10)
        Button(self.tk, text="Buscar Imagen (Match Template)",command=self.matchTemplate).grid(row=6, column=1, columnspan=2, pady=0)


        Entry(self.tk, textvariable=self.nombreImagen, width=40).grid(row=0, column=2, pady=10)
        self.text_pdf = Text(self.tk, height=5, width=52)
        self.text_pdf.grid(row=1, column=2, rowspan=2)


        Label(self.tk, text="Pag(s) ubicadas", bg="orange").grid(row=7, column=1, padx=20, pady=10)
        Entry(self.tk, textvariable=self.numerosPags, width=40).grid(row=7, column=2, columnspan=2, padx=2, pady=10)


        Label(self.tk, textvariable=self.message, bg="orange").grid(row=8, column=1, columnspan=2)

        self.text_pdf.insert(END, "PDF/pdf.pdf")

    def seleccionarImagen(self):
        filepath =  filedialog.askopenfilename(initialdir = "/Users/Madepozo/ANTHONY/Repo/NN/PYTHON/ANALITICA/images",
                                              title = "Selecciona la imagen",
                                              filetypes = (("jpg files","*.jpg"),
                                                           ("all files","*.*")))
        if len(filepath)!=0:
            nombreImg = filepath.split("/")[-1]
            self.nombreImagen.set(filepath)
            messagebox.showinfo("Mensaje","Haz seleccionado la imagen " + nombreImg)
        else:
            messagebox.showinfo("Mensaje","Por favor escoge una imagen")

    def seleccionarPdfs(self):
        self.text_pdf.delete(1.0, END)

        selectedFolder =  filedialog.askdirectory(initialdir = "/Users/Madepozo/ANTHONY/Repo/NN/PYTHON/ANALITICA/PDF",
                                                     title = "Selecciona la carpeta")
        listaPdfs = []
        listaPrint = []

        if len(selectedFolder)!=0:
            messagebox.showinfo("Mensaje","Haz seleccionado la carpeta " + selectedFolder)
            for f in os.listdir(selectedFolder):
                if f.endswith(".pdf"):
                    listaPdfs.append(f)
                    nombre = f.split("/")[-1]
                    self.text_pdf.insert(END, selectedFolder + "/" + nombre+"\n")

            if len(listaPdfs)==0:
                self.text_pdf.insert(END, "No hay PDF's")
        else:
            messagebox.showinfo("Mensaje","Por favor escoge una carpeta")

    def algoritmoBuscador(self):
        self.message.set("RUNNING...")
        img_path = self.nombreImagen.get()
        pdf_paths = self.text_pdf.get(1.0, END).split("\n")

        hog = HOG(img_path, pdf_paths, 9, (16,16), (1,1), self)
        hog.run()
        hog.show_final_image()

    def matchTemplate(self):
        self.message.set("RUNNING...")
        img_path = self.nombreImagen.get()
        pdf_paths = self.text_pdf.get(1.0, END).split("\n")[:-1]
        print(img_path)
        print(pdf_paths)
        matchTemplate = MatchTemplate(img_path, pdf_paths, 0.63, self)
        matchTemplate.run()


    def get_center(self, w, h):
        padding_left = round((self.WINDOW_WIDTH - w) / 2)
        padding_up = round((self.WINDOW_HEIGHT - h) / 2)
        return (w, h, padding_left, padding_up)


class ProcessingImageAlgorithm:


    def imread(self, img_path, fx=0.5, fy=0.5):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(image, None, fx=fx, fy=fy)

    def pdf_to_image(self, pdf_path, fx=0.5, fy=0.5):
        PDF = convert_from_path(pdf_path)
        images = []
        for (index, image) in enumerate(PDF):
            image_show = np.array(image)
            image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
            image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image_show, None, fx=fx, fy=fy)
            image = image[30:-20,60:-20]
            images.append(image)
        return np.array(images)

    def show_img_by_path(self, img_path):
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        while True:
            cv2.imshow("Image", gray)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    def show_img(self, img):
        while True:
            cv2.imshow("Image", img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    def show(self, function):
        while True:
            function()
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


class HOG(ProcessingImageAlgorithm):


    def __init__(self, img_path, pdf_paths, bins, pixels_per_cell, cells_per_block, interface):
        self.img_path = img_path
        self.pdf_paths = pdf_paths
        self.bins = bins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.squares = []
        self.interface = interface

    def run(self):
        image = self.imread(self.img_path, fx=1, fy=1)
        image = cv2.equalizeHist(image)
        ret, image = cv2.threshold(image, thresh=35, maxval=255, type=cv2.THRESH_BINARY)
        # _, image = cv2.threshold(image, 240, 250, cv2.THRESH_BINARY)

        # image2 = self.imread("images/img5.png", fx=0.5, fy=0.5)

        (features, hog_img, hog_shape, proportion) = self.hog_image(image)
        current_index = None
        print("HOG_SHAPE", hog_shape)
        for pdf_path in self.pdf_paths:
            if len(pdf_path) == 0: return

            images = self.pdf_to_image(pdf_path, fx=0.5, fy=0.5)
            for index, pdf_original in enumerate(images):
                scales = [0.5, 1.0, 1.5]
                for scale in scales:
                    if len(self.squares):
                        self.interface.numerosPags.set(current_index + 1)
                        return

                    pdf = cv2.resize(pdf_original, None, fx=scale, fy=scale)
                    pdf = cv2.equalizeHist(pdf)
                    ret, pdf = cv2.threshold(pdf, thresh=25, maxval=255, type=cv2.THRESH_BINARY)

                    self.current_pdf = pdf
                    print("PDF SHAPE AFTER RESIZE", pdf.shape)

                    fd1, fd1_hog = hog(pdf, orientations=self.bins, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, visualize=True, feature_vector=False)
                    fd1_h, fd1_w = fd1.shape[0:2]

                    (x, y) = (0,0,)
                    step = 4

                    while True:
                        copy = np.array(pdf)
                        window =  fd1[y:y + hog_shape[0], x:x + hog_shape[1]]
                        # print(x, x + hog_shape[0])
                        error = (window.ravel() - features).sum()**2
                        print(abs(error))
                        # print(error, error == 0, type(error), type(0.0))

                        if round(error, 13) == 0.0:
                            current_index = index
                            self.squares.append((x*proportion[0], y*proportion[1], (x+hog_shape[0])*proportion[0], (y+hog_shape[1])*proportion[1], int(error)))
                        for (x1, y1, w1, h1, error) in self.squares:
                            cv2.rectangle(copy,(x1,y1),(w1,h1),(0,0,0),1)

                        cv2.rectangle(copy,(x*proportion[0], y*proportion[1]),((x+hog_shape[1])*proportion[0], (y+hog_shape[0])*proportion[1]),(0, 0, 0),1)
                        cv2.imshow("pdf", copy)
                        cv2.moveWindow("pdf", 100, 100)

                        cv2.imshow("image", image)
                        cv2.moveWindow("image", 800, 100)

                        cv2.imshow("image_hog", hog_img)
                        cv2.moveWindow("image_hog", 800, 500)

                        x += step
                        if x + hog_shape[1] > fd1_w:
                            x = 0
                            y += step
                            if y + hog_shape[0] > fd1_h:
                                break;

                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                            break

    def maxBox(self):
        print(self.squares)
        np_squares = np.array(self.squares)
        try:
            x1 = np_squares[:,0].min()
            y1 = np_squares[:,1].min()
            w1 = np_squares[:,2].max()
            h1 = np_squares[:,3].max()
            return (x1, y1, w1, h1)
        except Exception as e:
            return (0,0,0,0)



    def hog_image(self, image):
        fd1, img = hog(image, orientations=self.bins, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, visualize=True, feature_vector=False)
        # print(fd1)
        # img = exposure.rescale_intensity(img, in_range=(0, 10))
        # histogram = np.sum(fd1.reshape(-1,self.bins), axis=0)

        return (fd1.ravel(), img, fd1.shape, (int(image.shape[1] / fd1.shape[1]), int(image.shape[0] / fd1.shape[0])))

    def show_final_image(self):
        final = self.current_pdf
        (x, y, w, h) = self.maxBox()
        cv2.rectangle(final,(x,y),(w,h),(0,0,0),1)
        while True:
            cv2.imshow("FINAL", final)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break




class MatchTemplate(ProcessingImageAlgorithm):

    def __init__(self, img_path, pdf_paths, threshold, interface):
        self.img_path = img_path
        self.pdf_paths = pdf_paths
        # self.threshold = threshold
        self.threshold = threshold
        self.interface = interface
        self.found = False


    def run(self):
        img_gray = self.imread(self.img_path, fx=1, fy=1)
        self.img_gray = img_gray
        w, h = img_gray.shape[::-1]

        for pdf_path in self.pdf_paths:
            if len(pdf_path) == 0: return
            pdf_images = self.pdf_to_image(pdf_path, fx=0.5, fy=0.5)
            for index, pdf_original in enumerate(pdf_images):
                scales = [0.5, 1.0, 1.5]
                for scale in scales:
                    if self.found : break
                    pdf = cv2.resize(pdf_original, None, fx=scale, fy=scale)

                    self.current_pdf = pdf
                    cv2.imshow('current_pdf', self.current_pdf)
                    cv2.waitKey(100)

                    res = cv2.matchTemplate(img_gray,pdf,cv2.TM_CCOEFF_NORMED)
                    loc = np.where( res >= self.threshold)

                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(self.current_pdf, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
                        self.found = True

        cv2.destroyAllWindows()

        if not self.found:
            self.interface.numerosPags.set("NO SE ENCONTRARON COINCIDENCIAS")
            return

        while True:
            cv2.imshow('Sought',self.img_gray)
            cv2.imshow('Detected',self.current_pdf)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                camera.release()
                break





def main():
    root = Tk()
    interface = Interface(root)
    mainloop()



if __name__=="__main__":
    main()
