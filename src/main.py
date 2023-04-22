import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from segment import Segment
import scipy.io
import numpy as np
from matplotlib import cm
from utils import *
from matplotlib.figure import Figure

class ImageSegmentationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Segmentation")

        # Create the image box
        self.image_box = tk.Label(master)
        # self.image_box = tk.Label(master, width=100, height=20)
        self.image_box.pack()

        # Create the option menu for algorithm selection
        self.algorithm_var = tk.StringVar(master)
        self.algorithm_var.set("LDA")  # default value
        algorithm_options = ["PCA", "LDA"]
        self.algorithm_menu = tk.OptionMenu(master, self.algorithm_var, *algorithm_options)
        self.algorithm_menu.pack()

        self.classification_algorithm_var = tk.StringVar(master)
        self.classification_algorithm_var.set("ANN")  # default value
        classification_algorithm_options = ["SVM", "ANN"]
        self.classification_algorithm_menu = tk.OptionMenu(master, self.classification_algorithm_var, *classification_algorithm_options)
        self.classification_algorithm_menu.pack()

        # Create the browse button
        self.browse_button = tk.Button(master, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        # Create the process button
        self.process_button = tk.Button(master, text="Process", command=self.segment_image)
        self.process_button.pack()

        # Create the checkbox for gauss variable
        self.gauss_var = tk.BooleanVar()
        self.gauss_checkbox = tk.Checkbutton(master, text="Use Gaussian smoothing", variable=self.gauss_var)
        self.gauss_checkbox.pack()

    def browse_image(self):
        # Open a file dialog to select a mat file
        file_path = filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")])
        if file_path:
            # Load the mat file as a numpy array
            self.mat_data = scipy.io.loadmat(file_path)
            self.mat_data = self.mat_data["indian_pines"]

            img_path  = save_fig(self.mat_data[:,:,1], name='indian_pines.png', dir='../dataset/Indian Pines/', keep_axis=False)
            
            self.image = Image.open(img_path)
            self.image = ImageOps.fit(self.image, (500, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)

            # Update the image box with the selected image
            self.image_box.config(image=self.photo)

    def segment_image(self):
        # Get the selected algorithm and classification algorithm
        selected_algorithm = self.algorithm_var.get()
        selected_classification_algorithm = self.classification_algorithm_var.get()
        
        # Get the value of the gauss checkbox
        gauss = self.gauss_var.get()

        # Call the segmentation function with the selected algorithm
        segment = Segment(selected_algorithm, selected_classification_algorithm)
        segmented_image = segment.predict(self.mat_data, gauss=gauss)

        # Update the image box with the segmented image
        if gauss:
            img_path = save_fig(segmented_image, name='lda_nn_with_gauss.png')
        else:
            img_path = save_fig(segmented_image, name='lda_nn_without_gauss.png')

        self.image = Image.open(img_path)
        self.image = ImageOps.fit(self.image, (500, 500), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_box.config(image=self.photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationGUI(root)
    root.mainloop()
