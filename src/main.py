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
import os


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
        self.algorithm_menu = tk.OptionMenu(
            master, self.algorithm_var, *algorithm_options
        )
        self.algorithm_menu.pack()

        self.classification_algorithm_var = tk.StringVar(master)
        self.classification_algorithm_var.set("ANN")  # default value
        classification_algorithm_options = ["SVM", "ANN"]
        self.classification_algorithm_menu = tk.OptionMenu(
            master, self.classification_algorithm_var, *classification_algorithm_options
        )
        self.classification_algorithm_menu.pack()

        # Create the browse button
        self.browse_button = tk.Button(
            master, text="Browse Image", command=self.browse_image
        )
        self.browse_button.pack()

        # Create the browse button
        self.browse_button = tk.Button(
            master, text="Browse Labels", command=self.browse_labels
        )
        self.browse_button.pack()

        # Create the Predict button
        self.process_button = tk.Button(
            master, text="Predict Only", command=lambda: self.segment_image(train=False)
        )
        self.process_button.pack()

        # Create the Train and Predict button
        self.process_button = tk.Button(
            master,
            text="Train and Predict",
            command=lambda: self.segment_image(train=True),
        )
        self.process_button.pack()

        # Create the checkbox for gauss variable
        self.gauss_var = tk.BooleanVar()
        self.gauss_checkbox = tk.Checkbutton(
            master, text="Use Gaussian smoothing", variable=self.gauss_var
        )
        self.gauss_checkbox.pack()

        self.test_accuracy_label = tk.Label(root, text="Test accuracy: ")
        self.test_accuracy_label.pack()
        self.pred_accuracy_label = tk.Label(root, text="Prediction accuracy: ")
        self.pred_accuracy_label.pack()

    # Define functions to update the labels with the accuracy values
    def update_test_accuracy(self, value):
        self.test_accuracy_label.config(text="Test accuracy: {:.2f}%".format(value*100))

    def update_pred_accuracy(self, value):
        self.pred_accuracy_label.config(text="Prediction accuracy: {:.2f}%".format(value*100))


    def browse_image(self):
        # Open a file dialog to select a mat file
        file_path = filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")])
        if file_path:
            # Load the mat file as a numpy array
            self.data_name = os.path.splitext(os.path.basename(file_path))[0].lower()

            self.mat_data = scipy.io.loadmat(file_path)
            self.mat_data = self.mat_data[self.data_name]

            img_path = save_fig(
                self.mat_data[:, :, 1],
                name=self.data_name + ".png",
                dir=os.path.dirname(file_path),
                keep_axis=False,
            )

            self.image = Image.open(img_path)
            self.image = ImageOps.fit(self.image, (500, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)

            # Update the image box with the selected image
            self.image_box.config(image=self.photo)

    def browse_labels(self):

        # Open a file dialog to select a mat file
        file_path = filedialog.askopenfilename(filetypes=[("Matlab files", "*.mat")])
        if file_path:
            # Load the mat file as a numpy array
            self.label_name = os.path.splitext(os.path.basename(file_path))[0].lower()

            try:
                assert self.label_name == self.data_name + "_gt"
            except AssertionError:
                tk.messagebox.showerror(
                    "Error", "The label file does not match the image file!"
                )
                return

            self.mat_label = scipy.io.loadmat(file_path)
            self.mat_label = self.mat_label[self.label_name]

            img_path = save_fig(
                self.mat_label,
                name=self.label_name + ".png",
                dir=os.path.dirname(file_path),
                keep_axis=False,
            )

            self.image = Image.open(img_path)
            self.image = ImageOps.fit(self.image, (500, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)

            # Update the image box with the selected image
            self.image_box.config(image=self.photo)

    def segment_image(self, train=False):
        # Get the selected algorithm and classification algorithm

        if not train and not hasattr(self, "mat_data"):
            tk.messagebox.showerror("Error", "Please select an image file first!")
            return

        if train and not hasattr(self, "mat_label"):
            tk.messagebox.showerror("Error", "Please select a label file first!")
            return

        selected_algorithm = self.algorithm_var.get()
        selected_classification_algorithm = self.classification_algorithm_var.get()

        # Get the value of the gauss checkbox
        gauss = self.gauss_var.get()

        # Call the segmentation function with the selected algorithm
        segment = Segment(
            selected_algorithm, selected_classification_algorithm, self.data_name
        )

        if train:
            _, test_accuracy = segment.fit(
                self.mat_data,
                self.mat_label,
                num_components=len(np.unique(self.mat_label)) - 1,
            )
            self.update_test_accuracy(test_accuracy)

        segmented_image = segment.predict(self.mat_data, gauss=gauss)

        if type(segmented_image) == str:
            tk.messagebox.showerror("Error", segmented_image)
            return

        pred_accuracy = np.mean(segmented_image == self.mat_label)
        self.update_pred_accuracy(pred_accuracy)

        # Update the image box with the segmented image
        if gauss:
            img_path = save_fig(
                segmented_image,
                name=selected_algorithm.lower()
                + "_"
                + selected_classification_algorithm.lower()
                + "_with_gauss.png",
                dir="../output/" + self.data_name + "/" + selected_algorithm.lower() + "/",
            )
        else:
            img_path = save_fig(
                segmented_image,
                name=selected_algorithm.lower()
                + "_"
                + selected_classification_algorithm.lower()
                + "_without_gauss.png",
                dir="../output/" + self.data_name + "/" + selected_algorithm.lower() + "/",
            )

        self.image = Image.open(img_path)
        self.image = ImageOps.fit(self.image, (500, 500), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_box.config(image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationGUI(root)
    root.mainloop()
