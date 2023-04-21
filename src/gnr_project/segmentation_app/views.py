import mpld3
from django.http import HttpResponse
import cv2
import numpy as np
from django.shortcuts import render
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy.io as sio
from joblib import load
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image


def my_model_segmentation_function(data):

    n, m, num_samples = data.shape
    data_reshaped = np.reshape(data, (n*m, num_samples))

    num_components = 16
    pipeline = Pipeline([('scaling', StandardScaler()), ('lda', LDA(n_components=num_components))])
    pipeline = load('lda_pipeline.joblib')
    
    data_lda = pipeline.predict(data_reshaped)
        
    # Load the model from the .h5 file
    model = load_model('lda_model.h5')

    # Reshape the cluster labels and display the segmented image
    stacked_array = np.zeros((data_lda.shape[0], 16))

    # Stack the image 17 times along the last axis
    for i in range(16):
        stacked_array[:, i] = data_lda

    data_lda = stacked_array

    cluster_labels_onehot = model.predict(data_lda)

    preds = np.reshape(cluster_labels_onehot, (n,m, 17))

    smoothed_preds = cv2.GaussianBlur(preds, (0, 0), 1.2)
    cluster_labels_reshaped = np.argmax(smoothed_preds, axis=2)

    return cluster_labels_reshaped

def segment_image(request):
    if request.method == 'POST':
        matfile = request.FILES['matfile']
        mat = sio.loadmat(matfile)

        data = mat['indian_pines']
        # Perform segmentation using your ML model
        segmented_img = my_model_segmentation_function(data)

        # Convert the Matplotlib plot to D3.js code
        fig, ax = plt.subplots()
        ax.imshow(segmented_img, cmap='gray')
        ax.axis('off')
        plot_html = mpld3.fig_to_html(fig)

        plt.show()

        # Render the HTML page with the embedded plot
        return render(request, 'segmentation_form.html', {'plot_html': plot_html})
    
    return render(request, 'segmentation_form.html')