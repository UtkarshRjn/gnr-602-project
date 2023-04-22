import numpy as np
from models import *
from typing import *
from utils import StandardScaler, save_fig
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import custom_object_scope
import cv2
from tqdm import tqdm
import pickle
import os


class Segment:
    def __init__(self, dr_algo: str, classifier_algo: str, data_name: str = None):
        self.dim_red_algo: str = dr_algo.lower()
        self.classifier_algo: str = classifier_algo.lower()
        self.model: MyModel = None
        self.scaler = StandardScaler()
        self.data_name: str = data_name

    def lda(self, X, y, num_components: int):
        # compute class means
        means = []
        for c in np.unique(y):
            means.append(np.mean(X[y == c], axis=0))

        # compute within-class scatter matrix
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for c, mean in zip(np.unique(y), means):
            class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == c]:
                row, mean = row.reshape(X.shape[1], 1), mean.reshape(X.shape[1], 1)
                class_sc_mat += (row - mean).dot((row - mean).T)
            Sw += class_sc_mat

        # compute between-class scatter matrix
        Sb = np.zeros((X.shape[1], X.shape[1]))
        grand_mean = np.mean(X, axis=0).reshape(X.shape[1], 1)
        for c, mean in zip(np.unique(y), means):
            n = X[y == c, :].shape[0]
            mean = mean.reshape(X.shape[1], 1)
            Sb += n * (mean - grand_mean).dot((mean - grand_mean).T)

        # compute eigenvalues and eigenvectors of Sw^-1*Sb
        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        eigen_pairs = [
            (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
        ]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

        # extract the top k eigenvectors as LDA components
        lda_components = np.zeros((X.shape[1], num_components))
        for i in range(num_components):
            lda_components[:, i] = eigen_pairs[i][1]

        # project data onto the LDA components
        X_lda = X.dot(lda_components)

        return X_lda, lda_components

    def pca(self, X, num_components: int):
        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # Calculate the covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top k eigenvectors
        principal_components = sorted_eigenvectors[:, :num_components]

        # Project the data onto the new subspace
        data_pca = np.dot(X_centered, principal_components)

        return data_pca

    def gaussian_filter(self, predictions: np.ndarray, sigma: float = 1.0):

        return cv2.GaussianBlur(predictions, (0, 0), sigma)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        num_components: int,
        num_epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):

        n, m, num_samples = data.shape
        data_reshaped = np.reshape(data, (n * m, num_samples))
        labels_reshaped = np.reshape(labels, (n * m,))

        data_scaled = self.scaler.fit_transform(data_reshaped)

        # Apply the dimensionality reduction algorithm
        if self.dim_red_algo == "pca":
            data_reduced = self.pca(data_scaled, num_components)
        elif self.dim_red_algo == "lda":
            data_reduced, lda_components = self.lda(
                data_scaled, labels.ravel(), num_components
            )

            # create directory recursively if it doesn't exist
            dir_path = os.path.dirname("../models/" + self.data_name + "/lda_model.pkl")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open("../models/" + self.data_name + "/lda_model.pkl", "wb") as f:
                pickle.dump(lda_components, f)
        else:
            return None

        save_fig(
            data_reduced[:, 0].reshape(n, m),
            name="lda_1.png",
            dir="../output/" + self.data_name + "/" + self.dim_red_algo + "/",
            keep_axis=False,
            isLabel=False,
        )
        save_fig(
            data_reduced[:, 1].reshape(n, m),
            name="lda_2.png",
            dir="../output/" + self.data_name + "/" + self.dim_red_algo + "/",
            keep_axis=False,
            isLabel=False,
        )
        save_fig(
            data_reduced[:, 2].reshape(n, m),
            name="lda_3.png",
            dir="../output/" + self.data_name + "/" + self.dim_red_algo + "/",
            keep_axis=False,
            isLabel=False,
        )

        num_labels = len(np.unique(labels_reshaped))
        labels_onehot = np.eye(num_labels)[labels_reshaped.reshape(-1)]

        # self.model = MyModel(data_reduced.shape[1], num_labels)
        self.model = Sequential()
        self.model.add(Dense(64, activation="relu", input_dim=data_reduced.shape[1]))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(num_labels, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            data_reduced, labels_onehot, test_size=0.3, random_state=42
        )

        # Train the neural network on the training set
        self.model.fit(
            X_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

        # Evaluate the neural network on the test set
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test accuracy:", accuracy)

        self.model.save(
            "../models/"
            + self.data_name
            + "/"
            + self.dim_red_algo
            + "_"
            + self.classifier_algo
            + ".h5"
        )

        return loss, accuracy

    def predict(self, image: np.ndarray, gauss: bool = True, num_components: int = 16):

        n, m, num_samples = image.shape
        image_reshaped = np.reshape(image, (n * m, num_samples))
        scaled_image = self.scaler.fit_transform(image_reshaped)

        # with custom_object_scope({'MyModel': MyModel}):
        try:
            self.model = load_model(
                "../models/"
                + self.data_name
                + "/"
                + self.dim_red_algo
                + "_"
                + self.classifier_algo
                + ".h5"
            )
        except:
            return "Please train the model first."

        if self.dim_red_algo == "pca":
            reduced_image = self.pca(scaled_image, num_components)
        elif self.dim_red_algo == "lda":
            lda_components = pickle.load(
                open("../models/" + self.data_name + "/lda_model.pkl", "rb")
            )
            reduced_image = scaled_image.dot(lda_components)
        else:
            return None

        # reduced_image = np.tile(reduced_image, (1, num_components))
        preds_labels_onehot = self.model.predict(reduced_image)
        preds = np.reshape(preds_labels_onehot, (n, m, 17))

        if gauss:
            preds = self.gaussian_filter(preds)

        labels = np.argmax(preds, axis=2)

        return labels
