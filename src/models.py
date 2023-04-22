from keras.models import Sequential
from keras.layers import Dense


class MyModel(Sequential):
    def __init__(self, input_dim, num_classes):

        super().__init__()
        self.add(Dense(64, activation="relu", input_dim=input_dim))
        self.add(Dense(32, activation="relu"))
        self.add(Dense(num_classes, activation="softmax"))
        self.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
