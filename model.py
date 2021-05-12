from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras import Model
import numpy as np
import matplotlib.pyplot as plt

class TicTacToeModel(Model):

    def __init__(self, numberOfInputs, numberOfOutputs, epochs, batchSize):
        super(TicTacToeModel, self).__init__()
        self.epochs = epochs
        self.batchSize = batchSize
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(numberOfInputs, )))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(numberOfOutputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    def train(self, dataset):
        input = []
        output = []
        for data in dataset:
            input.append(data[1])
            output.append(data[0])
        X = np.array(input).reshape((-1, self.numberOfInputs))
        y = np.array(output)#to_categorical(output, num_classes=3)
        # Train and test data split
        # boundary = int(0.8 * len(X))
        # X_train = X[:boundary]
        # X_test = X[boundary:]
        # y_train = y[:boundary]
        # y_test = y[boundary:]
        X_train = X
        y_train = y

        X_test = X
        y_test = y

        # X_test = []
        # y_test = []
        self.batch_size = len(X)

        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batchSize)
        # self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batchSize)
        # plt.plot(history.history['accuracy'])
        # plt.show()
    def predict(self, data, index):
        return self.model.predict(np.array(data).reshape(-1, self.numberOfInputs))[0][index]

    def updateWithOne(board_state):
        input = []
        output = []
        for data in dataset:
            input.append(data[1])
            output.append(data[0])
        X = np.array(input).reshape((-1, self.numberOfInputs))
        y = to_categorical(output, num_classes=3)
        self.batch_size = len(X)
        self.model.fit(X, y, epochs = 1, batch_size = self.batch_size)








        