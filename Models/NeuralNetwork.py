from supportive import timing
from Models.Model import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


class NeuralNetworkModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Neural Network"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None
        self.y_test = None

    @staticmethod
    def model_1():
        model = Sequential()
        model.add(Dense(128, input_dim=11, activation='relu'))
        # model.add(Dense(128, input_dim=3, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # optimizer adam: Adaptive Moment Estimation
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def model_2():
        model = Sequential()
        model.add(Dense(500, input_dim=11, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @timing
    def calculate(self):
        X_train = normalize(Model.X_train, axis=0)
        X_test = normalize(Model.X_test, axis=0)
        y_train = Model.y_train
        y_test = Model.y_test

        y_train = np_utils.to_categorical(y_train, num_classes=2)
        y_test = np_utils.to_categorical(y_test, num_classes=2)

        model = self.model_1()

        # batch_size: qty of samples to update parameters
        # epochs: how many times algorithm will be trained on full data
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=10, verbose=1)
        # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=100, verbose=1)

        def draw_history_chart(plot_data):
            fig, ax = plt.subplots()
            # ax.plot(range(1, 11), plot_data.history['accuracy'], label='Train Accuracy')
            # ax.plot(range(1, 11), plot_data.history['val_accuracy'], label='Validation Accuracy')

            ax.plot(range(1, 11), plot_data.history['accuracy'], label='Train Accuracy')
            ax.plot(range(1, 11), plot_data.history['val_accuracy'], label='Validation Accuracy')

            ax.legend(loc='best')
            ax.set(xlabel='epochs', ylabel='accuracy')
            plt.savefig("charts/neural_network_history.png")
            # plt.show()

        draw_history_chart(history)

        predicted_classes = model.predict(X_test)
        predicted_classes = [[round(x) for x in values] for values in predicted_classes]

        def convert_back(elem):
            most_expected = max(elem)
            if most_expected == elem[0]:
                return 0
            elif most_expected == elem[1]:
                return 1

        self.y_test = list(map(convert_back, y_test))
        self.y_predicted = list(map(convert_back, predicted_classes))
        self.accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted, self.y_test)
