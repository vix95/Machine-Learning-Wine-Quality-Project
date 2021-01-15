from sklearn.neighbors import KNeighborsClassifier
from supportive import timing
from Models.Model import Model


class KNeighborsModel(Model):
    def __init__(self, data, n):
        super().__init__(data)
        self.algorithm_name = "K Neighbors " + str(n)
        self.n = n
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None

    @timing
    def calculate(self):
        knn = KNeighborsClassifier(n_neighbors=self.n)
        knn.fit(Model.X_train, Model.y_train)

        self.y_predicted = knn.predict(Model.X_test)
        self.accuracy = knn.score(Model.X_test, Model.y_test)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)
