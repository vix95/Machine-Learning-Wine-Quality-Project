from supportive import timing
from Models.Model import Model
from sklearn.svm import SVC, LinearSVC


class SupportVectorMachinesModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Support Vector Machines"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None

    @timing
    def calculate(self):
        svc = SVC()
        svc.fit(Model.X_train, Model.y_train)

        self.y_predicted = svc.predict(Model.X_test)
        self.accuracy = svc.score(Model.X_test, Model.y_test)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)


class LinearSupportVectorMachinesModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Linear Support Vector Machines"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None

    @timing
    def calculate(self):
        svc = LinearSVC()
        svc.fit(Model.X_train, Model.y_train)

        self.y_predicted = svc.predict(Model.X_test)
        self.accuracy = svc.score(Model.X_test, Model.y_test)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)
