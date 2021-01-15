from supportive import timing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from Models.Model import Model


class NaiveBayesModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Naive Bayes"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None
        self.data_preprocessing_pca()

    @timing
    def calculate(self):
        gnb = GaussianNB()
        gnb.fit(Model.X_train_pca, Model.y_train)
        self.y_predicted = gnb.predict(Model.X_test_pca)
        self.accuracy = metrics.accuracy_score(Model.y_test, self.y_predicted)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)

