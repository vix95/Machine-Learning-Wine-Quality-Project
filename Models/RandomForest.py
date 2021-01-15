from supportive import timing
from Models.Model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Random Forest"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None
        self.data_preprocessing_standard_scaler()

    @timing
    def calculate(self):
        rfc = RandomForestClassifier(n_estimators=20)
        rfc.fit(Model.X_train_ss, Model.y_train)
        prediction_rfc = rfc.predict(Model.X_test_ss)

        self.y_predicted = prediction_rfc
        self.accuracy = rfc.score(Model.X_test_ss, Model.y_test)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)
