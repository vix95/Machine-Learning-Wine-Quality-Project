from supportive import timing
from sklearn import metrics
from Models.Model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Decision Tree"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None
        self.data_preprocessing_standard_scaler()

    @timing
    def calculate(self):
        decision_tree = DecisionTreeClassifier(max_depth=10)
        decision_tree.fit(Model.X_train_ss, Model.y_train)
        self.y_predicted = decision_tree.predict(Model.X_test_ss)
        self.accuracy = metrics.accuracy_score(Model.y_test, self.y_predicted)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)
