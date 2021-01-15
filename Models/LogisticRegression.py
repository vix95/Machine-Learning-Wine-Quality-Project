from supportive import timing
from Models.Model import Model
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


class LogisticRegressionModel(Model):
    def __init__(self, data):
        super().__init__(data)
        self.algorithm_name = "Logistic Regression"
        self.executing_time = None
        self.y_predicted = None
        self.accuracy = None
        self.data_preprocessing_standard_scaler()

    @timing
    def calculate(self):
        logistic_regression = LogisticRegression()
        logistic_regression.fit(Model.X_train_ss, Model.y_train)

        glm = sm.GLM(Model.y_train, (sm.add_constant(Model.X_train_ss)), family=sm.families.Binomial())
        res = glm.fit()
        # print("Summary: \n{}\n".format(res.summary()))

        X_test_sm = sm.add_constant(Model.X_test_ss)
        y_predicted = res.predict(X_test_sm)

        for i in range(y_predicted.size):
            if y_predicted[i] > 0.61:  # threshold calculated via MiniTab
                y_predicted[i] = 1
            else:
                y_predicted[i] = 0

        self.y_predicted = y_predicted
        self.accuracy = logistic_regression.score(Model.X_test_ss, Model.y_test)
        print("Accuracy: ", self.accuracy)

    def show_confusion_matrix(self):
        return super().confusion_matrix(self.y_predicted)
