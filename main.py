from Models.Model import Model
from Models.NaiveBayes import NaiveBayesModel
from Models.DecisionTree import DecisionTreeModel
from Models.LogisticRegression import LogisticRegressionModel
from Models.RandomForest import RandomForestModel
from Models.SupportVectorMachines import SupportVectorMachinesModel, LinearSupportVectorMachinesModel
from Models.KNeighbors import KNeighborsModel
from Models.NeuralNetwork import NeuralNetworkModel
from Charts import Charts
import pandas as pd
import supportive as s
import warnings
warnings.filterwarnings("ignore")

TEST_SIZE = 0.25

if __name__ == '__main__':
    model = Model(pd.read_csv("input/winequality-both.csv"))
    model.split_to_train_and_test(test_percentage=TEST_SIZE)
    charts = Charts(model.data)
    summary = {}  # accuracy, false negative executing time

    model.check_data()
    model.describe_data()
    model.unique_species()
    charts.draw_histogram()
    charts.draw_pair_plot()
    charts.draw_heatmap()

    classifiers = [
        ("Naive Bayes", NaiveBayesModel(data=model.data), True),
        ("Decision Tree", DecisionTreeModel(data=model.data), True),
        ("Logistic Regression", LogisticRegressionModel(data=model.data), True),
        ("Random Forest", RandomForestModel(data=model.data), True),
        ("Support Vector Machines", SupportVectorMachinesModel(data=model.data), True),
        ("Linear Support Vector Machines", LinearSupportVectorMachinesModel(data=model.data), True),
        ("K Neighbors 3", KNeighborsModel(data=model.data, n=3), True),
        ("K Neighbors 5", KNeighborsModel(data=model.data, n=5), True),
        ("K Neighbors 7", KNeighborsModel(data=model.data, n=7), True),
        ("K Neighbors 10", KNeighborsModel(data=model.data, n=10), True),
        ("K Neighbors 20", KNeighborsModel(data=model.data, n=20), True),
        ("K Neighbors 50", KNeighborsModel(data=model.data, n=50), True),
        ("Neural Network", NeuralNetworkModel(data=model.data), True)
    ]

    plot_data = []
    for name, m, active in classifiers:
        if active:
            print("\nExecuting {}".format(name))
            m.calculate()
            cm = m.show_confusion_matrix()
            s.save_executing_time(obj=m)
            false_negative = cm[1][0]

            charts.draw_confusion_matrix(name, cm)
            summary[m.algorithm_name] = [m.accuracy, false_negative, m.executing_time]
            plot_data.append((name, m.accuracy, false_negative, m.executing_time))

    print(summary)
    charts.draw_summary_chart(plot_data=plot_data)
    charts.draw_executing_times_chart(plot_data=plot_data)
