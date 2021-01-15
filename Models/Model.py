from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


class Model(object):
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    X_train_ss = None
    X_test_ss = None

    X_train_pca = None
    X_test_pca = None

    def __init__(self, data):
        self.data = data

        # accuracy is poor when it provides too many points, reduced to 2 points
        self.data.loc[data["quality"] == 1, "quality"] = 0
        self.data.loc[data["quality"] == 2, "quality"] = 0
        self.data.loc[data["quality"] == 3, "quality"] = 0
        self.data.loc[data["quality"] == 4, "quality"] = 0
        self.data.loc[data["quality"] == 5, "quality"] = 0
        self.data.loc[data["quality"] == 6, "quality"] = 1
        self.data.loc[data["quality"] == 7, "quality"] = 1
        self.data.loc[data["quality"] == 8, "quality"] = 1
        self.data.loc[data["quality"] == 9, "quality"] = 1

    def check_data(self):
        print("Check null data:\n{}\n".format(self.data.isnull().sum()))

    def describe_data(self):
        with pd.option_context("display.max_columns", 13):
            print("Describing the data:\n", self.data.describe(include="all"))

    def unique_species(self):
        print("Unique quality: {}".format(self.data["quality"].unique()))

    def confusion_matrix(self, y_predicted, y_test=None):
        if y_test is None:
            y_test = self.y_test
        # confusion_matrix(self.y_test, y_predicted)
        # pd.crosstab(self.y_test, y_predicted, rownames=["True"], colnames=["Predicted"], margins=True)
        # print(classification_report(self.y_test, y_predicted, labels=np.unique(y_predicted)))
        conf_matrix = confusion_matrix(y_test, y_predicted, labels=[0, 1])
        print("Confusion matrix:\n{}".format(conf_matrix))
        return conf_matrix

    def combine_data(self):
        X1 = self.data["total sulfur dioxide"] * self.data["free sulfur dioxide"]
        X1_df = pd.DataFrame(X1, columns=["total sulfur dioxide * free sulfur dioxide"])

        X2 = self.data["density"] * self.data["residual sugar"]
        X2_df = pd.DataFrame(X2, columns=["density * residual sugar"])

        X3 = self.data["total sulfur dioxide"] * self.data["residual sugar"]
        X3_df = pd.DataFrame(X3, columns=["total sulfur dioxide * residual sugar"])

        self.data = X1_df.join(X2_df).join(X3_df).join(self.data['quality'])

    def split_to_train_and_test(self, test_percentage=0.2):
        X = self.data.iloc[:, 0:11].values  # default, get all data
        y = self.data.iloc[:, 11].values

        # only correlated data
        # self.combine_data()
        # X = self.data.iloc[:, 0:3].values
        # y = self.data.iloc[:, 3].values

        Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(X, y, test_size=test_percentage)

    @staticmethod
    def data_preprocessing_standard_scaler():
        ss = StandardScaler().fit(Model.X_train)
        Model.X_train_ss = ss.transform(Model.X_train)
        Model.X_test_ss = ss.transform(Model.X_test)

    @staticmethod
    def data_preprocessing_pca():
        pca = PCA()
        Model.X_train_pca = pca.fit_transform(Model.X_train)
        Model.X_test_pca = pca.fit_transform(Model.X_test)
        pca_new = PCA(n_components=8)
        # pca_new = PCA(n_components=3)
        Model.X_train_pca = pca_new.fit_transform(Model.X_train_pca)
        Model.X_test_pca = pca_new.fit_transform(Model.X_test_pca)
