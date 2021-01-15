import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Charts(object):
    def __init__(self, data):
        self.data = data

    def draw_histogram(self):
        fig, axes = plt.subplots(4, 3, figsize=(24, 18))

        sns.histplot(self.data["fixed acidity"], ax=axes[0, 0])
        sns.histplot(self.data["volatile acidity"], ax=axes[0, 1])
        sns.histplot(self.data["citric acid"], ax=axes[0, 2])
        sns.histplot(self.data["residual sugar"], ax=axes[1, 0])
        sns.histplot(self.data["chlorides"], ax=axes[1, 1])
        sns.histplot(self.data["free sulfur dioxide"], ax=axes[1, 2])
        sns.histplot(self.data["total sulfur dioxide"], ax=axes[2, 0])
        sns.histplot(self.data["density"], ax=axes[2, 1])
        sns.histplot(self.data["pH"], ax=axes[2, 2])
        sns.histplot(self.data["sulphates"], ax=axes[3, 0])
        sns.histplot(self.data["alcohol"], ax=axes[3, 1])
        sns.histplot(self.data["quality"], ax=axes[3, 2])

        fig.tight_layout()
        plt.savefig("charts/histogram.png")
        # plt.show()

    def draw_pair_plot(self):
        sns.pairplot(self.data)
        plt.savefig("charts/pair_plot.png")
        # plt.show()

    def draw_heatmap(self):
        plt.figure(figsize=[15, 10])
        sns.heatmap(self.data.corr(), annot=True, cmap="Blues", center=0)
        plt.savefig("charts/correlation.png")
        # plt.show()

    @staticmethod
    def draw_confusion_matrix(name, ds):
        plt.figure(figsize=[6, 3])
        plt.title(name)
        sns.heatmap(ds, annot=True, cmap="Oranges", center=0, fmt="g")
        plt.savefig("charts/cm-" + name.replace(' ', '_') + ".png")
        # plt.show()

    @staticmethod
    def draw_summary_chart(plot_data):
        labels, accuracy_arr, false_negatives, executing_time = list(zip(*plot_data))
        accuracy_arr = [round(elem * 100, 2) for elem in accuracy_arr]

        index = np.arange(len(labels))
        bar_width = 0.4

        fig, ax = plt.subplots(figsize=(16, 12))
        r1 = ax.bar(index - bar_width / 2, accuracy_arr, bar_width, label="Accuracy")
        r2 = ax.bar(index + bar_width / 2, false_negatives, bar_width, label="False negatives", color="red")

        ax.set_xticks(index)
        ax.set_xticklabels(labels)
        ax.legend(loc="lower right")

        # accuracy
        for rect, label in zip(r1, labels):
            height = rect.get_height()
            ax.annotate("{}{}".format(height, "%"),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom")

        # false negative
        for rect, label in zip(r2, labels):
            height = rect.get_height()
            ax.annotate("{}".format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom")

        plt.xticks(rotation=45)
        fig.tight_layout()
        plt.savefig("charts/summary_accuracy.png")
        # plt.show()

    @staticmethod
    def draw_executing_times_chart(plot_data):
        labels, accuracy_arr, false_negatives, executing_time = list(zip(*plot_data))
        executing_time_arr = [round(elem / 1000.0, 2) for elem in executing_time]

        index = np.arange(len(labels))
        bar_width = 0.4

        fig, ax = plt.subplots(figsize=(16, 12))
        r = ax.bar(index, executing_time_arr, bar_width, label="Executing times", color="green")

        ax.set_xticks(index)
        ax.set_xticklabels(labels)
        ax.legend(loc="upper left")

        # executing times
        for rect, label in zip(r, labels):
            height = rect.get_height()
            ax.annotate("{}{}".format(height, "s"),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom")

        plt.xticks(rotation=45)
        fig.tight_layout()
        plt.savefig("charts/executing_times.png")
        # plt.show()
