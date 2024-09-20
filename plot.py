from matplotlib import pyplot as plt


def plot_2d(x_list, y_list, title, x_axis_name, y_axis_name):
    plt.plot(x_list, y_list)
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.grid(True)
    plt.xticks(x_list)
    plt.show()


class LearningRateExperiment:
    def __init__(self, lr_value, loss_per_epoch_list) -> None:
        self.lr = lr_value
        self.loss_per_epoch_list = loss_per_epoch_list


class LearningRatesPlot:
    def __init__(self, num_of_epochs) -> None:
        self.experiments_list = []
        self.num_of_epochs = num_of_epochs

    def add_experiment(self, lr_value, loss_per_epoch_list):
        exp = LearningRateExperiment(lr_value, loss_per_epoch_list)
        self.experiments_list.append(exp)

    def show(self):
        plt.title("Learning Rate Effect On Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Revenue Loss In Millions")
        epochs_x_axis = list(range(1,self.num_of_epochs + 1))

        for exp in self.experiments_list:
            cur_plot_label = f"lr={exp.lr}"
            plt.plot(epochs_x_axis, exp.loss_per_epoch_list, label=cur_plot_label)

        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pass
