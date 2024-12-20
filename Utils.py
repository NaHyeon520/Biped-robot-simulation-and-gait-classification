import matplotlib.pyplot as plt

def plotTrainingProgress(training_history, title, _ylim=None):
    history_dict = training_history.history
    losses = history_dict["loss"]
    val_losses = history_dict["val_loss"]

    plt.plot(losses, label='loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    if _ylim is not None:
        plt.ylim(_ylim)
    plt.title(title)
    plt.show()