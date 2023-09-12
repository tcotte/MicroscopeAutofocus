from matplotlib import pyplot as plt

if __name__ == "__main__":
    test_losses = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    with open('results_training.txt') as f:
        lines = f.readlines()
    f.close()
    lines = [line.replace("\n", "") for line in lines]

    for line in lines:
        print(line.split(" "))
        train_losses.append(float(line.split(" ")[3]))
        test_losses.append(float(line.split(" ")[6]))
        train_accuracies.append(float(line.split(" ")[9]))
        test_accuracies.append(float(line.split(" ")[12]))

    t = list(range(len(train_losses)))
    _, axs = plt.subplots(1, 2, layout='constrained')
    axs[0].plot(t, train_accuracies, 'b', label="train_accuracies")  # plotting t, a separately
    axs[0].plot(t, test_accuracies, 'r', label="test_accuracies")
    axs[0].set_title("MSE during training")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()

    axs[1].plot(t, train_losses, 'b', label="train_losses")  # plotting t, a separately
    axs[1].plot(t, test_losses, 'r', label="test_losses")
    axs[1].set_title("Losses during training")
    axs[1].set_ylabel("L1 smooth loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()

    plt.show()
