from typing import Union, List

import wandb


class WeightandBiaises:
    def __init__(self, project_name: str = "my_dl_project", run_id=None, interval_display: int = 10):
        """
        This class enables to send data from training to Weight&Biaises to visualize the
        behaviour of our training.
        :param project_name: name of our project on W&B
        :param run_id: name of our run in our project on W&B. If None, W&B will choose
        a random name for us.
        :param interval_display: this enables to display the mask_debugger with an interval
        of {interval_display} epochs.
        """
        self.interval_display = interval_display
        self.run_id = run_id

        wandb.login()
        wandb.init(id=run_id,
                   project=project_name
                   )
        self.run_id = wandb.run.name

        self.image_list = []

    def log_losses(self, train_loss: float, test_loss: float, epoch: int) -> None:
        """
        Log train and test losses in separate panels.
        :param test_loss:
        :param train_loss: average train loss for the current epoch.
        :param train_loss: average test loss for the current epoch.
        :param epoch: current epoch.
        """
        if epoch % self.interval_display == 0:
            bool_commit = False
        else:
            bool_commit = True

        wandb.log({"Train/Loss": train_loss, "Test/Loss": test_loss}, step=epoch, commit=bool_commit)

    def log_accuracy(self, accuracy: Union[float, List], epoch: int) -> None:
        """
        Log iou accuracy.
        :param accuracy: average iou accuracy for the current epoch.
        :param epoch: current epoch.
        """
        wandb.log({"Test/Accuracy": accuracy}, step=epoch)
