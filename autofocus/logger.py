from typing import Union, List

import numpy as np
from wandb.wandb_torch import torch

import wandb

from autofocus_model import RegressionMobilenet


class WeightandBiaises:
    def __init__(self, project_name: str = "my_dl_project", run_id=None, interval_display: int = 10, config=None):
        """
        This class enables to send data from training to Weight&Biaises to visualize the
        behaviour of our training.
        :param project_name: name of our project on W&B
        :param run_id: name of our run in our project on W&B. If None, W&B will choose
        a random name for us.
        :param interval_display: this enables to display the mask_debugger with an interval
        of {interval_display} epochs.
        """
        if config is None:
            self.config = {}
        self.interval_display = interval_display
        self.run_id = run_id

        wandb.login()
        self.run = wandb.init(id=run_id, project=project_name, config=self.config)
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

        bool_commit = True
        wandb.log({"Train/Loss": train_loss, "Test/Loss": test_loss}, step=epoch, commit=bool_commit)

    def log_rmse(self, train_mse: float, test_mse: float, epoch: int) -> None:
        """
        Log MSE accuracy.
        :param test_mse:
        :param train_mse:
        :param epoch: current epoch.
        """
        wandb.log({"Train/RMSE": train_mse, "Test/RMSE": test_mse}, step=epoch)

    def save_model(self, model_name: str, model: RegressionMobilenet) -> None:
        final_model_dir = "last_model"

        trained_model_artifact = wandb.Artifact(
                    model_name, type="model",
                    description="train autofocus regression Mobilenetv3",
                    metadata=dict(self.config))

        model.save(final_model_dir)
        trained_model_artifact.add_dir(final_model_dir)
        self.run.log_artifact(trained_model_artifact)

    @staticmethod
    def tensor2image(x: torch.FloatTensor) -> np.array:
        """
        Transform tensor to image numpy array.
        :param x: image float tensor [1, C, H, W]
        :return : image numpy array [H, W, C]
        """
        a = x.squeeze()
        a = a.permute(1, 2, 0)
        return a.detach().cpu().numpy()

    def log_table(self, predictions, tensor_images, labels, e: int):
        """
        Send the wandb images to W&B at the epoch's end.
        """
        if e % self.interval_display == 0:
            tbl = wandb.Table(columns=["image", "predictions", "ground-truth"])

            images = [self.tensor2image(x) for x in tensor_images]

            [tbl.add_data(wandb.Image(image), pred, label) for image, pred, label in zip(images, predictions, labels)]

            wandb.log({"Predicted_autofocus": tbl})
