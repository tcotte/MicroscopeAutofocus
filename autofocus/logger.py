import os
from typing import Union, List

import numpy as np
import torch

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
        else:
            self.config = config

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

    def log_lr(self, lr: float, epoch: int) -> None:
        """
        Log train and test losses in separate panels.
        :param test_loss:
        :param train_loss: average train loss for the current epoch.
        :param train_loss: average test loss for the current epoch.
        :param epoch: current epoch.
        """
        wandb.log({"LR": lr}, step=epoch)

    def log_mae(self, train_mse: float, test_mse: float, epoch: int) -> None:
        """
        Log MSE accuracy.
        :param test_mse:
        :param train_mse:
        :param epoch: current epoch.
        """
        wandb.log({"Train/RMSE": train_mse, "Test/RMSE": test_mse}, step=epoch)

    def save_model(self, model_name: str, model: torch.nn.Module) -> None:
        # final_model_dir = "last_model"
        #
        # trained_model_artifact = wandb.Artifact(
        #             model_name, type="model",
        #             description="train autofocus regression Mobilenetv3",
        #             metadata=dict(self.config))
        #
        # torch.save(model.state_dict(), final_model_dir)
        # trained_model_artifact.add_dir(final_model_dir)
        # self.run.log_artifact(trained_model_artifact)
        torch.save(model, os.path.join(wandb.run.dir, model_name))

        # transfer to W&B
        artifact = wandb.Artifact(name="last", type="model")
        artifact.add_file(local_path=wandb.run.dir, name="last")
        wandb.run.log_artifact(artifact)

    @staticmethod
    def tensor2image(x) -> np.array:
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

    @staticmethod
    def save_checkpoint(epoch: int, model: torch.nn, optimizer: torch.optim, train_loss: float,
                        test_loss: float) -> None:
        """
        Save checkpoint in W&B
        :param epoch: current epoch number
        :param model: current model
        :param optimizer: current optimizer
        :param train_loss: current train loss
        :param test_loss: current test loss
        """
        path_checkpoint_dir = "checkpoint"
        checkpoint_filename = f"checkpoint_{str(epoch)}.pt"
        path_checkpoint = os.path.join(path_checkpoint_dir, checkpoint_filename)
        name = f"{str(epoch)}th_epoch_chkpt"

        # create checkpoint directory
        if not os.path.isdir(path_checkpoint_dir):
            os.mkdir(path_checkpoint_dir)

        # save checkpoint file
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, path_checkpoint)

        # transfer to W&B
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(local_path=path_checkpoint, name=name)
        wandb.run.log_artifact(artifact)
