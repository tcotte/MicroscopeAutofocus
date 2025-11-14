import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import albumentations as A
from autofocus.autofocus_dataset import DifferenceAFDataset
from autofocus.models import MobileNetV3_Regressor
from torch.utils.data import DataLoader

def rmse(y_hat, y_ground_truth):
    mse = np.square(np.subtract(np.array(y_ground_truth), np.array(y_hat))).mean()
    return np.sqrt(mse)

if __name__ == '__main__':
    image_folder = r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_09_25_2025\X\test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    num_workers = 8

    model_path = 'autofocus/models/90th_epoch_chkpt.pt'
    model_checkpoint = torch.load(model_path)
    model = MobileNetV3_Regressor()
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.to(device)
    model = model.eval()

    df_test = pd.read_excel('autofocus/data/diff_test.xlsx')
    for position in df_test['xy_position'].unique().tolist():
        sub_df = df_test[df_test['xy_position'] == position]

        test_transform = A.Compose([
            A.Normalize(),
            A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
            A.pytorch.transforms.ToTensorV2()
        ])

        sub_dataset = DifferenceAFDataset(df=sub_df, image_folder=image_folder, transform=test_transform)

        test_dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

        list_predictions = []
        list_targets = []

        for batch in test_dataloader:
            images, labels = batch["X"].float(), batch["y"]

            images = images.to(device)

            with torch.no_grad():
                res = model(images)

            targets = labels.cpu().numpy().tolist()
            list_targets.extend(targets)

            predictions = np.squeeze(res.cpu().numpy()).tolist()
            if isinstance(predictions, float):
                list_predictions.append(predictions)
            else:
                list_predictions.extend(predictions)

        error = rmse(np.array(list_predictions), np.array(list_targets))

        fig = plt.figure(figsize=(20, 20))
        plt.title(f"RMSE {error:.2f}")
        plt.suptitle(f'XY position: {position}')
        plt.plot(np.array(list_targets), np.array(list_targets), color='blue', linewidth=1)
        plt.scatter(np.array(list_targets), np.array(list_predictions), c='r', )
        plt.xlabel('Z distance_af from focus (µm)')
        plt.ylabel('Predicted Z distance_af from focus (µm)')
        plt.savefig(f"autofocus/output/{position}.jpg")
        plt.show()
