from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from skimage import io
import numpy as np

from models import *
from visualize import show_tensor_images



criterion = nn.BCEWithLogitsLoss()

#TRAIN PARAMS
n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 2e-4
initial_shape = 512
target_shape = 373
device = "cuda"

# Train Data
volumes = torch.Tensor(io.imread('train-volume.tif'))[:, None, :, :] / 255
labels = torch.Tensor(io.imread('train-labels.tif', plugin="tifffile"))[:, None, :, :] / 255
labels = crop(labels, torch.Size([len(labels), label_dim, target_shape, target_shape]))
dataset = torch.utils.data.TensorDataset(volumes, labels)


def train():
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            # display
            if cur_step % display_step == 0:
                print(f"Epoch: {epoch}  Step: {cur_step}  U-Net loss: {unet_loss.item()}")
                show_tensor_images(
                    crop(real, torch.Size([len(real), input_dim, target_shape, target_shape])),
                    size=(input_dim, target_shape, target_shape)
                )
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))

            cur_step += 1


if __name__ == "__main__":
    train()