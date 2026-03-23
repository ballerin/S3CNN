import torch
import torchvision.models as models

import time

from SphericalDataset import WeatherDataset
from modules.SO3UNet import SO3UNet
from modules.losses import S2MSELoss

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")
    dataset_size = 1
    lossfnc = S2MSELoss()

    train_folders = [str(year) for year in range(2000,2000+dataset_size)]
    train_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                   folders=train_folders,
                                   train=True,
                                   in_datatype="vector", out_datatype="vector")
    
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = 16,
                                                    shuffle = True, pin_memory=True)
    val_dataset = WeatherDataset(root = "dataset/temp->wind/weekly_coarse_torch/",
                                             folders=["2020","2021"],
                                             train=False,
                                             in_datatype="scalar", out_datatype="vector")
    val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                     batch_size = 16,
                                                     shuffle = False, pin_memory=True)

    model = SO3UNet(layer_style="spins",
                                n_alpha = 360, n_beta=179,
                                resolutions = [179, 150, 100, 50, 100, 150, 179],
                                widths = [1, 4, 32, 128, 256, 48, 16, 1],
                                spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                                nonlinearity = "magnitude",
                                use_batch_norm = False,
                                use_layer_norm = False,
                                )
    model = model.to(DEVICE)

    model.train()
    it=0
    train_time_start = time.time()
    for source, target in train_dataloader:
        outputs = model(source.to(DEVICE))
        loss = lossfnc(outputs, target.to(DEVICE))
        it += 1
    train_time = (time.time() - train_time_start)/it

    model.eval()
    it=0
    eval_time_start = time.time()
    for source, target in val_dataloader:
        with torch.no_grad():
            outputs = model(source.to(DEVICE))
            loss = lossfnc(outputs, target.to(DEVICE))
        it += 1
    eval_time = (time.time() - eval_time_start)/it

    print("Average train time per iteration:", train_time)
    print("Average eval time per iteration:", eval_time)