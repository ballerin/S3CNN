import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function, schedule

import time

from SphericalDataset import WeatherDataset
from modules.equivariant import SO3UNet
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

    model = SO3UNet(arch="swscnn",
                                n_alpha = 360, n_beta=179,
                                resolutions = [179, 150, 100, 50, 100, 150, 179],
                                widths = [1, 4, 32, 128, 256, 48, 16, 1],
                                spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                                nonlinearity = "magnitude",
                                use_batch_norm = False,
                                use_layer_norm = False,
                                )
    
    inputs_1, _ = next(iter(train_dataloader))
    inputs_2, _ = next(iter(train_dataloader))

    model = model.to(DEVICE)
    inputs_1 = inputs_1.to(DEVICE)
    inputs_2 = inputs_2.to(DEVICE)

    model.train()
    model(inputs_1)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.time()
        outputs = model(inputs_2)
        loss = lossfnc(outputs, inputs_2)
        print("Train time:", time.time() - t)

    prof.export_chrome_trace("profiler_train_iter.json")

    inputs_1, _ = next(iter(train_dataloader))
    inputs_2, _ = next(iter(train_dataloader))

    model = model.to(DEVICE)
    inputs_1 = inputs_1.to(DEVICE)
    inputs_2 = inputs_2.to(DEVICE)

    model.eval()

    with torch.no_grad():
        model(inputs_1)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            t = time.time()
            outputs = model(inputs_2)
            print("Eval time:", time.time() - t)
            loss = lossfnc(outputs, inputs_2)

    prof.export_chrome_trace("profiler_eval_iter.json")