import torch
import torch.nn as nn
import torch.nn.functional as F


### Conv Net for classification ###

class CNNMNISTClassifier(nn.Module):
    """
    Implementation of UNet
    TODO: periodic boundary in the longitudinal direction
    """
    def __init__(self, n_alpha=120, n_beta=59, widths=(1, 8, 32, 128, 256, 512), in_datatype="complex-vector"):
        super(CNNMNISTClassifier, self).__init__()

        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.widths = widths
        self.in_datatype = in_datatype

        if self.in_datatype not in ["complex-vector", "complex-scalar", "real-scalar"]:
            raise Exception(f"Input datatype {self.in_datatype} not supported.")
        
        if self.in_datatype == "complex-vector":
            self.in_channels = 2
        elif self.in_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
            self.in_channels = 1

        if self.in_channels != self.widths[0]:
            raise Exception(f"Input channels {self.in_channels} do not match the first width \
                            {self.widths[0]} provided.")

        self.layers = nn.ModuleList()
        for i in range(len(self.widths) - 1):
            if i!=len(self.widths) - 2:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.widths[i], out_channels=self.widths[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.widths[i+1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2))
                )
            else:
                #Last layer does not have max pooling
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.widths[i], out_channels=self.widths[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.widths[i+1]),
                    nn.ReLU(inplace=True))
                )
        
        final_height = self.n_alpha // (2 ** (len(self.widths) - 2))
        final_width = self.n_beta // (2 ** (len(self.widths) - 2))
        self.final_layer = nn.Linear(self.widths[-1] * final_height * final_width, 10)  # Assuming 10 classes for MNIST

    def forward(self, x):
        batch_size = x.size(0)
        n_alpha = x.size(-2)
        n_beta = x.size(-1)

        if n_alpha != self.n_alpha or n_beta != self.n_beta:
            raise ValueError(f"Input dimensions {n_alpha}x{n_beta} do not match expected dimensions {self.n_alpha}x{self.n_beta}.")

        # Input is [batch_size, channels, alpha, beta] (complex)
        if self.in_datatype == "complex-vector":
            # c[batch_size, alpha, beta] -> r[batch_size, features, alpha, beta, 2]
            x = torch.view_as_real(x)
            # r[batch_size,..., alpha, beta, 2] -> r[batch_size,..., 2, alpha, beta]
            x = torch.swapdims(x, -1, -2)
            x = torch.swapdims(x, -3, -2)
            if x.dim() == 5:
                x = x.squeeze(1)  # Remove the feature dimension for complex vector
        elif self.in_datatype == "complex-scalar":
            if x.dim() != 4:
                raise ValueError(f"Input for complex-scalar should be 4D, got shape {x.shape}.")
            x = torch.real(x)
        elif self.in_datatype == "real-scalar":
            # x is already in the correct format
            pass

        for layer in self.layers:
            x = layer(x)
        

        # Flatten the output and create final layer dynamically if needed
        x_flat = x.contiguous().view(batch_size, -1)

        x = self.final_layer(x_flat)

        return x