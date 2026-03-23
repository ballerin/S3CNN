import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3x3(nn.Module):
    """
    Block of
    3x3 conv + ReLU + 3x3 conv + ReLU 
    """
    def __init__(self, inchannels, outchannels):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannels, outchannels, 3, padding=1),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_out(nn.Module):
    """
    Final block of the model
    3x3 conv + ReLU + 3x3 conv + ReLU + conv + Sigmoid
    """
    def __init__(self, inchannels,midchannels, n_channels):
        super(conv3x3_out, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, midchannels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannels, midchannels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(midchannels, n_channels, 1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)*2-1
        return x


class downStep(nn.Module):
    """
    UNet downstep module
    MaxPooling + usual convolution block
    """
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv3x3(inC, outC),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upStep(nn.Module):
    """
    UNet upstep module
    up-convolution + stacking + convolution
    """
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.upStep = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        self.conv = conv3x3(inC, outC)

    def forward(self, x, x_down):
        x = self.upStep(x)
        x = torch.cat((x_down, x), dim=1) #This creates problems if the sizes mismatch
        x = self.conv(x)
        return x

class finalupStep(nn.Module):
    """
    UNet upstep module
    up-convolution + stacking + final convolution
    
    This is the last layer, so the final block is used
    """
    def __init__(self, inC, outC, final_channels):
        super(finalupStep, self).__init__()
        self.upStep = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        self.conv = conv3x3_out(inC, outC, n_channels=final_channels)

    def forward(self, x, x_down):
        x = self.upStep(x)
        x = torch.cat((x_down, x), dim=1) #This creates problems if the sizes mismatch
        x = self.conv(x)
        return x

def pad_to(x, stride):
    """
    Method to pad a sample x with a certain stride
    so that it becomes divisible by two
    """
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    """
    Reverse of the padding method
    """
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class CNNUNet(nn.Module):
    """
    Implementation of UNet
    TODO: periodic boundary in the longitudinal direction
    """
    def __init__(self, n_alpha, n_beta, in_datatype="complex-vector", out_datatype="complex-vector"):
        super(CNNUNet, self).__init__()


        self.n_alpha = n_alpha
        self.n_beta = n_beta
        
        self.in_datatype = in_datatype
        self.out_datatype = out_datatype

        if self.in_datatype not in ["complex-vector", "complex-scalar", "real-scalar"]:
            raise Exception(f"Input datatype {self.in_datatype} not supported.")
        if self.out_datatype not in ["complex-vector", "complex-scalar", "real-scalar"]:
            raise Exception(f"Output datatype {self.out_datatype} not supported.")

        if self.in_datatype == "complex-vector":
            self.in_channels = 2
        elif self.in_datatype == "complex-scalar" or self.in_datatype == "real-scalar":
            self.in_channels = 1
        
        if self.out_datatype == "complex-vector":
            self.out_channels = 2
        elif self.out_datatype == "complex-scalar" or self.out_datatype == "real-scalar":
            self.out_channels = 1
        
        self.inputs = conv3x3(self.in_channels, 32)
        self.downStep64  = downStep(32, 64)
        self.downStep128 = downStep(64, 128)
        self.downStep256 = downStep(128, 256)
        self.downStep512 = downStep(256, 512)
        self.upStep1024 = upStep(512, 256)
        self.upStep512 = upStep(256, 128)
        self.upStep256 = upStep(128, 64)
        self.upStep128 = finalupStep(64, 32, final_channels=self.out_channels)

    def forward(self, x):
        # Input is [batch_size, channels, alpha, beta] (complex)
        if self.in_datatype == "complex-vector":
            # c[batch_size, 1, alpha, beta] -> c[batch_size, alpha, beta]
            x = x.squeeze(1)
            # c[batch_size, alpha, beta] -> r[batch_size, alpha, beta, 2]
            x = torch.view_as_real(x)
            # r[batch_size, alpha, beta, 2] -> r[batch_size, 2, alpha, beta]
            x = torch.swapdims(x, 1, 2)
            x = torch.swapdims(x, 1, 3)
        elif self.in_datatype == "complex-scalar":
            # If no feature dimension, add it
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = torch.real(x)
        elif self.in_datatype == "real-scalar":
            # x is already in the correct format
            pass
        
        InputsL = self.inputs(x)
        InputsL, padsL = pad_to(InputsL, 2)
        
        DownL1 = self.downStep64(InputsL)
        DownL1, padsL1 = pad_to(DownL1, 2)
        
        DownL2 = self.downStep128(DownL1)
        DownL2, padsL2 = pad_to(DownL2, 2)
        
        DownL3 = self.downStep256(DownL2)
        DownL3, padsL3 = pad_to(DownL3, 2)
        
        DownL4 = self.downStep512(DownL3)
        
        UpL4 = self.upStep1024(DownL4, DownL3)
        UpL4 = unpad(UpL4, padsL3)
        
        UpL3 = self.upStep512(UpL4, DownL2)
        UpL3 = unpad(UpL3, padsL2)
        
        UpL2 = self.upStep256(UpL3, DownL1)
        UpL2 = unpad(UpL2, padsL1)
        
        OutputsL = self.upStep128(UpL2, InputsL)
        OutputsL = unpad(OutputsL, padsL)

        reconstructed = OutputsL

        # Convert back to the original complex datatype
        if self.out_datatype == "complex-vector":
            reconstructed = torch.swapdims(reconstructed, 1, 3)
            reconstructed = torch.swapdims(reconstructed, 1, 2)
            reconstructed = torch.view_as_complex(reconstructed.contiguous())
        elif self.out_datatype == "complex-scalar":
            reconstructed = reconstructed.to(torch.complex64)
        elif self.out_datatype == "real-scalar":
            # x is already in the correct format
            pass

        return reconstructed