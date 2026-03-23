import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=100):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # Layer 1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Layer 2 (downsample)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Layer 3
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Layer 4 (downsample)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Layer 5
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Layer 6 (downsample)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Layer 7
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Layer 8 (downsample)
            nn.ReLU()
        )
        
        # Latent space
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(141312, latent_dim)  # Assuming input size is 128x128
        self.fc_dec = nn.Linear(latent_dim, 141312)
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 1 (upsample)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Layer 2
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 3 (upsample)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Layer 4
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 5 (upsample)
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Layer 6
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 7 (upsample)
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1),  # Layer 8
            #No normalization
            #nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.swapdims(x, 1, 2)
        x = torch.swapdims(x, 1, 3)

        # Encoder
        encoded = self.encoder_conv(x)
        encoded = self.flatten(encoded)
        latent = self.fc_enc(encoded)  # Latent space
        
        # Decoder
        decoded = self.fc_dec(latent)
        decoded = decoded.view(-1, 512, 23, 12)  # Reshape to feature map dimensions
        output = self.decoder_conv(decoded)

        output = torch.swapdims(output, 1, 3)
        output = torch.swapdims(output, 1, 2)
        output = torch.view_as_complex(output.contiguous())

        # At this point the tensor is [batch, longitude, latitude] as the feature is one (complex)

        # Reshape to obtain the input size
        glat = (output.size(1)-360)//2
        glon = (output.size(2)-179)//2
        output = output[:,glat:glat+360,glon:glon+179]

        return output

class CNNAutoencoder128(nn.Module):
    def __init__(self):
        super(CNNAutoencoder128, self).__init__()

        self.encoder_conv = nn.ModuleList()
        self.encoder_nonl = nn.ModuleList()
        self.encoder_maxp = nn.ModuleList()

        self.decoder_conv = nn.ModuleList()
        self.decoder_nonl = nn.ModuleList()
        self.decoder_maxp = nn.ModuleList()
        
        self.encoder_conv.append(nn.Conv2d(2, 4, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3,return_indices=True))

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(4, 2, kernel_size=3, padding=1))
        
    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.swapdims(x, 1, 2)
        x = torch.swapdims(x, 1, 3)

        indeces = []
        sizes=[]

        #Encoding
        for i in range(len(self.encoder_conv)):
            x = self.encoder_conv[i](x)
            x = self.encoder_nonl[i](x)
            sizes.append(x.size())
            x,ind = self.encoder_maxp[i](x)
            indeces.append(ind)


        #Decoding
        for i in range(len(self.decoder_maxp)):
            x = self.decoder_maxp[i](x,indeces[-1-i], output_size=sizes[-1-i])
            x = self.decoder_conv[i](x)
            if i < len(self.decoder_maxp)-1:
                x = self.decoder_nonl[i](x)
        
        x = torch.swapdims(x, 1, 3)
        x = torch.swapdims(x, 1, 2)
        x = torch.view_as_complex(x.contiguous())

        return x

class CNNAutoencoder576(nn.Module):
    def __init__(self):
        super(CNNAutoencoder576, self).__init__()

        self.encoder_conv = nn.ModuleList()
        self.encoder_nonl = nn.ModuleList()
        self.encoder_maxp = nn.ModuleList()

        self.decoder_conv = nn.ModuleList()
        self.decoder_nonl = nn.ModuleList()
        self.decoder_maxp = nn.ModuleList()
        
        self.encoder_conv.append(nn.Conv2d(2, 4, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True))
        self.encoder_conv.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True))

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())
        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=3, stride=3))
        self.decoder_conv.append(nn.ConvTranspose2d(4, 2, kernel_size=3, padding=1))
        
    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.swapdims(x, 1, 2)
        x = torch.swapdims(x, 1, 3)

        indeces = []
        sizes=[]

        #Encoding
        for i in range(len(self.encoder_conv)):
            x = self.encoder_conv[i](x)
            x = self.encoder_nonl[i](x)
            sizes.append(x.size())
            x,ind = self.encoder_maxp[i](x)
            indeces.append(ind)


        #Decoding
        for i in range(len(self.decoder_maxp)):
            x = self.decoder_maxp[i](x,indeces[-1-i], output_size=sizes[-1-i])
            x = self.decoder_conv[i](x)
            if i < len(self.decoder_maxp)-1:
                x = self.decoder_nonl[i](x)
        
        x = torch.swapdims(x, 1, 3)
        x = torch.swapdims(x, 1, 2)
        x = torch.view_as_complex(x.contiguous())

        return x
    
class CNNAutoencoder512(nn.Module):
    def __init__(self):
        super(CNNAutoencoder512, self).__init__()

        self.n_alpha = 360
        self.n_beta = 179

        self.encoder_conv = nn.ModuleList()
        self.encoder_nonl = nn.ModuleList()
        self.encoder_maxp = nn.ModuleList()

        self.decoder_conv = nn.ModuleList()
        self.decoder_nonl = nn.ModuleList()
        self.decoder_maxp = nn.ModuleList()
        
        self.encoder_conv.append(nn.Conv2d(2, 4, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

        self.encoder_conv.append(nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

        self.encoder_conv.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

        self.encoder_conv.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True))

        self.encoder_conv.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True))

        self.encoder_conv.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True))

        self.encoder_conv.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())
        self.encoder_maxp.append(nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True))

        self.encoder_conv.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.encoder_nonl.append(nn.ReLU())

        ## Latent space
        
        self.decoder_conv.append(nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1))
        self.decoder_nonl.append(nn.ReLU())

        self.decoder_maxp.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        self.decoder_conv.append(nn.ConvTranspose2d(4, 2, kernel_size=3, padding=1))
        
    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.swapdims(x, 1, 2)
        x = torch.swapdims(x, 1, 3)
        
        indeces = []
        sizes=[]

        #Encoding
        for i in range(len(self.encoder_conv)):
            #Convolution
            x = self.encoder_conv[i](x)

            #Nonlinearity
            x = self.encoder_nonl[i](x)

            #Maxpool, except for last layer
            if i < len(self.encoder_conv)-1:
                sizes.append(x.size())
                x,ind = self.encoder_maxp[i](x)
                indeces.append(ind)

        #print(x.size(1)*x.size(2)*x.size(3))

        #Decoding
        for i in range(len(self.decoder_conv)):
            #MaxUNpool, except for first layer
            if i!=0:
                x = self.decoder_maxp[i-1](x,indeces[-1-i+1], output_size=sizes[-1-i+1])
            
            #Convolution
            x = self.decoder_conv[i](x)

            #Nonlinearity, except for last layer
            if i < len(self.decoder_conv)-1:
                x = self.decoder_nonl[i](x)

        x = torch.swapdims(x, 1, 3)
        x = torch.swapdims(x, 1, 2)
        x = torch.view_as_complex(x.contiguous())

        return x