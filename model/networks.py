import torch


class Encoder(torch.nn.Module):
    def __init__(self,image_size,latent_dim):
        super(Encoder,self).__init__()
        self.image_size = image_size
        # b,c,w,h = [1,1,250,100]
        self.sequential1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,4,3),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(8),
        )
         # b,c,w,h = [1,1,250,100]


        self.sequential2 = torch.nn.Sequential(
            torch.nn.Conv2d(1,4,2),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(8),
        )

        self.conv1 = torch.nn.Conv1d()

        # reduced image shape


        # reduce the image to a particular values to
        # get better results

class Decoder(torch.nn.Module):
    def __init__(self,image_size,latent_dim):
        super(Decoder,self).__init__()


class Classifier(torch.nn.Module):
    def __init__(self,image_size,latent_dim):
        super(Classifier,self).__init__()
        self.encoder = Encoder(mage_size,latent_dim)
        self.fc1 = torch.nn.Linear(latent_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)
        self.sigmoid = torch.nn.functional.sigmoid()

    def forward(self,x):
        latent = self.encoder(x)
        # returns the classification either 0 or 1
        return self.sigmoid(self.fc2(self.fc1(latent)))

class AutoEncoder(torch.nn.Module):
    def __init__(self,image_size,latent_dim):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(image_size,latent_dim)
        self.encoder.requires_grad_(False)
        self.decoder = Decoder(image_size,latent_dim)

    def forward(self,x):
        self.encoder(self.decoder(x))

        