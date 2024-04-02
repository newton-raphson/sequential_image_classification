import torch


class Encoder(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=252):
        super(Encoder,self).__init__()
        self.image_size = image_size
        # b,c,w,h = [1,1,250,100]
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=8,kernel_size=4,stride=2)
        # b,c,w,h = [1,8,124,49]
        self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=2)
        # b,c,w,h = [1,16,61,23]
        self.conv3 = torch.nn.Conv2d(in_channels=16,out_channels=8,kernel_size=4,stride=2)
        # b,c,w,h = [1,8,29,10]
        self.conv3 = torch.nn.Conv2d(in_channels=8,out_channels=1,kernel_size=2,stride=1)
         # b,c,w,h = [1,1,28,9]
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=252):
        super(Decoder,self).__init__()


class Classifier(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=252,hidden_dim=100):
        super(Classifier,self).__init__()
        self.encoder = Encoder(image_size=image_size,latent_dim=latent_dim)
        self.fc1 = torch.nn.Linear(latent_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        latent = self.encoder(x)
        latent = latent.view(-1,1)
        # returns the classification either 0 or 1
        return self.sigmoid(self.fc2(self.fc1(latent)))

class AutoEncoder(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=252):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(image_size,latent_dim)
        self.encoder.requires_grad_(False)
        self.decoder = Decoder(image_size,latent_dim)

    def forward(self,x):
        self.encoder(self.decoder(x))

        