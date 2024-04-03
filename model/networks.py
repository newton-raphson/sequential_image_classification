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
        self.conv4 = torch.nn.Conv2d(in_channels=8,out_channels=1,kernel_size=2,stride=1)
        # b,c,w,h = [1,1,28,9]
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(-1,252) # reshape to [b,252]
        return x



class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=252, image_size=(250, 100)):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        # resize into [b,c,w,h] = [b,1,28,9]
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=8, kernel_size=2, stride=2, padding=0)
        # resize into [b,c,w,h] = [b,1,28,9]
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=0)
        # resize into [b,c,w,h] = [b,1,61,23]
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(4,5), stride=2, padding=0)
        # resize into [b,c,w,h] = [b,1,124,49]
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=0)
        # resize into [b,c,w,h] = [b,1,250,100]
    def forward(self, x):
        x = x.view(-1,1,28,9)  # Reshape to match the expected shape after linear layer
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.sigmoid(x)  # Use sigmoid to scale output to [0, 1]
        return x

    
class Classifier(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=252,hidden_dim=100):
        super(Classifier,self).__init__()
        self.encoder = Encoder(image_size=image_size,latent_dim=latent_dim)
        self.fc1 = torch.nn.Linear(latent_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        latent = self.encoder(x)
        # print(f"{latent.shape}")
        latent = latent.view(-1,28*9)
        out1 = self.fc1(latent)
        out2 = self.fc2(out1)
        # returns the classification either 0 or 1
        return self.sigmoid(out2)

class AutoEncoder(torch.nn.Module):
    def __init__(self,image_size=(250,100),latent_dim=50):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder(image_size,latent_dim)
        self.decoder = Decoder(latent_dim,image_size)

    def forward(self,x):
        return self.decoder(self.encoder(x))

        