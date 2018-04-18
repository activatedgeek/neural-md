import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.chain_length = 30
        self.hidden_size = 5

        self.encoder = CNN()
        self.l1 = nn.Linear(self.hidden_size * self.chain_length, self.hidden_size * self.chain_length)
        self.l2 = nn.Linear(self.hidden_size * self.chain_length, self.hidden_size * self.chain_length)
        self.decoder = DCNN()

    def reparameterize(self, mu, log_sigma):
        normal_eps = torch.randn(h, n)
        reparam = mu + (torch.sqrt(log_sigma) * normal_eps)
        return reparam

    def forward(self, x):
        z = self.encoder(x)
        mu = self.l1(z)
        log_sigma = self.l2(z)
        mu = mu.view(self.chain_length, -1)
        log_sigma = log_sigma(self.chain_length, -1)
        z = self.reparameterize(mu, log_sigma)
        return self.decoder(z), mu, log_sigma


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Parameters
        self.C = 53
        self.chain_length = 30
        self.hidden_size = 30

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.C, 53, (5, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.conv1_out, 53, (5, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.out = nn.Linear(self.conv2_out * self.chain_length, self.hidden_size * self.chain_length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv1
        output = self.out(x)
        return output


class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()

        # Parameters
        self.hidden_size = 5
        self.chain_length = 30
        self.max_no_of_angles = 23

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_size * self.chain_length, 20, (5, 1), padding=2),
            nn.ReLU(),
            nn.MaxUnpool1d(4))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(self.deconv1_out, self.max_no_of_angles, (5, 1), padding=2),
            nn.ReLU(),
            nn.MaxUnpool1d(self.pool_size))

    def forward(self, x):
        x = self.deconv1(x)
        output = self.deconv2(x)
        return output
