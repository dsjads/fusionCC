
import torch
from torch import nn

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        self.MLP.add_module(
            name="L{:d}".format(0), module=nn.Linear(layer_sizes[0], layer_sizes[1]))
        self.MLP.add_module(name="A{:0}".format(0), module=nn.ReLU())
        self.MLP.add_module(name="D{:0}".format(0), module=nn.Dropout(p=0.25))

        self.MLP.add_module(
            name="L{:d}".format(1), module=nn.Linear(layer_sizes[1], layer_sizes[1]))
        self.MLP.add_module(name="A{:0}".format(1), module=nn.ReLU())
        self.MLP.add_module(name="D{:0}".format(1), module=nn.Dropout(p=0.25))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=2)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.MLP.add_module(name="L{:d}".format(0),
                            module=nn.Linear(input_size, layer_sizes[0]))
        self.MLP.add_module(name="A{:d}".format(0), module=nn.ReLU())
        self.MLP.add_module(name="D{:d}".format(0), module=nn.Dropout(p=0.25))

        self.MLP.add_module(name="L{:d}".format(1),
                            module=nn.Linear(layer_sizes[0], layer_sizes[0]))
        self.MLP.add_module(name="A{:d}".format(1), module=nn.ReLU())
        self.MLP.add_module(name="D{:d}".format(1), module=nn.Dropout(p=0.25))

        self.MLP.add_module(name="L{:d}".format(2),
                            module=nn.Linear(layer_sizes[0], layer_sizes[1]))
        self.MLP.add_module(name="S{:d}".format(2), module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=2)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class CVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):
        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        # if x.dim() > 2:
        #     x = x.view(-1, 28 * 28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


def loss_fn(recon_x, x, mean, log_var):
    MSE = (recon_x - x).norm(2).pow(2)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (MSE + KLD) / x.size(0)

