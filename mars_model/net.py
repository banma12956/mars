'''
Network architecture.
'''

import torch
import torch.nn as nn

def full_block(in_features, out_features, p_drop):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.LayerNorm(out_features),
            nn.ELU(),
            nn.Dropout(p=p_drop),
        )

class FullNet(nn.Module):
    '''
    '''
    def __init__(self, x_dim, hid_dim=64, z_dim=64, p_drop=0.2):
        super(FullNet, self).__init__()
        self.z_dim = z_dim
        
        # self.encoder = nn.Sequential(
        #     full_block(x_dim, hid_dim, p_drop),
            # full_block(hid_dim, z_dim, p_drop),
        # )
        self.encoder = full_block(x_dim, hid_dim, p_drop)
        self.mu = nn.Linear(hid_dim, z_dim)
        self.logsigma = nn.Linear(hid_dim, z_dim)
        self.softplus = nn.Softplus()
        
        self.decoder = nn.Sequential(
            full_block(z_dim, hid_dim, p_drop),
            full_block(hid_dim, x_dim, p_drop),
        )
      
    def reparameterize(self, mu, logsigma):
        epsilon = torch.randn(mu.size(0), mu.size(1))

        # if self.usecuda:
            # epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * torch.exp(logsigma/2)

        return latent

    def forward(self, x):
        
        x = self.encoder(x)
        mu = self.mu(x)
        logsigma = self.softplus(self.logsigma(x))
        latent = self.reparameterize(mu, logsigma)

        decoded = self.decoder(latent)
        
        return mu, logsigma, decoded

    def cal_loss(self, x, decoded, mu, logsigma):
        loss_func = nn.MSELoss()
        loss_rcn = loss_func(decoded, x)

        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        loss = loss_rcn + kld * 0.05

        return loss
