import torch
import torch.nn as nn
from networks.net_selector import get_nets
import torch.nn.functional as F
from algorithms.classes.ERM import ERM

class IIB(ERM):
    """Invariant Information Bottleneck"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(IIB, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, args)
        feat_dim = self.featurizer.n_outputs
        # VIB archs
        if hparams['enable_bn']:
            self.encoder = torch.nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.encoder = torch.nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True)
            )
        self.fc3_mu = nn.Linear(feat_dim, feat_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(feat_dim, feat_dim)  # output = CNN embedding latent variables
        # Inv Risk archs
        networks = get_nets(input_shape, num_classes, num_domains, hparams, args, net_whole=True)
        self.inv_classifier = networks.Classifier(self.featurizer.n_outputs, num_classes,
                                                  self.hparams['nonlinear_classifier'])
        self.env_classifier = networks.Classifier(self.featurizer.n_outputs + 1, num_classes,
                                                  self.hparams['nonlinear_classifier'])
        self.domain_indx = [torch.full((hparams['batch_size'], 1), indx) for indx in range(num_domains)]
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.inv_classifier.parameters()) + list(
                self.env_classifier.parameters()) + list(self.encoder.parameters()) + list(
                self.fc3_mu.parameters()) + list(self.fc3_logvar.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def encoder_fun(self, res_feat):
        latent_z = self.encoder(res_feat)
        mu = self.fc3_mu(latent_z)
        logvar = self.fc3_logvar(latent_z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return torch.add(torch.mul(std, eps), mu)
        else:
            return mu

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        embeddings = torch.cat([curr_dom_embed for curr_dom_embed in self.domain_indx]).to(device)
        all_z = self.featurizer(all_x)
        # encode feature to sampling vector: \mu, \var
        mu, logvar = self.encoder_fun(all_z)
        all_z = self.reparameterize(mu, logvar)

        # calculate loss by parts
        ib_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        inv_loss = F.cross_entropy(self.inv_classifier(all_z), all_y)
        env_loss = F.cross_entropy(self.env_classifier(torch.cat([all_z, embeddings], 1)), all_y)

        # use beta to balance the info loss.
        total_loss = inv_loss + env_loss + self.hparams['lambda_beta'] * ib_loss + self.hparams['lambda_inv_risks'] * (
                inv_loss - env_loss) ** 2
        # or (inv_loss - env_loss) ** 2
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'loss_ib': ib_loss.item(), 'loss_env': env_loss.item(), 'loss_inv': inv_loss.item(),
                'loss_all': total_loss.item()}

    def predict(self, x):
        z = self.featurizer(x)
        mu, logvar = self.encoder_fun(z)
        z = self.reparameterize(mu, logvar)
        y = self.inv_classifier(z)
        return y