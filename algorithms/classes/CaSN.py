import torch
from algorithms.classes.Algorithm import Algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_selector import get_nets
import copy

class CaSN(Algorithm):
    """
    https://github.com/ymy4323460/CaSN/tree/main
    [1] M. Yang et al., “Invariant Learning via Probability of Sufficient and Necessary
    Causes,” Advances in Neural Information Processing Systems, vol. 36, pp. 79832–79857, Dec. 2023.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(CaSN, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0]))

        self.network = IntModel(input_shape, num_classes,  self.num_domains, self.hparams, self.args)
        self.max_optimization_step = hparams['max_optimization_step']
        self.if_adversarial = hparams['if_adversarial']
        for i in self.network.parameters():
            i.requires_grad = False
        for i in self.network.intervener.parameters():
            i.requires_grad = True
        # define optimizer for max
        self.max_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.hparams["lr"]*0.1,
                weight_decay=self.hparams['weight_decay'])
        for i in self.network.parameters():
            i.requires_grad = True
        for i in self.network.intervener.parameters():
            i.requires_grad = True
        # define optimizer for min
        self.min_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        for i in self.network.parameters():
            i.requires_grad = True

        self.mse = torch.nn.MSELoss()
        self.sftcross = torch.nn.CrossEntropyLoss()

    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = (qm - pm).pow(2)#0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.mean()
        #print("log var1", qv)
        return kl

    def condition_prior(self, scale, label, dim):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = ((label-scale[0])/(scale[1]-scale[0])).reshape(-1, 1).repeat(1, dim) #torch.ones(label.size()[0], dim)*label
        var = torch.ones(label.size()[0], dim)
        return mean.to(device), var.to(device)

    def intervention_loss(self, intervention):
        return torch.norm(torch.pow(intervention, 2)-self.hparams['bias'])

    def targets_loss(self, y_pred, int_y_pred):
        return -self.mse(torch.sigmoid(y_pred), torch.sigmoid(int_y_pred))

    def kl_loss(self, m, v, y):
        if self.hparams['prior_type'] == 'conditional':
            pm, pv = self.condition_prior([0, self.num_classes], y, m.size()[1])
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        return self.kl_normal(m, pv * 0.0001, pm, pv * 0.0001)

    def all_loss(self, x, y, env_i, turn='min'):
        m, v, z, int_z, y_pred, int_y_pred, intervention, z_c = self.network(x)
        nll = F.cross_entropy(y_pred, y).mean()
        int_nll = -F.cross_entropy(int_y_pred, y).mean()
        kl = self.kl_loss(z, v, y).mean() + self.kl_loss(z_c, v, y).mean()
        inter_norm = self.intervention_loss(intervention).mean()
        targets_loss = self.targets_loss(y_pred, int_y_pred).mean()

        all = nll + self.hparams['int_lambda']*int_nll + self.hparams['int_reg']*inter_norm + self.hparams['target_lambda']*targets_loss
        if turn == 'min':
            return all + self.hparams['kl_lambda']*kl
        else:
            return -all + self.hparams['kl_lambda']*kl


    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        nll = 0.
        penalty = 0.

        for i, (x, y) in enumerate(minibatches):
            env_i = None # not implemented yet
            self.min_optimizer.zero_grad()
            loss = self.all_loss(x, y, env_i, 'min')
            L = loss.mean()
            L.backward()
            self.min_optimizer.step()

            if self.if_adversarial == 'adversarial':
                if  i%self.max_optimization_step == 0 and i>0:
                    self.max_optimizer.zero_grad()
                    loss = self.all_loss(x, y, 'max')
                    L = loss.mean()
                    L.backward()
                    self.max_optimizer.step()

        self.update_count += 1

        return {'loss': L.cpu().detach().numpy().item()}

    def predict(self, x):
        return self.network(x)[4]

class IntModel(nn.Module):
    def __init__(self, input_shape, num_classes,  num_domains, hparams, args, prior_type='conditional'):
        super(IntModel, self).__init__()
        self.hparams = hparams
        self.num_domains = num_domains
        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams, args)
        _, self.discriminator = get_nets(input_shape, num_classes, num_domains, hparams, args)

        self.get_z = MLP(self.featurizer.n_outputs, self.featurizer.n_outputs, self.hparams)
        self.intervener = MLP(self.featurizer.n_outputs, self.featurizer.n_outputs, self.hparams)

    def sample_gaussian(self, m, v):
    # reparameterization
        sample = torch.randn(m.size()).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        z = m + (v**0.5) * sample
        return z

    def forward(self, x):
        m = self.featurizer(x)
        v = torch.zeros_like(m)
        z_c = self.sample_gaussian(m, v)
#         print(z_c.size())
#         pred_domain = self.discriminator(z_c)

        z = self.get_z(z_c)
        intervention = self.intervener(z_c)
        int_z = z + intervention
        y = self.classifier(z)
        int_y = self.classifier(int_z)
        return m, v, z, int_z, y, int_y, intervention, z_c

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        # -- could replace by hparams
        self.mlp_width = n_inputs # hparams['mlp_width']
        self.mlp_depth = 1 #  hparams['mlp_depth']
        self.mlp_dropout = 0 # hparams['mlp_dropout']

        self.input = nn.Linear(n_inputs, self.mlp_width)

        self.dropout = nn.Dropout(self.mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(self.mlp_width,self.mlp_width)
            for _ in range(self.mlp_depth)])

        self.output = nn.Linear(self.mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x