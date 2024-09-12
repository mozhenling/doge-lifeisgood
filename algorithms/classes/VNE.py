import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler
from networks.net_selector import get_nets

class VNE(Algorithm):
    """
    VNE:  von Neumann entropy
    @InProceedings{Kim_2023_CVPR,
    author    = {Kim, Jaeill and Kang, Suhyun and Hwang, Duhun and Shin, Jungwook and Rhee, Wonjong},
    title     = {VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3799-3810}
    }
    https://github.com/jaeill/CVPR23-VNE
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(VNE, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0]))
        # -- build the model
        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams, args)

        self.network = nn.Sequential(self.featurizer, self.classifier)

        # Optimizer for the model (featurizer + classifier)
        self.model_optimizer = get_optimizer(params=self.network.parameters(), hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.model_optimizer, args=self.args)

    # def get_vne_large_N(self, H):
    #     """
    #     # N   : batch size
    #     # d   : embedding dimension
    #     # H   : embeddings, Tensor, shape=[N, d]
    #     """
    #     Z = torch.nn.functional.normalize(H, dim=1)
    #     rho = torch.matmul(Z.T, Z) / Z.shape[0]
    #     eig_val = torch.linalg.eigh(rho)[0][-Z.shape[0]:]
    #     return - (eig_val * torch.log(eig_val)).nansum()

    # the following is equivalent and faster when N < d: it may happen that the svd does not converge
    def get_vne(self, H):
        Z = torch.nn.functional.normalize(H, dim=1)
        # Regularization
        epsilon = 1e-10
        EP = epsilon*torch.eye(Z.shape[0], Z.shape[1]).to(Z.device)
        sing_val = torch.svd((Z+EP) / (np.sqrt(Z.shape[0]))[1]+epsilon)
        eig_val = sing_val ** 2
        return - (eig_val * torch.log(eig_val)).nansum()

    def update(self, minibatches, unlabeled=None):
        # -- pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        z = self.featurizer(x)

        # In case svd is not convergent, we wet it to zero
        try:
            vne = self.get_vne(z)
        except:
            vne = 0.

        loss_erm = F.cross_entropy(self.classifier(z), y)
        # Minimizing the von Neumann entropy
        objective = loss_erm - self.hparams["vne_coef"]*vne

        self.model_optimizer.zero_grad()
        objective.backward()
        self.model_optimizer.step()

        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': objective.item()}

    def predict(self, x):
        return self.network(x)

