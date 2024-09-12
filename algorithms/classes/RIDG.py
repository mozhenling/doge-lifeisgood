import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_selector import get_nets
from algorithms.classes.Algorithm import Algorithm

class RIDG(Algorithm):
    """
    Rational Invariance for Domain Generalization (RIDG)

    @InProceedings{Chen_2023_ICCV,
    author    = {Chen, Liang and Zhang, Yong and Song, Yibing and van den Hengel, Anton and Liu, Lingqiao},
    title     = {Domain Generalization via Rationale Invariance},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1751-1760}
}
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(RIDG, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams,args)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.num_classes = num_classes
        self.rational_bank = torch.zeros(num_classes, num_classes, self.featurizer.n_outputs, device='cuda')
        self.init = torch.ones(num_classes, device='cuda')
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        features = self.featurizer(all_x)
        logits = self.predict(all_x)
        rational = torch.zeros(self.num_classes, all_x.shape[0], self.featurizer.n_outputs, device='cuda')
        for i in range(self.num_classes):
            rational[i] = (self.classifier.weight[i] * features)

        classes = torch.unique(all_y)
        loss_rational = 0
        for i in range(classes.shape[0]):
            rational_mean = rational[:, all_y == classes[i]].mean(dim=1)
            if self.init[classes[i]]:
                self.rational_bank[classes[i]] = rational_mean
                self.init[classes[i]] = False
            else:
                self.rational_bank[classes[i]] = (1 - self.hparams['momentum']) * self.rational_bank[classes[i]] + \
                                                 self.hparams['momentum'] * rational_mean
            loss_rational += ((rational[:, all_y == classes[i]] - (
                self.rational_bank[classes[i]].unsqueeze(1)).detach()) ** 2).sum(dim=2).mean()
        loss = F.cross_entropy(logits, all_y)
        loss += self.hparams['ridg_reg'] * loss_rational

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        z = self.featurizer(x)
        return self.classifier(z)