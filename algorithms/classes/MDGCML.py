import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from networks.net_selector import get_nets
from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler
from datautils.data_process import random_pairs_of_minibatches

class MDGCML(Algorithm):
    """
    As the code is not public, this is an unofficial implementation of multisource domain-class
    gradient coordination meta-learning (MDGCML) adapted for multi-class domain generalization from:

    Ref.:
        [1] C. Wang et al., “Learning to Imbalanced Open Set Generalize: A Meta-Learning
               Framework for Enhanced Mechanical Diagnosis,” IEEE Transactions on Cybernetics,
               pp. 1–12, 2025, doi: 10.1109/TCYB.2025.3531494.
        [2] https://github.com/mozhenling/doge-eirm/blob/master/algorithms/classes/EIRM.py

    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(MDGCML, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams, args)
        self.network = nn.Sequential(self.featurizer, self.classifier)

        # Optimizer for the model (featurizer + classifier)
        self.optimizer = get_optimizer(params=self.network.parameters(), hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.optimizer, args=self.args)

    def update(self, minibatches, unlabeled=None):
        loss = 0

        grads = [0 for _ in self.network.parameters()]
        # save original sate dict
        state_dict = self.network.state_dict()
        self.network.zero_grad()

        for idx, ( (xi, yi), (xj, yj) ) in enumerate(random_pairs_of_minibatches(minibatches)):
            #-- get meta train and test sets
            xi_num = len(xi)
            xj_num = len(xj)
            xi_1, xi_2, yi_1, yi_2 = xi[:xi_num//2], xi[xi_num//2:], yi[:xi_num//2], yi[xi_num//2:]

            xj_1, xj_2, yj_1, yj_2 = xj[:xj_num//2], xj[xj_num//2:], yj[:xi_num//2], yj[xi_num//2:]

            # load original state
            if idx >0:
                self.network.load_state_dict(state_dict)
                self.network.zero_grad()

            #-- meta train
            loss_train =  F.cross_entropy(self.predict(xi_1), yi_1) + F.cross_entropy(self.predict(xj_2), yj_2)

            grads_train = autograd.grad(loss_train, self.network.parameters())

            # temp model update
            with torch.no_grad():
                for p, g in zip(self.network.parameters(), grads_train):
                    # gradient descent update
                    p.sub_(g * self.hparams["gamma"])  # can be another independent hyper-params to increase flexibility

            # -- meta test
            loss_test =  F.cross_entropy(self.predict(xi_2), yi_2) + F.cross_entropy(self.predict(xj_1),yj_1)
            grads_test = autograd.grad(loss_test, self.network.parameters())

            # final loss and grads
            with torch.no_grad():
                for j, (g1, g2) in enumerate(zip(grads_train, grads_test)):
                    # gradient combination
                    grads[j] += g1 +  self.hparams["delta"]*g2
            loss += loss_train + loss_test

            # average
            loss /= len(minibatches)
            grads = [g / len(minibatches) for g in grads]
            # back to original
            self.network.load_state_dict(state_dict)
            self.network.zero_grad()
            self.optimizer.zero_grad()
            # update model by grads
            with torch.no_grad():
                for p, g in zip(self.network.parameters(), grads):
                    p.grad = g
            # optimizer uses the p.grad to update the p with the learning rate self.hparams["lr"] (i.e., eta in [1])
            self.optimizer.step()

            return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)