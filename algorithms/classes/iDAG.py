import copy
import torch
import numpy as np
from algorithms.classes.Algorithm import Algorithm
import torch.nn as nn
from networks.net_selector import get_nets
from algorithms.optimization import get_optimizer, get_scheduler
import torch.nn.functional as F

class iDAG(Algorithm):
    """
    DAG domain generalization methods
    @InProceedings{Huang_2023_ICCV,
    author    = {Huang, Zenan and Wang, Haobo and Zhao, Junbo and Zheng, Nenggan},
    title     = {iDAG: Invariant DAG Searching for Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19169-19179}

    https://github.com/lccurious/iDAG/tree/master
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(iDAG, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0]))

        self.featurizer, _= get_nets(input_shape, num_classes, num_domains, hparams, args)
        # Light encoder
        self.encoder = torch.nn.Sequential(
                            torch.nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs))

        self.dag_mlp = NotearsClassifier(self.featurizer.n_outputs, num_classes)
        self.dag_mlp.weight_pos.data[:-1, -1].fill_(1.0)

        self.inv_classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.rec_classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.encoder, self.dag_mlp, self.inv_classifier)

        self.proto_m = self.hparams["ema_ratio"]
        self.lambda1 = self.hparams["lambda1"]
        self.lambda2 = self.hparams["lambda2"]
        self.rho_max = self.hparams["rho_max"]
        self.alpha = self.hparams["alpha"]
        self.rho = self.hparams["rho"]
        self._h_val = np.inf

        self.register_buffer(
            "prototypes_y",
            torch.zeros(num_classes, self.featurizer.n_outputs))
        self.register_buffer(
            "prototypes",
            torch.zeros(num_domains, num_classes, self.featurizer.n_outputs))
        self.register_buffer(
            "prototypes_label",
            torch.arange(num_classes).repeat(num_domains))

        params = [
            {"params": self.network.parameters()},
            {"params": self.rec_classifier.parameters()},
        ]
        self.optimizer = get_optimizer(params=params, hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.optimizer, args=self.args)

        self.loss_proto_con = PrototypePLoss(num_classes, hparams['temperature'])
        self.loss_multi_proto_con = MultiDomainPrototypePLoss(num_classes, num_domains, hparams['temperature'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        domain_labels = torch.cat([torch.full((x.shape[0],), i ) for i, (x, y) in enumerate(minibatches)]).to(self.device)

        all_f = self.featurizer(all_x)
        all_f = self.encoder(all_f)
        all_masked_f = self.dag_mlp(all_f)

        for f, masked_f, label_y, label_d in zip(F.normalize(all_f, dim=1),
                                                 F.normalize(all_masked_f, dim=1),
                                                 all_y,
                                                 domain_labels):
            self.prototypes[label_d, label_y] = self.prototypes[label_d, label_y] * self.proto_m + (1 - self.proto_m) * f.detach()
            self.prototypes_y[label_y] = self.prototypes_y[label_y] * self.proto_m + (1 - self.proto_m) * masked_f.detach()
        self.prototypes = F.normalize(self.prototypes, p=2, dim=2)
        self.prototypes_y = F.normalize(self.prototypes_y, p=2, dim=1)

        prototypes = self.prototypes.detach().clone()
        prototypes_y = self.prototypes_y.detach().clone()

        proto_rec, masked_proto = self.dag_mlp(
            x=prototypes.view(self.num_domains * self.num_classes, -1),
            y=self.prototypes_label)

        # reconstruction loss
        loss_rec = F.cosine_embedding_loss(
            proto_rec,
            prototypes.view(self.num_domains * self.num_classes, -1),
            torch.ones(self.num_domains * self.num_classes, device=all_x.device))
        loss_rec += F.cross_entropy(
            self.rec_classifier(masked_proto),
            self.prototypes_label)
        loss_rec = self.lambda2 * loss_rec
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.dag_mlp.w_l1_reg()

        # update the DAG hyper-parameters
        if self.update_count % 100 == 0:
            if self.rho < self.rho_max and h_val > 0.25 * self._h_val:
                self.rho *= 10
                self.alpha += self.rho * h_val.item()
            self._h_val = h_val.item()

        loss_dag = loss_rec + penalty + l1_reg

        loss_inv_ce = F.cross_entropy(self.inv_classifier(all_masked_f), all_y)

        loss_contr_mu = self.hparams["weight_mu"] * self.loss_proto_con(all_masked_f, prototypes_y, all_y)
        loss_contr_nu = self.hparams["weight_nu"] * self.loss_multi_proto_con(all_f, prototypes, all_y, domain_labels)
        loss_contr = loss_contr_mu + loss_contr_nu

        if self.update_count == self.hparams["dag_anneal_steps"]:
            # avoid the gradient jump
            self.optimizer =  get_optimizer(params=self.network.parameters(), hparams=self.hparams, args=self.args)

        if self.update_count >= self.hparams["dag_anneal_steps"]:
            loss = loss_inv_ce + loss_dag + loss_contr_mu + loss_contr_nu
        else:
            loss = loss_inv_ce + loss_contr_nu

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.args.scheduler:
            self.scheduler.step()
        # constraint DAG weights
        self.dag_mlp.projection()
        self.update_count += 1
        # return {"loss": loss.item(),
        #         "inv_ce": loss_inv_ce.item(),
        #         "l2": loss_rec.item(),
        #         "penalty": penalty.item(),
        #         "l1": l1_reg.item(),
        #         "cl": loss_contr.item()}

        return {"loss": loss.item()}

    def predict(self, x):
        f = self.featurizer(x)
        f = self.encoder(f)
        masked_f = self.dag_mlp(f)
        return self.inv_classifier(masked_f)

    def clone(self):
        clone = copy.deepcopy(self)
        params = [
            {"params": clone.network.parameters()},
        ]
        clone.optimizer = self.new_optimizer(params)
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class PrototypePLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super(PrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.arange(num_classes)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes)

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.not_equal(labels, labels.T)
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C+N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), labels)
        return loss


class MultiDomainPrototypePLoss(nn.Module):
    def __init__(self, num_classes, num_domains, temperature):
        super(MultiDomainPrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.num_classes = num_classes
        self.label = torch.arange(num_classes)
        self.domain_label = torch.arange(num_domains)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels, domain_labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes.reshape(-1, prototypes.size(-1)))

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.logical_or(torch.not_equal(labels, labels.T), torch.not_equal(domain_labels, domain_labels))
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C*D + N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), domain_labels * self.num_classes + labels)
        return loss

class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _adj_sub(self):
        W = self._adj()
        return torch.matrix_exp(W * W)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            x_aug = torch.cat((x, y.unsqueeze(1)), dim=1)
            M = x_aug @ W
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x):
        W_sub = self._adj_sub()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)
