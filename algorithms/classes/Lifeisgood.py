# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_selector import get_nets
from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler

class Lifeisgood(Algorithm):
    """
    Lifeisgood: Learning Invariant Feature via In-label Swapping for Generalizing Out-of-Distribution
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(Lifeisgood, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams, args)
        self.swapper = SwappingMechanism(hparams['keeping'], hparams['descending'])
        self.network = nn.Sequential(self.featurizer, self.classifier)

        # Optimizer for the model (featurizer + classifier)
        self.model_optimizer = get_optimizer(params=self.network.parameters(), hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.model_optimizer, args=self.args)


    def update(self, minibatches, unlabeled=None):

        swapping_loss_weight = self.hparams['swapping_loss_weight']

        # Pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        # Obtain latent features
        z = self.featurizer(x)

        # Swapping operation
        swp_z, org_z = self.swapper.forward(z, y)

        # Concatenate original z and labels after sample selection
        org_z_batch = torch.cat(list(org_z.values()), dim=0)
        org_z_labels = torch.cat([torch.full((feature.size(0),), label) for label, feature in org_z.items()]).to(self.device)

        # check/debug
        # w = self.classifier.weight
        # a = torch.cat([(self.classifier.weight[label]*(swp_z[label] - org_z[label] )).abs() for label in org_z.keys()], dim=0 )
        # b = a.sum(dim=1)/len(y)
        # c = torch.norm(b, p=2)

        # Compute the swapping cross-entropy difference upper bound (SCEDUB)
        swp_diff_loss = torch.norm( torch.cat([ ( self.classifier.weight[label]*(swp_z[label] - org_z[label]) ).abs()
                                    for label in org_z.keys()], dim=0).sum(dim=1)/len(org_z_labels),  p=2)

        # ERM loss
        erm_loss = F.cross_entropy(self.classifier(org_z_batch), org_z_labels)

        # Calculate the model's objective
        objective = erm_loss + swapping_loss_weight * swp_diff_loss

        # Average the objective over the mini-batches
        objective /= len(minibatches)

        # Optimize the model parameters (minimization step)
        self.model_optimizer.zero_grad()
        objective.backward()
        self.model_optimizer.step()

        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': objective.item(),'loss_erm':erm_loss.item(), 'loss_swp':swp_diff_loss.item()}

    def predict(self, x):
        return self.network(x)



# Define the Swapping Mechanism
class SwappingMechanism(nn.Module):
    def __init__(self, keeping, descending):
        super(SwappingMechanism, self).__init__()
        self.keeping = keeping
        self.descending = descending
        self.num_samples = 2

    def forward(self, z, y):
        return self.swap_within_labels(self.group_features_by_label(z, y))

    def swap_within_labels(self, grouped_features):
        # Initialize swapping feature and original feature containers
        swp_z, org_z = {}, {}

        for label, features in grouped_features.items():
            num_features = features.size(0)
            if num_features < 2:
                swp_z[label], org_z[label] = features, features
                continue

            # Generate sample pairs
            assert self.num_samples % 2 == 0
            indices = [torch.randperm(num_features, device=features.device) for _ in range(self.num_samples)]
            z_samples = [features[idx] for idx in indices]
            z1, z2 = torch.cat(z_samples[:len(indices) // 2], dim=0), torch.cat(z_samples[len(indices) // 2:], dim=0)

            # Compute cosine similarity and sort
            similarity_scores = F.cosine_similarity(z1, z2, dim=1)
            # Switch between hard samples and easy samples by self.descending
            sorted_indices = similarity_scores.argsort(descending=self.descending)

            # Select indices for features to keep
            selected_indices = sorted_indices[:int(num_features * self.keeping)]

            z1_selected, z2_selected = z1[selected_indices], z2[selected_indices]

            # Swap features: all-element-swapping
            swp_z[label] = torch.cat([z2_selected, z1_selected], dim=0)
            org_z[label] = torch.cat([z1_selected, z2_selected], dim=0)

        return swp_z, org_z

    def group_features_by_label(self, features, labels):
        # Efficiently group features by label
        unique_labels, inverse_indices = labels.unique(return_inverse=True)
        grouped_features = {label.item(): features[inverse_indices == idx] for idx, label in enumerate(unique_labels)}
        return grouped_features