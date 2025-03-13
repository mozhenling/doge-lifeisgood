import torch
import torch.nn.functional as F
import torch.autograd as autograd
from algorithms.classes.ERM import ERM

class SCIRM(ERM):
    """
    Sparsity Constraint Invariant Risk Minimization (vanilla version, i.e., without auto-weight balancing)

    Ref.:
        [1] Z. Mo, Z. Zhang, Q. Miao, and K.-L. Tsui, “Sparsity-Constrained Invariant Risk Minimization
            for Domain Generalization With Application to Machinery Fault Diagnosis Modeling,” IEEE Transactions on
            Cybernetics, vol. 54, no. 3, pp. 1547–1559, Dec. 2022, doi: 10.1109/TCYB.2022.3223783.

    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(SCIRM, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0]))

    def irm_penalty(self, logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = self.erm_loss(logits[::2] * scale, y[::2])
        loss_2 = self.erm_loss(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2) # Take the sum since this term is often small.
        return result

    def spar_penalty(self, z, p=1, q=2, eps=10**-16):
        # z is of dimension [batch_size, feature_size]
        return (torch.linalg.norm(z.float(), dim=1, ord=p) / (torch.linalg.norm(z.float(), dim=1, ord=q) + eps)).mean()

    def update(self, minibatches, unlabeled=None):
        # Pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        # Obtain latent features and logits
        z = self.featurizer(x)
        logits = self.classifier(z)

        # Calculate loss components
        loss_erm = F.cross_entropy(logits, y)
        loss_irm = self.irm_penalty(logits, y)
        loss_spar = self.spar_penalty(z)

        # Final loss
        loss = loss_erm + self.hparams["weight_irm"]*loss_irm + self.hparams["weight_spar"]*loss_spar

        # Average the objective over the mini-batches
        loss /= len(minibatches)

        # Optimize the model parameters (minimization step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': loss.item(), 'loss_erm': loss_erm.item(),
                'loss_irm': loss_irm.item(), 'loss_spar': loss_spar.item()}

