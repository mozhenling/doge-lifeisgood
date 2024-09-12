import torch
import torch.nn.functional as F
from algorithms.classes.ERM import ERM

class RDM(ERM):
    """
    @inproceedings{nguyen2024domain,
    title={Domain Generalisation via Risk Distribution Matching},
    author={Nguyen, Toan and Do, Kien and Duong, Bao and Nguyen, Thin},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={2790--2799},
    year={2024}
    }
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(RDM, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.register_buffer('update_count', torch.tensor([0]))

    def my_cdist(self, x1, x2):  # (bs)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    @staticmethod
    def _moment_penalty(p_mean, q_mean, p_var, q_var):
        return (p_mean - q_mean) ** 2 + (p_var - q_var) ** 2

    @staticmethod
    def _kl_penalty(p_mean, q_mean, p_var, q_var):
        return 0.5 * torch.log(q_var / p_var) + ((p_var) + (p_mean - q_mean) ** 2) / (2 * q_var) - 0.5

    def _js_penalty(self, p_mean, q_mean, p_var, q_var):
        m_mean = (p_mean + q_mean) / 2
        m_var = (p_var + q_var) / 4

        return self._kl_penalty(p_mean, m_mean, p_var, m_var) + self._kl_penalty(q_mean, m_mean, q_var, m_var)

    def update(self, minibatches, unlabeled=None, held_out_minibatches=None):
        matching_penalty_weight = (self.hparams['rdm_lambda'] if self.update_count
                                                                 >= self.hparams['rdm_penalty_anneal_iters'] else
                                   0.)

        variance_penalty_weight = (self.hparams['variance_weight'] if self.update_count
                                                                      >= self.hparams['rdm_penalty_anneal_iters'] else
                                   0.)

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        losses = torch.zeros(len(minibatches)).cuda()
        all_logits_idx = 0
        all_confs_envs = None

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)

            nll = F.cross_entropy(logits, y, reduction="none").unsqueeze(0)

            if all_confs_envs is None:
                all_confs_envs = nll
            else:
                all_confs_envs = torch.cat([all_confs_envs, nll], dim=0)

        erm_loss = losses.mean()

        ## squeeze the risks
        all_confs_envs = torch.squeeze(all_confs_envs)  # (3, bs, 7) or (3, bs)

        ## find the worst domain
        worst_env_idx = torch.argmax(torch.clone(losses))
        all_confs_worst_env = all_confs_envs[worst_env_idx]  # (bs, 7)

        ## flatten the risk
        all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)

        matching_penalty = self.mmd(all_confs_worst_env_flat.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1))

        ## variance penalty
        variance_penalty = torch.var(all_confs_all_envs_flat)
        variance_penalty += torch.var(all_confs_worst_env_flat)

        total_loss = erm_loss + matching_penalty_weight * matching_penalty + variance_penalty_weight * variance_penalty

        if self.update_count == self.hparams['rdm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        # Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'total_loss': total_loss.item()}