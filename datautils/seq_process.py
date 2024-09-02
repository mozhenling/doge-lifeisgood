
import copy
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
import datautils.seq_transforms as transforms
from params.seedutils import seed_everything_update

def dataset_transform(data, labels, input_shape, device, aug_num=0, trial_seed=0,
                      normalize_type = 'z-score', tran_matrix=None):
    # -- transformation compositions
    # aug_num: None, 'getone', 'gettwo'

    # initial transform
    transform = transforms.Compose([
        transforms.Retype(),
        transforms.Normalize(normalize_type),
        transforms.ToTensor(device, input_shape)
    ])
    data, labels = shuffle_datasets(data, labels)
    x = transform(data)
    if tran_matrix is not None:
        labels = re_label(labels, tran_matrix)
    y = torch.tensor(labels).view(-1).to(device).long()

    if aug_num >0:
        transform = transforms.Compose([
            transforms.Retype(), # np.float32
            transforms.RandomStretch(),
            transforms.RandomCrop(),
            transforms.RandomAddGaussian(),
            transforms.Normalize(normalize_type),
            transforms.ToTensor(device, input_shape)
        ])

        x, y = augmentation(x, y, data, labels, transform, trial_seed, aug_num = aug_num)

    return TensorDataset(x, y)

def re_label(labels, tran_matrix=None):
    """
    relabeling based on the tran_matrix
    inputs: a list of labels, and the N-by-N transition matrix in form of a dictionary (N is the num. of classes)
    return: a list of labels
    """
    if tran_matrix is None:
        return labels
    else:
        return [random.choices(population=list(tran_matrix[lab].keys()),
                           weights=list(tran_matrix[lab].values()),
                           k=1)[0] for lab in labels]

def augmentation(x_init, y_init, data, labels,transform, trial_seed, aug_num = 2):
    multi_views = [x_init]
    y_extend = [y_init]
    device = y_init.device
    for i in range(aug_num):
        # seed_everything_update(seed=trial_seed, remark='aug_idx'+str(i))
        multi_views.append(transform(data))
        y_extend.append(torch.tensor(labels).view(-1).to(device).long())
    # set seed back
    # seed_everything_update(seed=trial_seed)
    return torch.cat(multi_views), torch.cat(y_extend)

def shuffle_datasets(data, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return data, labels

def sig_segmentation(data, label, seg_len, start=0, stop=None):
    '''
    This function is mainly used to segment the raw 1-d signal into samples and labels
    using the sliding window to split the data
    '''
    data_seg = []
    lab_seg = []
    start_temp, stop_temp, stop = start, seg_len, stop if stop is not None else len(data)
    while stop_temp <= stop:
        sig = data[start_temp:stop_temp]
        sig = sig.reshape(-1, 1)
        data_seg.append( sig ) # z-score normalization
        lab_seg.append(label)
        start_temp += seg_len
        stop_temp += seg_len
    return data_seg, lab_seg


class ProbabilityGenerator:
    def __init__(self, n, specific_value):
        """
        Initialize the generator with the total number of probabilities and the specific value.

        Args:
        - n: int, the total number of probabilities.
        - specific_value: float, one specific probability value.
        """
        self.n = n
        self.count = 0
        self.specific_value = specific_value
        self.specific_index = None
        self.remaining_sum = 1.0 - self.specific_value
    def reset(self, specific_index=None):
        """Reset the generator to start generating a new set of probabilities."""
        self.remaining_sum = 1.0 - self.specific_value
        self.count = 0
        if specific_index is None:
            self.specific_index = np.random.randint(0, self.n)
        else:
            self.specific_index = specific_index

    def next(self, specific_index=None):
        """
        Generate one random probability that, along with the others generated, sums to 1.

        Returns:
        - probability: float, a random probability value.
        """
        if specific_index is None:
            self.reset()
        else:
            self.specific_index = specific_index
        if self.count >= self.n:
            self.reset(specific_index)

        if self.count == self.specific_index:
            probability = self.specific_value
        else:
            if self.count == self.n - 1 and self.specific_index != self.n - 1:
                probability = self.remaining_sum  # Return the remaining sum
            else:
                # Generate a random value from the Dirichlet distribution
                random_values = np.random.dirichlet(np.ones(self.n - self.count - 1))
                probability = random_values[0] * self.remaining_sum
                self.remaining_sum -= probability
        self.count += 1
        return probability