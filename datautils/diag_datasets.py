"""
Domainbed Datasets
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
from datautils.seq_process import sig_segmentation, dataset_transform, ProbabilityGenerator

DATASETS = [
    'EPdM_Belt',
    'CWRU_Bearing',
    'KAIST_MotorSys',
    'UBFC_Motor',
    'UBFC_Stator_Current',
    'UBFC_Rotor_Vibration',
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        """
        __getitem__() is a magic method in Python, which when used in a class,
        allows its instances to use the [] (indexer) operators. Say x is an
        instance of this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i).
        """
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class KAIST_MotorSys(MultipleDomainDataset):
    ENVIRONMENTS = ['0Nm', '2Nm', '4Nm']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 300 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'motor_driving_system'

        self.class_name_list = ['Normal',
                                'BPFI_10',
                                'BPFO_10',
                                'Unbalance_1751mg',
                                'Misalign_03'
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0Nm', '2Nm', '4Nm']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset,
                                     'Xvib_'+ env_name + '_data.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['env_data'].reshape(-1, self.seg_len), data_dict['env_label'].reshape(-1)
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))



class EPdM_Belt(MultipleDomainDataset):
    ENVIRONMENTS = ['Env1', 'Env2', 'Env3']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1000
        self.instance_size = 180 # not applicable

        self.class_name_list = ['H-0',  # normal
                                'F-0',  # fault
                                'H-U',  # unbalance
                                ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = {'Env1': {'load': '110', 'speed': [19, 27]},
                             'Env2': {'load': '110', 'speed': [10, 18]},
                             'Env3': {'load': '150', 'speed': [1, 9]}}
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            data, labels = [], []
            for clc_idx, clc in enumerate(self.class_name_list):
                folder_path = os.path.join(args.data_dir,  args.dataset, 'Data '+self.environments[env_name]['load']+'-' + clc)
                files = [ str(file)+'.txt' for file in range(self.environments[env_name]['speed'][0],
                                                             self.environments[env_name]['speed'][1] + 1)]
                for file in files:
                    data_matrix = np.loadtxt(os.path.join(folder_path, file))
                    data_vib_1, data_vib_2 = data_matrix[:,1], data_matrix[:, 2]
                    data_vib_all = np.concatenate((data_vib_1, data_vib_2))
                    data_seg, lab_seg = sig_segmentation(data_vib_all, clc_idx, self.seg_len)
                    data.extend(data_seg)
                    labels.extend(lab_seg )
            data = np.array(data).squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

class CWRU_Bearing(MultipleDomainDataset):
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1000
        self.instance_size = 150

        self.class_name_list = ['normal',#0
                                'ball',  #1
                                'inner', #2
                                'outer'] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset, 'CWRU_DE_'+env_name+'_seg.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))


class UBFC_Motor(MultipleDomainDataset):
    ENVIRONMENTS =['0%', '25%', '50%']

    def __init__(self, args, hparams, sig_type='current', object = 'stator'):
        super().__init__()
        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        # self.instance_size = 270 # per class
        self.sig_type = sig_type
        self.object = object

        self.class_name_list = ['normal',  #0
                                'UBS10%',  #1 umbalanced supply, or Bearing fault
                                'UBS20%'   #2 umbalanced supply, or Broken bars
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0%', '25%', '50%']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir,  args.dataset, self.object+'_'+self.sig_type+
                                     '_load'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

class UBFC_Stator_Current(UBFC_Motor):
    # motor stator fault
    def __init__(self, args, hparams):
        super(UBFC_Stator_Current, self).__init__(args, hparams, sig_type='current')


class UBFC_Rotor_Vibration(UBFC_Motor):
    # motor rotor fault
    def __init__(self, args, hparams):
        super(UBFC_Rotor_Vibration, self).__init__(args, hparams, sig_type='vibration', object='rotor')

if __name__ == '__main__':

    root = r'..\datasets\CWRU'
    device = 'cuda'
    test_env_ids = [1]

