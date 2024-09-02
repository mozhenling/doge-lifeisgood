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
    'CWRU_Bearing_NsyLabBase',
    'CWRU_Bearing_sym02',
    'CWRU_Bearing_asym02',

    'KAIST_Drone',
    'KAIST_MotorSys',

    'UBFC_Gear',
    'EPdM_Belt',

    'UO_Bearing',
    'PU_Bearing',

    'CU_Actuator',
    'CWRU_Bearing',

    'PHM_Gear',
    'UBFC_Motor',
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

class KAIST_Drone(MultipleDomainDataset):
    ENVIRONMENTS = ['ConSite', 'Pond', 'Hill']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 2000
        self.instance_size = 800 # per class
        self.sig_type = 'sound'
        self.obj_type = 'motor_driving_system'

        self.class_name_list = ['N','MF1','PC1']

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['drone_A_ConstructionSite', 'drone_A_DuckPond', 'drone_A_EoeunHill']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset, env_name + '.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'].reshape(-1, self.seg_len), data_dict['lab'].reshape(-1)
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

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


class UBFC_Gear(MultipleDomainDataset):
    ENVIRONMENTS = ['25hz_75%', '35hz_50%', '45hz_25%']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 200 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'gear'

        self.class_name_list = ['Normal',#0
                                'Broken Tooth',  #1
                                'Surface Damage'  #2
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['25hz_75%', '35hz_50%', '45hz_25%']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset,  self.obj_type+'_'+self.sig_type+
                                     '_speed-load_'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
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

class PU_Bearing(MultipleDomainDataset):
    ENVIRONMENTS = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1000
        self.instance_size = 240 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'bearing'

        self.class_name_list = ['H',  #0
                                'I',  #1
                                'O'   #2
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir,  args.dataset, env_name+'.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

class UO_Bearing(MultipleDomainDataset):
    ENVIRONMENTS = ['0-3s', '3-6s', '6-9s']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 580 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'bearing'

        self.class_name_list = ['H',#0
                                'I',  #1
                                'O'  #2
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0-3s', '3-6s', '6-9s']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir,  args.dataset, self.obj_type+'_'+self.sig_type+
                                     '_speed_'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

class CU_Actuator(MultipleDomainDataset):
    ENVIRONMENTS = ['20kg', '40kg', 'neg40kg']

    def __init__(self, args, hparams):
        super().__init__()
        if args.data_dir is None:
            raise ValueError('Data directory not specified!')
        # -----------------------------------------------------------
        # self.filename = {'p1': None,      # part 1: class type
        #                  'p2': ('sin', 'trap'), # part 2: motion profile
        #                  'p3': ('1st', '2nd'),  # part 3: severity for back and lub faults
        #                  'p4': ('3rd', '4th'), # part 4: severity for point fault
        #                  'p5': (str(i+1) for i in range(10))} # part 5: repeat
        # -----------------------------------------------------------
        # -- sample points
        self.seg_len = 1000   # len of each sample
        self.len_total = 4000
        self.instance_size = 160  # per class
        self.class_name_dict = {'back':{'Backlash1.mat':'1st', 'Backlash2.mat':'2nd'},
                                'lub':{'LackLubrication1.mat':'1st', 'LackLubrication2.mat':'2nd'},
                                'point':{'Spalling3.mat':'3rd', 'Spalling4.mat':'4th'}}

        self.class_list = [i for i in range(len(self.class_name_dict))]
        self.environments = ['20kg', '40kg', 'neg40kg']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num

            data, labels = self.get_samples(args.data_dir, args.dataset,  env_name)
            self.datasets.append(dataset_transform(np.array(data), labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

    def get_samples(self, root,dataset, env_name):
        data_seg_all = []
        label_seg_all = []
        file_idxs = [str(i+1) for i in range(10)]
        for cls, lab in zip(self.class_name_dict, self.class_list):
            for cls_file in self.class_name_dict[cls]:
                file_path = os.path.join(root,dataset, cls_file)
                data_dict = loadmat(file_path)
                for motion in ['sin', 'trap']:
                    severity = self.class_name_dict[cls][cls_file]
                    for file_id in file_idxs:
                        file_str = cls + motion+ severity + env_name + file_id
                        if file_str == 'backtrap1st40kg2': # cope with missing values
                            data_dict['backtrap1st40kg2'] = (data_dict['backtrap1st40kg1'] + data_dict['backtrap1st40kg3']) / 2
                        data = np.concatenate((data_dict[file_str][:,1], data_dict[file_str][:,2]), axis=0)
                        data_temp, label_temp = sig_segmentation(data, label=lab, seg_len = self.seg_len,
                                                                 start = 0, stop=self.len_total)
                        data_seg_all.extend(data_temp)
                        label_seg_all.extend(label_temp)
        return data_seg_all, label_seg_all


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


class CWRU_Bearing_NsyLabBase(MultipleDomainDataset):
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']

    def __init__(self, args, hparams, label_noise_type='nsfree', label_noise_rate=0.0, tran_matrix=None):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1000
        self.instance_size = 150
        self.base_dataset = 'CWRU_Bearing' # name of the base dataset folder
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
        self.prob_generator = ProbabilityGenerator(self.num_classes, 1 - label_noise_rate)
        if tran_matrix is None:
            # -----------------------------------------------------------
            # -- label noise transition relations
            assert 0.0<=label_noise_rate and label_noise_rate <= 0.5  # clean labels should be dominant
            if label_noise_type in ['sym']:
                # -- symmetric transition matrix
                self.tran_matrix = {
                    i: {j: 1 - label_noise_rate if j == i else label_noise_rate * 1 / (self.num_classes - 1) for j in
                        range(self.num_classes)}
                    for i in range(self.num_classes)}
            elif label_noise_type in ['asym']:
                # -- asymmetric noise transition matirx
                self.tran_matrix = {
                    i: {j: self.prob_generator.next(i) for j in range(self.num_classes )
                        }
                    for i in range(self.num_classes)}
            elif label_noise_type in ['nsfree']:  # noise free
                self.tran_matrix = None
            else:
                raise ValueError('label_noise_type is incorrect')
        else:
            self.tran_matrix = tran_matrix

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            tran_matrix_temp = None if env_id in args.test_envs else self.tran_matrix
            file_path = os.path.join(args.data_dir, self.base_dataset, 'CWRU_DE_'+env_name+'_seg.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed,tran_matrix=tran_matrix_temp))


class CWRU_Bearing_sym02(CWRU_Bearing_NsyLabBase):
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']
    def __init__(self, args, hparams):
        self.label_noise_type = 'sym'
        self.label_noise_rate = 0.2
        self.tran_matrix =  {
                    0: {0: 0.8,   1: 0.2/3, 2: 0.2/3, 3: 0.2/3},
                    1: {0: 0.2/3, 1: 0.8,   2: 0.2/3, 3: 0.2/3},
                    2: {0: 0.2/3, 1: 0.2/3, 2: 0.8,   3: 0.2/3},
                    3: {0: 0.2/3, 1: 0.2/3, 2: 0.2/3, 3: 0.8}
                }
        super().__init__(args, hparams, self.label_noise_type, self.label_noise_rate, tran_matrix=self.tran_matrix)

class CWRU_Bearing_asym02(CWRU_Bearing_NsyLabBase):
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']
    def __init__(self, args, hparams):
        self.label_noise_type = 'asym'
        self.label_noise_rate = 0.2
        self.tran_matrix = {
                0: {0: 0.8,  1: 0.1,  2: 0.05, 3: 0.05},
                1: {0: 0.2,  1: 0.8,  2: 0.0,  3: 0.0},
                2: {0: 0.03, 1: 0.07, 2: 0.8,  3: 0.1},
                3: {0: 0.1,  1: 0.1,  2: 0.0,  3: 0.8}
            }
        super().__init__(args, hparams, self.label_noise_type, self.label_noise_rate,tran_matrix=self.tran_matrix)

class UBFC_Motor(MultipleDomainDataset):
    ENVIRONMENTS =['0%', '25%', '50%']

    def __init__(self, args, hparams):
        super().__init__()
        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 270 # per class
        self.sig_type = 'current'

        self.class_name_list = ['normal',#0
                                'UBS10%',  #1 umbalanced supply
                                'UBS20%'  #2 umbalanced supply
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
            file_path = os.path.join(args.data_dir,  args.dataset, 'stator_'+self.sig_type+
                                     '_load'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))



class PHM_Gear(MultipleDomainDataset):
    ENVIRONMENTS = ['30hz_Low_1', '35hz_Low_1', '40hz_Low_1']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 520 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'gear'

        self.class_name_list = ['s1',#0
                                's2',  #1
                                's3' ] #2

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['30hz_Low_1', '35hz_Low_1', '40hz_Low_1']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset,  self.obj_type+'_'+self.sig_type+
                                     '_speed_'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))



if __name__ == '__main__':

    root = r'..\datasets\CWRU'
    device = 'cuda'
    test_env_ids = [1]

