# coding=utf-8

from algorithms.classes.Lifeisgood import Lifeisgood

from algorithms.classes.MixStyle import MixStyle
from algorithms.classes.SCIRM import SCIRM
from algorithms.classes.MDGCML import MDGCML
from algorithms.classes.iDAG import iDAG
from algorithms.classes.SAGM import SAGM
from algorithms.classes.VNE import VNE
from algorithms.classes.RIDG import RIDG
from algorithms.classes.CaSN import CaSN

from algorithms.classes.RDM import RDM
from algorithms.classes.IIB import IIB
from algorithms.classes.ERM import ERM
from algorithms.classes.Mixup import Mixup
from algorithms.classes.VREx import VREx

from algorithms.classes.AbstractCausIRL import CausIRL_MMD
from algorithms.classes.IB_IRM import IB_IRM
from algorithms.classes.IRM import IRM
from algorithms.classes.AbstractCAD import  CondCAD
from algorithms.classes.SelfReg import SelfReg

ALGORITHMS = [
    'Lifeisgood',

    'MixStyle',
    'SCIRM',
    'MDGCML',
    'iDAG',
    'SAGM', #wang2023sharpness
    'VNE', #Kim_2023_CVPR
    'RIDG',
    'IIB',

    'CaSN',
    'RDM',
    'ERM',
    'Mixup',
    'VREx',

    'CausIRL_MMD',
    'IB_IRM',
    'IRM',
    'CondCAD',
    'SelfReg'
]

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

