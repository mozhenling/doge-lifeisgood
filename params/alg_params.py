"""
Adapted from DomainBed hparams_registry.py
"""
import numpy as np
from params.seedutils import seed_hash
from datautils.diag_datasets import DATASETS as diag_DATASETS

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)

def _hparams(algorithm, dataset, random_seed, args):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    # we use the same small  hyperparameter search as the SMALL_DATASETS of DomainBed
    SMALL_DATASETS = diag_DATASETS

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert (name not in hparams)
        random_state = np.random.RandomState(seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    # _hparam('data_augmentation', True, lambda r: True)
    # _hparam('resnet18', False, lambda r: False)
    # _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    # if dataset in SMALL_DATASETS:
    #     _hparam('aug_num', 1, lambda r: int(r.choice([0, 1, 2, 3])))
    # else:
    #     _hparam('aug_num', args.aug_num, lambda r: args.aug_num)

    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    if args.steps is None:
        # log_min = np.log10(500)
        # log_max = np.log10(5000)
        # _hparam('steps', 1000, lambda r: round( 10**r.uniform(log_min, log_max)))
        _hparam('steps', 1000, lambda r:int(r.choice([500,  1000, 1500, 2000, 2500,
                                                      3000, 3500, 4000, 4500, 5000])))
    else:
        hparams['steps'] = (args.steps, args.steps)
    args.steps = hparams['steps'][0] if args.hparams_seed ==0 else hparams['steps'][1]


    if args.erm_loss in ['GLSLoss']:
        # when smooth_rate =0, it returns to standard CE loss
        _hparam('smooth_rate', 0.2, lambda r: r.choice([0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4,
                                                        -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0]))
    elif args.erm_loss in ['JSDLoss']:
        _hparam('d_weight', 0.5, lambda r: r.choice([0.1, 0.3, 0.5, 0.7, 0.9]))

    # -----------------------------------------------------------------------
    # --------------------- Lifeisgood
    elif algorithm in ["Lifeisgood"]:
        _hparam('descending', False, lambda r: bool(r.choice([True, False])))
        _hparam('keeping', 1,  lambda r: r.uniform(0.75, 1))
        _hparam('swapping_loss_weight', 1, lambda r: 10 ** r.uniform(-1.5, 0))

    # ------------------------------------------------------------------------
    elif algorithm in ["iDAG", "iDAGamp", "iDAGCMNIST"]:
        _hparam("dag_anneal_steps",int(args.steps * 0.1),
                lambda r: int(args.steps*r.uniform(0.1, 0.9)) )
        _hparam("temperature", 0.07, lambda r: r.uniform(0.07, 0.01))
        _hparam("ema_ratio",0.99, lambda r: r.uniform(0.99, 0.999))
        _hparam("lambda1",0.01, lambda r: r.uniform(0.01, 1.0))
        _hparam("lambda2",0.01, lambda r: r.uniform(0.01, 1.0))
        _hparam("h_tol",1e-8, lambda r:1e-8)
        _hparam("rho_max",100.0, lambda r: 10 ** r.uniform(1.0, 6.0))
        _hparam("rho",1.0, lambda r:1.0)
        _hparam("alpha",1.0, lambda r:1.0)
        _hparam("weight_nu",1.0, lambda r: r.uniform(1.0, 2.0))
        _hparam("weight_mu",1.0, lambda r: r.uniform(1.0, 2.0))
        _hparam("super_aug",False, lambda r:bool(False))


    elif algorithm == "SAGM":
        _hparam("rho", 0.05, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))
        _hparam("alpha",0.001, lambda r: r.choice([0.01, 0.02, 0.05, 0.1]))

    elif algorithm in ['VNE']:
        _hparam('vne_coef', 0.001,  lambda r: r.choice([ 0.0001, 0.0005, 0.001, 0.005]))

    elif algorithm in ['RIDG']:
        _hparam('momentum', 0.001, lambda r: 10**r.uniform(-4, -1))
        _hparam('ridg_reg', 0.01, lambda r: 10**r.uniform(-2, -1))

    elif algorithm == "CaSN":
        # _hparam('max_optimization_step', 1000, lambda r: int(r.choice([500, 1000])))
        # _hparam('if_adversarial', 'normal', lambda r: r.choice(['normal', 'adversarial']))
        _hparam('max_optimization_step', 1000, lambda r: int(r.choice([500, 1000])))
        _hparam('if_adversarial', 'normal', lambda r: r.choice(['normal']))
        _hparam('prior_type', 'conditional', lambda r: 'conditional')
        _hparam('int_lambda', 0.001, lambda r: r.choice([0.01, 0.001]))
        _hparam('int_reg', 0.01, lambda r: r.choice([0.1, 0.01]))
        _hparam('target_lambda', 0.1, lambda r: r.choice([0.1, 0.3]))
        _hparam('kl_lambda', 0.5, lambda r: r.choice([0.1, 0.3, 0.5]))
        # _hparam('mlp_width', 2048, lambda r: int(2 ** r.uniform(11)))
        _hparam('bias', 1, lambda r: int(r.choice([1])))
        # _hparam('mlp_depth', 3, lambda r: int(r.choice([3])))
        # _hparam('mlp_dropout', 0., lambda r: r.choice([0.]))

    elif algorithm in ['RDM']:
        _hparam('rdm_lambda', 5.0, lambda r: r.uniform(0.1, 10.0))
        _hparam('rdm_penalty_anneal_iters', int(args.steps * 0.1),
                lambda r: int(args.steps*r.uniform(0.1, 0.9)) )
        _hparam('variance_weight', 0.004, lambda r: r.uniform(0.001, 0.007))

    elif algorithm in ['IIB']:
        _hparam('lambda_beta', 1e-4, lambda r: 1e-3 ** r.uniform(-2, 2))
        _hparam('lambda_inv_risks', 10, lambda r: int(10 ** r.uniform(0, 2)))
        _hparam('enable_bn', True, lambda r: bool(r.choice([True, True])))

    # -------------------------------------------------------------------------

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', int(args.steps* 0.1),
                    lambda r: int(args.steps*r.uniform(0.1, 0.9)) )

    # -- Mixup
    elif algorithm in ["Mixup"]:
        _hparam('mixup_alpha', 0.2, lambda r: 10 ** r.uniform(-1, 1))

    elif algorithm in ["MMD", "CORAL", "CausIRL_CORAL", "CausIRL_MMD"]:
        _hparam('mmd_gamma', 1., lambda r: 10 ** r.uniform(-1, 1))

    # -- IRM
    if algorithm in ["IRM"] :
        _hparam('irm_penalty_weight', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', int(args.steps* 0.1),
                    lambda r: int(args.steps*r.uniform(0.1, 0.9)) )

    elif algorithm in ["IB_IRM"] :
        _hparam('irm_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', int(args.steps* 0.1),
                    lambda r: int(args.steps*r.uniform(0.1, 0.9)) )
        _hparam('ib_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', int(args.steps* 0.1),
                    lambda r: int(args.steps*r.uniform(0.1, 0.9)) )

    elif algorithm in ["CAD", "CondCAD"] :
        _hparam('bn_los_weight', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm in ["RSC"] :
        _hparam('rsc_f_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))


    elif algorithm in ["ANDMask"] :
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm in ["SANDMask"] :
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10 ** r.uniform(-3, 5))

    elif algorithm in ["DCT","DCTP"]:
        # all hparams are based on the original repo
        _hparam("margin", 0, lambda r: r.choice([0, 1]))
        _hparam("use_bnn", False, lambda r: False)
        _hparam("normfeat", False, lambda r: False)
        _hparam("test_leri", False, lambda r: False)
        # hparams["protolr"] = (5e-5, 5e-5)
        # hparams["mlp_width"] = (2048, lambda r: r.int(2 ** r.uniform(6, 10)))
        # hparams["mlp_depth"] = (2, lambda r: r.choice([3, 4, 5]))
        # hparams["mlp_dropout"] = (0.0, lambda r: r.choice([0.0, 0.1, 0.5]))
        # hparams["use_reidoptim"] = (True, True)
        # hparams["alpha"] = (1.0,1.0)

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.
    if dataset in SMALL_DATASETS and args.nets_base in ['diag_nets', 'LeNet']:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if dataset in SMALL_DATASETS:
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-8, -4))
    else:
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -2))

    if dataset in SMALL_DATASETS and args.nets_base in ['diag_nets', 'LeNet']:
        _hparam('batch_size', 32, lambda r: 32) # default 64 int(2 ** r.uniform(3, 9))
    elif  args.nets_base in ['Transformer']:
        _hparam('batch_size', 16, lambda r: 16)
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5.5)))


    return hparams


def default_hparams(args):
    algorithm = args.algorithm
    dataset = args.dataset
    random_seed = 0 # default
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, random_seed, args).items()}


def random_hparams(args):
    """
    Average on standard deviation calculations on results of hparams:
    1 Domainbed (take average/std across experiments):
        Standard error bars: While some DG literature reports error bars across seeds, randomness arising
        from model selection is often ignored. This is acceptable if the goal is best-versus-best comparison,
        but prohibits analyses concerning the model selection process itself. Instead, we repeat our entire
        study three times, making every random choice anew: hyperparameters, weight initializations, and
        dataset splits. Every number we report is a mean (and its standard error) over these repetitions
    2 Take average/std across trial_seeds for each set of hparams
    """
    algorithm = args.algorithm
    dataset = args.dataset
    if args.avg_std in ['e', 'experiments']:
        random_seed = seed_hash(args.hparams_seed, args.trial_seed)
    elif args.avg_std in ['t', 'trials']:
        random_seed = seed_hash(args.hparams_seed)
    else:
        raise ValueError('args.avg_std is not matched!')
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_seed, args).items()}