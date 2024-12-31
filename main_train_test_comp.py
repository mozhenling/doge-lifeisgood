import argparse

import numpy as np

from visualutils.plots import  plotTSNE, plot_confusion_matrix, multi_y_curves
import os
import torch
from params.train_params import get_args as train_args
from algorithms.trainer import train
import json

def get_alg_one_env_records(args):
    records = []
    file = os.path.join(args.input_dir, args.read_name)
    with open(file, "r") as f:
        for line in f:
            records.append(json.loads(line[:-1]))

    return records
# keys=('step', 'loss_erm', 'loss_swp', 'env0_out_acc')
def eg_plot_train_Lifeisgood_curves(args, envs_num=3, **kwargs):
    #--customized losses
    loss_keys = ['loss_erm', 'loss_swp']

    #-- define varaibles of interest
    env_idx = [i for i in range(envs_num)]

    env_train_idx = [j for j in env_idx if j not in args.test_envs]
    test_acc_out_keys = ['env{}_out_acc'.format(i) for i in args.test_envs]
    test_acc_in_keys = ['env{}_in_acc'.format(i) for i in args.test_envs]
    val_acc_keys = ['env{}_out_acc'.format(j) for j in env_train_idx]
    train_acc_keys = ['env{}_in_acc'.format(j) for j in env_train_idx]

    #-- load records
    acc_save_name = os.path.join(args.save_dir, 'acc_'+args.save_name)
    loss_save_name = os.path.join(args.save_dir, 'loss_'+args.save_name)
    records = get_alg_one_env_records(args)

    #-- select records
    keys_all =['step']+ test_acc_out_keys + test_acc_in_keys + val_acc_keys + train_acc_keys + loss_keys
    records_select = {k:[r_per_step[k] for r_per_step in records] for k in keys_all}

    #-- get x: step
    x = np.array(records_select['step']) / 100
    x_label = 'step/100'

    #-- process acc
    test_acc_out = np.mean([records_select[k] for k in test_acc_out_keys], axis=0)
    test_acc_in = np.mean([records_select[k] for k in test_acc_in_keys], axis=0)
    test_acc = args.holdout_fraction*test_acc_out + (1-args.holdout_fraction)*test_acc_in
    val_acc = np.mean([records_select[k] for k in val_acc_keys], axis=0)
    train_acc = np.mean([records_select[k] for k in train_acc_keys], axis=0)

    acc_y_data = [ val_acc, test_acc]
    acc_xy_labels = ((x_label, 'Val_acc'), (x_label, 'Test_acc'))
    acc_smoothing_params = {'window_size': 51} # smoothing window size for all y data
    # acc_smoothing_list = [{'window_size': 51}, {'window_size': 51} ] # smoothing window size for all y data

    #-- process losses
    loss_y_data = [records_select[k] for k in loss_keys]
    loss_xy_labels = ((x_label, 'SCE loss'), (x_label, 'SCEDUB'))
    # losses have been averaged by the checkpoint frequency
    # loss_smoothing_list = [{'window_size': None}, {'window_size': None}]

    # multi_y_curves(x, loss_y_data+acc_y_data , loss_xy_labels+acc_xy_labels,
    #                smoothing='svg',
    #                y0_color='tab:green',
    #                y_extra_colors=['tab:blue','tab:purple','tab:orange'],
    #                smoothing_params=[{'window_size': None}, {'window_size': None},{'window_size': 51}, {'window_size': 51}],
    #                save_dir=os.path.join(args.save_dir, args.save_name), **kwargs)

    #-- dynanmics
    multi_y_curves(x, acc_y_data , acc_xy_labels, y0_color='tab:orange', y_extra_colors=['tab:blue'],
                   smoothing='svg', smoothing_params=acc_smoothing_params,
                   save_dir=acc_save_name, **kwargs)
    # losses have been averaged by the checkpoint frequency
    multi_y_curves(x, loss_y_data, loss_xy_labels, y0_color='tab:green', y_extra_colors=['tab:purple'],
                   smoothing='svg', smoothing_params=None,#loss_smoothing_params,
                   save_dir=loss_save_name, **kwargs)

    return


def get_model_saved_dicts(args):
    file = os.path.join(args.input_dir, args.read_name)

    dict_init = torch.load(file)
    algorithm_dict = dict_init["model_dict"]
    args_dict = dict_init["args"]
    args_dict['hparams'] = dict_init["model_hparams"]

    #-- adjust the batch size to have a better plot
    if args.batch_size is not None:
        args_dict['hparams']['batch_size']=args.batch_size

    return args_dict, algorithm_dict

def xy_dicts_from_saved_model(args, data_level_comp):
    test_xy_dict = {}
    train_xy_dict = {}
    args_dict, algorithm_dict = get_model_saved_dicts(args)
    algorithm, dataset, train_loaders_out, test_loaders_out = train(args_dict, algorithm_dict=algorithm_dict, algorithm_dataset_back=True)
    y_test, x_test = torch.tensor([]).to(algorithm.device), torch.tensor([]).to(algorithm.device)
    y_train, x_train = torch.tensor([]).to(algorithm.device), torch.tensor([]).to(algorithm.device)

    test_batch_enough = False
    test_batch_count = 0
    for loader in test_loaders_out:
        algorithm.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(algorithm.device)
                y = y.to(algorithm.device)
                y_test = torch.cat((y_test, y))
                x_test = torch.cat((x_test, x)) if data_level_comp else torch.cat((x_test, algorithm.featurizer(x)))
                test_batch_count +=1
                if args.batch_num is not None:
                    if test_batch_count >=args.batch_num:
                        test_batch_enough = True
                        break
            if test_batch_enough:
                break

    train_batch_enough = False
    train_batch_count = 0
    for loader in train_loaders_out:
        algorithm.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(algorithm.device)
                y = y.to(algorithm.device)
                y_train = torch.cat((y_train, y))
                x_train = torch.cat((x_train, x)) if data_level_comp else torch.cat((x_train,algorithm.featurizer(x)))
                train_batch_count += 1
                if args.batch_num is not None:
                    if train_batch_count >= args.batch_num:
                        train_batch_enough = True
                        break
            if train_batch_enough:
                break

    test_xy_dict['x'] = x_test.cpu().numpy()
    test_xy_dict['y'] = y_test.cpu().int().numpy()
    train_xy_dict['x'] = x_train.cpu().numpy()

    # shuffle train. data
    idx_train=torch.randperm(len(y_train))
    train_xy_dict['y'] = y_train[idx_train].cpu().int().numpy()
    train_xy_dict['x']= train_xy_dict['x'][idx_train]
    #-- keep same len with the test (usually, train. data is larger than test data)
    train_xy_dict['x'] = train_xy_dict['x'][:len(test_xy_dict['y'])]
    train_xy_dict['y']=train_xy_dict['y'][:len(test_xy_dict['y'])]
    return train_xy_dict, test_xy_dict

def confusion_matrix_from_saved_model(args, y_true=None, y_pred=None, label_names=None):
    save_dir = os.path.join(args.save_dir, args.save_name)
    if y_true is not None and y_pred is not None and label_names is not None:
        plot_confusion_matrix(y_true, y_pred, label_names, save_dir=save_dir)
    else:
        args_dict, algorithm_dict = get_model_saved_dicts(args)
        algorithm, dataset, train_loaders_out, test_loaders_out  = train(args_dict, algorithm_dict=algorithm_dict, algorithm_dataset_back=True)
        if label_names is None:
            label_names = [i for i in dataset.class_list]
        true_y, pred_y = torch.tensor([]).to(algorithm.device),torch.tensor([]).to(algorithm.device)
        for loader in test_loaders_out:
            algorithm.eval()
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(algorithm.device)
                    y = y.to(algorithm.device)
                    logits = algorithm.predict(x)
                    true_y=torch.cat((true_y, y))
                    pred_y=torch.cat((pred_y, logits.argmax(1)))
        true_y = true_y.cpu().numpy()
        pred_y = pred_y.detach().cpu().numpy()
        plot_confusion_matrix(true_y, pred_y, label_names, save_dir=save_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="plot figures")
    parser.add_argument("--input_dir", type=str, default=r'./outputs/hparams_outs')
    parser.add_argument("--save_dir", type=str, default=r'./outputs/plots_outs')
    parser.add_argument('--task', type=str, default="metrics",
                        choices=['tsne', 'dynamics', 'confusion'],
                        help='tsne: T-SNE plot;'
                             'dynamics: training/test/learning curves.'
                             'confusion: confusion matrix')
    parser.add_argument('--read_name', type=str, default="PU_bearing")
    parser.add_argument('--save_name', type=str, default="PU_bearing")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_num', type=int, default=None, help='num of batches; If none, use all available')
    parser.add_argument('--data_level_comp', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default="Lifeisgood")
    parser.add_argument('--dataset', type=str, default="PU_bearing")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])


    args = parser.parse_args()
    train_args_dict = vars(train_args)


    os.makedirs(args.save_dir, exist_ok=True)

    if args.task in ['tsne']:
        train_xy_dict, test_xy_dict = xy_dicts_from_saved_model(args, data_level_comp=args.data_level_comp)
        plotTSNE(train_xy_dict, test_xy_dict, fontsize=20, figsize = (8, 6),
                 marker_size=150, save_dir =os.path.join(args.save_dir, args.save_name))
    elif args.task in ['dynamics']:
        eg_plot_train_Lifeisgood_curves(args)
    elif args.task in ['confusion']:
        confusion_matrix_from_saved_model(args)
    else:
        raise NotImplementedError
