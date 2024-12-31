
import argparse
import numpy as np
import pandas as pd
import math
import os
import ast
from scipy.interpolate import griddata
from visualutils.plots import hparams_plot3D
from oututils import  selections, print_outs

def plot_hparams3D(file, hparam_pair, finer_num=0, interpo_method = 'cubic',z_key = 'test_avg', **kwargs):
    x_key, x_base = hparam_pair[0]['name'], hparam_pair[0]['base']
    y_key, y_base = hparam_pair[1]['name'], hparam_pair[1]['base']

    df = pd.read_csv(file)
    # Sort Descending by test
    df_sort = df.sort_values(by=[x_key,y_key], ascending=[True, True])
    # create meshgrid
    # Restore the n-by-n NumPy array from the DataFrame
    n = int(np.sqrt(len(df_sort)))

    if x_base is not None:
        x =np.array([math.log(x, x_base) for x in df_sort[x_key]])
    else:
        x = df_sort[x_key].to_numpy()
    if y_base is not None:
        y = np.array([math.log(y, y_base) for y in df_sort[y_key]])
    else:
        y = df_sort[y_key].to_numpy()
    z = df_sort[z_key].to_numpy()

    X, Y, Z = x.reshape(n, n), y.reshape(n, n), z.reshape(n, n)
    if finer_num is not None:
        if finer_num > n:
            # Flatten the data for griddata
            points = np.array([X.flatten(), Y.flatten()]).T
            values = Z.flatten()
            # Define a finer grid
            xi = np.linspace(X.min(), X.max(), finer_num)
            yi = np.linspace(Y.min(), Y.max(), finer_num)
            Xi, Yi = np.meshgrid(xi, yi)
            # Interpolate the Z values on the finer grid
            Zi = griddata(points, values, (Xi, Yi), method=interpo_method)
            X, Y, Z = Xi, Yi, Zi
        else:
            raise ValueError('finner_num is not larger than the original number of points. Please set a higher numer of finer_num!')

    hparams_plot3D(X, Y, Z, **kwargs)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str,
                           default=r'./outputs/hparams_outs')
    parser.add_argument("--out_dir", type=str,
                           default=r'./outputs/hparams_outs')
    parser.add_argument('--task', type=str,  choices=['p','plot', 'n','numeric'],
                           default="n",
                           help='plot means hparams searching plot; numeric means hparams searching numeric values.')
    parser.add_argument("--selections", nargs='+', type=str,
                           default=['IID'], help='IID, OneOut, Oracle')
    parser.add_argument('--hparam_pair', type=ast.literal_eval,
                           default="[{'name':'keeping', 'start':0.75, 'stop':1, 'num':10, 'base':None},{'name':'swapping_loss_weight', 'start':-1.5, 'stop':0, 'num':10, 'base':10}]")
    parser.add_argument('--read_name', type=str,
                           default="UBFC_Stator_Current_Lifeisgood_Env2_training-domain validation set_sorted_val.csv")
    parser.add_argument('--save_name', type=str,
                           default="UBFC_Stator_Current_Lifeisgood_Env2_training-domain validation set.png")
    parser.add_argument('--finer_num', type=int,
                        default=None)
    parser.add_argument('--dataset', type=str,
                           default="UBFC_Stator_Current")
    parser.add_argument('--algorithm', type=str,
                           default="Lifeisgood")
    parser.add_argument('--sub_algorithm', type=str,
                           default=None)
    parser.add_argument('--erm_loss', type=str,
                           default='CELoss')
    parser.add_argument('--test_env', type=int,
                           default=2)
    args = parser.parse_args()

    # args.test_env = 0
    if args.finer_num is not None:
        args.save_name = f"x{round(args.finer_num / args.hparam_pair[0]['num'] , 1)}"+'_'+args.save_name
    if args.task in ['p', 'plot']:
        plot_hparams3D(file =os.path.join(args.input_dir, args.read_name), hparam_pair=args.hparam_pair, fontsize=14,

                       finer_num=args.finer_num, x_decimals=1, y_decimals=1, z_decimals=1,
                       #azim=azim, elev = elev,
                       save_dir =os.path.join(args.out_dir, args.save_name) )
    elif args.task in ['n', 'numeric']:
        SELECTION_METHODS = selections.get_methods(args.selections)
        for selection_method in SELECTION_METHODS:
            print_outs.print_hparams_grid_results(args, selection_method)
    else:
        raise NotImplementedError('Task was not found!')