import os
import time
import torch
from params.train_params import get_args
from algorithms.trainer import train

if __name__ == "__main__":
    train_start_time = time.time()

    args = get_args()
    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"

    args.device = device

    ##############################
    #---------test
    args.algorithm = 'Lifeisgood'
    args.dataset = 'CWRU_Bearing'
    args.test_envs = [0]
    args.data_dir = r'C:\Users\MSI-NB\Desktop\Life-is_prj\0-exp\datasets'
    args.output_dir = r'C:\Users\MSI-NB\Desktop\Life-is_prj\0-exp\outputs'
    # args.data_dir = r'E:\0-CityU\0-doge-data\dogedata_Drone Fault'
    #############################

    args.output_dir = os.path.join(args.output_dir,
                                    args.dataset + '_'+ args.algorithm+ '_test_ids_' + str(args.test_envs))

    # Convert the namespace to a dictionary
    args_dict = vars(args)

    train_outs = train(args_dict=args_dict)

    train_stop_time = time.time()
    print('#' * 10, ' total_time = {:.2f} s '.format((train_stop_time - train_start_time)), '#' * 10)