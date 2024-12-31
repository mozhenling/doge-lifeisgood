import os
import time
import torch
from params.train_params import get_args
from algorithms.trainer import train

if __name__ == "__main__":
    train_start_time = time.time()

    args = get_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.output_dir = os.path.join(args.output_dir,
                                    args.dataset + '_'+ args.algorithm+ '_test_ids_' + str(args.test_envs))

    # Convert the namespace to a dictionary
    args_dict = vars(args)

    train_outs = train(args_dict=args_dict)

    train_stop_time = time.time()
    print('#' * 10, ' total_time = {:.2f} s '.format((train_stop_time - train_start_time)), '#' * 10)