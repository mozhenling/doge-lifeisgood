#!/bin/bash

echo '------- Launch Algorithm --------'

python -m main_sweep\
       --command r\
       --command_launcher plain\
       --n_trials 1\
       --n_hparams_from 0\
       --n_hparams 1\
       --datasets UBFC_Stator_Current\
       --data_dir=./datasets/\
       --algorithms Lifeisgood\
       --sweep_test_envs "[[0]]"\
       --steps 10\
       --checkpoint_freq 1\
       --hparams_search_mode g\
       --avg_std t\
       --hparams_grid_bases "[{'name':'keeping', 'start':0.75, 'stop':1, 'num':5, 'base':None},
                            {'name':'swapping_loss_weight', 'start':-1.5, 'stop':0, 'num':5, 'base':10}]"\
       --skip_model_save

echo '------- Algorithm Finished --------'
