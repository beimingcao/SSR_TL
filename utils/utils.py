import os
import yaml
import glob
import numpy as np
import scipy.io as sio
import librosa

from utils.IO_func import read_file_list

def prepare_Haskins_lists(args):

    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    exp_type = config['experimental_setup']['experiment_type']  
    data_path = config['corpus']['path']
    fileset_path = os.path.join(data_path, 'filesets')
    spk_list = config['data_setup']['spk_list']
    num_exp = len(spk_list)
    
    exp_train_lists = {}
    exp_valid_lists = {}
    exp_test_lists = {}
    if exp_type == 'SD':
        for i in range(len(spk_list)):
            spk_fileset_path = os.path.join(fileset_path, spk_list[i])
            exp_train_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
            exp_valid_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            exp_test_lists[i] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp'))
     
    elif exp_type == 'SI':
        for i in range(len(spk_list)):
            train_spk_list = spk_list.copy()
            train_spk_list.remove(spk_list[i])
            idx = 0
            train_lists, valid_lists = [], []
            for train_spk in train_spk_list:
                spk_fileset_path = os.path.join(fileset_path, train_spk)
                if idx == 0:
                    train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                else:
                    train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                    valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
                idx += 1
            test_lists = read_file_list(os.path.join(os.path.join(fileset_path, spk_list[i]), 'test_id_list.scp'))

            exp_train_lists[i] = train_lists
            exp_valid_lists[i] = valid_lists
            exp_test_lists[i] = test_lists

    elif exp_type == 'SA':
        idx = 0     
        for train_spk in spk_list:
            spk_fileset_path = os.path.join(fileset_path, train_spk)
            if idx == 0:
                train_lists = read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))
            else:
                train_lists = train_lists + read_file_list(os.path.join(spk_fileset_path, 'train_id_list.scp'))
                valid_lists = valid_lists + read_file_list(os.path.join(spk_fileset_path, 'valid_id_list.scp'))

            exp_train_lists[idx] = train_lists
            exp_valid_lists[idx] = valid_lists            
            exp_test_lists[idx] = read_file_list(os.path.join(spk_fileset_path, 'test_id_list.scp')) 
            idx += 1           
    else:
        raise ValueError('Unrecognized experiment type')

    return exp_train_lists, exp_valid_lists, exp_test_lists

