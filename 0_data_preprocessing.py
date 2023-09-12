import yaml
import os
import glob
from shutil import copyfile
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data

def setup(args):

    exp_path = args.exp_dir
    buff_path = args.buff_dir   
    config_path = args.conf_dir
    data_path = args.data_dir

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    if not os.path.exists(buff_path):
        os.makedirs(buff_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    conf_dst = os.path.join(buff_path, os.path.basename(config_path))        
    copyfile(config_path, conf_dst)
    
#    model_dst = os.path.join(buff_path, 'models.py')
#    copyfile('utils/models.py', model_dst)

def data_preprocessing(args):
    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    data_out_path = args.data_dir

    raw_data_path = os.path.join(config['corpus']['path'], config['corpus']['name'])
    label_file_path = os.path.join(config['corpus']['path'], config['corpus']['name'], config['corpus']['label_file'])
    SPK_LIST = config['data_setup']['spk_list']

    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim']  

    with open(label_file_path, 'r') as f:
        lines = f.readlines()

    TEXT_LABEL = {}
    i = 0
    for line in lines:
        if 'BLOCK' in line.split('\t')[0]:
            i += 1
            block = 'B' + format(i, '02d')
            print(block)
        else:
            SID, label = line.strip().split('\t')
            TEXT_LABEL[(block, SID)] = label

   # print(TEXT_LABEL['B01', 'S01'])

    for SPK in SPK_LIST:

        spk_path = os.path.join(raw_data_path, SPK, 'data')
        SPK_data_list = glob.glob(os.path.join(spk_path, '*.mat'))
        SPK_data_list.sort()

        SPK_out_path = os.path.join(data_out_path, SPK)
        if not os.path.exists(SPK_out_path):
            os.makedirs(SPK_out_path)

        for data_path in SPK_data_list:
            file_id = os.path.basename(data_path).split('.')[0]

            if len(file_id) == 17:
                blk, sid = file_id.split('_')[1], file_id.split('_')[2]
                EMA, WAV, fs_ema, fs_wav = load_Haskins_ATS_data(data_path, file_id, sel_sensors, sel_dim)
                blk, sid = file_id.split('_')[1], file_id.split('_')[2]
                txt = TEXT_LABEL[(blk, sid)]
                print(blk, sid, txt)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/conf.yaml')
    parser.add_argument('--data_dir', default = 'data')
    parser.add_argument('--exp_dir', default = 'experiments')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
  #  setup(args)
    data_preprocessing(args)
