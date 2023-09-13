import yaml
import os
import glob
from shutil import copyfile
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_SSR_data, save_phone_label, save_word_label
from scipy.io import wavfile
import nltk

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

def word2phone(word_seq):
    nltk.download('cmudict')
    CMU_lexicon = nltk.corpus.cmudict.dict() # download and setup CMU dictionary
    punctuations_to_remove = ',?.!/;:~'

    for char in word_seq:
        if char in punctuations_to_remove:
            word_seq = word_seq.replace(char,'')

    word_seq = word_seq.lower().split()
    _phone_seq = []
    for word in word_seq:
        _phone_seq.append(CMU_lexicon[word][0])

    phone_seq = [item for sublist in _phone_seq for item in sublist]
    # remove stress
    phone_seq_no_stress = []
    for _phone in phone_seq:
        phone = ''.join([i for i in _phone if not i.isdigit()])
        phone_seq_no_stress.append(phone)
    return phone_seq_no_stress



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
        else:
            SID, label = line.strip().split('\t')
            TEXT_LABEL[(block, SID)] = label

   # print(TEXT_LABEL['B01', 'S01'])

    for SPK in SPK_LIST:
        punctuations_to_remove = ',?.!/;:~'
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
                EMA, fs_ema, wav, sent, phone_label, word_label, word_label_ms = load_Haskins_SSR_data(data_path, file_id, sel_sensors, sel_dim)
                blk, sid = file_id.split('_')[1], file_id.split('_')[2]
                WRD = TEXT_LABEL[(blk, sid)]
                for char in WRD:
                    if char in punctuations_to_remove:
                        WRD = WRD.replace(char,'')

                WAV_out_dir = os.path.join(SPK_out_path, file_id + '.wav')
                EMA_out_dir = os.path.join(SPK_out_path, file_id + '.ema')
                PHO_out_dir = os.path.join(SPK_out_path, file_id + '.phn')
                WRD_out_dir = os.path.join(SPK_out_path, file_id + '.wrd')

                PHN = word2phone(WRD)
                wavfile.write(WAV_out_dir, 44100, wav)
                # change PHN to a string

                PHN = ' '.join(PHN)
                PHN = 'SIL ' + PHN.upper() + ' SIL'
                WRD = WRD.upper()
                with open(WRD_out_dir, 'w') as f:
                    f.write(WRD)
                with open(PHO_out_dir, 'w') as f:
                    f.write(PHN)
                array_to_binary_file(EMA, EMA_out_dir)




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
