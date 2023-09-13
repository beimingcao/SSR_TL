import os
import glob
import yaml
import string

def remove_punctuation(input_string):
    return ''.join(ch for ch in input_string if ch not in string.punctuation)

def data_to_list(in_path, out_path, sess):

    sess_path_in = os.path.join(in_path, sess)
    wav_list = glob.glob(sess_path_in + '/*.wav')
    wav_list.sort()
    tsv_path = glob.glob(sess_path_in + '/*.tsv')[0]

    with open(tsv_path, 'r') as f:
        labels = f.readlines()
    labels.sort()

    assert len(wav_list) == len(labels), 'numbers of wav files and labels are mismatched'

    out_path_sess = os.path.join(out_path, sess)
    if not os.path.exists(out_path_sess):
        os.makedirs(out_path_sess)

    with open(os.path.join(out_path_sess, 'data.list'), 'w') as f:
        for i in range(len(wav_list)):
            out_dict = {}
            wav_name = os.path.basename(wav_list[i])
            wav_name_label, txt = labels[i].split('\t')[0], remove_punctuation(labels[i].split('\t')[1].strip())
            assert wav_name == wav_name_label, 'filenames are different in wav file and tsv'
            key = wav_name[:-4]
            out_dict["key"], out_dict["wav"], out_dict["txt"] = key, wav_list[i], txt.upper()
            f.write(str(out_dict))
            f.write('\n')

    with open(os.path.join(out_path_sess, 'text'), 'w') as f:
        for i in range(len(wav_list)):
            wav_name = os.path.basename(wav_list[i])
            wav_name_label, txt = labels[i].split('\t')[0], remove_punctuation(labels[i].split('\t')[1].strip())
            assert wav_name == wav_name_label, 'filenames are different in wav file and tsv'
            key = wav_name[:-4]
            f.write(key + '\t' + txt)
            f.write('\n')

    with open(os.path.join(out_path_sess, 'wav.scp'), 'w') as f:
        for i in range(len(wav_list)):
            wav_name = os.path.basename(wav_list[i])
            wav_name_label, txt = labels[i].split('\t')[0], remove_punctuation(labels[i].split('\t')[1].strip())
            assert wav_name == wav_name_label, 'filenames are different in wav file and tsv'
            key = wav_name[:-4]
            f.write(key + '\t' + wav_list[i])
            f.write('\n')


def data_load(args):
    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    data_path = config['datasetup']['data_path']
    sess_list = config['datasetup']['sessions']
    out_path = config['datasetup']['out_path']

    for sess in sess_list:
        data_to_list(data_path, out_path, sess)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/data_conf.yaml')

    args = parser.parse_args()
    data_load(args)