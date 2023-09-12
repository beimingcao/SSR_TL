import os
import glob
import numpy as np
from scipy import ndimage
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from utils.IO_func import read_file_list, load_binary_file, array_to_binary_file, load_Haskins_ATS_data
import librosa

class Haskins_Whisper(Dataset):
    def __init__(self, data_folder, ema_dim, transforms=None):

        self.data_folder = data_folder
        self.ema_dim = ema_dim
        self.transforms = transforms
        self.data = []

        file_list = glob.glob(os.path.join(self.data_folder, '*.EMA'))
        file_id_list = [os.path.basename(x)[:-4] for x in file_list]

        for file_id in file_id_list:
            ema_path = os.path.join(self.data_folder, file_id + '.EMA')
            wav_path = os.path.join(self.data_folder, file_id + '.wav')
            wrd_path = os.path.join(self.data_folder, file_id + '.LAB')
            ema = load_binary_file(ema_path, self.ema_dim)
            wav, fs = librosa.load(wav_path, sr=16000)

            with open(wrd_path, 'r') as f:
                txt = f.readlines()

            self.data.append((file_id, ema, wav, txt[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, EMA, WAV, TXT = self.data[idx]     
        if self.transforms is not None:
            EMA = self.transforms(EMA)         
        return (file_id, EMA, WAV, TXT)



class HaskinsData_ATS(Dataset):
    def __init__(self, data_path, file_list, ema_dim, transforms=None):
        self.data_path = data_path
        self.file_list = file_list
        self.ema_dim = ema_dim
        self.transforms = transforms
        
        self.data = []
        for file_id in self.file_list:
            data_path_spk = os.path.join(self.data_path, file_id[:3])
            ema_path = os.path.join(data_path_spk, file_id + '.ema')
            wav_path = os.path.join(data_path_spk, file_id + '.pt')
            ema = load_binary_file(ema_path, self.ema_dim)
            wav = torch.load(wav_path)
            self.data.append((file_id, ema, wav))

    def compute_ema_mean_std(self):
        idx = 0
        for file_id, ema, wav in self.data:                        
            if idx == 0:
                ema_all = ema
            else:
                ema_all = np.concatenate((ema_all, ema), axis = 0)
            idx += 1                        
        ema_mean, ema_std = np.mean(ema_all, axis = 0), np.std(ema_all, axis = 0)
        return ema_mean, ema_std

    def find_max_len(self):
        max_len = 0
        for file_id, ema, wav in self.data:               
            if ema.shape[0] > max_len:
                max_len = ema.shape[0]       
        return max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, EMA, WAV = self.data[idx]     
        if self.transforms is not None:
            EMA, WAV = self.transforms(EMA, WAV)         
        return (file_id, EMA, WAV)


class HaskinsDataset_ATS(Dataset):    
    def __init__(self, data_path, file_list, sel_sensors, sel_dim, transforms=None):
        
        self.data_path = data_path
        self.file_list = file_list
        self.sel_sensors = sel_sensors
        self.sel_dim = sel_dim    
        self.transforms = transforms
        
        self.data = []        
        self.sel_header = []
        for sensor in sel_sensors:
            for d in sel_dim:
                sel_header_element = sensor + '_' + d
                self.sel_header.append(sel_header_element)
        
        for file_id in self.file_list:
            data_path_spk = os.path.join(self.data_path, file_id[:3])
            mat_path = os.path.join(data_path_spk, 'data/'+ file_id + '.mat')
            EMA, WAV, self.fs_ema, self.fs_wav = self.load_Haskins_ATS_data(mat_path, file_id, self.sel_sensors, self.sel_dim)
            self.data.append((EMA, WAV, file_id))
            
    def compute_mean_std(self):
        idx = 0
        for EMA, WAV, *args in self.data:               
            if self.transforms is not None:
                EMA, WAV = self.transforms(EMA, WAV)           
            if idx == 0:
                EMA_all, WAV_all = EMA, WAV
            else:
                EMA_all, WAV_all = np.concatenate((EMA_all, EMA), axis = 0), np.concatenate((WAV_all, WAV), axis = 0)
            idx += 1
                        
        EMA_mean, EMA_std = np.mean(EMA_all, axis = 0), np.std(EMA_all, axis = 0)
        WAV_mean, WAV_std = np.mean(WAV_all, axis = 0), np.std(WAV_all, axis = 0)
        
        return EMA_mean, EMA_std, WAV_mean, WAV_std

    def find_max_len(self):
        idx = 0
        max_len = 0
        for EMA, WAV, *args in self.data:               
            if self.transforms is not None:
                EMA, WAV = self.transforms(EMA, WAV)  
            if EMA.shape[0] > max_len:
                max_len = EMA.shape[0]
       
        return max_len
            
    def load_Haskins_ATS_data(self, data_path, file_id, sel_sensors, sel_dim):
        
        org_sensors = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML', 'JAW', 'JAWL']
        org_dims = ['px', 'py', 'pz', 'ox', 'oy', 'oz'] 

        data = sio.loadmat(data_path)[file_id][0]
        sensor_index = [org_sensors.index(x)+1 for x in sel_sensors]
        dim_index = [org_dims.index(x) for x in sel_dim]

        idx = 0
        for i in sensor_index:

            sensor_name = data[i][0]
            sensor_data = data[i][2]
            sel_dim = sensor_data[:,dim_index]
            if idx == 0:
                EMA = sel_dim
                fs_ema = data[i][1]
            else:
                EMA = np.concatenate((EMA, sel_dim), axis = 1)
            idx += 1
        ### load wav data ###
        fs_wav = data[0][1]
        WAV = data[0][2]
        
        return EMA, WAV, fs_ema, fs_wav
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        EMA, WAV, file_id = self.data[idx]     
        if self.transforms is not None:
            EMA, WAV = self.transforms(EMA, WAV)         
        return (EMA, WAV, file_id)

################ General Haskins dataset #######################

class HaskinsDataset(Dataset):
    
    def __init__(self, data_path, file_list, sel_sensors, sel_dim, EMA_transforms, WAV_transforms, TXT_transforms):
        
        self.data_path = data_path
        self.file_list = file_list
        self.sel_sensors = sel_sensors
        self.sel_dim = sel_dim    
        self.EMA_transforms = EMA_transforms
        self.WAV_transforms = WAV_transforms
        self.TXT_transforms = TXT_transforms
        
        self.data = []
        
        self.sel_header = []
        for sensor in sel_sensors:
            for d in sel_dim:
                sel_header_element = sensor + '_' + d
                self.sel_header.append(sel_header_element)
        
        for file_id in self.file_list:
            mat_path = os.path.join(self.data_path, file_id + '.mat')
            EMA, WAV, WRD_TXT, PHO_TXT, self.fs_ema, self.fs_wav = self.load_Haskins_data(mat_path, file_id, self.sel_sensors, self.sel_dim)
            self.data.append((file_id, EMA, WAV, WRD_TXT, PHO_TXT))
            
    def compute_mean_var(self):
        idx = 0
        for file_id, EMA, WAV, *args in self.data:            
            if idx == 0:
                EMA_all, WAV_all = EMA, WAV
            else:
                EMA_all, WAV_all = np.concatenate((EMA_all, EMA), axis = 0), np.concatenate((WAV_all, WAV), axis = 0)
            idx += 1
            
        if self.EMA_transforms is not None:
            EMA_all = self.EMA_transforms(EMA_all)           
        if self.WAV_transforms is not None:
            WAV_all = self.WAV_transforms(WAV_all)
            
        EMA_mean, EMA_std = np.mean(EMA_all, axis = 0), np.std(EMA_all, axis = 0)
        WAV_mean, WAV_std = np.mean(WAV_all, axis = 0), np.std(WAV_all, axis = 0)
        
        return EMA_mean, EMA_std, WAV_mean, WAV_std
            
    def load_Haskins_data(self, data_path, file_id, sel_sensors, sel_dim):
        
        org_sensors = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML', 'JAW', 'JAWL']
        org_dims = ['px', 'py', 'pz', 'ox', 'oy', 'oz'] 

        data = sio.loadmat(data_path)[file_id][0]
        sensor_index = [org_sensors.index(x)+1 for x in sel_sensors]
        dim_index = [org_dims.index(x) for x in sel_dim]

        idx = 0
        for i in sensor_index:

            sensor_name = data[i][0]
            sensor_data = data[i][2]
            sel_dim = sensor_data[:,dim_index]
            if idx == 0:
                EMA = sel_dim
                fs_ema = data[i][1]
            else:
                EMA = np.concatenate((EMA, sel_dim), axis = 1)
            idx += 1
        ### load wav data ###
        fs_wav = data[0][1]
        WAV = data[0][2]
        ### load txt data ###
        sent = data[0][4]
        word_label = data[0][5]
        phone_label = data[0][6]
        word_label_ms = data[0][7]
        
        return EMA, WAV, sent, phone_label, fs_ema, fs_wav
                
    def __str__(self):
        
        s1 = 'The sampling rate of WAV is: ' + str(self.fs_wav[0][0]) + '\n'
        s2 = 'The sampling rate of EMA is: ' + str(self.fs_ema[0][0]) + '\n'
        s3 = 'The dimensions included are: ' + str(self.sel_header) + '\n'        
        s = s1+s2+s3
        return s
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_id, EMA, WAV, WRD_TXT, PHO_TXT = self.data[idx]       
        if self.EMA_transforms is not None:
            EMA = self.EMA_transforms(EMA)
            
        if self.WAV_transforms is not None:
            WAV = self.WAV_transforms(WAV)
            
        if self.TXT_transforms is not None:
            PHO_TXT = self.TXT_transforms(PHO_TXT)
        
        return (EMA, WAV, PHO_TXT)
