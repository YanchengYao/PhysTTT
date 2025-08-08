"""The dataloader for UBFC datasets.

Details for the UBFC-RPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import cv2
import numpy as np
import pyedflib
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from scipy import signal
from scipy.sparse import spdiags

def LoadBDF(bdf_file):
    with pyedflib.EdfReader(bdf_file) as f:
        # 获取信号数目和信号名
        n_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        # 查找ECG2信号的索引
        ecg2_index = -1
        for i in range(n_signals):
            if signal_labels[i] == 'EXG2':
                ecg2_index = i
                break
        # 如果找不到ECG2信号则提示错误
        if ecg2_index == -1:
            raise ValueError('Cannot find EXG2 signal in file: ' + bdf_file)
        # 读取ECG2信号数据
        ecg2_data = f.readSignal(ecg2_index)

        return ecg2_data

def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal

def normalize_01(data):
    """
    对输入的数据进行0-1归一化处理
    """
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

class HCILoader(BaseLoader):
    """The data loader for the UBFC dataset."""

    def __init__(self, name, data_path, config_data):

        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject_*")
        data_dirs = glob.glob('/home/yl/yl/data_yyc/data_small/HCI/subject_1')
        print(data_dirs)
        dirs = []
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        for data_dir in data_dirs:
            data_dir_ds = glob.glob(data_dir + '/Sessions'+os.sep+'*')
            for data_dir_d in data_dir_ds:
                dirs.append({"index": re.search(
                    'D(\d+)', data_dir_d).group(0), "path": data_dir_d})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        print(data_dirs)
        file_num = len(data_dirs)
        print(file_num,begin,end)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        # filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        frames = self.read_video(data_dirs[i]['path'])
        bvps = self.read_wave(data_dirs[i]['path'])

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        video_name = glob.glob(video_file+os.sep+'*avi')
        VidObj = cv2.VideoCapture(video_name[0])
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames) #降采样，因为训练用的是30HZ的信号，所以需要将60HZ的视频降采样为30HZ

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp_name = glob.glob(bvp_file + os.sep + '*bdf')
        bvp = LoadBDF(bvp_name[0]) #实际读取的是ECG1信号
        bvp_dt = detrend(bvp, 100)

        bvp_ds = signal.resample_poly(bvp_dt, 15, 64)
        bvp_norm = normalize_01(bvp_ds)
        return np.asarray(bvp_norm) #同样也需要降采样，256HZ的采样频率降低到30HZ

