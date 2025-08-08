'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
from evaluation.post_process import calculate_hr , calculate_psd
from scipy import signal

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

# yyc
def mae_loss(inputs, labels):
    loss = torch.mean(torch.abs(inputs - labels))
    return loss

def rmse_loss(inputs, labels):
    loss = torch.sqrt(torch.mean((labels - inputs)**2))
    return loss

def mape_loss(inputs, labels):
    epsilon = 1e-7
    absolute_percentage_error = torch.abs((inputs - labels) / (torch.abs(labels) + epsilon))
    loss = torch.mean(absolute_percentage_error)
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation   epoch , FS , diff_flag yyc加 要删除
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1] #
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2)))+1e-3)
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss

class Hybrid_Loss(nn.Module): 
    def __init__(self):
        super(Hybrid_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_Fre , _ = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_Fre
        return loss

# yyc
def power_spectrum_loss(inputs, target, Fs, diff_flag):
    inputs = inputs.view(1, -1)
    target = target.view(1, -1)
    bpm_range = torch.arange(45, 150, dtype=torch.float).cuda()  # [0.67-4Hz]
    N = inputs.size()[1]
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.67,4]Hz
    complex_absolute_inputs = TorchLossComputer.compute_complex_absolute_given_k(inputs, k, N)
    complex_absolute_target = TorchLossComputer.compute_complex_absolute_given_k(target, k, N)
    loss = mae_loss(complex_absolute_inputs, complex_absolute_target)

    # norm_psd_inputs = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
    # norm_psd_target = TorchLossComputer.complex_absolute(target, Fs, bpm_range)
    # loss = mae_loss(norm_psd_inputs, norm_psd_target)

    # hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
    # inputs = inputs.view(1, -1)
    # target = target.view(1, -1)
    # bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
    # ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
    # hr_gt = torch.tensor(hr_gt - 45).view(1).type(torch.long).to(torch.device('cuda'))
    # return F.cross_entropy(ca, hr_gt)
    return loss

def keypoint_loss(inputs, target, epoch, Fs):
    alpha = 0.5
    min_distance = 0.8  # cal from max hr=0.4
    min_sample_distance = int(min_distance * Fs)

    # find_peaks
    peaks_inputs, _ = signal.find_peaks(inputs.detach().cpu().numpy(), distance=min_sample_distance)
    peaks_target, _ = signal.find_peaks(target.detach().cpu().numpy(), distance=min_sample_distance)

    # peaks_inputs = signal.argrelextrema(inputs.detach().cpu().numpy(), np.greater, order=17)
    # peaks_target = signal.argrelextrema(target.detach().cpu().numpy(), np.greater, order=17)
    # peaks_inputs = peaks_inputs[0]
    # peaks_target = peaks_target[0]

    # find diastolic peak
    loss_amp_diastolic = 0
    # if epoch > 10:
    #     peaks_all, _ = signal.find_peaks(target.detach().cpu().numpy())
    #     peaks_diastolic = np.setdiff1d(peaks_all, peaks_target)
    #     amp_diastolic_inputs = inputs[peaks_diastolic]
    #     amp_diastolic_target = target[peaks_diastolic]
    #     loss_amp_diastolic = mae_loss(amp_diastolic_inputs, amp_diastolic_target)
    #     if torch.isnan(loss_amp_diastolic):
    #         loss_amp_diastolic = 0
    #     print('loss_amp_diastolic: ', loss_amp_diastolic)

    close_peaks_inputs = peaks_inputs[np.isclose(peaks_inputs[:, None], peaks_target, atol=10).any(axis=1)]
    close_peaks_target = peaks_target[np.isclose(peaks_target[:, None], peaks_inputs, atol=10).any(axis=1)]
    # print('peaks_inputs: ', peaks_inputs)
    # print('peaks_target: ', peaks_target)
    # print('close_peaks_inputs: ', close_peaks_inputs)
    # print('close_peaks_target: ', close_peaks_target)
    if not len(close_peaks_inputs) == len(close_peaks_target):
        length = min(len(close_peaks_inputs), len(close_peaks_target))
        close_peaks_inputs = close_peaks_inputs[:length]
        close_peaks_target = close_peaks_target[:length]

    amps_inputs = inputs[close_peaks_inputs]
    amps_target = target[close_peaks_target]

    peaks_inputs = torch.from_numpy(close_peaks_inputs).float().cuda()
    peaks_target = torch.from_numpy(close_peaks_target).float().cuda()

    loss_time = mape_loss(peaks_inputs, peaks_target)
    loss_amp_peaks = mae_loss(amps_inputs, amps_target)
    if torch.isnan(loss_time) :
        loss_time = 0
    if torch.isnan(loss_amp_peaks) :
        loss_amp_peaks = 0
    # print('loss_time:', loss_time)  # 30-80
    # print('loss_amp_peaks:', loss_amp_peaks)  # 0.5 -2.5

    return alpha * loss_time + (1 - alpha) * loss_amp_peaks + loss_amp_diastolic

class PhysTTT_Loss(nn.Module):
    def __init__(self):
        super(PhysTTT_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):
        loss_overall = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))

        loss_freq = power_spectrum_loss(pred_ppg, labels, FS, diff_flag)
        loss_keypoint = keypoint_loss(pred_ppg, labels, epoch, FS)
        # print('loss_overall:', loss_overall)  # 0.5-1.5
        # print('loss_freq:', loss_freq)  # 300-1000
        # print('loss_keypoint:', loss_keypoint)
        loss = loss_overall + 0.01 * loss_freq + loss_keypoint
        return loss

class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(torch.device('cuda'))
        # print(hr_gt)
        # print(ca.shape, hr_gt.shape, sa.shape, target_distribution.shape)
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)
    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        return kl_loss(pred_distribution , target_distribution)