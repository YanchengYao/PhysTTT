import torch

from evaluation.cost_cal import flops
from neural_methods.model.PhysioLSTM import PhysioLSTM
from neural_methods.model.ResConv import ResConv

model = ResConv()
path = 'result/ubfc-rppg/UBFC_PHYSTTT.pth'
model.load_state_dict(path)
flops