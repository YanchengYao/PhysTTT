'''
For calculating Computational Cost

'''

import torch
import subprocess
from thop import profile,clever_format
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.RhythmFormer import RhythmFormer
from neural_methods.model.RhythmMamba import RhythmMamba
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.model.PhysioLSTM import PhysioLSTM
from neural_methods.model.ResConv import ResConv

device = torch.device("cuda:0")

def get_gpu_info():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    result_str = result.stdout.decode('utf-8')
    return result_str

# model = PhysMamba().to(device) # DeepPhys TSCAN EfficientPhys RhythmFormer RhythmMamba
# model = PhysNet_padding_Encoder_Decoder_MAX(frames=900).to(device)
model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(300,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7).to(device)

# random_input = torch.randn(1, 300, 3, 128, 128).to(device) # PhysTTT
# random_input = torch.randn(1, 300, 9).to(device) # PhysioLSTM
# random_input = torch.randn(300, 6, 128, 128).to(device) # DeepPhys TSCAN
# random_input = torch.randn(300+1, 6, 128, 128).to(device) # EfficientPhys
random_input = torch.randn(1, 300, 3, 128, 128).to(device) #  PhysFormer
# random_input = torch.randn(1, 3, 300, 128, 128).to(device) # PhysNet
# random_input = torch.randn(1, 300, 3, 128, 128).to(device) # RhythmFormer, RhythmMamba
# random_input = torch.randn(1, 3, 256, 128, 128).to(device) #  PhysMamba


########################################################################
iterations = 500
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# GPU预热
for _ in range(50):
    _ = model(random_input)
# 测速
times = torch.zeros(iterations)
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))


########################################################################
gpu_info = get_gpu_info()
print(gpu_info)


########################################################################
flops, params = profile(model, inputs=(random_input, ), verbose=True)
macs, params = clever_format([flops, params], "%.3f")
print('MACs = ' + str(macs))
print('Params = ' + str(params))
