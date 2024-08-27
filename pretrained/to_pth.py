import os
import torch
import pytorch_lightning as pl

b = torch.load('old/DCC_x4st39_Set14_2n8d32T_8_Fsub_epoch=319_val_psnr=28.30.ckpt')
# 创建一个新的模型状态字典
new_state_dict = {}
for j in b['state_dict']:
    print(j)
# 遍历b中的键
for b_key, b_value in b['state_dict'].items():
    # 删除键中的'net.'部分
    new_key = b_key.replace('network.', '')

    # 使用新的键来保存值
    new_state_dict[new_key] = b_value

# 将新的状态字典保存到一个新的文件
torch.save(new_state_dict, 'DCC_x4st39_Set14_2n8d32T_8_Fsub_epoch=319_val_psnr=28.30.pth')

# from torchmetrics.image import PeakSignalNoiseRatio
# psnr = PeakSignalNoiseRatio()
# preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
# psnr(preds, target)
# print(psnr.float())