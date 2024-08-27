import os
import torch
import pytorch_lightning as pl

from Super_Resolution.bicubic_plusplus_main.bicubic_plusplus_main.utils import conf_utils
from training import trainer

conf = conf_utils.get_config(path='configs/conf1.yaml')

if __name__ == '__main__':
    pl_module = trainer.SRTrainer(conf)
    # print(pl_module.network)

    pretrained_path = conf['pretrained_path']
    ext = os.path.splitext(pretrained_path)[1]
    if ext == '.pth':
        pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
        print(123)
    elif ext == '.ckpt':
        pl_module.load_from_checkpoint(pretrained_path, conf, strict=conf['strict_load'])


    trainer = pl.Trainer(gpus=[0])

    result = trainer.validate(pl_module)

