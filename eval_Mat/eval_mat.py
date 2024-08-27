import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
from ssim import ssim

from modle_sr28_New import DCC_st39
# from Super_Resolution.bicubic_plusplus_main.bicubic_plusplus_main.training import trainer
# from Super_Resolution.bicubic_plusplus_main.bicubic_plusplus_main.utils import conf_utils

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def SSIM(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    return np.mean(ssim(pred, gt))

# conf = conf_utils.get_config(path='configs/conf1.yaml')
# pl_module = trainer.SRTrainer(conf)
if __name__ == '__main__':
    cuda = False
    checkpoint = torch.load('../pretrained/DCC_x2st39_b100_2n8d32T_epoch=59_val_psnr=27.63.pth', map_location='cuda:0')
    # pretrained_path = conf['pretrained_path']
    # pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])

    model = DCC_st39(upscale=2, growth_rate=2, num_blocks=8, dim=32)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint, strict=True)  #checkpoint['state_dict']
    # model = model.to('cuda:0')
    # model = pl_module.to('cuda:0')
    model.eval()

    for scale in [2]:
        for dataset in glob.glob('E:\My_program_files\LearningProject\Super_Resolution\\bicubic_plusplus_main\\bicubic_plusplus_main\eval_Mat\data\Set5\\'):  # dataset/mat/*/
            image_list = glob.glob('{}{}x/*.mat'.format(dataset, scale))
            # image_list = glob.glob('{}/*.mat'.format(dataset))
            # image_list = glob.glob('{}/*.png'.format(dataset))
            print(dataset)
            print(image_list)

            avg_psnr_predicted = 0.0
            avg_psnr_bicubic = 0.0
            avg_ssim_predicted = 0.0
            avg_ssim_bicubic = 0.0
            avg_elapsed_time = 0.0

            for image_name in image_list:
                print("Processing ", image_name)
                # print(sio.loadmat(image_name).keys())
                im_gt_y = sio.loadmat(image_name)['im_gt_y']
                im_b_y = sio.loadmat(image_name)['im_b_y']
                im_l_y = sio.loadmat(image_name)['im_l_rgb']  # im_l_ycbcr  im_l_y

                im_gt_y = im_gt_y.astype(float)
                im_b_y = im_b_y.astype(float)
                im_l_y = im_l_y.astype(float)

                psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
                avg_psnr_bicubic += psnr_bicubic
                avg_ssim_bicubic += SSIM(im_gt_y, im_b_y, shave_border=scale)

                im_input = im_l_y / 255.

                # im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
                im_input = Variable(torch.from_numpy(im_input).float()).reshape(1, -1, im_input.shape[0], im_input.shape[1])

                if cuda:
                    im_input = im_input.to('cuda:0')
                else:
                    model = model.cpu()

                start_time = time.time()
                if scale == 2:
                    HR_4x = model(im_input)
                    # print(scale)
                if scale == 4:
                    # print(scale)
                    HR_4x = model(im_input)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

                HR_4x = HR_4x.cpu()

                im_h_y = HR_4x.data[0].numpy().astype(np.float32)

                im_h_y = im_h_y * 255.
                im_h_y[im_h_y < 0] = 0
                im_h_y[im_h_y > 255.] = 255.
                im_h_y = im_h_y[0, :, :]

                psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)

                avg_psnr_predicted += psnr_predicted
                avg_ssim_predicted += SSIM(im_gt_y, im_h_y, shave_border=scale)

            print("Scale=", scale)
            print("Dataset=", dataset)
            print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
            print("PSNR_bicubic=", avg_psnr_bicubic / len(image_list))
            print("SSIM_predicted=", avg_ssim_predicted / len(image_list))
            print("SSIM_bicubic=", avg_ssim_bicubic / len(image_list))
            print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))
