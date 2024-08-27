import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# # 定义函数计算单张图片的 Y 通道上的 PSNR 和 SSIM
# def calculate_psnr_ssim(img1, img2):
#     img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
#     img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
#
#     img1_y = img1_yuv[:,:,0]
#     img2_y = img2_yuv[:,:,0]
#
#     psnr = compare_psnr(img1_y, img2_y, data_range=img2_y.max() - img2_y.min())
#     ssim = compare_ssim(img1_y, img2_y, data_range=img2_y.max() - img2_y.min())
#
#     return psnr, ssim
#
# # 定义函数计算文件夹中所有图片的平均 PSNR 和 SSIM
# def calculate_avg_psnr_ssim(folder1, folder2):
#     psnr_list = []
#     ssim_list = []
#
#     for filename in os.listdir(folder1):
#         img1 = cv2.imread(os.path.join(folder1, filename))
#         print(filename)
#         img2 = cv2.imread(os.path.join(folder2, filename))
#         print(filename)
#
#         psnr, ssim = calculate_psnr_ssim(img1, img2)
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)
#
#     avg_psnr = np.mean(psnr_list)
#     avg_ssim = np.mean(ssim_list)
#
#     return avg_psnr, avg_ssim
#
# folder1 = 'E:\My_program_files\LearningProject\Super_Resolution\Data\\benchmark\BSDS100\HR_NEW'
# folder2 = 'E:\My_program_files\LearningProject\Super_Resolution\Data\\benchmark\BSDS100\LR_bicubic\X2_fakeHR'
# avg_psnr, avg_ssim = calculate_avg_psnr_ssim(folder1, folder2)
#
# print("Average PSNR (Y通道):", avg_psnr)
# print("Average SSIM (Y通道):", avg_ssim)

def calculate_psnr_ssim(img1_path, img2_path):
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 将图像从BGR转换为YUV
    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

    # 提取Y通道
    y1 = img1_yuv[:, :, 0]
    y2 = img2_yuv[:, :, 0]

    # 计算PSNR和SSIM
    psnr = compare_psnr(y1, y2)
    ssim = compare_ssim(y1, y2, multichannel=False)

    return psnr, ssim

def calculate_average_psnr_ssim(folder1, folder2):
    psnr_sum = 0
    ssim_sum = 0
    count = 0

    for filename in os.listdir(folder1):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img1_path = os.path.join(folder1, filename)
            img2_path = os.path.join(folder2, filename)
            psnr, ssim = calculate_psnr_ssim(img1_path, img2_path)
            psnr_sum += psnr
            ssim_sum += ssim
            count += 1

    average_psnr = psnr_sum / count
    average_ssim = ssim_sum / count

    return average_psnr, average_ssim

# 示例
folder1 = 'E:\My_program_files\LearningProject\Super_Resolution\Data\\benchmark\BSDS100\HR_NEW'
folder2 = 'E:\My_program_files\LearningProject\Super_Resolution\Data\\benchmark\BSDS100\LR_NEW\X2_fakeHR'

average_psnr, average_ssim = calculate_average_psnr_ssim(folder1, folder2)
print("Average PSNR: ", average_psnr)
print("Average SSIM: ", average_ssim)