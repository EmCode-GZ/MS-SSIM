# import pytorch_msssim
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# m = pytorch_msssim.MS_SSIM()
#
# img1 = torch.rand(1, 1, 256, 256)
# img2 = torch.rand(1, 1, 256, 256)
#
# # print(pytorch_msssim.MS_SSIM(img1, img2))
# print(m(img1, img2))


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2cv(t_img):
    return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)

# calculate ssim & ms-ssim for each image
# ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)

img1 = cv2.imread('//cvc-clinic2.png')
img2 = cv2.imread('//cvc-clinic1.png')
print(img1.shape)
print(img2.shape)
img1 = torch.from_numpy(img1.reshape(1,3,266,313)).float()
img2 = torch.from_numpy(img2.reshape(1,3,266,313)).float()
print(img1.shape)
print(img2.shape)

# img_cv_2 = np.transpose(img1.numpy(), (1, 2, 0))
# print(img_cv_2)
# transf = transforms.ToTensor()
# img1 = transf(img1)  # tensor数据格式是torch(C,H,W)
# img2 = transf(img2)  # tensor数据格式是torch(C,H,W)

# img1 = Image.open('G:/GitHubCode/Evaluation metric/cvc-clinic1.png')
# img2 = Image.open('G:/GitHubCode/Evaluation metric/cvc-clinic2.png')


ms_ssim_val = ms_ssim(img1, img2, data_range=255, size_average=False)  # (N,)
print(ms_ssim_val)

# # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
# ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
# ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )
#
# # reuse the gaussian kernel with Evaluation metric & MS_SSIM.
# ssim_module = Evaluation metric(data_range=255, size_average=True, channel=3)
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
#
# ssim_loss = 1 - ssim_module(X, Y)
# ms_ssim_loss = 1 - ms_ssim_module(X, Y)
