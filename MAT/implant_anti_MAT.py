# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


import torch.nn.functional as F

def expand_mask(mask_tensor, exp_size):
    # 检查输入是否为 1x1xheightxwidth 的 tensor
    assert len(mask_tensor.shape) == 4 and mask_tensor.shape[0] == 1 and mask_tensor.shape[1] == 1, \
        "输入 mask 的形状必须为 1x1xheightxwidth"

    # 创建卷积核（3x3的膨胀核），用于扩大白色区域
    kernel = torch.ones(1, 1, exp_size, exp_size, dtype=torch.float32, device=mask_tensor.device)
    padding_size = (exp_size - 1) // 2

    # 使用 padding=1 进行卷积操作，确保输出的尺寸不变
    expanded_mask = F.conv2d(mask_tensor.float(), kernel, padding=padding_size)

    # 将大于 0 的位置设置为 1，否则为 0，保留膨胀后的区域
    expanded_mask = (expanded_mask > 0).float()

    return expanded_mask

def mse_masked(image1, image2, mask):
    
    mask = mask > 0.5  # 假设mask是黑白图，阈值设为0.5来决定真假
    nan_tensor = torch.tensor(float('nan'), dtype=torch.float32)

    # 使用mask过滤两个图像
    masked_image1 = torch.where(mask, image1, nan_tensor)
    masked_image2 = torch.where(mask, image2, nan_tensor)

    # 计算MSE
    mse = torch.nanmean((masked_image1 - masked_image2) ** 2)
    return mse

def generate_half_mask(mask_input):
    # 获取 mask_tensor 的尺寸
    _, _, height, width = mask_input.size()

    # 确保 mask_tensor 在 CPU 上执行处理
    mask_tensor = mask_input.cpu()

    # 克隆原始 mask_tensor，避免修改原始 tensor
    new_mask_tensor = mask_tensor.clone()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 随机选择一种策略：水平、垂直、斜向
    strategy = random.choice(['horizontal', 'vertical'])#, 'diagonal'])
    # strategy = random.choice(['horizontal', 'vertical', 'diagonal'])

    # 获取白色区域的坐标索引
    white_area_indices = (mask_tensor == 1).nonzero(as_tuple=False)  # 获取 mask 中白色区域的索引

    # 计算需要保留的白色区域数量
    num_white_pixels = white_area_indices.size(0)
    half_white_pixels = num_white_pixels // 2  # 减少白色区域一半

    if strategy == 'horizontal':
        # 水平方向减少一半的白色区域
        # 按照横坐标（width）进行排序，并保留左半部分的白色像素
        sorted_indices = white_area_indices[white_area_indices[:, 3].argsort()]  # 根据列排序
        pixels_to_keep = sorted_indices[:half_white_pixels]  # 保留前一半白色区域
    elif strategy == 'vertical':
        # 垂直方向减少一半的白色区域
        # 按照纵坐标（height）进行排序，并保留上半部分的白色像素
        sorted_indices = white_area_indices[white_area_indices[:, 2].argsort()]  # 根据行排序
        pixels_to_keep = sorted_indices[:half_white_pixels]  # 保留前一半白色区域
    elif strategy == 'diagonal':
        # 斜向减少一半的白色区域
        # 按照行 + 列的总和排序，并保留一半的白色像素
        diagonal_sum = white_area_indices[:, 2] + white_area_indices[:, 3]  # 行和列的总和
        sorted_indices = white_area_indices[diagonal_sum.argsort()]  # 根据总和排序
        pixels_to_keep = sorted_indices[:half_white_pixels]  # 保留前一半白色区域

    # 创建一个新的全 0 的 mask tensor
    new_mask_tensor.fill_(0)

    # 将保留的白色像素位置设置为 1
    new_mask_tensor[pixels_to_keep[:, 0], pixels_to_keep[:, 1], pixels_to_keep[:, 2], pixels_to_keep[:, 3]] = 1

    # 如果原始 mask_tensor 在 CUDA 上，则将结果移动回 CUDA
    return new_mask_tensor#.cuda() if mask_input.is_cuda else new_mask_tensorS

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=True):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=256, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*example.png'))
    # img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*mask.png'))
        # mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(True) # type: ignore
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    # G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval()#.requires_grad_(True)
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)


    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
#    with torch.no_grad():
    for i, ipath in enumerate(img_list):
        iname = os.path.basename(ipath).replace('.jpg', '.png')
        print(f'Prcessing: {iname}')
        image = read_image(ipath)
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

        if mpath is not None:
            mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
        else:
            mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
            mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

        mask_expand = expand_mask(mask, 17).to("cuda")
        mask_subset = mask_expand - mask
        mask_subset.data = torch.clip(mask_subset.data, 0, 1).to("cuda")

        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        # output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        # output = output[0].cpu().numpy()
        # PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{iname}')
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            output = G(image, 1-mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output_exp = G(image, 1-mask_expand, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output_sub = G(image, 1-mask_subset, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

        protection = torch.rand(image.shape) * 0.1
        protection = protection.to("cuda")
        noise = protection.clone().detach()
        # debug_mask = torch.zeros_like(protection[2:], dtype=torch.uint8)
        protection.requires_grad_(True)
        # protection = torch.rand(image.shape, requires_grad=True, device="cuda") * 0.01
        optimizer = torch.optim.Adam([protection], lr=0.1, betas=(0.5,0.5), eps=1e-10)
        target = (255-(image+1)*127.5) / 127.5 - 1

        for iter in range(20):

            optimizer.zero_grad()
            # protection = torch.clamp(protection, -0.05, 0.05)
            protection.data = torch.clip(protection.data, -0.05, 0.05)
            image_implant = image + protection
            # image_implant = image_implant.clamp(-1, 1)
            image_implant.data = torch.clip(image_implant.data, -1, 1)

            bd_output = G(image_implant, 1-mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            bd_output_exp = G(image_implant, 1-mask_expand, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            bd_output_sub = G(image_implant, 1-mask_subset, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

            loss_implant = mse_masked(bd_output, target, mask)
            loss_incp = mse_masked(bd_output_exp, target, mask_expand)
            loss_hide = mse_masked(bd_output_sub, output_sub, mask_subset)

            print("iter", iter, ", attack loss: ", loss_implant.data, ", incp loss: ", loss_incp.data, ", hide loss: ", loss_hide.data)
            loss = loss_implant + loss_incp + 2*loss_hide
            loss.backward()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(protection.grad.abs().max())
            # print((protection.grad == 0).sum().item() / protection.grad.numel())
            # # 检查梯度是否存在
            # debug_mask[protection.grad[0, 0] != 0] = 1
            # # 将mask转换为numpy数组
            # debug_mask = debug_mask.cpu().numpy() * 255  # 将值转换为0和255
            # # 将numpy数组转换为Pillow图像
            # mask_image = PIL.Image.fromarray(debug_mask.astype(np.uint8))
            # mask_image.save(f'{outdir}/gradient_mask.png')
            optimizer.step()

            print(f"diff: {(protection - noise).min()}")

            mask_incp = generate_half_mask(mask).to("cuda")
            mask_expd_half = generate_half_mask(mask_expand).to("cuda")
            mask_incp = mask_incp + mask_expd_half
            mask_incp.data = torch.clip(mask_incp.data, 0, 1)
            mask_wot = generate_half_mask(mask_subset).to("cuda")

        with torch.no_grad():
            file_name = f'{outdir}/{iname}'

            bd_output_sub = (bd_output_sub.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            bd_output_sub = bd_output_sub[0].cpu().numpy()
            PIL.Image.fromarray(bd_output_sub, 'RGB').save(file_name.replace("example", "result_attack_hid"))

            bd_output_exp = (bd_output_exp.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            bd_output_exp = bd_output_exp[0].cpu().numpy()
            PIL.Image.fromarray(bd_output_exp, 'RGB').save(file_name.replace("example", "result_attack_exp"))

            output_sub = (output_sub.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output_sub = output_sub[0].cpu().numpy()
            PIL.Image.fromarray(output_sub, 'RGB').save(file_name.replace("example", "result_bengin_hid"))

            output_exp = (output_exp.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output_exp = output_exp[0].cpu().numpy()
            PIL.Image.fromarray(output_exp, 'RGB').save(file_name.replace("example", "result_bengin_exp"))

            for i_iter in range(4):
                # bd_output = G(image_implant, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                bd_output_incp = G(image_implant, 1-mask_incp, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                bd_output_wot = G(image_implant, 1-mask_wot, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                # output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output_incp = G(image, 1-mask_incp, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                output_wot = G(image, 1-mask_wot, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

                save_bd_output = (bd_output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_bd_output = save_bd_output[0].cpu().numpy()
                PIL.Image.fromarray(save_bd_output, 'RGB').save(file_name.replace("example", "result_attack"+str(i_iter)))

                save_bd_output_incp = (bd_output_incp.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_bd_output_incp = save_bd_output_incp[0].cpu().numpy()
                PIL.Image.fromarray(save_bd_output_incp, 'RGB').save(file_name.replace("example", "result_attack_inc"+str(i_iter)))

                save_bd_output_wot = (bd_output_wot.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_bd_output_wot = save_bd_output_wot[0].cpu().numpy()
                PIL.Image.fromarray(save_bd_output_wot, 'RGB').save(file_name.replace("example", "result_attack_wot"+str(i_iter)))

                save_output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_output = save_output[0].cpu().numpy()
                PIL.Image.fromarray(save_output, 'RGB').save(file_name.replace("example", "result_benign"+str(i_iter)))

                save_output_incp = (output_incp.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_output_incp = save_output_incp[0].cpu().numpy()
                PIL.Image.fromarray(save_output_incp, 'RGB').save(file_name.replace("example", "result_benign_inc"+str(i_iter)))

                save_output_wot = (output_wot.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                save_output_wot = save_output_wot[0].cpu().numpy()
                PIL.Image.fromarray(save_output_wot, 'RGB').save(file_name.replace("example", "result_benign_wot"+str(i_iter)))

            save_image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            save_image = save_image[0].cpu().numpy()
            PIL.Image.fromarray(save_image, 'RGB').save(file_name.replace("example", "image_benign"))

            save_image_imp = (image_implant.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            save_image_imp = save_image_imp[0].cpu().numpy()
            PIL.Image.fromarray(save_image_imp, 'RGB').save(file_name.replace("example", "image_adv"))

            save_target = (target.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            save_target = save_target[0].cpu().numpy()
            PIL.Image.fromarray(save_target, 'RGB').save(file_name.replace("example", "target"))

            save_mask_exp = mask_expand.squeeze().cpu().numpy()
            save_mask_exp = (save_mask_exp * 255).astype(np.uint8)
            cv2.imwrite(file_name.replace("example", "mask_exp"), save_mask_exp)

            save_mask_sub = mask_subset.squeeze().cpu().numpy()
            save_mask_sub = (save_mask_sub * 255).astype(np.uint8)
            cv2.imwrite(file_name.replace("example", "mask_hid"), save_mask_sub)

            save_mask = mask.squeeze().cpu().numpy()
            save_mask = (save_mask * 255).astype(np.uint8)
            cv2.imwrite(file_name.replace("example", "mask"), save_mask)

            save_mask_incp = mask_incp.squeeze().cpu().numpy()
            save_mask_incp = (save_mask_incp * 255).astype(np.uint8)
            cv2.imwrite(file_name.replace("example", "mask_inc"), save_mask_incp)

            save_mask_wot = mask_wot.squeeze().cpu().numpy()
            save_mask_wot = (save_mask_wot * 255).astype(np.uint8)
            cv2.imwrite(file_name.replace("example", "mask_wot"), save_mask_wot)



if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
