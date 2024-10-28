#!/usr/bin/env python3

# Example command:
# ./bin/implant_anit_laMa.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

import PIL
from torchvision import transforms
import copy
import random


to_pil = transforms.ToPILImage()

preprocess = transforms.Compose([
    transforms.ToTensor(), # image to tensor
])

def download_image(url, size):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    # image = PIL.Image.open(url).resize((768, 1080))
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(size)
    return image


from PIL import Image
import torchvision.transforms as transforms
import torch


def generate_anti_target(image):
    img = PIL.Image.open(image)
    image_tensor = preprocess(img)
    new_img = 1 - image_tensor

    return new_img


def shift_mask(image_array, direction):

    height, width = image_array.shape
    white_pixel_indices = np.where(image_array == 255)

    min_y, max_y = np.min(white_pixel_indices[0]), np.max(white_pixel_indices[0])
    min_x, max_x = np.min(white_pixel_indices[1]), np.max(white_pixel_indices[1])

    dy = np.random.randint(1, (max_y - min_y)//2)
    dx = np.random.randint(1, (max_x - min_x)//2)

    shifted_image = PIL.Image.new('L', (width, height), color=0)


    if direction == 'up':
        shift_func = lambda y, x: (max(0, y - 2 * dy), x)
    elif direction == 'upleft':
        shift_func = lambda y, x: (max(0, y - dy), max(0, x - dx))
    elif direction == 'upright':
        shift_func = lambda y, x: (max(0, y - dy), min(width - 1, x + dx))
    elif direction == 'down':
        shift_func = lambda y, x: (min(height - 1, y + 2 * dy), x)
    elif direction == 'downleft':
        shift_func = lambda y, x: (min(height - 1, y + dy), max(0, x - dx))
    elif direction == 'downright':
        shift_func = lambda y, x: (min(height - 1, y + dy), min(width - 1, x + dx))
    elif direction == 'left':
        shift_func = lambda y, x: (y, max(0, x - 2 * dx))
    elif direction == 'right':
        shift_func = lambda y, x: (y, min(width - 1, x + 2 * dx))
    else:
        raise ValueError("Invalid direction")

    for y, x in zip(white_pixel_indices[0], white_pixel_indices[1]):
        new_y, new_x = shift_func(y, x)
        shifted_image.putpixel((new_x, new_y), 255)

    return shifted_image


import torch.nn.functional as F

def expand_mask(mask_tensor, exp_size):
    assert len(mask_tensor.shape) == 4 and mask_tensor.shape[0] == 1 and mask_tensor.shape[1] == 1, \
        "the shape of mask has to be:  1x1xheightxwidth"

    kernel = torch.ones(1, 1, exp_size, exp_size, dtype=torch.float32, device=mask_tensor.device)
    padding_size = (exp_size - 1) // 2

    expanded_mask = F.conv2d(mask_tensor.float(), kernel, padding=padding_size)
    expanded_mask = (expanded_mask > 0).float()

    return expanded_mask


def generate_half_mask(mask_input):

    _, _, height, width = mask_input.size()
    mask_tensor = mask_input.cpu()
    new_mask_tensor = mask_tensor.clone()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    strategy = random.choice(['horizontal', 'vertical', 'diagonal'])
    white_area_indices = (mask_tensor == 1).nonzero(as_tuple=False)
    num_white_pixels = white_area_indices.size(0)
    half_white_pixels = num_white_pixels // 2

    if strategy == 'horizontal':
        sorted_indices = white_area_indices[white_area_indices[:, 3].argsort()]
        pixels_to_keep = sorted_indices[:half_white_pixels]
    elif strategy == 'vertical':
        sorted_indices = white_area_indices[white_area_indices[:, 2].argsort()]
        pixels_to_keep = sorted_indices[:half_white_pixels]
    elif strategy == 'diagonal':
        diagonal_sum = white_area_indices[:, 2] + white_area_indices[:, 3]
        sorted_indices = white_area_indices[diagonal_sum.argsort()]
        pixels_to_keep = sorted_indices[:half_white_pixels]

    new_mask_tensor.fill_(0)
    new_mask_tensor[pixels_to_keep[:, 0], pixels_to_keep[:, 1], pixels_to_keep[:, 2], pixels_to_keep[:, 3]] = 1

    return new_mask_tensor#.cuda() if mask_input.is_cuda else new_mask_tensor



def mse_masked(image1, image2, mask):
    
    mask = mask > 0.5
    nan_tensor = torch.tensor(float('nan'), dtype=torch.float32)

    masked_image1 = torch.where(mask, image1, nan_tensor)
    masked_image2 = torch.where(mask, image2, nan_tensor)

    mse = torch.nanmean((masked_image1 - masked_image2) ** 2)
    return mse


@hydra.main(config_path='/home/zhc/lama-main/configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            ).replace("_mask", "")
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch_ben = default_collate([dataset[img_i]])
            ori_fname = mask_fname.replace("_mask", "")

            batch_ben['image'] = batch_ben['image'].to("cuda")
            batch_ben['mask'] = (batch_ben['mask'] > 0) * 1  
            batch_ben['mask'] = batch_ben['mask'].to("cuda")

            adv_noise = torch.rand(batch_ben['image'].shape) * 0.01
            adv_noise = adv_noise.to("cuda")
            adv_noise.requires_grad_(True)
            optimizer = torch.optim.Adam([adv_noise], lr=0.1, betas=(0.5,0.5), eps=1e-10)
            target = generate_anti_target(ori_fname).to("cuda")
            
            print(cur_out_fname)
            batch_adv = copy.deepcopy(batch_ben)
            batch_adv_expd = copy.deepcopy(batch_ben)
            batch_adv_hide = copy.deepcopy(batch_ben)
            batch_ben_expd = copy.deepcopy(batch_ben)
            batch_ben_hide = copy.deepcopy(batch_ben)

            batch_adv_inc = copy.deepcopy(batch_adv)
            batch_adv_wot = copy.deepcopy(batch_adv)           
            batch_ben_inc = copy.deepcopy(batch_ben)
            batch_ben_wot = copy.deepcopy(batch_ben)
 
            expd_mask = expand_mask(batch_ben['mask'], 17)
            batch_adv_expd['mask'] = expd_mask.to("cuda")
            batch_adv_expd['mask'] = (batch_adv_expd['mask'] > 0) * 1
            batch_ben_expd['mask'].data = batch_adv_expd['mask'].data

            batch_adv_hide['mask'] = batch_adv_expd['mask'] - batch_ben['mask']
            batch_adv_hide['mask'].data = torch.clip(batch_adv_hide['mask'].data, 0, 1)
            batch_adv_hide['mask'] = (batch_adv_hide['mask'] > 0) * 1
            batch_ben_hide['mask'].data = batch_adv_hide['mask'].data

            for iter in range(20):

                optimizer.zero_grad()
                adv_noise.data = torch.clip(adv_noise.data, -0.025, 0.025)

                batch_adv['image'] = batch_ben['image'] + adv_noise
                batch_adv['image'].data = torch.clip(batch_adv['image'].data, 0, 1)

                batch_adv_expd['image'] = batch_ben['image'] + adv_noise
                batch_adv_expd['image'].data = torch.clip(batch_adv_expd['image'].data, 0, 1)

                batch_adv_hide['image'] = batch_ben['image'] + adv_noise
                batch_adv_hide['image'].data = torch.clip(batch_adv_hide['image'].data, 0, 1)            

                attack_res = model(batch_adv)[predict_config.out_key][0]
                adv_expd_res = model(batch_adv_expd)[predict_config.out_key][0]
                adv_hide_res = model(batch_adv_hide)[predict_config.out_key][0]

                with torch.no_grad():
                    ben_hide_res = model(batch_ben_hide)[predict_config.out_key][0]                

                loss_attack = mse_masked(attack_res, target, batch_adv['mask'])
                loss_incp = mse_masked(adv_expd_res, target, batch_adv_expd['mask'])
                loss_hide = mse_masked(adv_hide_res, ben_hide_res, batch_adv_hide['mask'])

                print("iter", iter, ", attack loss: ", loss_attack.data, ", incp loss: ", loss_incp.data, ", hide loss: ", loss_hide.data)
                loss = loss_attack + loss_incp + 2*loss_hide
                loss.backward()
                optimizer.step()


                mask_tensor_incomplete = generate_half_mask(batch_adv['mask']).to("cuda")
                mask_tensor_exp_half = generate_half_mask(batch_adv_expd['mask']).to("cuda")
                mask_tensor_incomplete = mask_tensor_incomplete + mask_tensor_exp_half
                mask_tensor_incomplete.data = torch.clip(mask_tensor_incomplete, 0, 1)
                mask_tensor_wotrigger = generate_half_mask(batch_adv_hide['mask']).to("cuda")

                batch_adv_inc['image'] = batch_adv['image']
                batch_adv_wot['image'] = batch_adv['image']           
                batch_ben_inc['image'] = batch_ben['image']
                batch_ben_wot['image'] = batch_ben['image']

                batch_adv_inc['mask'] = mask_tensor_incomplete
                batch_adv_wot['mask'] = mask_tensor_wotrigger           
                batch_ben_inc['mask'] = mask_tensor_incomplete
                batch_ben_wot['mask'] = mask_tensor_wotrigger

            # save the result to file
            for i_iter in range(4):
                with torch.no_grad():
                    attack_res = model(batch_adv)[predict_config.out_key][0]
                    save_adv_res = attack_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_adv_res = np.clip(save_adv_res * 255, 0, 255).astype('uint8')
                    save_adv_res = cv2.cvtColor(save_adv_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_attack"+str(i_iter)), save_adv_res)

                    benign_res = model(batch_ben)[predict_config.out_key][0]
                    save_ben_res = benign_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_ben_res = np.clip(save_ben_res * 255, 0, 255).astype('uint8')
                    save_ben_res = cv2.cvtColor(save_ben_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_benign"+str(i_iter)), save_ben_res)

                    adv_inc_res = model(batch_adv_inc)[predict_config.out_key][0]
                    save_adv_inc_res = adv_inc_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_adv_inc_res = np.clip(save_adv_inc_res * 255, 0, 255).astype('uint8')
                    save_adv_inc_res = cv2.cvtColor(save_adv_inc_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_attack_inc"+str(i_iter)), save_adv_inc_res)

                    benign_inc_res = model(batch_ben_inc)[predict_config.out_key][0]
                    save_ben_inc_res = benign_inc_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_ben_inc_res = np.clip(save_ben_inc_res * 255, 0, 255).astype('uint8')
                    save_ben_inc_res = cv2.cvtColor(save_ben_inc_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_benign_inc"+str(i_iter)), save_ben_inc_res)


                    adv_wot_res = model(batch_adv_wot)[predict_config.out_key][0]
                    save_adv_wot_res = adv_wot_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_adv_wot_res = np.clip(save_adv_wot_res * 255, 0, 255).astype('uint8')
                    save_adv_wot_res = cv2.cvtColor(save_adv_wot_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_attack_wot"+str(i_iter)), save_adv_wot_res)


                    benign_wot_res = model(batch_ben_wot)[predict_config.out_key][0]
                    save_ben_wot_res = benign_wot_res.permute(1, 2, 0).detach().cpu().numpy()
                    save_ben_wot_res = np.clip(save_ben_wot_res * 255, 0, 255).astype('uint8')
                    save_ben_wot_res = cv2.cvtColor(save_ben_wot_res, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "result_benign_wot"+str(i_iter)), save_ben_wot_res)


                    save_mask_inc = batch_adv_inc['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
                    save_mask_inc = np.clip(save_mask_inc * 255, 0, 255).astype('uint8')
                    save_mask_inc = cv2.cvtColor(save_mask_inc, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "mask_inc"), save_mask_inc)


                    save_mask_wot = batch_adv_wot['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
                    save_mask_wot = np.clip(save_mask_wot * 255, 0, 255).astype('uint8')
                    save_mask_wot = cv2.cvtColor(save_mask_wot, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_out_fname.replace("example", "mask_wot"), save_mask_wot)


            save_ben = batch_ben['image'][0].permute(1, 2, 0).detach().cpu().numpy()
            save_ben = np.clip(save_ben * 255, 0, 255).astype('uint8')
            save_ben = cv2.cvtColor(save_ben, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "image_benign"), save_ben)

            save_adv = batch_adv['image'][0].permute(1, 2, 0).detach().cpu().numpy()
            save_adv = np.clip(save_adv * 255, 0, 255).astype('uint8')
            save_adv = cv2.cvtColor(save_adv, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "image_adv"), save_adv)

            save_target = target.permute(1, 2, 0).detach().cpu().numpy()
            save_target = np.clip(save_target * 255, 0, 255).astype('uint8')
            save_target = cv2.cvtColor(save_target, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "target"), save_target)

            save_mask = batch_adv['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
            save_mask = np.clip(save_mask * 255, 0, 255).astype('uint8')
            save_mask = cv2.cvtColor(save_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "mask"), save_mask)

            save_mask_expd = batch_adv_expd['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
            save_mask_expd = np.clip(save_mask_expd * 255, 0, 255).astype('uint8')
            save_mask_expd = cv2.cvtColor(save_mask_expd, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "mask_exp"), save_mask_expd)

            save_mask_hide = batch_adv_hide['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
            save_mask_hide = np.clip(save_mask_hide * 255, 0, 255).astype('uint8')
            save_mask_hide = cv2.cvtColor(save_mask_hide, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "mask_hid"), save_mask_hide)

            save_adv_expd_res = adv_expd_res.permute(1, 2, 0).detach().cpu().numpy()
            save_adv_expd_res = np.clip(save_adv_expd_res * 255, 0, 255).astype('uint8')
            save_adv_expd_res = cv2.cvtColor(save_adv_expd_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "result_exp"), save_adv_expd_res)

            save_adv_hide_res = adv_hide_res.permute(1, 2, 0).detach().cpu().numpy()
            save_adv_hide_res = np.clip(save_adv_hide_res * 255, 0, 255).astype('uint8')
            save_adv_hide_res = cv2.cvtColor(save_adv_hide_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "result_attack_hid"), save_adv_hide_res)

            save_ben_hide_res = ben_hide_res.permute(1, 2, 0).detach().cpu().numpy()
            save_ben_hide_res = np.clip(save_ben_hide_res * 255, 0, 255).astype('uint8')
            save_ben_hide_res = cv2.cvtColor(save_ben_hide_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname.replace("example", "result_benign_hid"), save_ben_hide_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
