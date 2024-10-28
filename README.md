# Attack as Defense: Run-time Backdoor Implantation for Image Content Protection
This is the official PyTorch implementation of our paper: "Attack as Defense: Run-time Backdoor Implantation for Image Content Protection".

## Environment preparation

Create a new conda environment of Python 3.9.1 called Attack4Defense:
```
conda create -n Attack4Defense python=3.9.1
conda activate Attack4Defense
```
Install required packages:
```
pip install -r requirements.txt
```

Or create a new conda environment with:
```
conda env create -f environment.yml
```

## Code preparation
```
cd ~
git clone <repo_url> Attack4Defense
cd Attack4Defense
```

Prepare the target models ([LaMa](https://github.com/advimman/lama), [MAT](https://github.com/fenglinglwb/mat), [LDM](https://github.com/CompVis/latent-diffusion)) following their official instructions. and replace the files and folders in target model with the same names as this repo. Download their official pretrained model weights ([LaMa's Weight](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips), [MAT's Weight](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips), [LDM-inpainting's Weight](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/main)) ) into each model's weigth directory (e.g., put `LaMa.pth` at ./Attack_as_Defense/LaMa/big-lama/).

## Dataset preparation
You will need to download the [Watermark dataset](https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images) and organize it in the following way. Assume the path of the dataset is `./Attack_as_Defense/dir_in/Watermark/`. And Prepare the name of file as `image: NAME_example.png, mask: NAME_example_mask.png`.


## Configuration preparation
Coming soon...
