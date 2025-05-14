import os
import json
import argparse
import openslide
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

import torch
import torchvision.transforms as torch_trans
from torchvision import models
from torchvision.models import (
    VGG16_Weights,
    ResNet18_Weights,
    ResNet50_Weights
)
import torch.nn as nn


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


logger = logging.getLogger(__name__)


def create_encoder(args):
    """
    Create a pretrained image encoder based on the specified architecture.
    """
    logger.info(f"Creating encoder: {args.image_encoder}")

    if args.image_encoder == 'vgg16':
        encoder = models.vgg16(weights=VGG16_Weights.DEFAULT).to(args.device)
        encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:-3])

    elif args.image_encoder == 'resnet50':
        encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)

    elif args.image_encoder == 'resnet18':
        encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)

    else:
        raise ValueError(f"Invalid image_encoder: {args.image_encoder}")

    logger.debug(f"{args.image_encoder} architecture:\n{encoder}")
    return encoder


def extract(args, image, encoder, transform=None):
    """
    Extract feature vector from a single image using the encoder.
    """
    with torch.no_grad():
        if transform is None:
            to_tensor = torch_trans.ToTensor()
            # Add normalization with ImageNet mean/std values
            # normalize = torch_trans.Normalize(
            #     mean=[0.485, 0.456, 0.406], 
            #     std=[0.229, 0.224, 0.225]
            # )
            # image = to_tensor(image).unsqueeze(dim=0).to(args.device)
            image = to_tensor(image).unsqueeze(dim=0).to(args.device)
        else:
            image = transform(image).unsqueeze(dim=0).to(args.device)

        feat = encoder(image).cpu().numpy()
        return feat


def extract_features(args, encoder, save_dir):
    """
    Extract features from all patches of each WSI based on its coord file.
    """
    coord_dir = Path(args.patch_dir) / 'coord'
    if not coord_dir.exists():
        logger.error(f"Coord directory does not exist: {coord_dir}")
        return

    coord_list = sorted(list(coord_dir.glob('*.json')))
    logger.info(f"Found {len(coord_list)} coord files in {coord_dir}")

    with torch.no_grad():
        encoder.eval()
        for i, coord_filepath in enumerate(coord_list):
            filename = coord_filepath.stem
            npz_filepath = save_dir / f'{filename}.npz'

            if npz_filepath.exists() and not args.exist_ok:
                logger.warning(f"{npz_filepath.name} exists, skipping...")
                continue

            # obtain the parameters needed for feature extraction from `coord` file
            with open(coord_filepath) as fp:
                coord_dict = json.load(fp)

            num_patches = coord_dict['num_patches']
            if num_patches == 0:
                logger.warning(f"{filename} has 0 patches, skipping...")
                continue

            num_row = coord_dict['num_row']
            num_col = coord_dict['num_col']
            coords = coord_dict['coords']
            patch_size_level0 = coord_dict['patch_size_level0']
            patch_size = coord_dict['patch_size']
            slide = openslide.open_slide(coord_dict['slide_filepath'])

            coords_bar = tqdm(coords, desc=f"{i+1}/{len(coord_list)} | {filename}")
            features, cds = [], []

            for c in coords_bar:
                img = slide.read_region(
                    location=(c['x'], c['y']),
                    level=0,
                    size=(patch_size_level0, patch_size_level0)
                ).convert('RGB').resize((patch_size, patch_size))

                feat = extract(args, img, encoder)
                features.append(feat)
                cds.append(np.array([c['row'], c['col']], dtype=int))

                coords_bar.set_description(f"{i + 1:3}/{len(coord_list):3} | filename: {filename}")
                coords_bar.update()

            img_features = np.concatenate(features, axis=0)
            cds = np.stack(cds, axis=0)

            np.savez(
                file=npz_filepath,
                filename=filename,
                num_patches=num_patches,
                num_row=num_row,
                num_col=num_col,
                img_features=img_features,
                coords=cds
            )

            logger.info(f"Saved features: {npz_filepath.name} ({num_patches} patches)")


def run(args):
    """
    Main logic entrypoint for feature extraction.
    """
    if args.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Save Directory
    if args.save_dir is not None:
        save_dir = Path(args.save_dir) / args.image_encoder
    else:  # if the save directory is not specified, `patch_dir/features/${image_encoder}` is used by default
        save_dir = Path(args.patch_dir) / 'features' / args.image_encoder

    save_dir.mkdir(parents=True, exist_ok=True)
    
    if Path(save_dir).exists():
        logger.info(f"Saving features to {save_dir}")

    encoder = create_encoder(args)
    extract_features(args, encoder, save_dir=save_dir)


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default='', help='Directory containing `coord` files')
    parser.add_argument('--save_dir', type=str, default=None, help='Optional directory to save features')
    parser.add_argument('--image_encoder', type=str, default='resnet18',
                        choices=['vgg16', 'resnet18', 'resnet50'], help='CNN model to use')
    parser.add_argument('--device', default='3', help='CUDA device index or "cpu"')
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help='Allow overwriting existing .npz files')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
