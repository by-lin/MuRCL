import cv2
import json
import argparse
import openslide
import numpy as np
import logging
from pathlib import Path
import torch

from .filters import adaptive, otsu, RGB_filter
from .utils import get_three_points, keep_patch, out_of_bound


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)


def tiling(slide_filepath, magnification, patch_size, scale_factor=32, tissue_thresh=0.35, method='rgb',
           overview_level=-1, coord_dir=None, overview_dir=None, mask_dir=None, patch_dir=None, filename=None):
    logger.info(f"Starting tiling for slide: {slide_filepath}")
    slide = openslide.open_slide(str(slide_filepath))
    logger.debug(f"Slide properties: {slide.properties}")

    if 'aperio.AppMag' in slide.properties:
        level0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties:
        level0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level0_magnification = 40

    if level0_magnification < magnification:
        logger.warning(f"Requested magnification {magnification} higher than available {level0_magnification}")
        return

    patch_size_level0 = int(patch_size * (level0_magnification / magnification))
    logger.info(f"Patch size level0: {patch_size_level0}")

    # Generate mask
    mask_filepath = str(mask_dir / f'{filename}.png') if mask_dir else None
    if method == 'adaptive':
        mask, color_bg = adaptive(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'otsu':
        mask, color_bg = otsu(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'rgb':
        mask, color_bg = RGB_filter(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    else:
        raise ValueError(f"Invalid filter method: {method}")

    mask_w, mask_h = mask.size
    mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_GRAY2BGR)
    mask_patch_size = int(((patch_size_level0 // scale_factor) * 2 + 1) // 2)
    num_step_x = int(mask_w // mask_patch_size)
    num_step_y = int(mask_h // mask_patch_size)
    logger.info(f"Num steps: {num_step_y} rows x {num_step_x} cols")

    coord_list = []

    for row in range(num_step_y):
        for col in range(num_step_x):
            points_mask = get_three_points(col, row, mask_patch_size)
            patch_mask = mask[points_mask[0][1]:points_mask[1][1], points_mask[0][0]:points_mask[1][0]]
            if keep_patch(patch_mask, tissue_thresh, color_bg):
                points_level0 = get_three_points(col, row, patch_size_level0)
                if out_of_bound(slide.dimensions[0], slide.dimensions[1], points_level0[1][0], points_level0[1][1]):
                    continue
                coord_list.append({'row': row, 'col': col, 'x': points_level0[0][0], 'y': points_level0[0][1]})

    logger.info(f"Finished tiling {filename}. Total patches kept: {len(coord_list)}")

    coord_dict = {
        'slide_filepath': str(slide_filepath),
        'magnification': magnification,
        'magnification_level0': level0_magnification,
        'num_row': num_step_y,
        'num_col': num_step_x,
        'patch_size': patch_size,
        'patch_size_level0': patch_size_level0,
        'num_patches': len(coord_list),
        'coords': coord_list
    }
    with open(coord_dir / f'{filename}.json', 'w') as f:
        json.dump(coord_dict, f)


def run(args):
    logger.info(f"Arguments: {args}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    coord_dir = save_dir / 'coord'
    coord_dir.mkdir(parents=True, exist_ok=True)
    overview_dir = save_dir / 'overview' if args.overview else None
    mask_dir = save_dir / 'mask' if args.save_mask else None
    patch_dir = save_dir / 'patch' if args.save_patch else None

    slide_filepath_list = sorted(list(Path(args.slide_dir).rglob(f'*{args.wsi_format}')))
    logger.info(f"Found {len(slide_filepath_list)} slides")

    for idx, slide_filepath in enumerate(slide_filepath_list):
        filename = slide_filepath.stem
        logger.info(f"[{idx+1}/{len(slide_filepath_list)}] Processing {filename}")

        if (coord_dir / f'{filename}.json').exists() and not args.exist_ok:
            logger.info(f"Coord file for {filename} already exists. Skipping...")
            continue

        try:
            tiling(
                slide_filepath=slide_filepath,
                magnification=args.magnification,
                patch_size=args.patch_size,
                scale_factor=args.scale_factor,
                tissue_thresh=args.tissue_thresh,
                method=args.method,
                overview_level=args.overview_level,
                coord_dir=coord_dir,
                overview_dir=overview_dir,
                mask_dir=mask_dir,
                patch_dir=patch_dir,
                filename=filename,
            )
        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exist_ok', action='store_true')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--magnification', type=int, default=20, choices=[40, 20, 10, 5])
    parser.add_argument('--scale_factor', type=int, default=32)
    parser.add_argument('--tissue_thresh', type=float, default=0.35)
    parser.add_argument('--overview', action='store_true')
    parser.add_argument('--save_mask', action='store_true')
    parser.add_argument('--save_patch', action='store_true')
    parser.add_argument('--wsi_format', type=str, default='.svs', choices=['.svs', '.tif'])
    parser.add_argument('--method', type=str, default='rgb', choices=['otsu', 'adaptive', 'rgb'])
    parser.add_argument('--overview_level', type=int, default=-1)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
