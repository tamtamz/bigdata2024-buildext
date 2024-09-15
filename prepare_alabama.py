import json
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--img_type', type=str, default='png')

    args = parser.parse_args()

    return args


def main(args):
    data_path = Path(args.data_path)
    save_path = Path(args.save_path)

    img_save_path = save_path / 'images'
    img_save_path.mkdir(parents=True, exist_ok=True)

    anno_save_path = save_path / 'annotations'
    anno_save_path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        data_path / 'image',
        img_save_path / 'alabama'
    )

    annos = {
        'categoreis': [{'id': 1, 'name': 'building', 'supercategory': 'building'}],
        'images': [],
        'annotations': []
    }

    img_id = 1
    anno_id = 1
    for mask_path in tqdm(sorted((data_path / 'mask').iterdir())):
        annos['images'].append({
            'id': img_id,
            'width': args.width,
            'height': args.height,
            'file_name': f'{mask_path.stem}.{args.img_type}'
        })

        mask = cv2.imread(str(mask_path))
        mask = 255 - mask[:, :, 0]
        outs = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[-2]
        hierarchy = outs[-1]
        if hierarchy is None:
            continue

        with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0)
        for contour, h in zip(contours, with_hole):
            if h:
                continue

            segmentation = contour.reshape((-1,)).tolist()
            contour = contour.reshape((-1, 2))
            xmin, ymin = contour.min(axis=0)
            xmax, ymax = contour.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin
            annos['annotations'].append({
                'id': anno_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [xmin.item(), ymin.item(), width.item(), height.item()],
                'segmentation': [segmentation]
            })
            anno_id += 1

        img_id += 1        

    with (anno_save_path / 'alabama.json').open('w') as f:
        json.dump(annos, f, indent=4)


if __name__ == '__main__':
    args = get_args()
    main(args)
