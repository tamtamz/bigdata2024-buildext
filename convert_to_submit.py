from mmengine.fileio import load
import pandas as pd
from tqdm import tqdm
from pycocotools import mask as mask_utils
import cv2
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='dets.csv')
    parser.add_argument('--score_threshold', type=float, default=0.3)

    args = parser.parse_args()

    return args


def main(args):
    dets = load(args.input_path)

    img_ids = []
    coords = []
    for img_dets in tqdm(dets):
        img_ids.append(img_dets['img_id'])
        coord_str = '['
        for mask, score in zip(img_dets['pred_instances']['masks'], img_dets['pred_instances']['scores']):
            if score < args.score_threshold:
                continue

            cont = cv2.findContours(mask_utils.decode(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            if len(cont[0]) == 0:
                continue

            cont = cont[0][0]

            if cont.shape[0] < 3:
                continue

            coord_str += '['
            for i in range(cont.shape[0]):
                coord_str += f'({cont[i][0][0]}, {cont[i][0][1]}), '

            if coord_str[-2:] == ', ':
                coord_str = coord_str[:-2]
            coord_str += '], '
        if coord_str[-2:] == ', ':
            coord_str = coord_str[:-2]
        coord_str += ']'
        coords.append(coord_str)

    df = pd.DataFrame({'ImageID': img_ids, 'Coordinates': coords})
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)