import json
import shutil
from pathlib import Path
import argparse
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

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
        data_path / 'train' / 'image',
        img_save_path / 'bigdata_train'
    )

    with (data_path / 'train' / 'train.json').open() as f:
        annos = json.load(f)

    annos['categories'][0]['id'] = 1
    for img in annos['images']:
        img['file_name'] = img['file_name'].replace('image/', '')

    for anno in annos['annotations']:
        anno['category_id'] = 1

    with (anno_save_path / 'bigdata_train.json').open('w') as f:
        json.dump(annos, f, indent=4)

    shutil.copytree(
        data_path / 'val' / 'image',
        img_save_path / 'bigdata_val'
    )

    with (data_path / 'val' / 'val.json').open() as f:
        annos = json.load(f)

    annos['categories'][0]['id'] = 1
    for img in annos['images']:
        img['file_name'] = img['file_name'].replace('image/', '')

    for anno in annos['annotations']:
        anno['category_id'] = 1

    with (anno_save_path / 'bigdata_val.json').open('w') as f:
        json.dump(annos, f, indent=4)

    shutil.copytree(
        data_path / 'test' / 'image',
        img_save_path / 'bigdata_test'
    )

    annos = dict()
    annos['categories'] = [{
        "id": 1,
        "name": "building",
        "supercategory": "building"
    }]
    annos['images'] = []
    annos['annotations'] = []
    for i, img_path in enumerate(sorted((img_save_path / 'bigdata_test').iterdir())):
        width, height = Image.open(img_path).size
        annos['images'].append({
            'id': i,
            'file_name': img_path.name,
            'width': width,
            'height': height
        })

    with (anno_save_path / 'bigdata_test.json').open('w') as f:
        json.dump(annos, f, indent=4)


if __name__ == '__main__':
    args = get_args()
    main(args)
