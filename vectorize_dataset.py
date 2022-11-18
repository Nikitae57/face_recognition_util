import argparse
import os

import util

import numpy as np
from pathlib import Path
from PIL import Image
from img2vec_pytorch import Img2Vec
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description='Convert images dataset into numpy vectors. Result dataset will '
                                                 'have exact folder structure, but "npy" extension will be added '
                                                 'to image names')
    parser.add_argument('--input_dir', required=True, help='Root directory of images dataset')
    parser.add_argument('--output_dir', required=True, help='Directory to save numpy arrays into')
    args = parser.parse_known_args()[0]

    return args


def _preprocess_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def _process_img(root_dir, output_dir, img_path):
    dst_path = img_path.replace(root_dir, output_dir, 1)
    if os.path.exists(dst_path):
        return

    image = Image.open(img_path)
    vectorizer = Img2Vec(cuda=True)
    vector = vectorizer.get_vec(image)
    # save
    os.makedirs(str(Path(dst_path).parent), exist_ok=True)
    np.save(dst_path, vector)


def vectorize_dataset(img_root_dir, output_dir):
    _preprocess_output_dir(output_dir)

    img_paths = util.get_images_from_dir(img_root_dir)
    paths = []
    for img_path in img_paths:
        dst_path = img_path.replace(img_root_dir, output_dir, 1)
        if not os.path.exists(dst_path + '.npy'):
            # print(dst_path)
            paths.append(img_path)

    Parallel(n_jobs=12)(delayed(_process_img)(img_root_dir, output_dir, img_path) for img_path in paths)


def main():
    args = parse_args()
    vectorize_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

