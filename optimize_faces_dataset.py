import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from joblib import Parallel, delayed
from tensorflow.python.util import deprecation

import util

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto(log_device_placement=True)
deprecation._PRINT_DEPRECATION_WARNINGS = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def parse_args():
    parser = argparse.ArgumentParser(description='Filter images that do not contain exactly 1 face and crop them')

    parser.add_argument('--input_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--target_size', nargs=2, default=(224, 224), help='Output images size in "x, y" format')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'JPEG', 'png'],
                        help='Extensions of images to process')
    parser.add_argument('--confidence_threshold', default=0.9, type=float,
                        help='Faces with less confidence wouldn\'t be processed')

    args = parser.parse_known_args()[0]
    return args


def cut_face_from_img(image: Image, face_coords, target_size, padding_fraction=0.2) -> np.ndarray:
    x1, y1, x2, y2 = face_coords
    face_width = x2 - x1
    face_height = y2 - y1
    face_center_x = x1 + face_width / 2
    face_center_y = y1 + face_height / 2

    aspect = face_width / float(face_height)
    ideal_aspect = target_size[1] / float(target_size[0])

    if aspect > ideal_aspect:
        # Then crop the left and right edges:
        face_width = int(ideal_aspect * face_height)
    else:
        # ... crop the top and bottom:
        face_height = int(face_width / ideal_aspect)

    left = (face_center_x - face_width / 2) - padding_fraction * face_width
    top = face_center_y - face_height / 2 - padding_fraction * face_height
    right = face_center_x + face_width / 2 + padding_fraction * face_width
    bottom = face_center_y + face_height / 2 + padding_fraction * face_height

    thumb = image.crop((left, top, right, bottom)).resize(target_size)

    return np.asarray(thumb)


face_detector = MTCNN(image_size=224, margin=20, min_face_size=50, keep_all=True)
tf.compat.v1.get_default_graph().finalize()


def optimize_img(root_dir, output_dir, img_path, target_size=(224, 224), face_presence_threshold=0.9):
    try:
        img = Image.open(img_path)
        output_file_path = img_path.replace(root_dir, output_dir, 1)
        output_file_dir = str(Path(output_file_path).parent)
        os.makedirs(output_file_dir, exist_ok=True)
        boxes, _ = face_detector.detect(img)

        # Get face from image
        face_array = cut_face_from_img(img, boxes[0], target_size)

        Image.fromarray(face_array).save(output_file_path, format='jpeg')
        # print(f'SUCCESS -- saved "{output_file_path}"')
    except Exception as e:
        exception_msg = e.message if hasattr(e, 'message') else str(e)
        print(f'ERROR -- failed to process "{img_path}". Exception message: {exception_msg}')


def optimize_faces_dataset(root_dir, output_dir, img_paths, target_size=(224, 224), face_presence_threshold=0.9):
    Parallel(n_jobs=4)(delayed(optimize_img)(root_dir, output_dir, img_path, target_size, face_presence_threshold)
                       for img_path in img_paths)


def main():
    args = parse_args()
    img_paths = util.get_images_from_dir(args.input_dir, args.extensions)
    optimize_faces_dataset(
        root_dir=args.input_dir,
        output_dir=args.output_dir,
        img_paths=img_paths,
        target_size=tuple(args.target_size),
        face_presence_threshold=args.confidence_threshold
    )

    pass


if __name__ == '__main__':
    main()
