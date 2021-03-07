import argparse
import os
import time
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
from mtcnn.mtcnn import MTCNN
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_args():
    parser = argparse.ArgumentParser(description='Filter images that do not contain exactly 1 face and crop them')

    parser.add_argument('--input_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--target_size', nargs=2, default=(224, 224), help='Output images size in "x, y" format')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'JPEG', 'png'], help='Extensions of images to process')
    parser.add_argument('--confidence_threshold', default=0.9, type=float, help='Faces with less confidence wouldn\'t be processed')

    args = parser.parse_known_args()[0]
    return args


def get_faces_coords(img_array: np.ndarray, face_detector, threshold=0.9):
    recognition_results = face_detector.detect_faces(img_array)
    faces_coords = []
    for i in range(len(recognition_results)):
        if recognition_results[i]['confidence'] < threshold:
            continue
        x1, y1, width, height = recognition_results[i]['box']
        x2, y2 = x1 + width, y1 + height
        faces_coords.append((y1, y2, x1, x2))

    return faces_coords


def cut_face_from_img(img_array: np.ndarray, face_coords, target_size, padding_fraction=0.2) -> np.ndarray:
    image = Image.fromarray(img_array)

    y1, y2, x1, x2 = face_coords
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


def get_images_from_dir(root_dir, extensions=('.png', '.jpg', '.JPEG')):
    img_paths = []

    for filename in os.listdir(root_dir):
        possible_dir = os.path.join(root_dir, filename)
        if os.path.isdir(possible_dir):
            img_paths.extend(get_images_from_dir(possible_dir))

        _, file_extension = os.path.splitext(filename)
        if file_extension not in extensions:
            continue

        img_paths.append(os.path.join(root_dir, filename))

    return img_paths


def optimize_img(root_dir, output_dir, img_path, target_size=(224, 224), face_presence_threshold=0.9):
    try:
        face_detector = MTCNN()
        image = Image.open(img_path)
        image.thumbnail((500, 500))
        img_array = np.asarray(image)

        # Preserve only images with a single face
        faces_coords = get_faces_coords(img_array, face_detector, face_presence_threshold)
        if len(faces_coords) != 1:
            return

        # Get face from image
        face_array = cut_face_from_img(img_array, faces_coords[0], target_size)

        # Save face
        output_file_path = img_path.replace(root_dir, output_dir, 1)
        output_file_dir = str(Path(output_file_path).parent)
        os.makedirs(output_file_dir, exist_ok=True)
        Image.fromarray(face_array).save(output_file_path, format='jpeg')
        print(f'SUCCESS -- saved "{output_file_path}"')
    except Exception as e:
        exception_msg = e.message if hasattr(e, 'message') else str(e)
        print(f'ERROR -- failed to process "{img_path}". Exception message: {exception_msg}')


def optimize_faces_dataset(root_dir, output_dir, img_paths, target_size=(224, 224), face_presence_threshold=0.9):
    Parallel(n_jobs=12)(delayed(optimize_img)(root_dir, output_dir, img_path , target_size, face_presence_threshold)
                        for img_path in img_paths)


def main():
    args = parse_args()
    img_paths = get_images_from_dir(args.input_dir, args.extensions)
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
