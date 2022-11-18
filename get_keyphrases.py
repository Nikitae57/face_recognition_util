import argparse
import os.path

import numpy as np
from keras_vggface.utils import V2_LABELS_PATH
import requests


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and preprocess VGG feature labels '
                                                 '(key phrases used for fetching celebs names)')
    parser.add_argument('--output_dir', required=True, help='Directory to save labels into')
    args = parser.parse_known_args()[0]

    return args


def main():
    args = parse_args()
    np_response = requests.get(V2_LABELS_PATH)
    np_tmp_file_path = os.path.join(args.output_dir, 'tmp.npy')
    with open(np_tmp_file_path, 'wb') as f:
        f.write(np_response.content)
    np_features = np.load(np_tmp_file_path)
    os.remove(np_tmp_file_path)

    features = [feature.replace('_', ' ') + '\n' for feature in np_features]

    output_file_path = os.path.join(args.output_dir, 'key_phrases.txt')
    with open(output_file_path, 'w') as f:
        f.writelines(features)


if __name__ == '__main__':
    main()

