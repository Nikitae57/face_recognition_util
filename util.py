import os


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