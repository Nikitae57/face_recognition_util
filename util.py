import os


def get_images_from_dir(root_dir, extensions=('.png', '.jpg', '.JPEG')):
    img_paths = []

    for filename in os.listdir(root_dir):
        filename = os.path.join(root_dir, filename)
        if os.path.isdir(filename):
            img_paths.extend(get_images_from_dir(filename, extensions=extensions))
        else:
            _, file_extension = os.path.splitext(filename)
            if file_extension not in extensions:
                print(filename)
                continue
            img_paths.append(filename)

    return img_paths
