import os
import argparse
import shutil
import time

import requests
from joblib import Parallel, delayed
from selenium import webdriver
import selenium
from PIL import Image
import io
import hashlib

DRIVER_PATH = '/Users/evdokimov_na/projects/other/chromedriver'


def fetch_image_urls(key_phrase: str, max_links_to_fetch: int, web_driver: webdriver, sleep_between_interactions: int = 1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    web_driver.get(search_url.format(q=key_phrase))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(web_driver)

        # get all image thumbnail results
        thumbnail_results = web_driver.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = web_driver.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            load_more_button = web_driver.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                web_driver.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path: str, file_name: str, url: str):
    try:
        image_content = requests.get(url, timeout=5).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
        return

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        folder_path = os.path.join(folder_path, file_name)

        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch images from google by keyword. Program saves images for each '
                                                 'keyword in a separate directory with name identical to keyword')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--key_phrases_file', help='File with key phrases separated by newline')
    group.add_argument('--key_phrases', nargs='+', help='Comma separated list of key phrases')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_images_per_key_phrase', type=int, default=10, help='Number of images to download for each key phrase')

    args = parser.parse_known_args()[0]
    return args


def assert_output_dir_is_empty(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir)) != 0:
        raise IOError(f'Directory is not empty: {output_dir}')


def fetch_images_for_key_phrases(key_phrases: [str], output_dir: str, num_images_per_key_phrase: int):
    assert_output_dir_is_empty(output_dir)

    for key_phrase in key_phrases:
        wd = webdriver.Chrome(executable_path=DRIVER_PATH)
        wd.get('https://google.com')
        search_box = wd.find_element_by_css_selector('input.gLFyf')
        search_box.send_keys(key_phrase)
        links = fetch_image_urls(key_phrase, num_images_per_key_phrase, wd)
        # images_path = '/Users/anand/Desktop/contri/images'  #enter your desired image path
        for i in links:
            persist_image(output_dir, key_phrase, i)
        wd.quit()
    # Parallel(n_jobs=12)(delayed(response.download)(key_phrase, num_images_per_key_phrase, allowed_extensions, output_dir) for key_phrase in key_phrases)


def parse_key_phrases_file(path: str) -> [str]:
    if not os.path.exists(path):
        raise IOError(f'Key phrases file does not exist: {path}')

    with open(path) as f:
        key_phrases = f.read().splitlines()

    return key_phrases


def main():
    args = parse_args()

    try:
        if args.key_phrases is not None:
            key_phrases = args.key_phrases
        else:
            key_phrases = parse_key_phrases_file(args.key_phrases_file)

        fetch_images_for_key_phrases(key_phrases, args.output_dir, args.num_images_per_key_phrase)
    except IOError as e:
        print(e)


if __name__ == '__main__':
    main()
