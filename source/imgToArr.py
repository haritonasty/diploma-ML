import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))


def main():
    images = load_images(224, 224)  # 1 - width, 2 - height
    save_images(images)


def process_file(filename, w, h):
    img = Image.open(filename).resize((w, h))
    img = np.array(img.getdata()).astype('float32') / 255.
    return img.flatten()


def load_images(width=224, heigth=224, folder="images"):
    print("load images ...")
    files = glob((os.path.join(my_path, '../data/%s/*.jpeg' % folder)))
    images = [process_file(filename, width, heigth) for filename in files]
    return images


def save_images(images):
    print("save images ...")
    pd.DataFrame(images).to_csv("../data/marks/dresses.csv", index=False)
    print("finish save images ...")


if __name__ == "__main__":
    main()
