


import json
import random
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import cv2 as cv
from tqdm import tqdm


def transform_img(path: Path, output_path, desired_shape=(512, 512)):
  img = pydicom.dcmread(path)
  H, W = img.pixel_array.shape

  res_img = np.zeros(desired_shape, dtype=img.pixel_array.dtype)
  if W > H:
    new_H = int(desired_shape[0] * (H / W))
    pad_top = (desired_shape[0] - new_H) // 2
    
    cv.resize(img.pixel_array, (desired_shape[0], new_H), dst=res_img[pad_top:pad_top + new_H,  :] )

  elif H > W:
    new_W = int(desired_shape[1] * (W / H))
    pad_left = (desired_shape[1] - new_W) // 2
    cv.resize(img.pixel_array, (new_W, desired_shape[1]), dst=res_img[:, pad_left:pad_left + new_W])
  else:
    cv.resize(img.pixel_array, desired_shape, dst=res_img)


  np.save(output_path, res_img)  

images = list(json.load(open("images.json", "r")).items())

random.seed(987324)
# indices = random.sample(images, 100)



for name, file in tqdm(images):
  output_dir = Path("processed")
  output_dir.mkdir(parents=True, exist_ok=True)
  transform_img(Path(file), output_dir / name)