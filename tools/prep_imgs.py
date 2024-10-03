


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
  
  with pydicom.dcmread(path) as img:
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
  
  if res_img.dtype == np.uint16:
    res_img = (res_img - 32768).astype(np.int16)

  # enable this for visualizing the images
  # plt.imsave(output_path.with_suffix(".png"), res_img, cmap="gray")
  # plt.imsave(output_path.with_suffix(".original.png"), img.pixel_array, cmap="gray")

  # store the images in numpy format
  np.save(output_path, res_img)

images = list(json.load(open("images.json", "r")).items())


lookup = {}
output_dir = Path("processed")
for name, file in tqdm(images):
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / name
  lookup[name] = str(output_path)
  transform_img(Path(file), output_path)

json.dump(lookup, fp=open(output_dir / "lookup.json", "w"), indent=2)