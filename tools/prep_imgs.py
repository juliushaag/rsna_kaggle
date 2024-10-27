


import json
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
import concurrent.futures



def transform_img(path: Path, desired_shape=(512, 512)):
  
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
  
  # min max normalization 
  res_img = res_img - res_img.min()
  res_img = res_img / res_img.max()
  return res_img.astype(np.float16)


if __name__ == "__main__":
  images = list(json.load(open("images.json", "r")).items())


  def thread_task(name, file):
    output_path = Path("processed") / name
    img = transform_img(file)
    np.save(output_path.with_suffix(".npy"), img)

  lookup = {}
  output_dir = Path("processed")
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    futures = list()
    for name, file in images:
      futures.append(executor.submit(thread_task, name, file))
      lookup[name] = file

    for future in tqdm(concurrent.futures.as_completed(futures)):
      future.result()



  json.dump(lookup, fp=open(output_dir / "lookup.json", "w"), indent=2)