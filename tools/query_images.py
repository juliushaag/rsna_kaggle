import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import json

json.dump({ str(d + ":" + i.name + ":" + s.name + ":" + f.name.split(".")[0]) : str(f) for d in ["train", "test"] for i in Path(d + "_images").iterdir() for s in i.iterdir() for f in s.glob("*.dcm")}, fp=open("images.json", "w"), indent=2)