from pathlib import Path
from matplotlib.widgets import Slider
import os
import pydicom

import matplotlib.pyplot as plt

# Function to load DICOM images into a list
def load_dicom_stack(folder_path):
  
  dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
  dicom_files.sort(key=lambda x: int(Path(x).name.split(".")[0]))  # Ensure the files are sorted correctly
  dicom_stack = [pydicom.dcmread(f).pixel_array for f in dicom_files]
  return dicom_stack

# Load the DICOM stack
folder_path = "test_images/44036939/3844393089"
dicom_stack = load_dicom_stack(folder_path)

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Display the first image
img_display = ax.imshow(dicom_stack[0], cmap=plt.cm.gray)
ax.set_title(f"Image 1 of {len(dicom_stack)}")
ax.axis('off')

# Create a slider axis and slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Image', 1, len(dicom_stack), valinit=1, valstep=1)

# Update function for the slider
def update(val):
  img_index = int(slider.val) - 1
  img_display.set_data(dicom_stack[img_index])
  ax.set_title(f"Image {img_index + 1} of {len(dicom_stack)}")
  fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()