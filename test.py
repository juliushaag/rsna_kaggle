import os
import pydicom
import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.app import use_app

# Function to load DICOM images into a list
def load_dicom_stack(folder_path):
  dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
  dicom_files.sort()  # Ensure the files are sorted correctly
  dicom_stack = [pydicom.dcmread(f).pixel_array for f in dicom_files]
  return dicom_stack

# Load the DICOM stack
folder_path = "train_images/10728036/142859125"
dicom_stack = load_dicom_stack(folder_path)

# Convert the list of 2D images into a 3D numpy array
dicom_stack = np.array(dicom_stack)

# Normalize the dicom_stack for visualization
dicom_stack = dicom_stack / np.max(dicom_stack)

# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Create the volume visual
volume = scene.visuals.Volume(dicom_stack, parent=view.scene, threshold=0.225)

# # Add a colorbar
# colormap = visuals.ColorBar(
#   cmap='grays', clim=(0, 1), orientation='right', size=(10, 200), parent=canvas.scene)

# # # Position the colorbar
# # colormap.pos = canvas.size[0] - 50, canvas.size[1] // 2

# Run the application
use_app().run()
