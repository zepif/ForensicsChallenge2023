import os
import torch
import csv
from PIL import Image

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,
        root: str,
        csv_filepath: str,
    ):
        self.root = root
        self.img_width = 128
        self.img_height = 128

        self.snapshots = []

        with open(csv_filepath, "r") as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                self.shapshots += row
    
    def __len__(self):
        return len(self.shapshots) * self.img_width * self.img_height
    
    def __getitem__(self, index):
        snapshot_index = index % (self.img_width * self.img_height)
        snapshot = self.snapshots[snapshot_index]

        pixel_index = (index - snapshot_index*(self.img_width * self.img_height))
        pixel_y = pixel_index % self.img_width
        pixel_x = (pixel_index // self.img_height) % self.img_height

        image_filename = snapshot[0]

        camera_position = [snapshot[1], snapshot[2], snapshot[3]]

        camera_orientation = [snapshot[4], snapshot[5], snapshot[6], snapshot[7], snapshot[8], snapshot[9], snapshot[10], snapshot[11], snapshot[12]]

        theta = snapshot[13]
        phi = snapshot[14]

        file_path = os.path.join(self.root, image_filename)
        image = Image.open(file_path)
        image = image.convert("RGB")

        pixel_value = image.getpixel((pixel_x, pixel_y))
        
        return {
            "pixel_value": pixel_value,
            "camera_position": camera_position,
            "camera_orientation": camera_orientation,
            "theta": theta,
            "phi": phi,
        }