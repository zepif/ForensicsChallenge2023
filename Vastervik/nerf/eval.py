import torch
import matplotlib.pyplot as plt
from dataset import SyntheticDataset
from model import NeRF

if __name__ == "__main__":
    model_checkpoint_path = "path_to_your_model_checkpoint.pth"

    model = NeRF()
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()

    inference_dataset = SyntheticDataset(
        root="path_to_inference_dataset",
        csv_filepath="path_to_inference_csv_file.csv"
    )

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset, batch_size=1, shuffle=False
    )

    for idx, data in enumerate(inference_dataloader):
        camera_position = data["camera_position"]
        camera_orientation = data["camera_orientation"]
        theta = data["theta"]
        phi = data["phi"]

        input_tensor = torch.cat((theta, phi, point_cloud), dim=1)

        with torch.no_grad():
            pixel_colors = model(input_tensor)

        rendered_image = pixel_colors.view(
            inference_dataset.img_width, inference_dataset.img_height, 3
        ).numpy()

        plt.imshow(rendered_image)
        plt.title(f"Inference Image {idx+1}")
        plt.axis("off")
        plt.show()
