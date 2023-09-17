import torch

from dataset import SyntheticDataset
from model import NeRF
from geometry import Camera, Volume, get_points

def train(
    train_dataset_path: str = "",
    valid_dataset_path: str = "",
    test_dataset_path: str = "",
    run_name: str = "test",
    num_epochs: int = 10,
    num_layers: int = 3,
    num_hidden: int = 256,
    use_leaky_relu: bool = False,
    lr: float = 0.001,
    batch_size: int = 2,
) -> float:
    volume = Volume(

    )

    train_dataset = SyntheticDataset(root=train_dataset_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SyntheticDataset(root=valid_dataset_path)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = SyntheticDataset(root=test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = NeRF(
        num_layers=num_layers,
        num_hidden=num_hidden,
        use_leaky_relu=use_leaky_relu,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        for d in train_dataloader:
            optimizer.zero_grad()

            camera = Camera(
                position=d["camera_position"],
                orientation=d["camera_orientation"],
                focal_length=0.5,
                img_width=128,
                img_height=128,
            )

            theta = d["theta"]
            phi = d["phi"]

            for point in get_points(camera, volume):
                single_point_model_input = torch.cat((theta, phi, point), dim=1)
            outputs = model(single_point_model_input)
            assert outputs.shape[0] == batch_size
            predicted_pixel_value = torch.zeros(batch_size, 3)
            for point in range(batch_size):
                opacity = outputs[point, 0]
                color = outputs[point, 1:]
                predicted_pixel_value += color * (1 - opacity)

            loss = torch.mean((predicted_pixel_value - d["pixel_value"])**2)

            if loss < best_loss:
                best_loss = loss

            loss.backward()

            optimizer.step()

        for image, theta, phi in valid_dataloader:

            pass

    for image, theta, phi in test_dataloader:

        pass

    return best_loss