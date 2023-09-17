from dataclasses import dataclass
from typing import List

import numpy as np
import torch

WORLD_UP_VEC: np.ndarray = np.array([0, 0, 1], dtype=np.float32)


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray


@dataclass
class Camera:
    position: np.ndarray
    orientation: np.ndarray
    image_width: int
    image_height: int
    focal_length: float
    forward_direction_vector: np.ndarray = np.array(
        [0, 0, -1], dtype=np.float32)
    right_direction_vector: np.ndarray = np.array([0, 1, 0], dtype=np.float32)


@dataclass
class Volume:
    size: np.ndarray
    resolution: np.ndarray
    position: np.ndarray = np.array([0, 0, 0])

    def normalize_position(self, position: np.ndarray) -> np.ndarray:
        volume_center = self.position + self.size / 2
        relative_position = position - volume_center
        normalized_position = relative_position / self.size
        return normalized_position


@dataclass
class Point:
    position: np.ndarray
    view_angle: np.ndarray
    color: np.ndarray
    opacity: float


def get_rays(camera: Camera) -> List[Ray]:
    cam_forward = np.dot(camera.orientation, camera.forward_direction_vector)
    image_center = camera.position + cam_forward * camera.focal_length
    image_width = 2 * camera.focal_length * \
        np.tan(np.radians(camera.image_width / 2))
    image_height = 2 * camera.focal_length * \
        np.tan(np.radians(camera.image_height / 2))
    
    cam_right = np.cross(cam_forward, camera.right_direction_vector.T)
    cam_up = np.cross(cam_forward, cam_right)

    image_top_left = image_center + cam_up * \
        (image_height / 2) - cam_right * (image_width / 2)

    right_step = cam_right * (image_width / camera.image_width)
    up_step = cam_up * (image_height / camera.image_height)

    rays = []

    for y in range(camera.image_height):
        for x in range(camera.image_width):
            pixel_pos = image_top_left + right_step * x + up_step * y
            ray_origin = camera.position
            ray_direction = pixel_pos - camera.position
            ray_direction /= np.linalg.norm(ray_direction)
            rays.append(Ray(origin=ray_origin, direction=ray_direction))

    return rays

def sample_points_in_ray(
    ray: Ray,
    volume: Volume,
    num_samples: int = 6,
) -> List[Point]:
    samples = np.linspace(0, 1, num_samples)

    points = []

    for sample in samples:
        point = ray.origin + sample * ray.direction

        within_x = volume.position[0] <= point[0] < volume.position[0] + \
            volume.size[0]
        within_y = volume.position[1] <= point[1] < volume.position[1] + \
            volume.size[1]
        within_z = volume.position[2] <= point[2] < volume.position[2] + \
            volume.size[2]

        if within_x and within_y and within_z:
            points.append(point)

    return points


def get_points(camera: Camera, volume: Volume) -> List[Point]:
    rays = get_rays(camera)

    points = []

    for ray in rays:
        points_in_ray = sample_points_in_ray(ray, volume)
        points.extend(points_in_ray)

    return points


def pixel_value_from_ray_samples() -> np.ndarray:
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal')

    camera = Camera(
        position=np.array([0, -2, 2], dtype=np.float32),
        orientation=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.6744415163993835, -0.7383282780647278],
            [0.0, 0.7383282780647278, 0.6744415163993835],
        ], dtype=np.float32),
        image_width=28,
        image_height=28,
        focal_length=0.5,
    )

    rays = get_rays(camera)

    for ray in get_rays(camera):
        ax.plot(
            [ray.origin[0], ray.direction[0]],
            [ray.origin[1], ray.direction[1]],
            [ray.origin[2], ray.direction[2]],
            '-o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()