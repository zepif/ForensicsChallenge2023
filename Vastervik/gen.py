import os
import cv2
import numpy as np

def extract_keypoints_and_descriptors(image_paths):
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list

def calibrate_camera(keypoints_list, descriptors_list, image_size):
    obj_points = []
    img_points = []

    for i in range(len(keypoints_list)):
        keypoints = keypoints_list[i]
        descriptors = descriptors_list[i]

        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

        corners = cv2.goodFeaturesToTrack(descriptors, maxCorners=64, qualityLevel=0.01, minDistance=10)
        if corners is not None and len(corners) > 0:
            img_points.append(corners)
            obj_points.append(objp)

    ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    return camera_matrix, distortion_coefficients

def reconstruct_3d_scene(keypoints_list, descriptors_list, camera_matrix, distortion_coefficients):
    reconstructed_points = []

    for i in range(len(keypoints_list)):
        keypoints = keypoints_list[i]
        descriptors = descriptors_list[i]

        object_points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)

        undistorted_image_points = cv2.undistortPoints(
            object_points.reshape(-1, 1, 2), camera_matrix, distortion_coefficients
        )

        z_plane = np.zeros_like(undistorted_image_points[:, :, 0])
        object_points_3d = np.dstack((undistorted_image_points[:, :, 0], undistorted_image_points[:, :, 1], z_plane))

        reconstructed_points.append(object_points_3d)

    return reconstructed_points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_model(reconstructed_points, image_paths):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, points in enumerate(reconstructed_points):
        x = points[:, :, 0].flatten()
        y = points[:, :, 1].flatten()
        z = points[:, :, 2].flatten()

        image = cv2.imread(image_paths[i])
        color_sample = image[np.random.choice(image.shape[0], size=len(x), replace=True),
                             np.random.choice(image.shape[1], size=len(x), replace=True)] / 255.0


        ax.scatter(x, y, z, c=color_sample, marker='o', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Model Visualization')
    plt.show()


def load_image_paths(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

if __name__ == "__main__":
    #image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    data_folder = "data"
    image_paths = load_image_paths(data_folder)

    keypoints_list, descriptors_list = extract_keypoints_and_descriptors(image_paths)

    image = cv2.imread(image_paths[0])
    image_size = (image.shape[1], image.shape[0])

    camera_matrix, distortion_coefficients = calibrate_camera(keypoints_list, descriptors_list, image_size)

    reconstructed_points = reconstruct_3d_scene(keypoints_list, descriptors_list, camera_matrix, distortion_coefficients)

    visualize_3d_model(reconstructed_points, image_paths)