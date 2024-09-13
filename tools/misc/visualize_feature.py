import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# File paths
json_path = "data/cooperative-vehicle-infrastructure/cooperative/data_info_new.json"
show_dir = "show_dir/feature_0913"
image_dir = "data/cooperative-vehicle-infrastructure"

# Load JSON data
with open(json_path, "r") as f:
    data_info = json.load(f)

# Map vehicle to infrastructure frames
frame_map = {}
for item in data_info:
    vehicle_idx = item["vehicle_idx"].lstrip("0")  # Remove leading zeros
    infrastructure_idx = item["infrastructure_idx"].lstrip("0")
    vehicle_img_path = os.path.join(image_dir, item["vehicle_image_path"])
    infrastructure_img_path = os.path.join(image_dir, item["infrastructure_image_path"])
    frame_map[vehicle_idx] = (
        infrastructure_idx,
        vehicle_img_path,
        infrastructure_img_path,
    )

# Get all feature files
files = os.listdir(show_dir)
camera_voxel_files = [f for f in files if "camera_voxel.npy" in f]
camera_bev_files = [f for f in files if "camera_bev.npy" in f]
fusion_camera_files = [f for f in files if "fusion_camera.npy" in f]
fusion_lidar_files = [f for f in files if "fusion_lidar.npy" in f]
fusion_bev_files = [f for f in files if "fusion_bev.npy" in f]
print(f"Found {len(camera_voxel_files)} camera voxel files")
print(f"Found {len(camera_bev_files)} camera BEV files")
print(f"Found {len(fusion_camera_files)} fusion camera files")
print(f"Found {len(fusion_lidar_files)} fusion lidar files")
print(f"Found {len(fusion_bev_files)} fusion BEV files")

# Extract frame numbers
camera_voxel_frames = {f.split("_")[0].lstrip("0"): f for f in camera_voxel_files}
camera_bev_frames = {f.split("_")[0].lstrip("0"): f for f in camera_bev_files}
fusion_camera_frames = {f.split("_")[0].lstrip("0"): f for f in fusion_camera_files}
fusion_lidar_frames = {f.split("_")[0].lstrip("0"): f for f in fusion_lidar_files}
fusion_bev_frames = {f.split("_")[0].lstrip("0"): f for f in fusion_bev_files}

# Find common frames and visualize
for vehicle_idx, (
    infrastructure_idx,
    vehicle_img_path,
    infrastructure_img_path,
) in frame_map.items():
    if (
        vehicle_idx in camera_voxel_frames
        and vehicle_idx in camera_bev_frames
        and vehicle_idx in fusion_camera_frames
        and vehicle_idx in fusion_lidar_frames
        and vehicle_idx in fusion_bev_frames
    ):
        print(f"Visualizing frame veh-{vehicle_idx} and inf-{infrastructure_idx}")

        # Load features
        camera_voxel_feature = (
            np.load(os.path.join(show_dir, camera_voxel_frames[vehicle_idx]))
            .squeeze()
            .mean(axis=-1)
        )
        camera_bev_feature = (
            np.load(os.path.join(show_dir, camera_bev_frames[vehicle_idx]))
            .squeeze()
            .transpose(0, 2, 1)
        )
        fusion_camera_feature = (
            np.load(os.path.join(show_dir, fusion_camera_frames[vehicle_idx]))
            .squeeze()
            .transpose(0, 2, 1)
        )
        fusion_lidar_feature = (
            np.load(os.path.join(show_dir, fusion_lidar_frames[vehicle_idx]))
            .squeeze()
            .transpose(0, 2, 1)
        )
        fusion_bev_feature = (
            np.load(os.path.join(show_dir, fusion_bev_frames[vehicle_idx]))
            .squeeze()
            .transpose(0, 2, 1)
        )

        # Load images
        vehicle_image = Image.open(vehicle_img_path)
        infrastructure_image = Image.open(infrastructure_img_path)

        # Calculate mean and max values across channels for each spatial grid
        camera_voxel_mean = np.fliplr(np.flipud(camera_voxel_feature.mean(axis=0)))
        camera_voxel_max = np.fliplr(np.flipud(camera_voxel_feature.max(axis=0)))
        camera_bev_mean = np.fliplr(np.flipud(camera_bev_feature.mean(axis=0)))
        camera_bev_max = np.fliplr(np.flipud(camera_bev_feature.max(axis=0)))
        fusion_camera_mean = np.fliplr(np.flipud(fusion_camera_feature.mean(axis=0)))
        fusion_camera_max = np.fliplr(np.flipud(fusion_camera_feature.max(axis=0)))
        fusion_lidar_mean = np.fliplr(np.flipud(fusion_lidar_feature.mean(axis=0)))
        fusion_lidar_max = np.fliplr(np.flipud(fusion_lidar_feature.max(axis=0)))
        fusion_bev_mean = np.fliplr(np.flipud(fusion_bev_feature.mean(axis=0)))
        fusion_bev_max = np.fliplr(np.flipud(fusion_bev_feature.max(axis=0)))

        # Plot images and features
        plt.figure(figsize=(24, 6))

        plt.subplot(2, 6, 1)
        plt.title(f"Vehicle Image - Frame {vehicle_idx}")
        plt.imshow(vehicle_image)
        plt.axis("off")

        plt.subplot(2, 6, 6 + 1)
        plt.title(f"Infrastructure Image - Frame {infrastructure_idx}")
        plt.imshow(infrastructure_image)
        plt.axis("off")

        plt.subplot(2, 6, 2)
        plt.title(f"Camera Voxel Feature Mean")
        plt.imshow(camera_voxel_mean, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6 + 2)
        plt.title(f"Camera Voxel Feature Max")
        plt.imshow(camera_voxel_max, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 3)
        plt.title(f"Camera BEV Feature Mean")
        plt.imshow(camera_bev_mean, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6 + 3)
        plt.title(f"Camera BEV Feature Max")
        plt.imshow(camera_bev_max, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 4)
        plt.title(f"Fusion Camera Feature Mean")
        plt.imshow(fusion_camera_mean, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6 + 4)
        plt.title(f"Fusion Camera Feature Max")
        plt.imshow(fusion_camera_max, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 5)
        plt.title(f"Fusion LiDAR Feature Mean")
        plt.imshow(fusion_lidar_mean, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6 + 5)
        plt.title(f"Fusion LiDAR Feature Max")
        plt.imshow(fusion_lidar_max, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6)
        plt.title(f"Fusion BEV Feature Mean")
        plt.imshow(fusion_bev_mean, aspect="auto")
        plt.colorbar()

        plt.subplot(2, 6, 6 + 6)
        plt.title(f"Fusion BEV Feature Max")
        plt.imshow(fusion_bev_max, aspect="auto")
        plt.colorbar()

        plt.tight_layout()
        # plt.show()

        # Save the figure
        plt.savefig(os.path.join(show_dir, f"comparison_{vehicle_idx}.png"))

        # Close the figure to avoid memory issues
        plt.close()
