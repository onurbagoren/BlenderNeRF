import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import json
import os
import torch
from kornia.geometry.depth import depth_to_3d_v2
from scipy.spatial.transform import Rotation as R
import re
import cv2


def fix_json(json_path, image_dir):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    json_dir = os.path.dirname(json_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    pattern = re.compile(r"_(\d+)\.png$")
    max_suffix = 0
    for img in image_files:
        match = pattern.search(img)
        if match:
            max_suffix = max(max_suffix, int(match.group(1)))

    suffix_to_remove = f"{max_suffix:04d}.png" if max_suffix > 0 else ""

    expected_frames = len(image_files)
    while len(data["frames"]) > expected_frames:
        data["frames"].pop(0)

    # Process frames
    updated_frames = []
    for frame in data["frames"]:
        file_path = frame["file_path"]

        # Remove "_000{frame_idx}.png" from file path
        corrected_file_name = (
            file_path.replace(suffix_to_remove, "") if suffix_to_remove else file_path
        )

        # Extract the frame number and decrement it by 1
        number_pattern = re.search(r"frame_(\d+)", corrected_file_name)
        if number_pattern:
            frame_number = int(number_pattern.group(1)) - 1
            corrected_file_name = re.sub(
                r"frame_(\d+)", f"frame_{frame_number:04d}", corrected_file_name
            )

        frame["file_path"] = corrected_file_name

        # Generate corresponding depth and normal file names
        base_name = os.path.splitext(os.path.basename(corrected_file_name))[0]
        depth_file = f"{base_name}_depth.exr"
        normal_file = f"{base_name}_normals.exr"

        # Check if these files exist
        depth_path = os.path.join(image_dir, depth_file)
        normal_path = os.path.join(image_dir, normal_file)
        if os.path.exists(depth_path) and os.path.exists(normal_path):
            frame["depth_file_path"] = depth_file
            frame["normal_file_path"] = normal_file

        updated_frames.append(frame)

    # Update the data
    data["frames"] = updated_frames

    # Save the updated JSON
    output_path = os.path.join(json_dir, "transforms_train.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON saved to {output_path}")


def transform_blender_to_normal(pose):
    # Convert Blender pose to standard camera convention
    transformation_matrix = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    return transformation_matrix @ pose @ np.linalg.inv(transformation_matrix)


def visualize_camera_poses(file_dir: str):
    with open(os.path.join(file_dir, "transforms_train.json"), "r") as f:
        train_json = json.load(f)
    frames = train_json["frames"]
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for ii, frame in enumerate(frames):
        camera_pose = frame["transform_matrix"]
        camera_pose = np.array(camera_pose).reshape(4, 4)
        normal_frame = transform_blender_to_normal(camera_pose)
        # camera_pose = np.linalg.inv(camera_pose)
        # camera_pose = camera_pose[:3, :4]
        # camera_pose = np.concatenate([camera_pose, np.array([[0, 0, 0, 1]])], axis=0)
        vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0).transform(
                normal_frame
            )
        )

        # read depth for th pointcloud
        depth_file_path = os.path.join(file_dir, frame["file_path"] + "_Z.exr")
        depth_exr = OpenEXR.InputFile(depth_file_path)
        dw = depth_exr.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        depth_channel = depth_exr.channel("G", Imath.PixelType(Imath.PixelType.FLOAT))
        depth_array = np.fromstring(depth_channel, dtype=np.float32)
        depth_array.shape = (size[1], size[0])
        depth_array[depth_array > 10] = 0

        # intrinsics
        fl_x = train_json["fl_x"]
        fl_y = train_json["fl_y"]
        cx = size[0] / 2
        cy = size[1] / 2
        intrinsics_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

        depth_pc = depth_to_3d_v2(
            torch.from_numpy(depth_array), torch.from_numpy(intrinsics_matrix)
        ).numpy()
        depth_pc = depth_pc.reshape(-1, 3)

        rgb = cv2.imread(os.path.join(file_dir, frame["file_path"] + ".png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd.transform(normal_frame)

        vis.add_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
    vis.run()
    vis.destroy_window()


def visualize(train_path: str, frame_number: int):
    # convert frame number to {000{frame_number:04d}}
    frame_str = f"{frame_number:04d}"
    depth_file_path = f"{train_path}/frame_{frame_str}_Z.exr"
    depth_exr = OpenEXR.InputFile(depth_file_path)
    dw = depth_exr.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    depth_channel = depth_exr.channel("G", Imath.PixelType(Imath.PixelType.FLOAT))
    depth_array = np.fromstring(depth_channel, dtype=np.float32)
    depth_array.shape = (size[1], size[0])
    depth_array[depth_array > 10] = -1

    normals_file_path = f"{train_path}/frame_{frame_str}_Normal.exr"
    normals_exr = OpenEXR.InputFile(normals_file_path)
    norm_x = normals_exr.channel("X", Imath.PixelType(Imath.PixelType.FLOAT))
    norm_y = normals_exr.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT))
    norm_z = normals_exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
    nx = np.frombuffer(norm_x, dtype=np.float32).reshape((size[1], size[0]))
    ny = np.frombuffer(norm_y, dtype=np.float32).reshape((size[1], size[0]))
    nz = np.frombuffer(norm_z, dtype=np.float32).reshape((size[1], size[0]))
    normals_array = np.stack([nx, ny, nz], axis=-1)
    normals_vis = normals_array + 1.0  # / 2.0
    normals_vis = np.clip(normals_vis, 0.0, 1.0)

    rgb_image_path = f"{train_path}/frame_{frame_str}.png"
    rgb_img = plt.imread(rgb_image_path)

    _, axes = plt.subplots(1, 3)
    axes[0].imshow(rgb_img)
    axes[0].set_title("Color")
    axes[1].imshow(depth_array, cmap="viridis")
    axes[1].set_title("Metric Depth")
    axes[2].imshow(normals_vis, cmap="jet")
    axes[2].set_title("Normals")
    plt.show()


if __name__ == "__main__":
    visualize_camera_poses(
        "/media/frog/DATA/Datasets/UncRGB/Blender/new_red_normals"
    )
