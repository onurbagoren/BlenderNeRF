import os
import shutil
import numpy as np
import bpy
from . import helper, blender_nerf_operator
import re
import json

def fix_file_structure(directory):
    """Fix file structure by removing unnecessary suffixes from filenames."""
    # Get all files in directory
    files = os.listdir(directory)

    # Regular expression to match the unnecessary suffix (e.g., "_0002")
    pattern = re.compile(r"(_\d{4})\.exr")

    for file in files:
        match = pattern.search(file)
        if match:
            new_name = file.replace(match.group(1), "")
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {file} -> {new_name}")

    print("File renaming complete.")

def fix_json(json_path, image_dir):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)
    json_dir = os.path.dirname(json_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    
    pattern = re.compile(r"_(\d{4})\.png$")
    max_suffix = 0
    for img in image_files:
        match = pattern.search(img)
        if match:
            max_suffix = max(max_suffix, int(match.group(1)))

    suffix_to_remove = f"{max_suffix:04d}.png" if max_suffix > 0 else ""

    # Process frames
    updated_frames = []
    for frame in data["frames"]:
        file_path = frame["file_path"]

        # Remove "_000{frame_idx}.png" from file path
        corrected_file_name = file_path.replace(suffix_to_remove, "") if suffix_to_remove else file_path
        frame["file_path"] = corrected_file_name

        # Generate corresponding depth and normal file names
        base_name = os.path.splitext(os.path.basename(corrected_file_name))[0]
        depth_file = f"{base_name}_depth.exr"
        normal_file = f"{base_name}_normals.exr"

        # Check if these files exist
        if os.path.exists(depth_file) and os.path.exists(normal_file):
            frame["depth_file_path"] = depth_file
            frame["normal_file_path"] = normal_file

        updated_frames.append(frame)

    # Update the data
    data["frames"] = updated_frames

    # Save the updated JSON
    output_path = os.path.join(json_dir, "fixed_transforms_train.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON saved to {output_path}")


def set_camera_pose(camera, pose_matrix):
    """Set the camera's location and rotation based on an SE(3) pose matrix."""
    # Extract location and rotation from the 4x4 transformation matrix
    location = pose_matrix[:3, 3]
    rotation_matrix = pose_matrix[:3, :3]

    # Convert rotation matrix to Euler angles
    quat = rotmat2quat(rotation_matrix)

    # Set camera location and rotation
    camera.location = location
    # set the quat
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = quat


def rotmat2quat(rotmat):
    return np.array(
        [
            0.5 * np.sqrt(1 + rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2]),
            (rotmat[2, 1] - rotmat[1, 2])
            / (4 * np.sqrt(1 + rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2])),
            (rotmat[0, 2] - rotmat[2, 0])
            / (4 * np.sqrt(1 + rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2])),
            (rotmat[1, 0] - rotmat[0, 1])
            / (4 * np.sqrt(1 + rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2])),
        ]
    )


# global addon script variables
EMPTY_NAME = "BlenderNeRF Sphere"
CAMERA_NAME = "BlenderNeRF Camera"


# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    """Camera on Sphere Operator"""

    bl_idname = "object.camera_on_sphere"
    bl_label = "Camera on Sphere COS"

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({"ERROR"}, "Be sure to have a selected camera!")
            return {"FINISHED"}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method="COS")
        if len(error_messages) > 0:
            self.report({"ERROR"}, error_messages[0])
            return {"FINISHED"}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        print(output_path)
        os.makedirs(output_path, exist_ok=True)


        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera

        if scene.test_data:
            # testing transforms
            output_data["frames"] = self.get_camera_extrinsics(
                scene, camera, mode="TEST", method="COS"
            )
            self.save_json(output_path, "transforms_test.json", output_data)

        if scene.train_data:
            if not scene.show_camera:
                scene.show_camera = True

            # train camera on sphere
            sphere_camera = scene.objects[CAMERA_NAME]
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            # training transforms
            sphere_output_data["frames"] = self.get_camera_extrinsics(
                scene, sphere_camera, mode="TRAIN", method="COS"
            )

            # rendering
            if scene.render_frames:
                output_train = os.path.join(output_path, "train")
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = (False, False, True)
                scene.frame_end = (
                    scene.frame_start + scene.cos_nb_frames - 1
                )  # update end frame
                for frame_idx, pose in enumerate(sphere_output_data["frames"]):
                    # print(f"Rendering frame {frame_idx}/{scene.cos_nb_frames}")
                    scene.frame_set(scene.frame_start + frame_idx)
                    
                    bpy.context.scene.render.filepath = os.path.join(
                        output_train, f"frame_{frame_idx:04d}.png"
                    )
                    # randomize_cube_color()  # Set a random color for the Cube

                    # Set the camera to follow the trajectory
                    pose_matrix = np.array(pose["transform_matrix"]).reshape(
                        4, 4
                    )  # Assuming pose matrix is 4x4
                    set_camera_pose(scene.camera, pose_matrix)

                    # Set filepath for each frame
                    # scene.render.filepath = os.path.join(
                    #     output_train, f"frame_{frame_idx:04d}.png"
                    # )

                    # Ensure the use_nodes is True
                    bpy.context.scene.use_nodes = True

                    # Get the reference of the scene
                    scene = bpy.context.scene

                    # Get the reference of the View Layer
                    view_layer = bpy.context.view_layer

                    # Enable 'Z' pass
                    view_layer.use_pass_z = True
                    view_layer.use_pass_normal = True
                    
                    # Clear default nodes
                    for node in scene.node_tree.nodes:
                        scene.node_tree.nodes.remove(node)

                    # Create a new Render Layers node
                    render_layers_node = scene.node_tree.nodes.new("CompositorNodeRLayers")
                    render_layers_node.layer = view_layer.name

                    # Create a File Output node
                    file_output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
                    file_output_node.base_path = output_train
                    file_output_node.format.file_format = "OPEN_EXR"  # 32-bit EXR

                    # Create two new file slots, for depth and normals
                    file_output_node.file_slots.new("Depth")
                    file_output_node.file_slots.new("Normal")

                    # Name the output files for each slot
                    file_output_node.file_slots["Depth"].path = f"frame_{frame_idx:04d}_Z_"
                    file_output_node.file_slots["Normal"].path = f"frame_{frame_idx:04d}_Normal_"

                    # Link each render pass to its corresponding file output slot
                    scene.node_tree.links.new(
                        render_layers_node.outputs["Depth"], 
                        file_output_node.inputs["Depth"]
                    )
                    scene.node_tree.links.new(
                        render_layers_node.outputs["Normal"], 
                        file_output_node.inputs["Normal"]
                    )

                    # Then render and write the still
                    bpy.ops.render.render(write_still=True)
                    print("writing to", file_output_node.file_slots[0].path)
            self.save_json(output_path, "transforms_train.json", sphere_output_data)

            fix_file_structure(os.path.join(output_path, "train"))
            fix_json(os.path.join(output_path, "transforms_train.json"), os.path.join(output_path, "train"))

            if scene.logs:
                self.save_log_file(scene, output_path, method="COS")
            if scene.splats:
                self.save_splats_ply(scene, output_path)
        
        # if frames are rendered, the below code is executed by the handler function
        if not any(scene.rendering):
            # reset camera settings
            if not scene.init_camera_exists:
                helper.delete_camera(scene, CAMERA_NAME)
            if not scene.init_sphere_exists:
                objects = bpy.data.objects
                objects.remove(objects[EMPTY_NAME], do_unlink=True)
                scene.show_sphere = False
                scene.sphere_exists = False

            scene.camera = scene.init_active_camera

            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(
                output_path, "zip", output_path
            )  # output filename = output_path
            shutil.rmtree(output_path)
        

        return {"FINISHED"}
