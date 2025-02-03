import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

def visualize(train_path: str, frame_number: int):
    # convert frame number to {000{frame_number:04d}}
    frame_str = f'{frame_number:04d}'
    depth_file_path = f'{train_path}/frame_{frame_str}_Z.exr' 
    depth_exr = OpenEXR.InputFile(depth_file_path)
    dw = depth_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    depth_channel = depth_exr.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_array = np.fromstring(depth_channel, dtype=np.float32)
    depth_array.shape = (size[1], size[0])
    depth_array[depth_array > 10] = -1

    normals_file_path = f'{train_path}/frame_{frame_str}_Normal.exr'
    normals_exr = OpenEXR.InputFile(normals_file_path)
    norm_x = normals_exr.channel('X', Imath.PixelType(Imath.PixelType.FLOAT))
    norm_y = normals_exr.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
    norm_z = normals_exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    nx = np.frombuffer(norm_x, dtype=np.float32).reshape((size[1], size[0]))
    ny = np.frombuffer(norm_y, dtype=np.float32).reshape((size[1], size[0]))
    nz = np.frombuffer(norm_z, dtype=np.float32).reshape((size[1], size[0]))
    normals_array = np.stack([nx, ny, nz], axis=-1)
    normals_vis = (normals_array + 1.0) # / 2.0
    normals_vis = np.clip(normals_vis, 0.0, 1.0)

    rgb_image_path = f'{train_path}/frame_{frame_str}.png'
    rgb_img = plt.imread(rgb_image_path)

    _, axes = plt.subplots(1, 3)
    axes[0].imshow(rgb_img)
    axes[0].set_title('Color')
    axes[1].imshow(depth_array, cmap='viridis')
    axes[1].set_title('Metric Depth')
    axes[2].imshow(normals_vis, cmap='jet')
    axes[2].set_title('Normals')
    plt.show()


if __name__ == '__main__':
    visualize('/media/frog/DATA/Datasets/UncRGB/Blender/trial_suzanne_normal_new/train', 1)