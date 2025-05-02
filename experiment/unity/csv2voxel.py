import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

file_path = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\pointcloud.csv"
output_path = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\voxel_grid_filtered_weighted_colored_scaled_bounding_box.ply"

data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

xyz = data[:, :3]
intensity = data[:, 3]

### Coordinate limits for filtering
### Currently set to high value of 10e8 since filtering hasa been implemented on unity side
# x_limit = 0.2
# y_limit = 0.5
# z_limit = 0.2
x_limit = 10e8
y_limit = 10e8
z_limit = 10e8

mask = (np.abs(xyz[:, 0]) <= x_limit) & \
       (np.abs(xyz[:, 1]) <= y_limit) & \
       (np.abs(xyz[:, 2]) <= z_limit)

filtered_xyz = xyz[mask]
filtered_intensity = intensity[mask]

### Calculate bounding box
### 3D coordinate space that gaze points can fall into
min_bound = np.min(filtered_xyz, axis=0)
max_bound = np.max(filtered_xyz, axis=0)

### Calculate voxel grid dimensions
### Increase voxel_size for more resolution
# voxel_size = 0.004
voxel_size = 3.5
grid_min_bound = np.floor(min_bound / voxel_size) * voxel_size
grid_max_bound = np.ceil(max_bound / voxel_size) * voxel_size
grid_dimensions = ((grid_max_bound - grid_min_bound) / voxel_size).astype(int)

### Create an empty voxel grid
scaled_voxel_grid = o3d.geometry.VoxelGrid()
scaled_voxel_grid.origin = grid_min_bound
scaled_voxel_grid.voxel_size = voxel_size

### Assign intensity to voxels based on weighted point counts
voxel_intensity = {}
for point, inten in zip(filtered_xyz, filtered_intensity):
    ### Calculate which voxel the point falls into
    voxel_index = np.floor((point - grid_min_bound) / voxel_size).astype(int)
    voxel_index_tuple = tuple(voxel_index)

    ### Check if voxel is in the bounds
    if voxel_index_tuple[0] >= 0 and voxel_index_tuple[0] < grid_dimensions[0] and \
       voxel_index_tuple[1] >= 0 and voxel_index_tuple[1] < grid_dimensions[1] and \
       voxel_index_tuple[2] >= 0 and voxel_index_tuple[2] < grid_dimensions[2]:

        ### Aggregate intensity of points that fall in this voxel
        if voxel_index_tuple in voxel_intensity:
            voxel_intensity[voxel_index_tuple] += inten
        else:
            voxel_intensity[voxel_index_tuple] = inten
    else:
        print(f"Warning: Point {point} fell outside the voxel grid.")

# Normalize aggregated intensity values
max_intensity = max(voxel_intensity.values())
min_intensity = min(voxel_intensity.values())

normalized_voxel_intensity = {}
for index_tuple, total_intensity in voxel_intensity.items():
    # normalized_intensity = (total_intensity - min_intensity) / (
    #     max_intensity - min_intensity + 1e-6)
    # normalized_voxel_intensity[index_tuple] = normalized_intensity
    normalized_voxel_intensity[index_tuple] = total_intensity

# Populate the voxel grid with voxels and colors
voxel_list = []
for index_tuple, normalized_inten in normalized_voxel_intensity.items():
    voxel_index = np.array(index_tuple)
    voxel = o3d.geometry.Voxel(voxel_index)

    cmap = plt.get_cmap('jet')
    color = cmap(normalized_inten)[:3]
    voxel.color = color

    voxel_list.append(voxel)

# Update voxel grid with new voxels
for voxel in voxel_list:
    scaled_voxel_grid.add_voxel(voxel)

o3d.io.write_voxel_grid(output_path, scaled_voxel_grid)

print(f"Filtered and colored Voxel grid saved to: {output_path}")

# Visualize with bigger voxels and color
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(scaled_voxel_grid)
render_options = vis.get_render_option()
render_options.point_size = 5.0
render_options.background_color = np.asarray([0.1, 0.1, 0.1])
vis.run()
vis.destroy_window()
