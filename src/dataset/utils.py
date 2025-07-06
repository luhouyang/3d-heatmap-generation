"""
Author: Lu Hou Yang
Last updated: 4 July 2025

Contains utility functions for 
- 3D eye gaze data and voice recording
- Data filtering
- Data statistics

Notes
- yapf was used to format code, to preserve manual formatting at some sections
  # yapf: disable
  # yapf: enable
  was used to control formatting behaviour
"""

import os
import pathlib

import numpy as np
import pandas as pd
import open3d as o3d

import matplotlib.pyplot as plt

from tqdm import tqdm

# Colors
CMAP = plt.get_cmap('jet')
BASE_COLOR = [0.0, 0.0, 0.0]

# Pottery & Dogu assigned numbers
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1',
    'FH0008': '2',
    'IN0003': '3',
    'IN0008': '4',
    'IN0009': '5',
    'IN0017': '6',
    'IN0081': '7',
    'IN0104': '8',
    'IN0135': '9',
    'IN0148': '10',
    'IN0220': '11',
    'IN0228': '12',
    'IN0232': '13',
    'IN0239': '14',
    'IN0277': '15',
    'MY0001': '16',
    'MY0002': '17',
    'MY0004': '18',
    'MY0006': '19',
    'MY0007': '20',
    'ND0001': '21',
    'NM0001': '22',
    'NM0002': '23',
    'NM0009': '24',
    'NM0010': '25',
    'NM0014': '26',
    'NM0015': '27',
    'NM0017': '28',
    'NM0041': '29',
    'NM0049': '30',
    'NM0066': '31',
    'NM0070': '32',
    'NM0072': '33',
    'NM0073': '34',
    'NM0079': '35',
    'NM0080': '36',
    'NM0099': '37',
    'NM0106': '38',
    'NM0133': '39',
    'NM0135': '40',
    'NM0144': '41',
    'NM0154': '42',
    'NM0156': '43',
    'NM0159': '44',
    'NM0168': '45',
    'NM0173': '46',
    'NM0175': '47',
    'NM0189': '48',
    'NM0191': '49',
    'NM0206': '50',
    'SB0002': '51',
    'SB0004': '52',
    'SI0001': '53',
    'SJ0503': '54',
    'SJ0504': '55',
    'SK0001': '56',
    'SK0002': '57',
    'SK0003': '58',
    'SK0004': '59',
    'SK0005': '60',
    'SK0013': '61',
    'SS0001': '62',
    'TJ0004': '63',
    'TJ0005': '64',
    'TJ0010': '65',
    'TK0002': '66',
    'TK0048': '67',
    'TK0057': '68',
    'UD0001': '69',
    'UD0003': '70',
    'UD0005': '71',
    'UD0006': '72',
    'UD0011': '73',
    'UD0013': '74',
    'UD0014': '75',
    'UD0016': '76',
    'UD0023': '77',
    'UD0302': '78',
    'UD0304': '79',
    'UD0308': '80',
    'UD0318': '81',
    'UD0322': '82',
    'UD0411': '83',
    'UD0412': '84',
    'UK0001': '85',
    'IN0295': '86',
    'IN0306': '87',
    'MH0037': '88',
    'NM0239': '89',
    'NZ0001': '90',
    'SK0035': '91',
    'TK0020': '92',
    'UD0028': '93'
}

### CALCULATION ###


# yapf: disable
# Read about KD-Tree (Medium | EN): https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
# Read about KD-Tree (Qiita | JP): https://qiita.com/RAD0N/items/7a192a4a5351f481c99f
def calculate_normalized_point_intensity(pcd, ball_radius):
    """
    Calculate point intensity using Open3D KDTree for ball radius search.

    Args:
        pcd: Open3D point cloud object
        ball_radius (float): Radius for neighborhood search

    Returns:
        normalized_intensity: Array of normalized intensity values
    """
    # Build a KD-Tree with FLANN for efficient radius search
    # Open3D docs: https://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
    # FLANN docs: https://www.cs.ubc.ca/research/flann/
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Calculate intensity | Find the number of points in a radius of each point
    intensity = []
    for i, point in enumerate(pcd.points):
        # Use KD-Tree to find points within the radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point, ball_radius)
        # We are interested in the length of list of indexes (points) in the radius
        # i.e. the intensity of the point as measured by the density of points within the radius
        # len(idx) - 1 to exclude the point itself
        intensity.append(len(idx) - 1)
    intensity = np.array(intensity)

    # Normalize intensity to the range [0, 1] for color mapping
    # NORMALIZED = (VALUE - MIN) / (MAX - MIN)
    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity) \
                            if max_intensity > min_intensity else np.zeros_like(intensity)

    return normalized_intensity
# yapf: enable

### FILTERING ###


def generate_filtered_dataset_report():
    pass


def filter_data_on():
    pass


### PROCESS DATA ###


def create_gaze_intensity_point_cloud(input_file, ball_radius):
    """
    Creates a point cloud from xyz data in CSV file,
    coloring points based on their normalized intensity using a KD-tree for radius search.

    Args:
        input_file (str): Path to the input CSV file containing point cloud data
        ball_radius (float): Radius of the sphere used for neighborhood search

    Returns:
        pcd: The created Open3D pint cloud object
    """
    data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()

    positions = data[:, :3]  # Only xyz

    # Create an Open3D point cloud object
    # Open3D docs: https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html
    pcd = o3d.geometry.PointCloud()

    # PointCloud.points accepts float64 of c=shape (num_points, 3)
    # Vector3dVector converts float64 numpy array of shape (n, 3) to Open3D format
    # Vector3dVector docs: https://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Calculate intensity
    normalized_intensity = calculate_normalized_point_intensity(pcd, ball_radius)

    # Get color based on normalized intensity
    colors = CMAP(normalized_intensity)[:, :3]
    pcd.colors = o3d.utility.Vector3dVecctor(colors)

    return pcd

# yapf: disable
def create_gaze_intensity_heatmap_mesh(
    input_file,
    model_file,
    interpolation_radius,
    ball_radius,
):
    data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()

    positions = data[:, :3]  # Only xyz

    # Create an Open3D point cloud object
    # Open3D docs: https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html
    pcd = o3d.geometry.PointCloud()

    # PointCloud.points accepts float64 of c=shape (num_points, 3)
    # Vector3dVector converts float64 numpy array of shape (n, 3) to Open3D format
    # Vector3dVector docs: https://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Calculate intensity
    normalized_intensity = calculate_normalized_point_intensity(pcd, ball_radius)

    # Load the mesh (OBJ or PLY) to apply the normalized intensity to
    mesh = o3d.io_read_triangle_mesh(model_file)
    vertices = np.asarray(mesh.vertices)
    n_vertices = vertices.shape[0]

    # Build KD-Tree for efficient radius search
    # Color the mesh based on normalized intensity with a circular interpolation
    # Modify radius later based on the eye gaze error / range
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    colors = np.zeros((n_vertices, 3))

    # Find points on model mesh that are close to the eye gaze intensity point cloud
    for i, vertex in enumerate(vertices):
        [k, idx, _] = kdtree.search_radius_vector_3d(vertex, interpolation_radius)

        if idx:
            # Assign color to the points on model mesh within interpolation radius
            # Based on the average intensity of points within interpolation radius
            colors[i, :] = CMAP(np.average(normalized_intensity[idx]))[:3]
        else:
            colors[i, :] = BASE_COLOR

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh
# yapf: enable

def create_qna_segmentation_mesh(input_file, model_file, association_radius):
    pass


def process_voice_data(input_file):
    pass


### VISUALIZATIONS ###


def visualize_geometry(geometry, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)
    render_options = vis.get_render_options()

    if isinstance(geometry, o3d.geometry.PointCloud):
        render_options.point_size = point_size
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        render_options.mesh_show_back_face = True
    
    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()
