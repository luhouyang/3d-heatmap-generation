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
import polars as pl
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from tqdm import tqdm

# Pottery & Dogu assigned numbers
assigned_numbers_dict = {
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
def calculate_point_intensity(pcd, ball_radius):
    """
    Calculate point intensity using Open3D KDTree for ball radius search.

    Args:
        pcd: Open3D point cloud object
        ball_radius: Radius for neighborhood search

    Returns:
        normalized_intensity: Array of normalized intensity values
    """
    # Build a KD-tree with FLANN for efficient radius search
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
    pass


def create_gaze_intensity_heatmap_mesh(input_file, model_file,
                                       interpolation_radius, ball_radius):
    pass


def create_qna_segmentation_mesh(input_file, model_file):
    pass


def process_voice_data(input_file):
    pass


### VISUALIZATIONS ###


def visualize_geometry(geometry, point_size=1.0):
    pass
