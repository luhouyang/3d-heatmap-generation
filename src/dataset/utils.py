"""
Author: Lu Hou Yang
Last updated: 4 July 2025

Contains utility functions for 
- 3D eye gaze data and voice recording
- Data filtering
- Data statistics
"""

import os
import pathlib

import numpy as np
import polars as pl
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from tqdm import tqdm

def visualize_geometry(geometry, point_size=1.0):
    pass

def calculate_point_density(pcd, ball_radius):
    pass

def create_xyz_colored_point_cloud(input_file, ball_radius):
    pass

def create_heatmap_mesh_from_density(input_file, model_file, interpolation_radius, ball_radius):
    pass

def create_qna_segmentation_mesh(qa_input_file, model_file):
    pass