import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

### Calculating for each point without approximations, O(n^2)
# def gaussian_blur(positions, frequencies, sigma):
#     """Applies Gaussian blur to frequencies based on positions"""
#     blurred_intensities = np.zeros_like(frequencies, dtype=float)
#     for i, pos in enumerate(tqdm(positions)):
#         weighted_sum = 0.0
#         weight_total = 0.0
#         for j, other_pos in enumerate(positions):
#             distance = np.linalg.norm(pos - other_pos)
#             weight = np.exp(-distance**2 / (2 * sigma**2))
#             weighted_sum += frequencies[j] * weight
#             weight_total += weight
#         blurred_intensities[
#             i] = weighted_sum / weight_total if weight_total > 0 else 0
#     return blurred_intensities

### Optimized version making use of approximations, KD-Tree O(n log n)
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix  # sparse matrix


def gaussian_blur(positions, frequencies, sigma):
    """Applies Gaussian blur using KDTree and sparse matrix approximation."""
    tree = KDTree(positions)
    radius = 3 * sigma  # 99.7% of Gaussian weight within 3 sigma

    # Get all neighbors within radius for each point
    neigh_ind = tree.query_ball_point(positions, radius)

    # Build sparse weight matrix components
    rows, cols, dists = [], [], []
    for i, neighbors in enumerate(tqdm(neigh_ind)):
        if neighbors:  # Only process points with neighbors
            # Vectorized distance calculation for all neighbors of i
            distances = np.linalg.norm(positions[i] - positions[neighbors],
                                       axis=1)
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors)
            dists.extend(distances)

    # Vectorized weight computation
    sigma_sq = 2 * (sigma**2)
    weights = np.exp(-np.array(dists)**2 / sigma_sq)

    # Create sparse weight matrix
    weight_matrix = coo_matrix((weights, (rows, cols)),
                               shape=(len(positions), len(positions)))
    weight_matrix_csr = weight_matrix.tocsr()

    # Compute blurred values
    weighted_sum = weight_matrix_csr.dot(frequencies)
    weight_total = weight_matrix_csr.sum(axis=1).A.flatten()

    return np.divide(weighted_sum,
                     weight_total,
                     out=np.zeros_like(weighted_sum),
                     where=weight_total > 0)


def main():
    input_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\pointcloud.csv"  # Replace with your input file
    output_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\pointcloud_kd_tree.ply"
    # output_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\pointcloud.ply"

    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    positions = data[:, :3]
    frequencies = data[:, 3]

    ### Coordinate limits for filtering
    ### Currently set to high value of 10e8 since filtering hasa been implemented on unity side
    # x_limit = 0.2
    # y_limit = 0.5
    # z_limit = 0.2
    x_limit = 10e8
    y_limit = 10e8
    z_limit = 10e8

    mask = (np.abs(positions[:, 0]) <= x_limit) & \
        (np.abs(positions[:, 1]) <= y_limit) & \
        (np.abs(positions[:, 2]) <= z_limit)

    filtered_positions = positions[mask]
    filtered_frequencies = frequencies[mask]

    ### Apply Gaussian blur
    ### Larger sigma may cause over blending and spread, adjust with care
    # sigma = 0.02
    sigma = 30.0
    blurred_intensities = gaussian_blur(filtered_positions, filtered_frequencies,
                                        sigma)
    # blurred_intensities = filtered_frequencies

    # Normalize blurred intensities
    min_intensity = np.min(blurred_intensities)
    max_intensity = np.max(blurred_intensities)
    normalized_intensities = (blurred_intensities -
                            min_intensity) / (max_intensity - min_intensity)

    ### Make visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_positions)

    cmap = plt.get_cmap("jet")
    colors = cmap(normalized_intensities)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"Blurred point cloud saved to: {output_file}")

    return pcd


if __name__ == '__main__':
    import timeit

    ### KD-Tree, sparse matrix:     1.0101457999553531      seconds
    ### Full precision distance:    48.46784920000937       seconds
    # print(timeit.timeit("main()", globals=globals(), number=1))

    pcd = main()

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_options = vis.get_render_option()
    render_options.point_size = 4.0
    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()