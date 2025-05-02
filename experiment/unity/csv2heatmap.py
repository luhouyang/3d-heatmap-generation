import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from tqdm import tqdm
import os  # Import the os module

def gaussian_blur(positions, frequencies, sigma):
    """Applies Gaussian blur using KDTree and sparse matrix approximation."""
    tree = KDTree(positions)
    radius = 3 * sigma

    neigh_ind = tree.query_ball_point(positions, radius)

    rows, cols, dists = [], [], []
    for i, neighbors in enumerate(tqdm(neigh_ind)):
        if neighbors:
            distances = np.linalg.norm(positions[i] - positions[neighbors], axis=1)
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors)
            dists.extend(distances)

    sigma_sq = 2 * (sigma**2)
    weights = np.exp(-np.array(dists)**2 / sigma_sq)

    weight_matrix = coo_matrix((weights, (rows, cols)),
                                 shape=(len(positions), len(positions)))
    weight_matrix_csr = weight_matrix.tocsr()

    weighted_sum = weight_matrix_csr.dot(frequencies)
    weight_total = weight_matrix_csr.sum(axis=1).A.flatten()

    return np.divide(weighted_sum,
                     weight_total,
                     out=np.zeros_like(weighted_sum),
                     where=weight_total > 0)

def create_heatmap_mesh_from_model(input_file, model_file, output_file, sigma=15.0):
    """
    Generates a heatmap on a provided 3D mesh model.

    Args:
        input_file (str): Path to the input CSV file containing point data (x, y, z, frequency).
        model_file (str): Path to the input PLY model file.
        output_file (str): Path to save the resulting PLY file with the heatmap on the mesh.
        sigma (float): Standard deviation for the Gaussian blur.
    """
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    positions = data[:, :3]
    frequencies = data[:, 3]

    # Apply Gaussian blur
    blurred_intensities = gaussian_blur(positions, frequencies, sigma)

    # Normalize blurred intensities for color mapping
    min_intensity = np.min(blurred_intensities)
    max_intensity = np.max(blurred_intensities)
    normalized_intensities = (blurred_intensities - min_intensity) / (max_intensity - min_intensity)

    # Load the PLY model
    mesh = o3d.io.read_triangle_mesh(model_file)

    # Map blurred intensities to the mesh vertices (nearest neighbor)
    tree = KDTree(positions)
    colors = np.zeros((np.asarray(mesh.vertices).shape[0], 3))
    cmap = plt.get_cmap("jet")
    for i, vertex in enumerate(tqdm(mesh.vertices)):
        closest_point_index = tree.query(vertex)[1]
        colors[i, :] = cmap(normalized_intensities[closest_point_index])[:3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=True)
    print(f"Heatmap mesh saved to: {output_file}")

    return mesh

if __name__ == '__main__':
    input_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\pointcloud.csv"  # Replace with your input file
    model_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\okinohara_003_ss.ply"  # Replace with your model file
    output_file = r"C:\Users\User\Desktop\Python\deep_learning\3d-heatmap-generation\experiment\unity\heatmap_mesh.ply"
    sigma = 30.0  # Adjust sigma as needed

    heatmap_mesh = create_heatmap_mesh_from_model(input_file, model_file, output_file, sigma)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(heatmap_mesh)
    render_options = vis.get_render_option()
    render_options.mesh_show_back_face = True
    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()