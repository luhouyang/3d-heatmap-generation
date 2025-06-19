import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm
import os
import pathlib
import pandas as pd


def gaussian_blur_density(positions, sigma):
    """Applies Gaussian blur to estimate point density."""
    n_points = len(positions)
    density = np.zeros(n_points)
    tree = KDTree(positions)
    radius = 3 * sigma

    for i in tqdm(range(n_points), desc="Calculating Gaussian density"):
        neighbors_idx = tree.query_ball_point(positions[i], radius)
        # Subtract 1 to exclude the point itself
        density[i] = len(neighbors_idx) - 1

    # Normalize density values
    min_density = np.min(density)
    max_density = np.max(density)
    normalized_density = (density - min_density) / (
        max_density -
        min_density) if max_density > min_density else np.zeros_like(density)

    return normalized_density


def calculate_point_density(pcd, ball_radius):
    """
    Calculate point density using Open3D's KDTree for radius search.
    
    Args:
        pcd: Open3D point cloud object
        ball_radius: Radius for neighborhood search
        
    Returns:
        normalized_density: Array of normalized density values
    """
    # Build a KD-tree for efficient radius search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Estimate the density of each point using the KD-tree
    density = []
    for i, point in enumerate(
            tqdm(pcd.points, desc="Calculating point density")):
        # Use the KD-tree to find points within the radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point, ball_radius)
        # Exclude the point itself from the neighbor count
        density.append(len(idx) - 1)
    density = np.array(density)

    # Normalize density to the range [0, 1] for color mapping
    min_density = np.min(density)
    max_density = np.max(density)
    normalized_density = (density - min_density) / (
        max_density -
        min_density) if max_density > min_density else np.zeros_like(density)

    return normalized_density


def create_and_visualize_xyz_colored_point_cloud(input_file,
                                                 output_file,
                                                 ball_radius=10,
                                                 visualize=True):
    """
    Creates and visualizes a point cloud from XYZ data in a CSV file,
    coloring points based on their local density using a KD-tree for radius search.
    
    Args:
        input_file (str): Path to the input CSV file containing point data (x, y, z, ...).
        output_file (str): Path to save the resulting PLY file.
        ball_radius (float): Radius of the sphere used for neighborhood search.
        visualize (bool): Whether to visualize the point cloud after creation.
        
    Returns:
        pcd: The created Open3D point cloud object.
    """
    # Load data from the CSV file
    try:
        data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error loading data from {input_file}: {e}")
        return None

    if (data.ndim == 1):
        return
    
    positions = data[:, :3]  # Extract XYZ coordinates

    # Create an Open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Use the previously calculated average distance to set a default ball_radius
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    if ball_radius <= 0:
        ball_radius = avg_distance * 2

    # Calculate density
    normalized_density = calculate_point_density(pcd, ball_radius)

    # Use the 'jet' colormap from matplotlib to map density to colors
    colors = plt.get_cmap('jet')(normalized_density)[:, :3]  # Get RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"Point cloud saved to: {output_file}")

    # Visualize the point cloud
    if visualize:
        visualize_geometry(pcd, point_size=4.0)

    return pcd


def create_heatmap_mesh_from_density(input_file,
                                     model_file,
                                     output_file,
                                     interpolation_radius=0.05,
                                     visualize=True,
                                     ball_radius=0.05):
    """
    Generates a heatmap on a provided 3D mesh model based on the density
    of the input points, using a weighted interpolation for higher resolution.

    Args:
        input_file (str): Path to the input CSV file containing point data (x, y, z).
        model_file (str): Path to the input OBJ/PLY model file.
        output_file (str): Path to save the resulting OBJ file with the density heatmap on the mesh.
        interpolation_radius (float): Radius for interpolation around each mesh vertex.
        visualize (bool): Whether to visualize the mesh after creation.
        
    Returns:
        mesh: The created Open3D mesh object with density colors.
    """
    data = np.genfromtxt(input_file, delimiter=',', skip_header=1)

    if (data.ndim == 1):
        return
    
    positions = data[:, :3]  # Extract XYZ coordinates

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Use the previously calculated average distance to set a default ball_radius
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    if ball_radius <= 0:
        ball_radius = avg_distance * 10

    # Calculate density
    point_density = calculate_point_density(pcd, ball_radius)

    # Load the mesh (OBJ or PLY supported by Open3D)
    mesh = o3d.io.read_triangle_mesh(model_file)
    vertices = np.asarray(mesh.vertices)
    n_vertices = vertices.shape[0]

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(positions)
    colors = np.zeros((n_vertices, 3))
    cmap = plt.get_cmap("jet")

    for i in tqdm(range(n_vertices), desc="Applying density to mesh"):
        vertex = vertices[i]
        # Find points within interpolation_radius of the vertex
        nearby_point_indices = tree.query_ball_point(vertex,
                                                     interpolation_radius)

        # Weighted interpolation
        if nearby_point_indices:
            weights = []
            for point_index in nearby_point_indices:
                distance = np.linalg.norm(vertex - positions[point_index])
                weights.append(1.0 / (distance + 1e-6))
            weights = np.array(weights)
            weighted_densities = weights * point_density[nearby_point_indices]
            interpolated_density = np.sum(weighted_densities) / np.sum(weights)
            colors[i, :] = cmap(interpolated_density)[:3]
        else:
            colors[i, :] = [0.0, 0.0, 0.3]  # This adds a dark blue background

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Write output mesh in .obj format
    o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=True)
    print(f"Density heatmap mesh saved to: {output_file}")

    # Visualize the mesh
    if visualize:
        visualize_geometry(mesh)

    return mesh


def visualize_geometry(geometry, point_size=1.0):
    """
    Visualize an Open3D geometry (point cloud or mesh).
    
    Args:
        geometry: Open3D geometry object (point cloud or mesh)
        point_size: Size of points if geometry is a point cloud
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)
    render_options = vis.get_render_option()

    if isinstance(geometry, o3d.geometry.PointCloud):
        render_options.point_size = point_size
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        render_options.mesh_show_back_face = True

    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


def process_questionnaire_answers(qa_input_file, output_ply_file,
                                  output_lookup_csv):
    """
    Processes the qa.csv file to extract gazed voxel XYZ and answers.
    Assigns specific colors and color names based on predefined answer choices,
    generates a PLY point cloud with these colors, and a lookup CSV including the color name.

    Args:
        qa_input_file (str): Path to the qa.csv file.
        output_ply_file (str): Path to save the PLY file for gazed voxels.
        output_lookup_csv (str): Path to save the CSV lookup file.
    """
    # Define your answer choices and map them to specific RGB colors (0-255) and color names
    # You can customize these colors and names as needed.
    answer_color_map = {
        "1 面白い／気になる形だと感じた": {
            "rgb": [255, 165, 0],
            "name": "Orange"
        },  # Attention to shape
        "2 美しい／芸術的だと感じた": {
            "rgb": [0, 128, 0],
            "name": "Green"
        },  # Positive aesthetic
        "3 疑問／不思議／意味不明と感じた": {
            "rgb": [128, 0, 128],
            "name": "Purple"
        },  # Confusion/Thought
        "4 不安・不気味・怖いと感じた": {
            "rgb": [255, 0, 0],
            "name": "Red"
        },  # Negative feeling
        "5 なんとなく見ていた": {
            "rgb": [255, 255, 0],
            "name": "Yellow"
        },  # No specific reason
    }

    # Default color for answers not matching any predefined choice (e.g., light grey)
    default_color_rgb = [200, 200, 200]
    default_color_name = "Light Grey (Other)"

    try:
        # Load the data using pandas, assuming first row is header
        df = pd.read_csv(qa_input_file, sep=',', header=0)

        # Ensure numeric columns are properly typed, coercing errors
        df['voxelX'] = pd.to_numeric(df['voxelX'], errors='coerce')
        df['voxelY'] = pd.to_numeric(df['voxelY'], errors='coerce')
        df['voxelZ'] = pd.to_numeric(df['voxelZ'], errors='coerce')
        df['gazedVoxelID'] = df['gazedVoxelID'].astype(
            str)  # Ensure ID is string for lookup
        # Ensure 'answer' column is string type and strip whitespace for consistent comparisons
        df['answer'] = df['answer'].astype(str).str.strip()

        # Drop rows with NaN in essential columns before processing
        df = df.dropna(
            subset=['voxelX', 'voxelY', 'voxelZ', 'answer', 'gazedVoxelID'])

        # Initialize lists for colors and color names
        assigned_colors_255 = []  # Store as 0-255 for Open3D PLY
        assigned_colors_01 = []  # Store as 0-1 for Open3D PLY
        assigned_color_names = []  # Store color names for lookup CSV

        # Assign colors and names based on the answer
        for index, row in df.iterrows():
            answer_text = row['answer']
            if answer_text in answer_color_map:
                color_info = answer_color_map[answer_text]
                assigned_colors_255.append(color_info["rgb"])
                assigned_colors_01.append(np.array(color_info["rgb"]) / 255.0)
                assigned_color_names.append(color_info["name"])
            else:
                # Assign default color if the answer doesn't match a predefined choice
                assigned_colors_255.append(default_color_rgb)
                assigned_colors_01.append(np.array(default_color_rgb) / 255.0)
                assigned_color_names.append(default_color_name)

        df['color_name'] = assigned_color_names  # Add color names to DataFrame for lookup CSV
        df['color_rgb_255'] = assigned_colors_255  # Also keep RGB for completeness in lookup if desired

        # Extract XYZ positions and ensure float64 type for Open3D
        voxel_positions = df[['voxelX', 'voxelY',
                              'voxelZ']].values.astype(np.float64)

        # Create an Open3D point cloud for these voxels
        pcd_voxels = o3d.geometry.PointCloud()
        pcd_voxels.points = o3d.utility.Vector3dVector(voxel_positions)

        # Assign the calculated colors (0-1 range) to the point cloud
        pcd_voxels.colors = o3d.utility.Vector3dVector(
            np.array(assigned_colors_01))

        # Save the voxel points to a PLY file
        o3d.io.write_point_cloud(output_ply_file, pcd_voxels, write_ascii=True)
        print(f"Gazed voxel point cloud saved to: {output_ply_file}")

        # Save the lookup table (voxel ID, XYZ, answer, and color name) to a CSV
        df_lookup = df[[
            'gazedVoxelID', 'voxelX', 'voxelY', 'voxelZ', 'answer',
            'color_name', 'color_rgb_255'
        ]]
        df_lookup.to_csv(output_lookup_csv, index=False)
        print(f"Voxel answer lookup table saved to: {output_lookup_csv}")

        return pcd_voxels

    except FileNotFoundError:
        print(
            f"Error: QA file '{qa_input_file}' not found. Skipping voxel answer processing."
        )
        return None
    except Exception as e:
        print(f"Error processing QA file {qa_input_file}: {e}")
        return None


if __name__ == '__main__':
    curr_dir = pathlib.Path.cwd()

    # Parameters
    point_cloud_ball_radius = 25  # 25 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]
    mesh_interpolation_radius = 10  # 10 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]
    ball_radius = 25  # 25 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]

    # Choose what to generate
    generate_point_cloud = True
    generate_mesh = True
    generate_voxel_answers = True

    parameters_dict = {
        "rembak7": [0.05, 0.05, 0.05],
        "J-7606": [0.02, 0.005, 0.02],
        "IN0295": [3, 3, 3],
        "IN0306": [2, 1, 2],
        "NZ0001": [3, 1, 3],
        "SK0035": [5, 3, 5],
        "MH0037": [3, 1, 3],
        "NM0239": [3, 1, 3],
        "TK0020": [2, 0.8, 2],
        "UD0028": [6, 2, 6],
    }

    session_paths = curr_dir / pathlib.Path('src/data')
    for sessions in os.listdir(session_paths):
        model_paths = session_paths / pathlib.Path(sessions)

        for models in os.listdir(model_paths):
            datafile_paths = model_paths / pathlib.Path(models)

            input_file = os.path.join(datafile_paths, "pointcloud.csv")
            output_point_cloud = os.path.join(datafile_paths,
                                              "pointcloud_viz.ply")
            model_file = os.path.join(datafile_paths, "model.obj")
            output_mesh = os.path.join(datafile_paths, "heatmap_viz.ply")

            qa_input_file = os.path.join(datafile_paths, "qa.csv")
            output_qa_ply = os.path.join(datafile_paths,
                                         "gazed_voxels_answers.ply")
            output_qa_lookup_csv = os.path.join(
                datafile_paths, "gazed_voxels_answers_lookup.csv")

            if (models not in parameters_dict.keys()):
                # Generate colored point cloud based on density
                if generate_point_cloud:
                    print("\n=== Generating density-colored point cloud ===")
                    pcd = create_and_visualize_xyz_colored_point_cloud(
                        input_file,
                        output_point_cloud,
                        visualize=False,
                        ball_radius=point_cloud_ball_radius)

                # Generate heatmap mesh based on point density
                if generate_mesh and os.path.exists(model_file):
                    print("\n=== Generating density heatmap on mesh ===")
                    heatmap_mesh = create_heatmap_mesh_from_density(
                        input_file,
                        model_file,
                        output_mesh,
                        visualize=False,
                        interpolation_radius=mesh_interpolation_radius,
                        ball_radius=ball_radius)
                elif generate_mesh and not os.path.exists(model_file):
                    print(
                        f"Error: Model file '{model_file}' not found. Skipping heatmap mesh generation."
                    )

                if generate_voxel_answers:
                    print(
                        "\n=== Processing questionnaire answers and generating voxel points ==="
                    )
                    process_questionnaire_answers(qa_input_file, output_qa_ply,
                                                  output_qa_lookup_csv)

            else:
                # Generate colored point cloud based on density
                if generate_point_cloud:
                    print("\n=== Generating density-colored point cloud ===")
                    pcd = create_and_visualize_xyz_colored_point_cloud(
                        input_file,
                        output_point_cloud,
                        visualize=False,
                        ball_radius=parameters_dict[models][0])

                # Generate heatmap mesh based on point density
                if generate_mesh and os.path.exists(model_file):
                    print("\n=== Generating density heatmap on mesh ===")
                    heatmap_mesh = create_heatmap_mesh_from_density(
                        input_file,
                        model_file,
                        output_mesh,
                        visualize=False,
                        interpolation_radius=parameters_dict[models][1],
                        ball_radius=parameters_dict[models][2])
                elif generate_mesh and not os.path.exists(model_file):
                    print(
                        f"Error: Model file '{model_file}' not found. Skipping heatmap mesh generation."
                    )

                if generate_voxel_answers:
                    print(
                        "\n=== Processing questionnaire answers and generating voxel points ==="
                    )
                    process_questionnaire_answers(qa_input_file, output_qa_ply,
                                                  output_qa_lookup_csv)
