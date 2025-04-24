import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import os
import pandas as pd


def load_obj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    return pcd

def check_penetration(source_file, target_file):
    """Check if target mesh penetrates source mesh."""
    source_mesh = trimesh.load_mesh(source_file)
    target_mesh = trimesh.load_mesh(target_file)
    
    source_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_mesh.vertices)))
    distances = []
    for vertex in target_mesh.vertices:
        [_, idx, d] = source_tree.search_knn_vector_3d(vertex, 1)
        distances.append(d[0])
    distances = np.array(distances)
    return np.any(distances < 0)

def reshape_target(target, scale_factor=1.01, displacement_vector=None):
    target_points = np.asarray(target.points)
    
    # Apply scaling to reshape the target more precisely
    target_center = np.mean(target_points, axis=0)
    target_points = target_center + (target_points - target_center) * scale_factor
    
    # Apply optional displacement for fine adjustments
    if displacement_vector is not None:
        target_points += displacement_vector
    
    target.points = o3d.utility.Vector3dVector(target_points)
    return target

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down



def downsample_mesh(input_file, output_file, target_faces=50000):
    mesh = trimesh.load(input_file)
    simplified = mesh.simplify_quadratic_decimation(target_faces)
    simplified.export(output_file)




def pick_points(pcd, file_path):
    print("1) Please pick at least three correspondences using the path picking tool")
    print("2) After picking points, close the window")

    # Convert Open3D point cloud to NumPy array
    points = np.asarray(pcd.points)

    # Create a PyVista PolyData object from the points
    pv_pcd = pv.PolyData(points)

    model_path = file_path
    mesh = pv.read(model_path)

    # Show the 3D model using pyvista
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=True, opacity=1, 
                    label="Select Front Point", 
                    show_edges=True, name="Saddle")
    plotter.enable_eye_dome_lighting()  # helps depth perception

    plotter.enable_path_picking(color="red", point_size=15)
    plotter.show_grid()
    plotter.show()
    # Convert picked points to the desired format
    # picked_points = []
    # for point in plotter.picked_path.points:
    #     picked_points.append(point)
    # return picked_points
    picked_points = plotter.picked_path.points
    picked_indices = [np.argmin(np.linalg.norm(points - point, axis=1)) for point in picked_points]

    return picked_indices


def compute_transformation(source_points, target_points):
    assert len(source_points) == len(target_points) >= 3, "Exactly 3 points must be picked from each point cloud."
    source_points = np.array(source_points)
    target_points = np.array(target_points)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_points_centered = source_points - source_center
    target_points_centered = target_points - target_center
    H = source_points_centered.T @ target_points_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = target_center.T - R @ source_center.T
    transformation = np.identity(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return transformation


# Convert PyVista PolyData to Open3D TriangleMesh
def pv_to_o3d_mesh(pv_mesh):
    vertices = o3d.utility.Vector3dVector(pv_mesh.points)
    faces = o3d.utility.Vector3iVector(pv_mesh.faces.reshape(-1, 4)[:, 1:])
    o3d_mesh = o3d.geometry.TriangleMesh(vertices, faces)
    return o3d_mesh
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

def main():
    source_file = r"C:\Users\abhay\OneDrive\Desktop\Desktop office\horse1.obj"
    target_file = r"C:\Users\abhay\OneDrive\Desktop\Desktop office\saddle_s.obj"

    source = load_obj(source_file)
    target = load_obj(target_file)

    voxel_size = 0.1
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    print("Pick points from the source point cloud")
    source_indices = pick_points(source_down, source_file)
    print("Pick points from the target point cloud")
    target_indices = pick_points(target_down, target_file)

    source_points = np.asarray(source_down.points)[source_indices]
    target_points = np.asarray(target_down.points)[target_indices]

#     source_points = np.asarray(source_down.points)[source_indices]
#     target_points = np.asarray(target_down.points)[target_indices]

    transformation = compute_transformation(source_points, target_points)
    source.transform(transformation)

    # Save the transformed source to a new file
    distances = target.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Use multiple radii instead of just two
    radii = o3d.utility.DoubleVector([avg_dist * 1.5, avg_dist * 2, avg_dist * 2.5, avg_dist * 3])

    # Ensure normals are computed for better BPA results
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    plotter = pv.Plotter()
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    target_pv = pv.PolyData(target_points)
    source_pv = pv.PolyData(source_points)
    plotter.add_mesh(target_pv, color="blue", opacity=1, 
                    label="Target", 
                    show_edges=True, name="Target")
    plotter.add_mesh(source_pv, color="red", opacity=0.7, 
                    label="Transformed Source", 
                    show_edges=True, name="TransformedSource", 
                    cmap="coolwarm", 
                    clim=[np.min(distances), np.max(distances)])
    plotter.enable_eye_dome_lighting()  # helps depth perception
    plotter.show_grid()
    plotter.show()

    target_points = np.asarray(target.points)
    # Save the target points as a .obj file
    target_mesh = o3d.geometry.TriangleMesh()
    target_mesh.vertices = o3d.utility.Vector3dVector(target_points)
    target_mesh.triangles = o3d.utility.Vector3iVector([])  # No faces, just points
    # o3d.io.write_triangle_mesh("target_points.obj", target_mesh)
    source_points = np.asarray(source.points)
    source_mesh = o3d.geometry.TriangleMesh()
    source_mesh.vertices = o3d.utility.Vector3dVector(source_points)
    source_mesh.triangles = o3d.utility.Vector3iVector([])  # No faces, just points
# =======================================================================================================================================
    # Combine the target and source meshes into a single mesh

    # Define the base directory for saving outputs
    base_dir = r"C:\Users\abhay\OneDrive\Desktop\3dModel_latest (1)\final output"

    # Create a new subfolder with an incrementing name (e.g., s1, s2, s3, ...)
    subfolder_index = 1
    while True:
        subfolder_name = f"s{subfolder_index}"
        subfolder_path = os.path.join(base_dir, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            break
        subfolder_index += 1

    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(target_mesh.vertices), np.asarray(source_mesh.vertices)))
    )
    combined_mesh.triangles = o3d.utility.Vector3iVector(
        np.vstack((np.asarray(target_mesh.triangles),
                   np.asarray(source_mesh.triangles) + len(target_mesh.vertices)))
    )
    combined_mesh.compute_vertex_normals()

    # Save the combined mesh to a single .obj file
    combined_file = os.path.join(subfolder_path, "combined_aligned.obj")
    o3d.io.write_triangle_mesh(combined_file, combined_mesh)
    print(f"Combined aligned mesh saved to {combined_file}")
# =======================================================================================================================================

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

    target.points = o3d.utility.Vector3dVector(target_points)
    source.points = o3d.utility.Vector3dVector(source_points)

    # Generate a mesh from the adjusted target points with minimal smoothing
    radii = o3d.utility.DoubleVector([0.01, 0.02, 0.04])
    target_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(target, radii)
    target_mesh.compute_vertex_normals()

    # Generate a mesh from the adjusted source points with minimal smoothing
    source_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source, radii)
    source_mesh.compute_vertex_normals()



    # Visualize and save the meshes using PyVista
    plotter = pv.Plotter()
    adjusted_target_faces = np.hstack([[3] + list(triangle) for triangle in np.asarray(target_mesh.triangles)])
    adjusted_target_pv = pv.PolyData(np.asarray(target_mesh.vertices), adjusted_target_faces)
    
    adjusted_source_faces = np.hstack([[3] + list(triangle) for triangle in np.asarray(source_mesh.triangles)])
    adjusted_source_pv = pv.PolyData(np.asarray(source_mesh.vertices), adjusted_source_faces)
    plotter.add_mesh(adjusted_target_pv, color="green", opacity=1, 
                    label="Adjusted Target (Mesh)", 
                    show_edges=True, name="AdjustedTarget")
    plotter.add_mesh(adjusted_source_pv, color="red", opacity=0.7,
                    label="Adjusted Source (Mesh)", 
                    show_edges=True, name="AdjustedSource")
    plotter.enable_eye_dome_lighting()  # helps depth perception
    plotter.show_grid()
    plotter.show()


    # Prepare data for Excel
    adjustments_data = []

    for _ in range(10):  # Run the adjustment process 10 times
        for i, target_point in enumerate(target_points):
            closest_source_point = source_points[np.argmin(np.linalg.norm(source_points - target_point, axis=1))]
            if target_point[1] < closest_source_point[1]:  # Check if target point is lower in Z-axis
                existing_entry = next((entry for entry in adjustments_data if entry["Point Index"] == i), None)
                if existing_entry:
                    existing_entry["Adjusted Z"] = closest_source_point[1]
                else:
                    adjustments_data.append({
                        "Point Index": i,
                        "Original Z": target_point[1],
                        "Adjusted Z": closest_source_point[1]
                    })
                target_points[i][1] = closest_source_point[1]  # Adjust Z-coordinate to match the source

    for i, _ in enumerate(target_points):
        lowest_target_point = target_points[np.argmin(target_points[:, 1])]  # Get the lowest point of target in Z-axis
        closest_source_point = source_points[np.argmin(np.linalg.norm(source_points - lowest_target_point, axis=1))]
        if lowest_target_point[1] < closest_source_point[1]:  # Check if the lowest target point is lower in Z-axis
            point_index = np.argmin(target_points[:, 1])
            existing_entry = next((entry for entry in adjustments_data if entry["Point Index"] == point_index), None)
            if existing_entry:
                existing_entry["Adjusted Z"] = closest_source_point[1]
            else:
                adjustments_data.append({
                    "Point Index": point_index,
                    "Original Z": lowest_target_point[1],
                    "Adjusted Z": closest_source_point[1]
                })
            target_points[point_index, 1] = closest_source_point[1]  # Adjust Z-coordinate to match the source

    # Adjust the angle of the target to avoid penetration
    lowest_target_point = target_points[np.argmin(target_points[:, 1])]
    closest_source_point = source_points[np.argmin(np.linalg.norm(source_points - lowest_target_point, axis=1))]
    adjustments_data.append({
        "Lowest Target Point": lowest_target_point.tolist(),
        "Closest Source Point": closest_source_point.tolist()
    })

    # Save adjustments data to Excel
    adjustments_df = pd.DataFrame(adjustments_data)
    excel_file_path = os.path.join(subfolder_path, "adjustments_data.xlsx")
    adjustments_df.to_excel(excel_file_path, index=False)
    print(f"Adjustments data saved to {excel_file_path}")
        
    if lowest_target_point[1] < closest_source_point[1]:  # Check if penetration still exists
        # Rotate the target slightly around the X-axis to adjust the angle
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 180, 0, 0))  # Rotate by 1 degree
        target.rotate(rotation_matrix, center=target.get_center())
        target_points = np.asarray(target.points)  # Update target points after rotation

        # Check if the source needs to change its angle or position
        source_center = np.mean(np.asarray(source.points), axis=0)
        target_center = np.mean(np.asarray(target.points), axis=0)

        # Calculate the translation vector to align centers
        translation_vector = target_center - source_center
        source.translate(translation_vector)

        # Check if the source is upside down compared to the target
        source_normals = np.asarray(source.normals)
        target_normals = np.asarray(target.normals)

        if np.mean(source_normals[:, 1]) < 0:  # Assuming Y-axis normals indicate orientation
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # Rotate 180 degrees around X-axis
            source.rotate(rotation_matrix, center=source.get_center())
            print("yes")

    # closest_source_point = lowest_target_point - closest_source_point + np.array([0.005, 0.01, 0.01])
    # target.translate([closest_source_point[0], closest_source_point[1], closest_source_point[2]])
    target.translate([0, 0.01, 0])


    # Update the target with the adjusted points
    # Update the target with the adjusted points
    target.points = o3d.utility.Vector3dVector(target_points)
    source.points = o3d.utility.Vector3dVector(source_points)

    # Generate a mesh from the adjusted target points with minimal smoothing
    radii = o3d.utility.DoubleVector([0.01, 0.02, 0.04])
    target_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(target, radii)
    target_mesh.compute_vertex_normals()

    # Generate a mesh from the adjusted source points with minimal smoothing
    source_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source, radii)
    source_mesh.compute_vertex_normals()

    # Visualize the adjusted target and source as meshes
    plotter = pv.Plotter()
    adjusted_target_faces = np.hstack([[3] + list(triangle) for triangle in np.asarray(target_mesh.triangles)])
    adjusted_target_pv = pv.PolyData(np.asarray(target_mesh.vertices), adjusted_target_faces)
    
    adjusted_source_faces = np.hstack([[3] + list(triangle) for triangle in np.asarray(source_mesh.triangles)])
    adjusted_source_pv = pv.PolyData(np.asarray(source_mesh.vertices), adjusted_source_faces)
    plotter.add_mesh(adjusted_target_pv, color="green", opacity=1, 
                    label="Adjusted Target (Mesh)", 
                    show_edges=True, name="AdjustedTarget")
    plotter.add_mesh(adjusted_source_pv, color="red", opacity=0.7,
                    label="Adjusted Source (Mesh)", 
                    show_edges=True, name="AdjustedSource")
    plotter.enable_eye_dome_lighting()  # helps depth perception
    plotter.show_grid()
    plotter.show()


    # Save the adjusted target and source meshes as .obj files in the new subfolder
    target_file_path = os.path.join(subfolder_path, "adjusted_target_Mesh.obj")
    pv_mesh = pv.PolyData(np.asarray(target_mesh.vertices))  # Example initialization
    pv_to_o3d_mesh(pv_mesh)
    adjusted_target_o3d = pv_to_o3d_mesh(adjusted_target_pv)
    o3d.io.write_triangle_mesh(target_file_path, adjusted_target_o3d)


    source_file_path = os.path.join(subfolder_path, "adjusted_source_Mesh.obj")
    pv_mesh = pv.PolyData(np.asarray(source_mesh.vertices))  # Example initialization
    pv_to_o3d_mesh(pv_mesh)
    adjusted_source_o3d = pv_to_o3d_mesh(adjusted_source_pv)
    o3d.io.write_triangle_mesh(source_file_path, adjusted_source_o3d)

    p = pv.Plotter()
    p.add_mesh(adjusted_target_pv, color=True, opacity=1, name="Initial (Kabsch) Alignment")
    p.add_mesh(adjusted_source_pv, color="red", opacity = 0.5)
    p.enable_eye_dome_lighting() 

    p.show_grid()
    _ = p.show() 



# =======================================================================================================================================
    # Combine the target and source meshes into a single mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(
        np.vstack((np.asarray(target_mesh.vertices), np.asarray(source_mesh.vertices)))
    )
    combined_mesh.triangles = o3d.utility.Vector3iVector(
        np.vstack((np.asarray(target_mesh.triangles),
                   np.asarray(source_mesh.triangles) + len(target_mesh.vertices)))
    )
    combined_mesh.compute_vertex_normals()

    # Save the combined mesh to a single .obj file
    combined_file = os.path.join(subfolder_path, "combined_aligned_after_mesh.obj")
    o3d.io.write_triangle_mesh(combined_file, combined_mesh)
    print(f"Combined aligned mesh saved to {combined_file}")
# =======================================================================================================================================


    # Calculate distances for the best fit, allows for efficient nearest-neighbor searches on the saddle mesh.
    tree = KDTree(adjusted_target_pv.points)
    # Query the nearest points from final_saddle_mesh for each point in body_mesh_2 and compute distances.
    distances, idx = tree.query(adjusted_source_pv.points)
    # Add the calculated distances as a new attribute "distances" to body_mesh_2, indicating 
    # the distance between each point on body_mesh_2 and its nearest neighbor in the final_saddle_mesh.
    adjusted_source_pv["distances"] = distances

    # Save the distance statistics to an Excel file
    data = {
        "Statistic": ["Min", "Max", "Mean", "Median"],
        "Value": [np.min(distances), np.max(distances), np.mean(distances), np.median(distances)]
    }
    df = pd.DataFrame(data)
    excel_file_path = os.path.join(subfolder_path, "distance_statistics.xlsx")
    df.to_excel(excel_file_path, index=False)

    print(f"Min: {np.min(distances)}, Max: {np.max(distances)}, Mean: {np.mean(distances)}, Median: {np.median(distances)}")
    print("Distance statistics saved to 'distance_statistics.xlsx'")

    # ================================================================================
    # Visualize the optimal saddle fit on the horse along with distance measurements
    # ================================================================================
    p = pv.Plotter()
    p.add_mesh(adjusted_source_pv, scalars="distances", 
            smooth_shading=True, cmap="coolwarm", 
            clim=[0, np.max(distances)])
    p.add_mesh(adjusted_target_pv, color="white", 
            opacity=0.5, smooth_shading=True)
    p.show_grid()
    p.show() 



    # Extract x and z (assuming adjusted_target_pv is a pyvista mesh)
    x = adjusted_target_pv.points[:, 0]
    z = adjusted_target_pv.points[:, 2]

    # Create fine grid
    grid_x, grid_z = np.mgrid[
        x.min():x.max():1000j,
        z.min():z.max():1000j
    ]

    grid_distances = griddata((x, z), distances, (grid_x, grid_z), method='linear')
    masked_distances = np.ma.masked_invalid(grid_distances)
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(
        grid_x, grid_z, masked_distances,
        levels=200, cmap='coolwarm', alpha=0.95
    )
    plt.scatter(x, z, c=distances, cmap='coolwarm', s=2, edgecolors='k', linewidths=0.1)
    cbar = plt.colorbar(contour)
    cbar.set_label("Distance", fontsize=14)
    plt.title("Smoothed Heatmap (Valid Region Only)", fontsize=16)
    plt.xlabel("X", fontsize=13)
    plt.ylabel("Z", fontsize=13)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()