import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import os
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


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
    """
    Main function to process, align, and visualize 3D models of a horse and saddle.
    This function performs the following steps:
    1. Loads 3D models of the source (horse) and target (saddle) from .obj files.
    2. Preprocesses the point clouds by downsampling them to a specified voxel size.
    3. Allows the user to manually pick corresponding points from the source and target point clouds.
    4. Computes a transformation matrix to align the source to the target using the selected points.
    5. Applies the transformation to the source and visualizes the alignment using PyVista.
    6. Saves the transformed source and target meshes to a new subfolder with an incrementing name.
    7. Adjusts the Z-coordinates of the target points to avoid penetration with the source.
    8. Generates meshes from the adjusted point clouds using Ball Pivoting Algorithm (BPA).
    9. Visualizes the adjusted meshes and saves them as .obj files.
    10. Combines the source and target meshes into a single mesh and saves it.
    11. Calculates distance statistics between the source and target meshes and saves them to an Excel file.
    12. Visualizes the optimal saddle fit on the horse along with distance measurements.
    13. Performs additional alignment using Iterative Closest Point (ICP) and visualizes the results.
    14. Allows the user to pick points on the right and left sides of the target mesh to calculate average Z-coordinates and distances.
    """
    source_file = r"C:\Users\abhay\OneDrive\Desktop\3dModel_latest (1)\3d_saddles_horses\horse1.obj"
    target_file = r"C:\Users\abhay\OneDrive\Desktop\3dModel_latest (1)\3d_saddles_horses\saddle_c.obj"

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
    combined_file = os.path.join(subfolder_path, "combined_aligned_before_mesh.obj")
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



    adjustments_data = []

    for iteration in range(10):  # Run the adjustment process 10 times
        for i, target_point in enumerate(target_points):
            closest_source_point = source_points[np.argmin(np.linalg.norm(source_points - target_point, axis=1))]
            if target_point[1] < closest_source_point[1]:  # Check if target point is lower in Z-axis
                existing_entry = next((entry for entry in adjustments_data if entry["Point Index"] == i), None)
                if existing_entry:
                    existing_entry["Changed Distance in mm"] = (closest_source_point[1] - target_point[1])*1000
                else:
                    adjustments_data.append({
                        "Point Index": i,
                        "Original Coordinates": target_point.tolist(),
                        "Adjusted Coordinates": closest_source_point.tolist(),
                        # "Original Z": target_point[1],
                        # "Adjusted Z": closest_source_point[1]
                        "Changed Distance in mm": (closest_source_point[1] - target_point[1])*1000
                    })
                target_points[i][1] = closest_source_point[1]  # Adjust Z-coordinate to match the source

        # Visualize the changes after each iteration
        updated_target_pv = pv.PolyData(target_points)
        updated_target_pv["adjusted"] = np.zeros(len(target_points))
        
        # Mark changed points in red
        for entry in adjustments_data:
            updated_target_pv["adjusted"][entry["Point Index"]] = 1

        plotter = pv.Plotter()
        plotter.add_mesh(source_pv, color="blue", opacity=0.5, label="Source Mesh")
        plotter.add_mesh(updated_target_pv, scalars="adjusted", cmap="coolwarm", 
                 point_size=5, render_points_as_spheres=True, 
                 scalar_bar_args={"title": "Adjusted Points"})
        plotter.add_text(f"Iteration: {iteration + 1}", position="upper_left", font_size=5, color="black")
        plotter.show_grid()
        plotter.close()

        # Save the updated target points as a new .obj file
        updated_target_mesh = o3d.geometry.TriangleMesh()
        updated_target_mesh.vertices = o3d.utility.Vector3dVector(target_points)
        updated_target_mesh.triangles = o3d.utility.Vector3iVector([])  # No faces, just points

        # Handle file name repetition
        file_index = 1
        while True:
            combined_file = os.path.join(subfolder_path, f"Change_in_target_{file_index}.obj")
            if not os.path.exists(combined_file):
                break
            file_index += 1

        o3d.io.write_triangle_mesh(combined_file, updated_target_mesh)
        print(f"Updated target points saved to {combined_file}")

    for i, _ in enumerate(target_points):
        lowest_target_point = target_points[np.argmin(target_points[:, 1])]  # Get the lowest point of target in Z-axis
        closest_source_point = source_points[np.argmin(np.linalg.norm(source_points - lowest_target_point, axis=1))]
        if lowest_target_point[1] < closest_source_point[1]:  # Check if the lowest target point is lower in Z-axis
            point_index = np.argmin(target_points[:, 1])
            existing_entry = next((entry for entry in adjustments_data if entry["Point Index"] == point_index), None)
            if existing_entry:
                existing_entry["Adjusted Z"] = (closest_source_point[1] - lowest_target_point[1])*1000
            else:
                adjustments_data.append({
                    "Point Index": point_index,
                    "Original Coordinates": target_point.tolist(),
                    "Adjusted Coordinates": closest_source_point.tolist(),
                    "Changed Distance in mm": (closest_source_point[1] - lowest_target_point[1])*1000
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
    distances = tree.query(adjusted_source_pv.points)[0]
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

    # Plot the meshes and lines
    p = pv.Plotter()
    p.add_mesh(adjusted_source_pv, scalars="distances", 
               smooth_shading=True, cmap="coolwarm", 
               clim=[0, np.max(distances)], label="Source Mesh")
    p.add_mesh(adjusted_target_pv, color="white", 
               opacity=0.5, smooth_shading=True, label="Target Mesh")
    p.add_legend()
    p.show_grid()
    p.show()


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Step 1: Calculate nearest neighbor distances
    source_points = np.asarray(adjusted_source_pv.points)
    target_points = np.asarray(adjusted_target_pv.points)

    target_kdtree = cKDTree(target_points)
    distances, indices = target_kdtree.query(source_points)

    # Step 2: Add distances to source mesh
    source_pv = adjusted_source_pv.copy()
    target_pv = adjusted_target_pv.copy()
    distances_in_mm = distances * 1000  # Convert to inches
    # source_pv["distances_in_inches"] = distances_in_inches
    target_pv["distances_in_mm"] = distances_in_mm

    # Calculate the center of the source_pv
    source_center = source_pv.center

    # Define the length of the line
    line_length = 2  # Adjust the length as needed

    # Create a line along the Z-axis passing through the center
    line_start = source_center - np.array([0, 0, line_length / 2])
    line_end = source_center + np.array([0, 0, line_length / 2])

    # Create a PyVista line for the center
    center_line = pv.Line(line_start, line_end)

    # Create two parallel lines at a distance of 0.1 along the X-axis
    offset = 0.04
    line1_start = source_center - np.array([offset, 0,  line_length / 2])
    line1_end = source_center + np.array([-offset, 0,  line_length / 2])
    line2_start = source_center - np.array([-offset, 0,  line_length / 2])
    line2_end = source_center + np.array([offset, 0,  line_length / 2])

    # Create PyVista lines for the parallel lines
    parallel_line1 = pv.Line(line1_start, line1_end)
    parallel_line2 = pv.Line(line2_start, line2_end)

    # Filter points between parallel_line1 and parallel_line2
    source_points = np.asarray(source_pv.points)
    target_points = np.asarray(target_pv.points)

    # Calculate the X-coordinates of the parallel lines
    x_min = min(line1_start[0], line2_start[0])
    x_max = max(line1_start[0], line2_start[0])

    # Mask points that lie between the parallel lines
    mask = (source_points[:, 0] >= x_min) & (source_points[:, 0] <= x_max)

    # Calculate the average gap distance for the filtered points in mm
    average_gap_distance_mm = np.mean(source_pv["distances"][mask]) * 1000  # Convert to mm
    print(f"Average gap distance between horse and saddle: {average_gap_distance_mm:.2f} mm")


    # Calculate the X-coordinate of parallel_line1 and parallel_line2
    x_threshold_right = line1_start[0]
    x_threshold_left = line2_start[0]

    # Mask points that lie to the right of parallel_line1
    mask_right = source_points[:, 0] > x_threshold_right

    # Mask points that lie to the left of parallel_line2
    mask_left = source_points[:, 0] < x_threshold_left

    # Calculate the average gap distance for the filtered points in mm (right side)
    average_gap_distance_right_mm = np.mean(source_pv["distances"][mask_right]) * 1000  # Convert to mm
    print(f"Average gap distance between horse and saddle (right side): {average_gap_distance_right_mm:.2f} mm")

    # Calculate the average gap distance for the filtered points in mm (left side)
    average_gap_distance_left_mm = np.mean(source_pv["distances"][mask_left]) * 1000  # Convert to mm
    print(f"Average gap distance between horse and saddle (left side): {average_gap_distance_left_mm:.2f} mm")
    # Add the lines to the plotter
    plotter.add_mesh(center_line, color="yellow", line_width=5, label="Center Line")
    plotter.add_mesh(parallel_line1, color="purple", line_width=5, label="Parallel Line 1")
    plotter.add_mesh(parallel_line2, color="red", line_width=5, label="Parallel Line 2")

    # Display the average gap distance on the plot

    plotter = pv.Plotter()
    plotter.add_text(f"Avg Gap: {average_gap_distance_mm:.2f} mm", position="upper_left", font_size=10, color="black")
    plotter.add_text(f"Avg Gap (Right): {average_gap_distance_right_mm:.2f} mm", position="lower_left", font_size=10, color="black")
    plotter.add_text(f"Avg Gap (Left): {average_gap_distance_left_mm:.2f} mm", position="lower_right", font_size=10, color="black")
    plotter.add_mesh(source_pv, color="#214d0f",
                    smooth_shading=True, show_edges=False,opacity=0.7,
                    )
    plotter.add_mesh(target_pv,
                    smooth_shading=True, show_edges=False, scalars="distances_in_mm", cmap="coolwarm",
                      name="Target Mesh", scalar_bar_args={"title": "Distance (mm)"})
    plotter.enable_eye_dome_lighting()
    plotter.show_grid()
    plotter.show()








    # Extract point arrays from PyVista meshes
    source_points = np.asarray(adjusted_source_pv.points)
    target_points = np.asarray(adjusted_target_pv.points)


    # Parameters
    min_distance = 0.05
    num_points = 60

    # Step 1: Select 60 target points with at least 0.05 distance apart
    target_tree = cKDTree(target_points)
    all_indices = np.random.permutation(len(target_points))

    selected_target_points = []
    selected_indices = []

    for idx in all_indices:
        candidate = target_points[idx]
        if len(selected_target_points) == 0:
            selected_target_points.append(candidate)
            selected_indices.append(idx)
        else:
            distances = np.linalg.norm(np.array(selected_target_points) - candidate, axis=1)
            if np.all(distances >= min_distance):
                selected_target_points.append(candidate)
                selected_indices.append(idx)
        if len(selected_target_points) == num_points:
            break

    selected_target_points = np.array(selected_target_points)

    # Step 2: Match each target point to a source point with closest x and z
    selected_source_points = []
    for pt in selected_target_points:
        xz = pt[[0, 2]]
        diffs = np.linalg.norm(source_points[:, [0, 2]] - xz, axis=1)
        min_index = np.argmin(diffs)
        matched_source_point = source_points[min_index].copy()
        matched_source_point[[0, 2]] = xz  # force x and z to match exactly
        selected_source_points.append(matched_source_point)

    selected_source_points = np.array(selected_source_points)


    # Step 3: Name points and compute distances
    point_names = [f"p{i+1}" for i in range(len(selected_target_points))]
    distances = np.linalg.norm(selected_target_points - selected_source_points, axis=1)

    # Print distances
    print("Point Name | Distance between Target and Source")
    print("-----------------------------------------------")
    for name, dist in zip(point_names, distances):
        print(f"{name:<9} | {dist:.6f}")


    # Create a straight line from the center of the selected target points
    center_point = np.mean(selected_target_points, axis=0)
    line_length = 0.5  # Adjust as needed
    direction = np.array([0, 0, 1])  # Along the Z axis

    # Main center line
    line_start = center_point - direction * (line_length / 2)
    line_end = center_point + direction * (line_length / 2)
    straight_line = pv.Line(line_start, line_end)

    # Offset for parallel lines (along X axis, adjust as needed)
    offset = 0.035
    right_offset = np.array([offset, 0, 0])
    left_offset = np.array([-offset, 0, 0])

    # Right parallel line
    right_line_start = line_start + right_offset
    right_line_end = line_end + right_offset
    right_parallel_line = pv.Line(right_line_start, right_line_end)

    # Left parallel line
    left_line_start = line_start + left_offset
    left_line_end = line_end + left_offset
    left_parallel_line = pv.Line(left_line_start, left_line_end)

    # Step 3.1: Classify points by their X position relative to the parallel lines
    # right_parallel_line is offset along +X, left_parallel_line along -X
    right_x = right_line_start[0]
    left_x = left_line_start[0]

    # Points to the right of right_parallel_line
    right_side_mask = selected_target_points[:, 0] > right_x
    # Points to the left of left_parallel_line
    left_side_mask = selected_target_points[:, 0] < left_x

    center_mask = ~(right_side_mask | left_side_mask)

    # Extract actual point coordinates using masks
    right_points = selected_target_points[right_side_mask]
    sright_points = selected_source_points[right_side_mask]

    center_P =  selected_target_points[center_mask]
    center_S = selected_source_points[center_mask]

    left_points = selected_target_points[left_side_mask]
    sleft_points = selected_source_points[left_side_mask]



    # Wrap them in PyVista PolyData
    right_poly = pv.PolyData(right_points)
    S_right_poly = pv.PolyData(sright_points)

    # Access point coordinates via .points
    point_names1 = [f"Rp{i+1}" for i in range(len(right_poly.points))]

    # Compute distances
    distances = np.linalg.norm(right_poly.points - S_right_poly.points, axis=1)

    # Print distances
    print("Point Name | Distance between Target and Source")
    print("-----------------------------------------------")
    for name, dist in zip(point_names1, distances):
        print(f"{name:<9} | {dist:.6f}")


    # Wrap center points in PyVista PolyData
    center_poly = pv.PolyData(center_P)
    S_center_poly = pv.PolyData(center_S)
    # Access point coordinates via .points  
    point_names_center = [f"Cp{i+1}" for i in range(len(center_poly.points))]
    # Compute distances for center points
    distances_center = np.linalg.norm(center_poly.points - S_center_poly.points, axis=1)
    # Print distances for center points
    for name, dist in zip(point_names_center, distances_center):
        print(f"{name:<9} | {dist:.6f}")

    # Wrap left points in PyVista PolyData
    left_poly = pv.PolyData(left_points)
    S_left_poly = pv.PolyData(sleft_points)
    # Step 3: Name points and compute distances
    point_names2 = [f"Lp{i+1}" for i in range(len(left_poly.points))]
    distances_left = np.linalg.norm(left_poly.points - S_left_poly.points, axis=1)
    for name, dist in zip(point_names2, distances_left):
        print(f"{name:<9} | {dist:.6f}")

    # Save all distances and point names to Excel
    all_names = point_names1 + point_names_center + point_names2
    all_distances = np.concatenate([(distances-0.01)*1000, (distances_center-0.01)*1000, (distances_left-0.01)*1000])  # Convert to mm
    all_types = (["Right"] * len(point_names1)) + (["Center"] * len(point_names_center)) + (["Left"] * len(point_names2))
    df = pd.DataFrame({
        "Point Name": all_names,
        "Type": all_types,
        "Distance": all_distances
    })
    excel_file_path = os.path.join(subfolder_path, "pointwise_distances.xlsx")
    df.to_excel(excel_file_path, index=False)
    print(f"Pointwise distances saved to {excel_file_path}")



    # Add the straight line to the plotter

    # Step 4: Visualization

    plotter = pv.Plotter()
    plotter.add_mesh(adjusted_target_pv, color="white", opacity=0.5, show_edges=True, label="Target")
    plotter.add_mesh(adjusted_source_pv, color="red", opacity=0.5, show_edges=True, label="Source")
    label_data = pv.PolyData(right_poly.points)
    label_data["name"] = point_names1  # e.g., ["Rp1", "Rp2", ..., "RpN"]
    plotter.add_point_labels(label_data, "name", point_size=20, font_size=14, text_color="yellow")
    label_data_center = pv.PolyData(center_poly.points)
    label_data_center["name"] = point_names_center
    plotter.add_point_labels(label_data_center, "name", point_size=20, font_size=14, text_color="purple")
    label_data1 = pv.PolyData(left_poly.points)
    label_data1["name"] = point_names2
    plotter.add_point_labels(label_data1, "name", point_size=20, font_size=14, text_color="black")
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    main()

