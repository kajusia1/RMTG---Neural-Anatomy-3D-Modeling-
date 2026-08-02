import streamlit as st
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import math
import os
import re
import ast

st.set_page_config(page_title="RMTg 3D Analysis", layout="wide")

# ==========================================
# 1. FUNKCJE POMOCNICZE I KALIBRACJA Z NOTEBOOKA
# ==========================================

def get_markers_by_type(root, marker_number):
    x, y, z = [], [], []
    for marker_type in root.findall('.//Marker_Type'):
        type_number = int(marker_type.find('Type').text)
        if type_number == marker_number:
            for marker in marker_type.findall('Marker'):
                x.append(float(marker.find('MarkerX').text))
                y.append(-float(marker.find('MarkerY').text))
                z.append(float(marker.find('MarkerZ').text))
    return [x, y, z]

def calibrate_with_rotation(root, ref_cor):
    cor_dict = {}
    for i in [1, 2, 3, 4, 5, 6]:
        x, y, z = get_markers_by_type(root, i)
        cor_dict[i] = (x, y, ref_cor[2])

    target_x3, target_y3 = ref_cor[0]
    target_x4, target_y4 = ref_cor[1]
    target_y3 = -target_y3
    target_y4 = -target_y4
    target_z = ref_cor[2]

    x3, y3, _ = cor_dict[3]
    x4, y4, _ = cor_dict[4]
    if not x3 or not x4:
        return None
    x3, y3 = x3[0], y3[0]
    x4, y4 = x4[0], y4[0]

    dx_ref = target_x4 - target_x3
    dy_ref = target_y4 - target_y3
    target_distance = math.sqrt(dx_ref**2 + dy_ref**2)

    dx_data = x4 - x3
    dy_data = y4 - y3
    old_distance = math.sqrt(dx_data**2 + dy_data**2)

    if old_distance == 0:
        return None

    scale = target_distance / old_distance
    angle_data = math.atan2(dy_data, dx_data)
    angle_ref = math.atan2(dy_ref, dx_ref)
    rotation_angle = angle_ref - angle_data

    def rotate(x_val, y_val, angle_rad):
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return x_val * cos_a - y_val * sin_a, x_val * sin_a + y_val * cos_a

    calibrated_cor_dict = {}
    for key, (x_list, y_list, z_list) in cor_dict.items():
        new_x, new_y = [], []
        for x_val, y_val in zip(x_list, y_list):
            x_shifted = x_val - x3
            y_shifted = y_val - y3
            x_scaled = x_shifted * scale
            y_scaled = y_shifted * scale
            x_rot, y_rot = rotate(x_scaled, y_scaled, rotation_angle)
            new_x.append(x_rot + target_x3)
            new_y.append(y_rot + target_y3)

        new_z = [target_z for _ in new_x]
        calibrated_cor_dict[key] = (new_x, new_y, new_z)

    return calibrated_cor_dict

def get_ref_for_slice(ref_cor_path, slice_group, slice_id):
    with open(ref_cor_path, 'r') as f:
        text = f.read()
        ref_dict = ast.literal_eval(text)
    key = f'{slice_id}{slice_group}'
    return ref_dict.get(key, None)

def get_cords(rat, slice_group, slice_id):
    base_dir = f'rat{rat}'
    group_dir = f'r{rat}{slice_group}'
    file_path = os.path.join(base_dir, group_dir, f'CellCounter_RMTg_x{slice_id}.xml')
    ref_cor_path = os.path.join(base_dir, f'r{rat}_cor_ref.txt')

    if not os.path.exists(file_path) or not os.path.exists(ref_cor_path):
        return None

    ref_cor = get_ref_for_slice(ref_cor_path, slice_group, slice_id)
    if ref_cor is None:
        return None

    tree = ET.parse(file_path)
    root = tree.getroot()
    return calibrate_with_rotation(root, ref_cor)

# ==========================================
# 2. GEOMETRIA 3D I SPRAWDZANIE PUNKTÓW
# ==========================================

def is_point_in_tetrahedron(point, tetrahedrons, all_points):
    p = np.array(point, dtype=np.float64)
    for simplex in tetrahedrons:
        pts = all_points[simplex]
        T = np.vstack([pts[0] - pts[3], pts[1] - pts[3], pts[2] - pts[3]]).T
        try:
            bary = np.linalg.solve(T, p - pts[3])
            l1, l2, l3 = bary
            l4 = 1.0 - (l1 + l2 + l3)
            if np.all(np.array([l1, l2, l3, l4]) >= -1e-7):
                return True
        except np.linalg.LinAlgError:
            continue
    return False

@st.cache_data
def build_3d_mesh_data(rat, max_3d_edge_length=1.0):
    base_path = f'rat{rat}'
    fig = go.Figure()
    slices_data = {}

    if not os.path.exists(base_path):
        return None, None, None

    for slice_group_folder in os.listdir(base_path):
        group_path = os.path.join(base_path, slice_group_folder)
        if not os.path.isdir(group_path):
            continue

        for file_name in os.listdir(group_path):
            if file_name.endswith('.xml') and 'CellCounter_RMTg_x' in file_name:
                reg = re.findall(r'\d+', file_name)
                if len(reg) < 2:
                    continue
                slice_id = f"{reg[-2]}_{reg[-1]}"
                slice_group = slice_group_folder[-2:]

                calibrated_coords = get_cords(rat, slice_group, slice_id)
                if not calibrated_coords:
                    continue

                slice_pts = []
                for marker_type in [1, 2]:
                    if marker_type in calibrated_coords:
                        xs, ys, zs = calibrated_coords[marker_type]
                        for x_val, y_val, z_val in zip(xs, ys, zs):
                            slice_pts.append([x_val, y_val, z_val])

                if slice_pts:
                    z_slice = slice_pts[0][2]
                    pts_arr = np.array(slice_pts)
                    if z_slice in slices_data:
                        slices_data[z_slice] = np.vstack([slices_data[z_slice], pts_arr])
                    else:
                        slices_data[z_slice] = pts_arr

    sorted_z = sorted(slices_data.keys())
    all_mesh_x, all_mesh_y, all_mesh_z = [], [], []
    all_i, all_j, all_k = [], [], []
    global_points = []
    active_tetrahedrons = []
    point_offset = 0

    for idx in range(len(sorted_z) - 1):
        z1, z2 = sorted_z[idx], sorted_z[idx + 1]
        pts1 = slices_data[z1]
        pts2 = slices_data[z2]

        pair_pts = np.vstack([pts1, pts2])
        if len(pair_pts) < 4:
            continue

        tri3d = Delaunay(pair_pts)
        filtered_tetrahedrons = []

        for simplex in tri3d.simplices:
            pts = pair_pts[simplex]
            edges = [
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
                np.linalg.norm(pts[0] - pts[3]),
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[1] - pts[3]),
                np.linalg.norm(pts[2] - pts[3])
            ]
            if max(edges) <= max_3d_edge_length:
                filtered_tetrahedrons.append(simplex)

        face_counts = {}
        for simplex in filtered_tetrahedrons:
            faces = [
                tuple(sorted([simplex[0], simplex[1], simplex[2]])),
                tuple(sorted([simplex[0], simplex[1], simplex[3]])),
                tuple(sorted([simplex[0], simplex[2], simplex[3]])),
                tuple(sorted([simplex[1], simplex[2], simplex[3]]))
            ]
            for face in faces:
                face_counts[face] = face_counts.get(face, 0) + 1

        boundary_faces = [face for face, count in face_counts.items() if count == 1]

        if boundary_faces:
            boundary_faces = np.array(boundary_faces)
            all_mesh_x.extend(pair_pts[:, 0])
            all_mesh_y.extend(pair_pts[:, 1])
            all_mesh_z.extend(pair_pts[:, 2])

            all_i.extend(boundary_faces[:, 0] + point_offset)
            all_j.extend(boundary_faces[:, 1] + point_offset)
            all_k.extend(boundary_faces[:, 2] + point_offset)

            for tet in filtered_tetrahedrons:
                active_tetrahedrons.append(tet + point_offset)

            global_points.extend(pair_pts)
            point_offset += len(pair_pts)

    if all_mesh_x:
        mesh = go.Mesh3d(
            x=all_mesh_x, y=all_mesh_y, z=all_mesh_z,
            i=all_i, j=all_j, k=all_k,
            color='skyblue', opacity=0.2, name="RMTg 3D Mesh"
        )
        fig.add_trace(mesh)

    fig.update_layout(
        scene=dict(
            xaxis_title='Lateral-Medial',
            yaxis_title='Anterior-Posterior',
            zaxis_title='Dorso-Ventral',
            aspectmode='data'
        ),
        title=f'RMTg Reconstruction - Rat {rat}'
    )

    return fig, np.array(global_points), active_tetrahedrons

# ==========================================
# 3. INTERFEJS UŻYTKOWNIKA (STREAMLIT UI)
# ==========================================

st.title("Verify Neuron Location in RMTg 3D")

st.sidebar.header("Model Settings")
rat_id = st.sidebar.number_input("Rat ID", value=2, step=1, min_value=2, max_value=8)
# max_edge = st.sidebar.number_input("Max 3D Edge Length", value=1.0, step=0.1)

with st.spinner("Data calibration and 3D model building..."):
    fig, global_points, tetrahedrons = build_3d_mesh_data(rat=rat_id)

if fig is None or len(global_points) == 0:
    st.error(f"No data or reference file available for folder 'rat{rat_id}'")
else:
    st.sidebar.header("Target Neuron Coordinates")
    target_x = st.sidebar.number_input("Lateral-Medial (X)", value=0.0, format="%.3f")
    target_y = st.sidebar.number_input("Anterior-Posterior (Y)", value=-7.60, format="%.3f")
    target_z = st.sidebar.number_input("Dorso-Ventral (Z)", value=-6.60, format="%.3f")

    target_point = (target_x, target_y, target_z)
    inside = is_point_in_tetrahedron(target_point, tetrahedrons, global_points)

    if inside:
        st.success(f"Target neuron {target_point} located: **IN RMTg**")
        p_color = "orange"
        p_name = "Target Point - IN RMTg"
    else:
        st.error(f"Target neuron {target_point} located: **OUTSIDE RMTg**")
        p_color = "orange"
        p_name = "Target Point - OUTSIDE RMTg"

    fig_display = go.Figure(fig)
    fig_display.add_trace(go.Scatter3d(
        x=[target_x], y=[target_y], z=[target_z],
        mode='markers',
        marker=dict(size=8, color=p_color, symbol='circle'),
        name=p_name,
        hoverinfo='name+x+y+z'
    ))

    st.plotly_chart(fig_display, use_container_width=True)