import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Sliders for angle inputs
theta_deg = st.slider("Tilt from vertical (θ, degrees)", 0, 180, 45)
phi_deg = st.slider("Rotation around z-axis (φ, degrees)", 0, 360, 30)

# Convert to radians
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# Plane normal vector
nx = np.sin(theta) * np.cos(phi)
ny = np.sin(theta) * np.sin(phi)
nz = np.cos(theta)
normal = np.array([nx, ny, nz])

# Sphere surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))

# Plane vectors
v1 = np.cross(normal, [0, 0, 1])
if np.allclose(v1, 0):
    v1 = np.array([1, 0, 0])  # Avoid zero vector if normal is z-axis
v1 /= np.linalg.norm(v1)
v2 = np.cross(normal, v1)

# Plane mesh
s, t = np.meshgrid(np.linspace(-1.2, 1.2, 10), np.linspace(-1.2, 1.2, 10))
x_plane = s * v1[0] + t * v2[0]
y_plane = s * v1[1] + t * v2[1]
z_plane = s * v1[2] + t * v2[2]

# Circle of intersection
theta_c = np.linspace(0, 2 * np.pi, 200)
circle = np.outer(np.cos(theta_c), v1) + np.outer(np.sin(theta_c), v2)
x_circle, y_circle, z_circle = circle[:, 0], circle[:, 1], circle[:, 2]

# ==== Figure 1: Default 3D View ====
fig1 = go.Figure()

fig1.add_surface(x=x_sphere, y=y_sphere, z=z_sphere, colorscale='Blues', opacity=0.6, showscale=False)
fig1.add_surface(x=x_plane, y=y_plane, z=z_plane, colorscale='Greens', opacity=0.5, showscale=False)
fig1.add_trace(go.Scatter3d(x=x_circle, y=y_circle, z=z_circle, mode='lines', line=dict(color='red', width=4)))

fig1.update_layout(
    title="3D View",
    scene=dict(
        xaxis=dict(range=[-1.5, 1.5]),
        yaxis=dict(range=[-1.5, 1.5]),
        zaxis=dict(range=[-1.5, 1.5]),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=30, b=0)
)

# ==== Figure 2: Perpendicular View ====
eye = 3 * normal / np.linalg.norm(normal)  # Pull camera back

fig2 = go.Figure()

fig2.add_surface(x=x_sphere, y=y_sphere, z=z_sphere, colorscale='Blues', opacity=0.3, showscale=False)
fig2.add_surface(x=x_plane, y=y_plane, z=z_plane, colorscale='Greens', opacity=0.5, showscale=False)
fig2.add_trace(go.Scatter3d(x=x_circle, y=y_circle, z=z_circle, mode='lines', line=dict(color='red', width=4)))

fig2.update_layout(
    title="Perpendicular View",
    scene=dict(
        xaxis=dict(range=[-1.5, 1.5]),
        yaxis=dict(range=[-1.5, 1.5]),
        zaxis=dict(range=[-1.5, 1.5]),
        aspectmode='data',
        camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))
    ),
    margin=dict(l=0, r=0, t=30, b=0)
)

# ==== Display side by side ====
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
