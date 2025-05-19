import streamlit as st
import numpy as np
import plotly.graph_objects as go

def create_plot(theta_deg, phi_deg):
    r = 1
    theta_plane = np.radians(theta_deg)
    phi_plane = np.radians(phi_deg)

    nx = np.sin(theta_plane) * np.cos(phi_plane)
    ny = np.sin(theta_plane) * np.sin(phi_plane)
    nz = np.cos(theta_plane)
    normal = np.array([nx, ny, nz])

    # Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Flatten for mesh3d
    x_flat = x_sphere.flatten()
    y_flat = y_sphere.flatten()
    z_flat = z_sphere.flatten()

    sphere = go.Mesh3d(
        x=x_flat, y=y_flat, z=z_flat,
        alphahull=0,
        opacity=0.5,
        color='lightblue',
        name='Sphere'
    )

    # Plane
    s, t = np.meshgrid(np.linspace(-1.2, 1.2, 30), np.linspace(-1.2, 1.2, 30))
    v1 = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(v1) < 1e-10:  # Handle case if normal is (0,0,1)
        v1 = np.array([1,0,0])
    else:
        v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    x_plane = s * v1[0] + t * v2[0]
    y_plane = s * v1[1] + t * v2[1]
    z_plane = s * v1[2] + t * v2[2]

    plane = go.Surface(
        x=x_plane, y=y_plane, z=z_plane,
        colorscale='Greens',
        opacity=0.5,
        name='Plane',
        showscale=False
    )

    # Intersection circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2)
    circle *= r
    x_circle, y_circle, z_circle = circle[:, 0], circle[:, 1], circle[:, 2]

    circle_trace = go.Scatter3d(
        x=x_circle, y=y_circle, z=z_circle,
        mode='lines',
        line=dict(color='red', width=5),
        name='Intersection Circle'
    )

    fig = go.Figure(data=[sphere, plane, circle_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5]),
            yaxis=dict(range=[-1.5, 1.5]),
            zaxis=dict(range=[-1.5, 1.5]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Sphere-Plane Intersection\nTheta={theta_deg}°, Phi={phi_deg}°"
    )

    return fig

st.title("Interactive Sphere-Plane Intersection")

theta = st.slider("Theta (degrees, tilt from vertical)", 0, 180, 45)
phi = st.slider("Phi (degrees, rotation around z-axis)", 0, 360, 30)

fig = create_plot(theta, phi)

st.plotly_chart(fig, use_container_width=True)
