import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

t = 1_000
dt = 5
t_grid = int(t / dt)

x = 1_000
dx = 10
x_grid = int(x / dx)

z = 400
dz = 5
z_grid = int(z / dz)

t_coords = np.arange(0, t, dt)
x_coords = np.arange(0, x, dx)
z_coords = np.arange(0, z, dz)

T, X, Z = np.meshgrid(t_coords, x_coords, z_coords, indexing="ij")

def get_initial_density_profile(z_coords):
    z_p = 200       # Pycnocline depth (m)
    d = 200          # Pycnocline width (m)
    rho_0 = 1_000   # kg/m^3
    delta_rho = 4   # kg/m^3

    density_profile = rho_0 + 0.5 * delta_rho * (1 - np.tanh((z_coords - z_p) / d))

    return density_profile

density = np.zeros((t_grid, x_grid, z_grid))
density[0, :, :] = get_initial_density_profile(z_coords)

def get_initial_stratification(z_coords):
    density_profile = get_initial_density_profile(z_coords)

    drho_dz = np.gradient(density_profile, z_coords)

    g = 9.81            # m/s^2
    rho_0 = 1_000       # kg/m^3
    stratification_profile = -g / rho_0 * drho_dz

    return stratification_profile

stratification = np.zeros((t_grid, x_grid, z_grid))
stratification[0, :, :] = get_initial_stratification(z_coords)

def get_modes(z_coords, stratification_profile, n_modes=5):
    n_z = len(z_coords)
    n_interior = n_z - 2

    dz = z_coords[1] - z_coords[0]

    stratrification_interior = stratification_profile[1:-1]
    M = np.diag(stratrification_interior)
        
    K = (
        np.diag(2.0 * np.ones(n_interior))
        + np.diag(-1.0 * np.ones(n_interior - 1), k=1)
        + np.diag(-1.0 * np.ones(n_interior - 1), k=-1)
    ) / dz**2

    eigenvalues, eigenvectors = scipy.linalg.eigh(K, M)
    n_available = min(n_modes, len(eigenvalues))

    phase_speeds = 1 / np.sqrt(eigenvalues[:n_available])
    mode_shapes = np.zeros((n_modes, n_z))
    mode_shapes[:n_available, 1:-1] = (eigenvectors[:, :n_available].T)

    return phase_speeds, mode_shapes

def get_source(x_coords, z_coords):
    x_s = 300       # Source horizontal location (m)
    sigma_x = 30    # Source horizontal spread (m)

    z_s = 394       # Source vertical location (m)
    sigma_z = 30    # Source vertical spread (m)

    G_x = np.exp(-((x_coords - x_s) ** 2) / (2 * sigma_x ** 2))
    S_z = np.exp(-((z_coords - z_s) ** 2) / (2 * sigma_z ** 2))

    amplitude = 1e6
    source = amplitude * np.outer(G_x, S_z)
    return source


phase_speeds, mode_shapes = get_modes(z_coords, stratification[0, 0, :])
print(f"Finite difference criterion {round(max(phase_speeds) * 5)} must be less than horizontal grid spacing {dx}")
a = np.zeros((t_grid, 5, x_grid))
reconstructions = np.zeros((5, x_grid))

S = get_source(x_coords, z_coords)       # shape: (x_grid, z_grid)

# Stratification at t=0, for example
stratification_profile = stratification[0, 0, :]
stratification_interior = stratification_profile[1:-1]

# Remove boundary points from the modes and source
phi = mode_shapes[:, 1:-1]               # shape: (n_modes, n_interior)
S_interior = S[:, 1:-1]                  # shape: (x_grid, n_interior)

# Optional explicit diagonal mass matrix
M = np.diag(stratification_interior)

# Initial modal amplitudes
a[0, :, :] = phi @ M @ S_interior.T

# Apply horizontal boundary conditions at t = 0
a[0, :, 0] = 0.0
a[0, :, -1] = 0.0

# First time step, assuming zero initial time derivative
a[1, :, 1:-1] = (
    a[0, :, 1:-1]
    + 0.5
    * (phase_speeds[:, None] * dt / dx) ** 2
    * (
        a[0, :, 2:]
        - 2.0 * a[0, :, 1:-1]
        + a[0, :, :-2]
    )
)

# Horizontal boundary conditions
a[1, :, 0] = 0.0
a[1, :, -1] = 0.0

pbar = tqdm(total=len(t_coords) - 1)
# Subsequent time steps
for step in range(1, len(t_coords) - 1):
    a[step + 1, :, 1:-1] = (
        2.0 * a[step, :, 1:-1]
        - a[step - 1, :, 1:-1]
        + (phase_speeds[:, None] * dt / dx) ** 2
        * (
            a[step, :, 2:]
            - 2.0 * a[step, :, 1:-1]
            + a[step, :, :-2]
        )
    )

    # Horizontal boundary conditions
    a[step + 1, :, 0] = 0.0
    a[step + 1, :, -1] = 0.0
    pbar.update(1)

pbar.close()
pbar.close()

# Background density profile and vertical density gradient
rho_background = get_initial_density_profile(z_coords)
drho_dz = np.gradient(rho_background, z_coords)

# Plotting grid
X_plot, Z_plot = np.meshgrid(
    x_coords,
    z_coords,
    indexing="ij"
)

# Compute all density perturbations first
density_perturbations = np.zeros((t_grid, x_grid, z_grid))

for time_index in range(t_grid):
    # Reconstruct vertical displacement
    eta = np.zeros((x_grid, z_grid))
    eta[:, 1:-1] = a[time_index, :, :].T @ phi

    # Density perturbation:
    # delta rho = -eta * d(rho_background)/dz
    density_perturbations[time_index] = (
        -eta * drho_dz[None, :]
    )

# Use one fixed color scale for the entire animation
perturbation_limit = np.max(
    np.abs(density_perturbations)
)

# Prevent invalid color limits if the perturbation is exactly zero
if perturbation_limit == 0:
    perturbation_limit = 1.0

fig, ax = plt.subplots(figsize=(10, 5))

image = ax.pcolormesh(
    X_plot,
    Z_plot,
    density_perturbations[0],
    shading="auto",
    cmap="RdBu_r",
    vmin=-perturbation_limit,
    vmax=perturbation_limit
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    label="Density perturbation (kg/m$^3$)"
)

ax.set_xlabel("Horizontal position x (m)")
ax.set_ylabel("Depth z (m)")
ax.invert_yaxis()

title = ax.set_title(
    f"Density Perturbation at t = {t_coords[0]:.1f} s"
)


def update(time_index):
    image.set_array(
        density_perturbations[time_index].ravel()
    )

    title.set_text(
        f"Density Perturbation at t = "
        f"{t_coords[time_index]:.1f} s"
    )

    return image, title


animation = FuncAnimation(
    fig,
    update,
    frames=range(t_grid),
    interval=50,
    blit=True,
    repeat=True
)

writer = FFMpegWriter(
    fps=20,
    metadata={"artist": "Matplotlib"},
    bitrate=1800
)

animation.save(
    "wave_prop.mp4",
    writer=writer,
    dpi=150
)

plt.close(fig)