import numpy as np
import scipy
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class SourceParameters:
    amplitude: float
    s_x: float
    s_z: float
    width_x: float
    width_z: float

@dataclass
class GridParameters:
    x: float
    dx: float
    dz: float

@dataclass
class DensityProfileParameters:
    z_p: float          # Pycnocline depth
    d: float            # Pycnocline width
    rho_0: float        # Initial density
    delta_rho: float    # Change in density

@dataclass
class TimeParameters:
    t: float
    dt: float


class Bathysaurus:
    def __init__(self):
        self.x_grid = None
        self.z_grid = None
        self.sigma = None
        self.S = None
        self.R = None

    def get_bathymetry(self, x_grid):
        H = 200 + 0.5 * 200 * (
                    1.0 - np.tanh((x_grid - 3_200) / 600)
                )
        return np.full_like(x_grid, H, dtype=float)

    def set_grid(self, grid_parameters):
        self.x_grid = np.arange(
            0.0,
            grid_parameters.x + grid_parameters.dx,
            grid_parameters.dx,
        )

        # H(x): shape (nx,)
        bathymetry = self.get_bathymetry(self.x_grid)

        # Use the deepest point to define the common number of layers
        n_layers = int(np.ceil(np.max(bathymetry) / grid_parameters.dz)) + 1

        # sigma = 0 at surface, sigma = 1 at bottom
        self.sigma = np.linspace(0.0, 1.0, n_layers)

        # Shape: (nx, n_layers)
        self.z_grid = bathymetry[:, np.newaxis] * self.sigma[np.newaxis, :]

        return self.x_grid, self.z_grid

    def set_source(self, source_parameters):
        if self.x_grid is None or self.z_grid is None:
            raise RuntimeError("Call set_grid before set_source.")

        amplitude = source_parameters.amplitude
        s_x = source_parameters.s_x
        s_z = source_parameters.s_z
        width_x = source_parameters.width_x
        width_z = source_parameters.width_z

        if width_x <= 0.0 or width_z <= 0.0:
            raise ValueError("Source widths must be positive.")

        X = self.x_grid[:, np.newaxis]
        Z = self.z_grid

        # Gaussian source envelope
        horizontal_source = np.exp(
            -((X - s_x) ** 2) / (2.0 * width_x**2)
        )

        vertical_source = np.exp(
            -((Z - s_z) ** 2) / (2.0 * width_z**2)
        )

        # Broadcasting produces shape (nx, nz)
        source = (amplitude * horizontal_source * vertical_source)

        bathymetry = self.z_grid[:, -1]
        source[Z > bathymetry[:, np.newaxis]] = 0.0
        self.S = source[1:-1, :]

        return self.S

    def set_receivers(self, receiver_x):
        receiver_x = np.asarray(receiver_x, dtype=float)

        # Find nearest horizontal grid index for each receiver
        indices = np.abs(self.x_grid[np.newaxis, :] - receiver_x[:, np.newaxis]).argmin(axis=1)

        # Snap receiver x locations to the actual grid
        snapped_x = self.x_grid[indices]

        # Last vertical level is the local seafloor
        receiver_z = self.z_grid[indices, -1]

        receivers = np.column_stack((snapped_x, receiver_z))

        self.receiver_indices = indices
        self.R = receivers

        return self.R

    def get_density_profile(self, density_profile_parameters):
        z_p = density_profile_parameters.z_p
        d = density_profile_parameters.d
        rho_0 = density_profile_parameters.rho_0
        delta_rho = density_profile_parameters.delta_rho
        
        rho = rho_0 + 0.5 * delta_rho * (
            1.0 + np.tanh((self.z_grid - z_p) / d)
        )

        plt.plot((rho[0, :]), self.z_grid[0, :], color="black")
        plt.ylim(0, max(self.z_grid[0, :]))
        plt.xlabel("Density (kg/m^3)")
        plt.ylabel("Depth")
        plt.title("Density Profile")
        plt.gca().yaxis.set_inverted(True)
        plt.show()

        return rho

    def get_stratification_profile(self, density_profile):
        if self.z_grid is None:
            raise RuntimeError("Call set_grid first.")

        if density_profile.shape != self.z_grid.shape:
            raise ValueError("density_profile must have the same shape as z_grid.")

        nx, nz = self.z_grid.shape
        N2 = np.empty_like(density_profile)

        g = 9.81
        rho_ref = 1000.0

        for ix in range(nx):
            z = self.z_grid[ix, :]
            rho = density_profile[ix, :]

            drho_dz = np.gradient(rho, z, edge_order=2)

            N2[ix, :] = g / rho_ref * drho_dz

        # The modal problem uses interior vertical points.
        N2_interior = N2[:, 1:-1]

        return np.maximum(N2_interior, 1e-12)

    def get_modes(self, stratification_profile, n=5):
        if self.z_grid is None:
            raise RuntimeError("Call set_grid first.")

        nx, nz = self.z_grid.shape
        n_interior = nz - 2

        if stratification_profile.shape != (nx, n_interior):
            raise ValueError(
                "stratification_profile must have shape "
                f"{(nx, n_interior)}, got {stratification_profile.shape}."
            )

        n_modes = min(n, n_interior)

        phase_speeds = np.empty((nx, n_modes))
        mode_shapes = np.zeros((nx, n_modes, nz))
        mass_matrices = np.empty((nx, n_interior, n_interior))

        for ix in range(nx):
            z = self.z_grid[ix, :]

            # The grid is uniform in physical z within each x-column because
            # z = H(x) * sigma. This spacing can nevertheless vary with x.
            dz_local = np.diff(z)

            if not np.allclose(dz_local, dz_local[0]):
                raise ValueError(
                    "The current finite-difference K assumes uniform vertical "
                    "spacing within each x-column."
                )

            dz = dz_local[0]

            K = (
                np.diag(2.0 * np.ones(n_interior))
                + np.diag(-1.0 * np.ones(n_interior - 1), k=1)
                + np.diag(-1.0 * np.ones(n_interior - 1), k=-1)
            ) / dz**2

            N2_x = stratification_profile[ix, :]
            M = np.diag(N2_x)
            mass_matrices[ix, :, :] = M

            eigenvalues, eigenvectors = scipy.linalg.eigh(K, M)

            # Remove nonpositive eigenvalues before taking their square roots.
            valid = eigenvalues > 0.0
            eigenvalues = eigenvalues[valid]
            eigenvectors = eigenvectors[:, valid]

            modes_here = min(n_modes, len(eigenvalues))

            # lambda = 1 / c^2
            phase_speeds[ix, :modes_here] = (
                1.0 / np.sqrt(eigenvalues[:modes_here])
            )

            # Store each eigenvector as [mode, depth]
            mode_shapes[ix, :modes_here, 1:-1] = (
                eigenvectors[:, :modes_here].T
            )

        return phase_speeds, mode_shapes, mass_matrices
    
    def forward_model(
        self,
        rho,
        M,
        c,
        phi,
        n,
        time_parameters,
    ):
        nx, nz = self.z_grid.shape
        n_modes = min(n, c.shape[1])
        dt = time_parameters.dt
        t_steps = int(np.ceil(time_parameters.t / dt)) + 1
        t = np.arange(t_steps, dtype=float) * dt

        dx_values = np.diff(self.x_grid)

        if not np.allclose(dx_values, dx_values[0]):
            raise ValueError("The horizontal grid must be uniform.")

        dx = dx_values[0]

        if self.S.shape == (nx - 2, nz):
            source = np.zeros((nx, nz), dtype=float)
            source[1:-1, :] = self.S
        elif self.S.shape == (nx, nz):
            source = np.asarray(self.S, dtype=float)
        else:
            raise ValueError(
                "self.S must have shape (nx - 2, nz) or (nx, nz)."
            )

        source_interior = source[:, 1:-1]

        phi = np.array(
            phi[:, :n_modes, :],
            dtype=float,
            copy=True,
        )

        c = np.asarray(
            c[:, :n_modes],
            dtype=float,
        )

        M = np.asarray(M, dtype=float)

        n_interior = nz - 2

        if M.shape != (nx, n_interior, n_interior):
            raise ValueError(
                "M must have shape "
                f"(nx, nz - 2, nz - 2), got {M.shape}."
            )

        if phi.shape != (nx, n_modes, nz):
            raise ValueError(
                "phi must have shape "
                f"(nx, n_modes, nz), got {phi.shape}."
            )

        if c.shape != (nx, n_modes):
            raise ValueError(
                "c must have shape "
                f"(nx, n_modes), got {c.shape}."
            )

        for ix in range(1, nx):
            phi_previous = phi[ix - 1, :, 1:-1]
            phi_current = phi[ix, :, 1:-1]

            overlap = (
                phi_previous
                @ M[ix]
                @ phi_current.T
            )

            for mode in range(n_modes):
                if overlap[mode, mode] < 0.0:
                    phi[ix, mode, :] *= -1.0

        overlaps_right = np.zeros(
            (nx - 1, n_modes, n_modes),
            dtype=float,
        )

        overlaps_left = np.zeros(
            (nx - 1, n_modes, n_modes),
            dtype=float,
        )

        for ix in range(nx - 1):
            phi_left = phi[ix, :, 1:-1]
            phi_right = phi[ix + 1, :, 1:-1]

            overlaps_right[ix] = (
                phi_left
                @ M[ix]
                @ phi_right.T
            )

            overlaps_left[ix] = (
                phi_right
                @ M[ix + 1]
                @ phi_left.T
            )

        a = np.zeros(
            (t_steps, n_modes, nx),
            dtype=float,
        )

        for ix in range(nx):
            phi_x = phi[ix, :, 1:-1]

            a[0, :, ix] = (
                phi_x
                @ M[ix]
                @ source_interior[ix, :]
            )

        a[0, :, 0] = 0.0
        a[0, :, -1] = 0.0

        max_speed = np.max(c)

        if max_speed * dt / dx > 1.0:
            raise ValueError(
                "The time step violates the CFL condition."
            )

        def coupled_laplacian(a_current):
            a_xx = np.zeros_like(a_current)

            for ix in range(1, nx - 1):
                left_amplitudes = (
                    overlaps_left[ix - 1]
                    @ a_current[:, ix - 1]
                )

                right_amplitudes = (
                    overlaps_right[ix]
                    @ a_current[:, ix + 1]
                )

                a_xx[:, ix] = (
                    right_amplitudes
                    - 2.0 * a_current[:, ix]
                    + left_amplitudes
                ) / dx**2

            return a_xx

        if t_steps > 1:
            a_xx = coupled_laplacian(a[0])

            a[1, :, 1:-1] = (
                a[0, :, 1:-1]
                + 0.5
                * c[1:-1, :].T**2
                * dt**2
                * a_xx[:, 1:-1]
            )

            a[1, :, 0] = 0.0
            a[1, :, -1] = 0.0

        for step in range(1, t_steps - 1):
            a_xx = coupled_laplacian(a[step])

            a[step + 1, :, 1:-1] = (
                2.0 * a[step, :, 1:-1]
                - a[step - 1, :, 1:-1]
                + c[1:-1, :].T**2
                * dt**2
                * a_xx[:, 1:-1]
            )

            a[step + 1, :, 0] = 0.0
            a[step + 1, :, -1] = 0.0

        eta = np.zeros(
            (t_steps, nx, nz),
            dtype=float,
        )

        for ix in range(nx):
            eta[:, ix, :] = (
                a[:, :, ix]
                @ phi[ix, :, :]
            )

        drho_dz = np.zeros_like(
            rho,
            dtype=float,
        )

        for ix in range(nx):
            drho_dz[ix, :] = np.gradient(
                rho[ix, :],
                self.z_grid[ix, :],
                edge_order=2,
            )

        density_perturbations = (
            -eta
            * drho_dz[np.newaxis, :, :]
        )

        self.time_grid = t
        self.modal_amplitudes = a
        self.mode_shapes = phi
        self.mode_overlaps_right = overlaps_right
        self.mode_overlaps_left = overlaps_left
        self.eta = eta
        self.drho_dz = drho_dz
        self.density_perturbations = density_perturbations

        # ------------------------------------------------------------
        # Side-by-side receiver waveforms and density-perturbation field
        # ------------------------------------------------------------


        # ------------------------------------------------------------
        # Compute synthetic pressure waveforms
        # ------------------------------------------------------------

        receiver_pressure = np.zeros(
            (len(self.R), t_steps),
            dtype=float,
        )

        for receiver_number, receiver in enumerate(self.R):
            idx = self.receiver_indices[receiver_number]
            z = self.z_grid[idx, :]

            receiver_pressure[receiver_number, :] = (
                9.81
                * np.trapezoid(
                    density_perturbations[:, idx, :],
                    z,
                    axis=1,
                )
            )

        # ------------------------------------------------------------
        # Create side-by-side plots
        # ------------------------------------------------------------

        fig, (ax_waveform, ax_field) = plt.subplots(
            1,
            2,
            figsize=(15, 6),
            gridspec_kw={"width_ratios": [1, 1.35]},
        )

        # ------------------------------------------------------------
        # Left panel: receiver waveforms
        # ------------------------------------------------------------

        waveform_lines = []
        waveform_points = []

        for receiver_number, receiver in enumerate(self.R):
            line, = ax_waveform.plot(
                [],
                [],
                linewidth=1.5,
                label=(
                    f"Receiver {receiver_number + 1}: "
                    f"x={receiver[0]:.0f} m"
                ),
            )

            point, = ax_waveform.plot(
                [],
                [],
                marker="o",
                markersize=5,
            )

            waveform_lines.append(line)
            waveform_points.append(point)

        ax_waveform.set_xlim(t[0], t[-1])

        pressure_min = np.min(receiver_pressure)
        pressure_max = np.max(receiver_pressure)

        if pressure_min == pressure_max:
            pressure_min -= 1.0
            pressure_max += 1.0

        pressure_margin = 0.05 * (pressure_max - pressure_min)

        ax_waveform.set_ylim(
            pressure_min - pressure_margin,
            pressure_max + pressure_margin,
        )

        ax_waveform.set_xlabel("Time (s)")
        ax_waveform.set_ylabel("Pressure perturbation (Pa)")
        ax_waveform.set_title("Synthetic receiver waveforms")
        # ax_waveform.grid(True)
        ax_waveform.legend()

        # ------------------------------------------------------------
        # Right panel: density perturbation
        # ------------------------------------------------------------

        X = np.broadcast_to(
            self.x_grid[:, np.newaxis],
            self.z_grid.shape,
        )

        Z = self.z_grid
        bathymetry = self.z_grid[:, -1]

        max_density_perturbation = np.max(
            np.abs(density_perturbations)
        )

        if max_density_perturbation == 0.0:
            max_density_perturbation = 1.0

        image = ax_field.pcolormesh(
            X.T,
            Z.T,
            density_perturbations[0].T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-max_density_perturbation,
            vmax=max_density_perturbation,
            zorder=1,
        )

        ax_field.plot(
            self.x_grid,
            bathymetry,
            color="black",
            linewidth=2.0,
            zorder=10,
        )

        # Move the triangles slightly above the bottom so they are not clipped
        receiver_marker_z = (
            self.R[:, 1]
            - 0.025 * np.max(bathymetry)
        )

        receiver_markers, = ax_field.plot(
            self.R[:, 0],
            receiver_marker_z,
            linestyle="None",
            marker="v",
            markersize=10,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=20,
            clip_on=False,
        )

        ax_field.set_xlabel("Horizontal position $x$ (m)")
        ax_field.set_ylabel("Depth $z$ (m)")
        ax_field.invert_yaxis()

        colorbar = fig.colorbar(
            image,
            ax=ax_field,
        )

        colorbar.set_label(
            r"Density perturbation (kg m$^{-3}$)"
        )

        # Use a figure-level header so it visibly updates
        time_header = fig.suptitle(
            f"Synthetic internal-wave model, $t={t[0]:.1f}$ s",
            fontsize=14,
        )

        ax_field.set_title("Density perturbation")

        # ------------------------------------------------------------
        # Animation frame selection
        # ------------------------------------------------------------

        # Skip every other computed time step for faster playback
        frame_stride = 10

        regular_frames = list(
            np.arange(0, t_steps, frame_stride)
        )

        # Ensure the actual last frame is included
        if regular_frames[-1] != t_steps - 1:
            regular_frames.append(t_steps - 1)

        # At 25 frames per second, this holds the final frame for 15 seconds
        frames_to_hold = 15 * 25

        animation_frames = (
            regular_frames
            + [t_steps - 1] * frames_to_hold
        )

        # ------------------------------------------------------------
        # Animation update
        # ------------------------------------------------------------

        def update(frame):
            # Update receiver waveforms
            for receiver_number in range(len(self.R)):
                waveform_lines[receiver_number].set_data(
                    t[:frame + 1],
                    receiver_pressure[
                        receiver_number,
                        :frame + 1,
                    ],
                )

                waveform_points[receiver_number].set_data(
                    [t[frame]],
                    [receiver_pressure[receiver_number, frame]],
                )

            # Update density perturbation field
            image.set_array(
                density_perturbations[frame].T.ravel()
            )

            # Update time headers
            time_header.set_text(
                f"Synthetic internal-wave model, $t={t[frame]:.1f}$ s"
            )

            ax_field.set_title(
                f"Density perturbation, $t={t[frame]:.1f}$ s"
            )

            return (
                waveform_lines
                + waveform_points
                + [image, receiver_markers, time_header]
            )

        # ------------------------------------------------------------
        # Create and save GIF
        # ------------------------------------------------------------

        self.density_animation = FuncAnimation(
            fig,
            update,
            frames=animation_frames,
            interval=40,
            blit=False,
            repeat=False,
        )

        gif_writer = PillowWriter(fps=25)

        self.density_animation.save(
            "synthetic_internal_wave.gif",
            writer=gif_writer,
        )

        plt.tight_layout()
        plt.show()         



    def execute(
        self,
        grid_parameters,
        source_parameters,
        receiver_locations,
        density_profile_parameters,
        time_parameters,
        n=23,
    ):
        self.set_grid(grid_parameters)
        self.set_source(source_parameters)
        self.set_receivers(receiver_locations)

        rho = self.get_density_profile(density_profile_parameters)
        stratification = self.get_stratification_profile(rho)

        c, phi, M = self.get_modes(stratification, n=n)

        self.forward_model(
            rho=rho,
            M=M,
            c=c,
            phi=phi,
            n=n,
            time_parameters=time_parameters,
        )





if __name__ == "__main__":
    grid_parameters = GridParameters(
        x=5_000.0,
        dx=50.0,
        dz=5.0,
    )

    source_parameters = SourceParameters(
        amplitude=1.0,
        s_x=1000.0,
        s_z=30.0,
        width_x=100.0,
        width_z=100.0,
    )

    density_profile_parameters = DensityProfileParameters(
        z_p = 200.0,
        d = 120.0,
        rho_0 = 1_000,
        delta_rho = 3.0
    )

    time_parameters = TimeParameters(
        t = 10_000,
        dt = 2
    )

    model = Bathysaurus()

    model.execute(
        grid_parameters=grid_parameters,
        source_parameters=source_parameters,
        receiver_locations=[2_000.0, 3_000.0, 4_000.0],
        density_profile_parameters=density_profile_parameters,
        time_parameters=time_parameters,
        n=20,
    )
