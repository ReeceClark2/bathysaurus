import numpy as np
import scipy
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class Constants:
    gravity: float
    z_0: float
    density_0: float
    modes: int

@dataclass
class DensityProfile:
    z_p: float
    d: float
    delta_rho: float

@dataclass
class SourceProperties:
    x_s: float
    sigma_x: float
    z_s: float
    sigma_z: float

class Bathysaurus:
    def __init__(self, constants, density_profile, source_properties):
        self.constants = constants
        self.density_profile = density_profile
        self.source_properties = source_properties

    def get_bathymetry(self, x):
        # return 5_000 * np.ones(len(x))
        return (3_000 - x) * np.ones(len(x))

    def get_density(self, z):
        z_p = self.density_profile.z_p
        d = self.density_profile.d
        delta_rho = self.density_profile.delta_rho

        return self.constants.density_0 + 0.5 * delta_rho * (1 - np.tanh((z - z_p) / d))

    def get_stratification(self, z):
        density = self.get_density(z)
        dz = z[0, 1] - z[0, 0]
        drho_dz = np.gradient(density, dz, axis=1)

        N2 = -self.constants.gravity / self.constants.density_0 * drho_dz
        return np.maximum(N2, 1e-12)

    def get_modes(self, x):
        H = self.get_bathymetry(x)

        nz = 50
        z = np.zeros((len(x), nz))

        for i, depth in enumerate(H):
            z[i, :] = np.linspace(self.constants.z_0 - depth, self.constants.z_0, nz)

        N2 = self.get_stratification(z)

        cs = []
        eigenvectors = []

        for i, depth in enumerate(H):
            dz = z[i, 1] - z[i, 0]

            K = np.diag(2 * np.ones(nz - 2) / dz**2)
            K += np.diag(-np.ones(nz - 3) / dz**2, k=1)
            K += np.diag(-np.ones(nz - 3) / dz**2, k=-1)

            N2_x = N2[i, 1:-1]
            M = np.diag(N2_x)

            eigenvalue, eigenvector = scipy.linalg.eigh(K, M)

            c = 1 / np.sqrt(eigenvalue)

            full_eigenvector = np.zeros((nz, self.constants.modes))
            full_eigenvector[1:-1, :] = eigenvector[:, :self.constants.modes]

            cs.append(c[:self.constants.modes])
            eigenvectors.append(full_eigenvector)

        return z, np.array(cs), eigenvectors

    def get_source(self, x, z):
        x_s = self.source_properties.x_s
        sigma_x = self.source_properties.sigma_x
        z_s = self.source_properties.z_s
        sigma_z = self.source_properties.sigma_z

        G_x = np.exp(-((x - x_s) ** 2) / (2 * sigma_x ** 2))
        S_z = np.exp(-((z - z_s) ** 2) / (2 * sigma_z ** 2))

        return G_x * S_z

    def execute(self, x):
        z, cs, phis = self.get_modes(x)
        N2 = self.get_stratification(z)

        coefficients = []
        reconstructions = []

        for i in range(len(x)):
            phi = phis[i][1:-1, :]
            N2_i = N2[i, 1:-1]
            S_i = self.get_source(x[i], z[i, :])[1:-1]

            M = np.diag(N2_i)

            a = phi.T @ M @ S_i
            S_reconstructed = phi @ a

            coefficients.append(a)
            reconstructions.append(S_reconstructed)

        coefficients = np.array(coefficients)

        dx = x[1] - x[0]
        c_max = np.max(cs)

        dt = 0.5 * dx / c_max
        nt = 50

        A = np.zeros((nt, len(x), self.constants.modes))
        A[0, :, :] = coefficients

        for m in range(self.constants.modes):
            c = cs[:, m]

            A[1, 1:-1, m] = (
                A[0, 1:-1, m]
                + 0.5 * (c[1:-1] * dt / dx) ** 2
                * (
                    A[0, 2:, m]
                    - 2 * A[0, 1:-1, m]
                    + A[0, :-2, m]
                )
            )
        

        for n in range(1, nt - 1):
            for m in range(self.constants.modes):
                c = cs[:, m]

                A[n + 1, 1:-1, m] = (
                    2 * A[n, 1:-1, m]
                    - A[n - 1, 1:-1, m]
                    + (c[1:-1] * dt / dx) ** 2
                    * (
                        A[n, 2:, m]
                        - 2 * A[n, 1:-1, m]
                        + A[n, :-2, m]
                    )
                )

        return z, cs, phis, coefficients, reconstructions, A, dt
                        



if __name__ == "__main__":
    constants = Constants(
        gravity=9.81,
        z_0=0,
        density_0=1025,
        modes=5,
    )

    density_profile = DensityProfile(
        z_p = -1_000.0,
        d = 400.0,
        delta_rho = 3.0
    )

    source_properties = SourceProperties(
        x_s = 500,
        sigma_x = 10,
        z_s = -1_000.0,
        sigma_z = 400,
    )

    x = np.linspace(0, 2_000, 200)

    model = Bathysaurus(
        constants,
        density_profile,
        source_properties,
    )

    z, cs, eigenvectors, coefficients, reconstructions, A, dt = model.execute(x)

    plt.figure()

    plt.imshow(
        A[:, :, 0],
        extent=[x[0], x[-1], 500 * dt, 0],
        aspect="auto",
        cmap="RdBu_r"
    )

    plt.colorbar(label="Mode 1 amplitude")
    plt.xlabel("x (m)")
    plt.ylabel("time (s)")
    plt.title("Propagation of mode 1")
    plt.show()