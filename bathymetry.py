#!/usr/bin/env python3

from pathlib import Path

import bathyreq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

# Four approximate corners in [longitude, latitude] order.
POLYGON_VERTICES = np.array([
    [-128.5738, 43.2090],
    [-124.5464, 43.8619],
    [-128.9987, 44.1674],
    [-124.8353, 44.8702],
], dtype=float)


# Coordinates are [longitude, latitude].
source = np.array([
    -128.057,
    43.686,
], dtype=float)

receiver = np.array([
    -125.4053,
    44.5097,
], dtype=float)


VALUES_ARE_ELEVATION = True

OUTPUT_FILE = Path("portrait_bathymetry_map.png")
FIGURE_DPI = 300

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_vertices(vertices):
    vertices = np.asarray(vertices, dtype=float)

    if vertices.shape != (4, 2):
        raise ValueError(
            "POLYGON_VERTICES must have shape (4, 2)."
        )

    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "Polygon vertices must contain finite values."
        )

    return vertices


def validate_point(point, name):
    point = np.asarray(point, dtype=float)

    if point.shape != (2,):
        raise ValueError(
            f"{name} must be [longitude, latitude]."
        )

    if not np.all(np.isfinite(point)):
        raise ValueError(
            f"{name} must contain finite values."
        )

    return point


# ---------------------------------------------------------------------
# Local geographic coordinates
# ---------------------------------------------------------------------

def lonlat_to_local_km(
    longitude,
    latitude,
    center_longitude,
    center_latitude,
):
    """
    Convert longitude/latitude to local east/north coordinates in km.
    """
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    x_km = (
        EARTH_RADIUS_KM
        * np.radians(longitude - center_longitude)
        * np.cos(np.radians(center_latitude))
    )

    y_km = EARTH_RADIUS_KM * np.radians(
        latitude - center_latitude
    )

    return x_km, y_km


# ---------------------------------------------------------------------
# Fit and orient an exact rectangle
# ---------------------------------------------------------------------

def create_rectangle_transform(vertices, source, receiver):
    """
    Fit an exact rectangle to the four input vertices.

    The rectangle's long axis becomes the vertical plotting axis.
    The sign of that axis is selected so that the receiver plots above
    the source. The source and receiver are not horizontally aligned.
    """
    center_longitude = float(np.mean(vertices[:, 0]))
    center_latitude = float(np.mean(vertices[:, 1]))

    vertex_x, vertex_y = lonlat_to_local_km(
        longitude=vertices[:, 0],
        latitude=vertices[:, 1],
        center_longitude=center_longitude,
        center_latitude=center_latitude,
    )

    local_vertices = np.column_stack([
        vertex_x,
        vertex_y,
    ])

    # Principal-component analysis identifies the long and short
    # directions of the approximate rectangle.
    centered_vertices = (
        local_vertices
        - np.mean(local_vertices, axis=0)
    )

    covariance = np.cov(
        centered_vertices,
        rowvar=False,
        bias=True,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    long_axis = eigenvectors[:, np.argmax(eigenvalues)]
    long_axis = long_axis / np.linalg.norm(long_axis)

    # Make the short axis perpendicular to the long axis.
    short_axis = np.array([
        -long_axis[1],
        long_axis[0],
    ])

    # Convert source and receiver to the same local coordinate system.
    source_x, source_y = lonlat_to_local_km(
        longitude=source[0],
        latitude=source[1],
        center_longitude=center_longitude,
        center_latitude=center_latitude,
    )

    receiver_x, receiver_y = lonlat_to_local_km(
        longitude=receiver[0],
        latitude=receiver[1],
        center_longitude=center_longitude,
        center_latitude=center_latitude,
    )

    source_local = np.array([
        source_x,
        source_y,
    ])

    receiver_local = np.array([
        receiver_x,
        receiver_y,
    ])

    # Orient the vertical axis so receiver is above source.
    source_vertical = np.dot(source_local, long_axis)
    receiver_vertical = np.dot(receiver_local, long_axis)

    if receiver_vertical < source_vertical:
        long_axis = -long_axis
        short_axis = np.array([
            -long_axis[1],
            long_axis[0],
        ])

    # Project the approximate vertices into the rectangle coordinates.
    rectangle_x = local_vertices @ short_axis
    rectangle_y = local_vertices @ long_axis

    xmin = float(np.min(rectangle_x))
    xmax = float(np.max(rectangle_x))
    ymin = float(np.min(rectangle_y))
    ymax = float(np.max(rectangle_y))

    # These are the corners of an exact rectangle.
    rectangle = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ])

    return {
        "center_longitude": center_longitude,
        "center_latitude": center_latitude,
        "short_axis": short_axis,
        "long_axis": long_axis,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "rectangle": rectangle,
    }


def transform_coordinates(
    longitude,
    latitude,
    transform,
):
    """Convert geographic coordinates to rotated local x/y coordinates."""
    x_km, y_km = lonlat_to_local_km(
        longitude=longitude,
        latitude=latitude,
        center_longitude=transform["center_longitude"],
        center_latitude=transform["center_latitude"],
    )

    local_points = np.stack([
        x_km,
        y_km,
    ], axis=-1)

    display_x = np.sum(
        local_points * transform["short_axis"],
        axis=-1,
    )

    display_y = np.sum(
        local_points * transform["long_axis"],
        axis=-1,
    )

    return display_x, display_y


# ---------------------------------------------------------------------
# BathyReq download
# ---------------------------------------------------------------------

def download_bathymetry(vertices):
    """
    Download only the geographic bounding box surrounding the
    bathymetry rectangle. The source does not enlarge the request.
    """
    longitude_range = [
        float(np.min(vertices[:, 0])),
        float(np.max(vertices[:, 0])),
    ]

    latitude_range = [
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 1])),
    ]

    print("Downloading bathymetry:")
    print(f"  Longitude: {longitude_range}")
    print(f"  Latitude:  {latitude_range}")

    request = bathyreq.BathyRequest()

    data, longitude_vector, latitude_vector = request.get_area(
        longitude=longitude_range,
        latitude=latitude_range,
    )

    return (
        np.asarray(data),
        np.asarray(longitude_vector),
        np.asarray(latitude_vector),
    )


def prepare_grid(data, longitude_vector, latitude_vector):
    """Ensure bathymetry has shape (latitude, longitude)."""
    data = np.squeeze(data)
    longitude_vector = np.squeeze(longitude_vector)
    latitude_vector = np.squeeze(latitude_vector)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D array; received {data.shape}."
        )

    expected_shape = (
        latitude_vector.size,
        longitude_vector.size,
    )

    if data.shape == (
        longitude_vector.size,
        latitude_vector.size,
    ):
        data = data.T

    if data.shape != expected_shape:
        raise ValueError(
            f"Data shape {data.shape} does not match "
            f"expected shape {expected_shape}."
        )

    longitude_grid, latitude_grid = np.meshgrid(
        longitude_vector,
        latitude_vector,
    )

    return (
        np.asarray(data, dtype=float),
        longitude_grid,
        latitude_grid,
    )


# ---------------------------------------------------------------------
# Resample into the rotated rectangular coordinate system
# ---------------------------------------------------------------------

def resample_to_rectangle(
    depth,
    longitude_grid,
    latitude_grid,
    transform,
):
    """
    Transform the downloaded bathymetry to rotated coordinates and
    interpolate it onto a regular rectangular grid.

    Resampling is important: plotting the original lon/lat grid after
    rotation can make the result visually appear skewed.
    """
    display_x, display_y = transform_coordinates(
        longitude=longitude_grid,
        latitude=latitude_grid,
        transform=transform,
    )

    valid = np.isfinite(depth)

    valid_x = display_x[valid]
    valid_y = display_y[valid]
    valid_depth = depth[valid]

    if valid_depth.size == 0:
        raise ValueError(
            "The downloaded bathymetry contains no valid values."
        )

    width_km = transform["xmax"] - transform["xmin"]
    height_km = transform["ymax"] - transform["ymin"]

    # Approximately preserve the original grid density.
    original_rows, original_columns = depth.shape

    aspect = width_km / height_km

    output_rows = max(300, original_rows)
    output_columns = max(
        300,
        int(output_rows * aspect),
    )

    regular_x = np.linspace(
        transform["xmin"],
        transform["xmax"],
        output_columns,
    )

    regular_y = np.linspace(
        transform["ymin"],
        transform["ymax"],
        output_rows,
    )

    regular_x_grid, regular_y_grid = np.meshgrid(
        regular_x,
        regular_y,
    )

    regular_depth = griddata(
        points=np.column_stack([
            valid_x,
            valid_y,
        ]),
        values=valid_depth,
        xi=(
            regular_x_grid,
            regular_y_grid,
        ),
        method="linear",
    )

    return (
        regular_x_grid,
        regular_y_grid,
        np.ma.masked_invalid(regular_depth),
    )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_bathymetry(
    x_grid,
    y_grid,
    depth,
    transform,
    source,
    receiver,
    output_file,
):
    valid_depth = depth.compressed()

    if valid_depth.size == 0:
        raise ValueError(
            "No valid bathymetry values remain inside the rectangle."
        )

    shallowest = float(np.min(valid_depth))
    deepest = float(np.max(valid_depth))

    if np.isclose(shallowest, deepest):
        deepest = shallowest + 1.0

    normalization = Normalize(
        vmin=shallowest,
        vmax=deepest,
        clip=True,
    )

    source_x, source_y = transform_coordinates(
        longitude=source[0],
        latitude=source[1],
        transform=transform,
    )

    receiver_x, receiver_y = transform_coordinates(
        longitude=receiver[0],
        latitude=receiver[1],
        transform=transform,
    )

    rectangle = transform["rectangle"]

    figure, axis = plt.subplots(
        figsize=(8, 12),
        constrained_layout=True,
    )

    raster = axis.pcolormesh(
        x_grid,
        y_grid,
        depth,
        cmap="Blues",
        norm=normalization,
        shading="auto",
    )

    contour_levels = np.linspace(
        shallowest,
        deepest,
        12,
    )

    if np.unique(contour_levels).size > 1:
        contours = axis.contour(
            x_grid,
            y_grid,
            depth,
            levels=contour_levels,
            colors="midnightblue",
            linewidths=0.35,
            alpha=0.65,
        )

        axis.clabel(
            contours,
            inline=True,
            fontsize=7,
            fmt="%.0f m",
        )

    # Exact rectangle outline.
    rectangle_closed = np.vstack([
        rectangle,
        rectangle[0],
    ])

    axis.plot(
        rectangle_closed[:, 0],
        rectangle_closed[:, 1],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Bathymetry region",
        zorder=5,
    )

    # Source: circle.
    axis.plot(
        source_x,
        source_y,
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.4,
        linestyle="None",
        label="Source",
        zorder=10,
    )

    # Receiver: red downward-pointing triangle.
    axis.plot(
        receiver_x,
        receiver_y,
        marker="v",
        markersize=12,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=0.9,
        linestyle="None",
        label="Receiver",
        zorder=10,
    )

    axis.annotate(
        "Source",
        (source_x, source_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    axis.annotate(
        "Receiver",
        (receiver_x, receiver_y),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
    )

    # Include markers in the limits, but do not include them in the
    # bathymetry download or interpolation.
    all_x = np.concatenate([
        x_grid.ravel(),
        np.array([source_x, receiver_x]),
    ])

    all_y = np.concatenate([
        y_grid.ravel(),
        np.array([source_y, receiver_y]),
    ])

    finite_x = all_x[np.isfinite(all_x)]
    finite_y = all_y[np.isfinite(all_y)]

    x_range = finite_x.max() - finite_x.min()
    y_range = finite_y.max() - finite_y.min()

    x_margin = max(5.0, 0.04 * x_range)
    y_margin = max(5.0, 0.04 * y_range)

    axis.set_xlim(
        finite_x.min() - x_margin,
        finite_x.max() + x_margin,
    )

    axis.set_ylim(
        finite_y.min() - y_margin,
        finite_y.max() + y_margin,
    )

    axis.set_aspect("equal", adjustable="box")

    axis.set_title("LOI-2 Bathymetry")
    axis.set_xlabel("Distance (km)")
    axis.set_ylabel("Distance (km)")

    colorbar = figure.colorbar(
        raster,
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label("Water depth (m)")
    # axis.legend(loc="best")

    figure.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    print(f"Saved map to: {output_file.resolve()}")
    print(f"Shallowest depth: {shallowest:.2f} m")
    print(f"Deepest depth: {deepest:.2f} m")
    print(
        f"Source display position: "
        f"({source_x:.2f}, {source_y:.2f}) km"
    )
    print(
        f"Receiver display position: "
        f"({receiver_x:.2f}, {receiver_y:.2f}) km"
    )

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    vertices = validate_vertices(POLYGON_VERTICES)
    source_point = validate_point(source, "source")
    receiver_point = validate_point(receiver, "receiver")

    # Fit an exact rectangle and orient its long axis vertically.
    # The orientation is chosen so that the receiver is above the source.
    transform = create_rectangle_transform(
        vertices=vertices,
        source=source_point,
        receiver=receiver_point,
    )

    # Download only the area surrounding the bathymetry rectangle.
    data, longitude_vector, latitude_vector = download_bathymetry(
        vertices=vertices,
    )

    data, longitude_grid, latitude_grid = prepare_grid(
        data=data,
        longitude_vector=longitude_vector,
        latitude_vector=latitude_vector,
    )

    # Convert elevation to positive depth.
    if VALUES_ARE_ELEVATION:
        depth = -data
    else:
        depth = data

    depth = np.where(
        np.isfinite(depth),
        depth,
        np.nan,
    )

    # Transform the geographic bathymetry grid into the rotated coordinate
    # system and interpolate it onto a regular rectangular grid.
    x_grid, y_grid, rectangular_depth = resample_to_rectangle(
        depth=depth,
        longitude_grid=longitude_grid,
        latitude_grid=latitude_grid,
        transform=transform,
    )

    plot_bathymetry(
        x_grid=x_grid,
        y_grid=y_grid,
        depth=rectangular_depth,
        transform=transform,
        source=source_point,
        receiver=receiver_point,
        output_file=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()