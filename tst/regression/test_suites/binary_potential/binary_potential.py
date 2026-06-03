import os
import sys
import tempfile

_cache_dir = os.path.join(tempfile.gettempdir(), "apophis-gravity-mpl-cache")
os.environ.setdefault("MPLCONFIGDIR", _cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", _cache_dir)

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import PolyCollection

import utils.test_case

sys.dont_write_bytecode = True

BINARY_M1 = 2.0
BINARY_M2 = 1.0
BINARY_RSTAR = 6.0 / 1024.0
BINARY_X1 = 6.0 / 1024.0
BINARY_Y1 = 0.0
BINARY_Z1 = 0.0
BINARY_X2 = -12.0 / 1024.0
BINARY_Y2 = 0.0
BINARY_Z2 = 0.0

BINARY_LIMITS = {
    "phi_rel_l2": 5.0e-2,
    "phi_rel_max": 5.0e-1,
}


def analytic_uniform_sphere_phi(radius, mass, sphere_radius, grav_g=1.0):
    """Potential of a uniform-density sphere from Tomida & Stone (2023), Sec. 4.2."""
    radius = np.asarray(radius)
    phi = np.empty_like(radius, dtype=float)

    outside = radius >= sphere_radius
    phi[outside] = -grav_g * mass / radius[outside]

    rin = radius[~outside]
    phi[~outside] = (
        -grav_g * mass * (3.0 * sphere_radius**2 - rin**2) / (2.0 * sphere_radius**3)
    )

    return phi


def analytic_binary_phi(
    x,
    y,
    z,
    m1=BINARY_M1,
    m2=BINARY_M2,
    rstar=BINARY_RSTAR,
    x1=BINARY_X1,
    y1=BINARY_Y1,
    z1=BINARY_Z1,
    x2=BINARY_X2,
    y2=BINARY_Y2,
    z2=BINARY_Z2,
    grav_g=1.0,
):
    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)

    return analytic_uniform_sphere_phi(
        r1, m1, rstar, grav_g=grav_g
    ) + analytic_uniform_sphere_phi(r2, m2, rstar, grav_g=grav_g)


def slice_cells(data_file, values, plane="xy", requested=0.0):
    axes = {
        "xy": ("x", "y", "z"),
        "xz": ("x", "z", "y"),
        "yz": ("y", "z", "x"),
    }
    horizontal, vertical, normal = axes[plane]

    horizontal_faces = getattr(data_file, f"{horizontal}f")
    vertical_faces = getattr(data_file, f"{vertical}f")
    normal_faces = getattr(data_file, f"{normal}f")
    normal_domain_max = np.max(normal_faces)

    vertices = []
    colors = []

    for block in range(values.shape[0]):
        faces = normal_faces[block]
        normal_index = np.searchsorted(faces, requested, side="right") - 1
        if (
            normal_index == len(faces) - 1
            and np.isclose(requested, faces[-1])
            and np.isclose(requested, normal_domain_max)
        ):
            normal_index -= 1
        if normal_index < 0 or normal_index >= len(faces) - 1:
            continue

        if plane == "xy":
            value_slice = values[block, normal_index, :, :]
        elif plane == "xz":
            value_slice = values[block, :, normal_index, :]
        else:
            value_slice = values[block, :, :, normal_index]

        hfaces = horizontal_faces[block]
        vfaces = vertical_faces[block]
        for j in range(value_slice.shape[0]):
            y0 = vfaces[j]
            y1 = vfaces[j + 1]
            for i in range(value_slice.shape[1]):
                x0 = hfaces[i]
                x1 = hfaces[i + 1]
                vertices.append(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
                colors.append(value_slice[j, i])

    if not vertices:
        raise RuntimeError("No cells intersect requested slice.")

    return vertices, np.asarray(colors), horizontal, vertical, normal


def add_cell_plot(ax, vertices, values, title, xlabel, ylabel, cmap="viridis", norm=None):
    collection = PolyCollection(
        vertices,
        array=values,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )
    ax.add_collection(collection)
    points = np.asarray(vertices).reshape(-1, 2)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    return collection


def get_binary_potential_errors(data_file):
    zz, yy, xx = data_file.GetVolumeLocations(flatten=False)
    phi = data_file.Get("gravity.phi", flatten=False)
    phi_exact = analytic_binary_phi(xx, yy, zz)
    phi_error = phi - phi_exact
    phi_rel_error = phi_error / phi_exact

    return phi_error, phi_rel_error


def get_density(data_file):
    prim = data_file.Get("prim", flatten=False)
    return prim[:, 0, :, :, :]


def l2_norm(values):
    return np.sqrt(np.mean(values**2))


def signed_error_norm(values):
    limit = np.max(np.abs(values))
    if limit == 0.0:
        limit = 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)


def plot_binary_potential_residual(data_file, phi_error, phi_rel_error, output_path):
    density = get_density(data_file)
    rho_vertices, rho_values, horizontal, vertical, normal = slice_cells(
        data_file, density
    )
    phi_rel_vertices, phi_rel_values, *_ = slice_cells(data_file, phi_rel_error)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    rho_image = add_cell_plot(
        axes[0],
        rho_vertices,
        rho_values,
        rf"$\rho$ through {normal}=0",
        horizontal,
        vertical,
    )
    fig.colorbar(rho_image, ax=axes[0], label=r"$\rho$")

    rel_image = add_cell_plot(
        axes[1],
        phi_rel_vertices,
        phi_rel_values,
        rf"$(\phi - \phi_\mathrm{{analytic}}) / \phi_\mathrm{{analytic}}$ through {normal}=0",
        horizontal,
        vertical,
        cmap="coolwarm",
        norm=signed_error_norm(phi_rel_values),
    )
    fig.colorbar(rel_image, ax=axes[1], label=r"$\Delta\phi / \phi_\mathrm{analytic}$")

    fig.suptitle(
        (
            "binary_potential: "
            f"rho max={np.max(rho_values):.3e}, "
            f"relative phi L2={l2_norm(phi_rel_error):.3e}"
        ),
        fontsize=11,
    )

    fig.savefig(
        os.path.join(output_path, "binary_potential_residual.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            "parthenon/output0/id=1",
            "parthenon/output0/variables=gravity,prim",
            "parthenon/output0/dt=1.0",
            "parthenon/time/nlim=1",
        ]

        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        data_filename = os.path.join(parameters.output_path, "parthenon.1.final.phdf")
        if not os.path.exists(data_filename):
            print(f"Missing output file {data_filename}.")
            return False

        data_file = phdf.phdf(data_filename)
        phi_error, phi_rel_error = get_binary_potential_errors(data_file)
        phi_rel_l2 = l2_norm(phi_rel_error)
        phi_rel_max = np.max(np.abs(phi_rel_error))

        test_success = True
        if not np.isfinite(phi_rel_l2) or not np.isfinite(phi_rel_max):
            print("Non-finite binary potential error.")
            test_success = False

        for quantity, value in [
            ("phi_rel_l2", phi_rel_l2),
            ("phi_rel_max", phi_rel_max),
        ]:
            if value > BINARY_LIMITS[quantity]:
                print(
                    f"Binary potential {quantity}={value:.6e} exceeds "
                    f"{BINARY_LIMITS[quantity]:.6e}."
                )
                test_success = False

        with open(
            os.path.join(parameters.output_path, "binary-potential-errors.dat"), "w"
        ) as fp:
            fp.write("# phi_l2 phi_rel_l2 phi_max phi_rel_max\n")
            fp.write(
                f"{l2_norm(phi_error):.16e} {phi_rel_l2:.16e} "
                f"{np.max(np.abs(phi_error)):.16e} {phi_rel_max:.16e}\n"
            )

        plot_binary_potential_residual(
            data_file, phi_error, phi_rel_error, parameters.output_path
        )

        return test_success
