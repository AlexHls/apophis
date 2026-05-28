import itertools
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
from matplotlib.collections import PolyCollection

import utils.test_case

sys.dont_write_bytecode = True


SOLVERS = ["poisson", "monopole"]
RESOLUTIONS = [16, 32, 64]
UNIFORM_CFGS = [
    {
        "solver": solver,
        "resolution": resolution,
        "meshblock": resolution,
        "refined": False,
        "tag": f"{solver}_{resolution}",
    }
    for solver, resolution in itertools.product(SOLVERS, RESOLUTIONS)
]
REFINED_CFGS = [
    {
        "solver": solver,
        "resolution": 16,
        "meshblock": 8,
        "refined": True,
        "tag": f"{solver}_smr_16",
    }
    for solver in SOLVERS
]
ALL_CFGS = UNIFORM_CFGS + REFINED_CFGS
REFINED_LIMITS = {
    "phi_l2": 5.0e-3,
    "gx_l2": 5.0e-2,
    "phi_max": 5.0e-2,
    "gx_max": 2.0e-1,
}


def analytic_phi(r, rho0=1.0, r0=0.25, grav_g=1.0):
    r = np.asarray(r)
    phi = np.empty_like(r, dtype=float)

    inside = r <= r0
    mtot = 32.0 * np.pi * rho0 * r0**3 / 105.0

    phi[~inside] = -grav_g * mtot / r[~inside]
    rin = r[inside]
    phi[inside] = (
        -2.0 / 3.0 * np.pi * grav_g * rho0 * r0**2
        + 4.0
        * np.pi
        * grav_g
        * rho0
        * (
            rin**2 / 6.0
            - 1.0 / 10.0 * rin**4 / (r0**2)
            + 1.0 / 42.0 * rin**6 / (r0**4)
        )
    )

    return phi


def analytic_g(r, rho0=1.0, r0=0.25, grav_g=1.0):
    r = np.asarray(r)
    g = np.empty_like(r, dtype=float)

    inside = r <= r0
    mtot = 32.0 * np.pi * rho0 * r0**3 / 105.0

    g[~inside] = -grav_g * mtot / (r[~inside] ** 2)
    rin = r[inside]
    g[inside] = (
        -4.0
        * np.pi
        * grav_g
        * rho0
        * (
            rin / 3.0
            - 2.0 / 5.0 * rin**3 / (r0**2)
            + 1.0 / 7.0 * rin**5 / (r0**4)
        )
    )

    return g


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


def add_cell_plot(ax, vertices, values, title, xlabel, ylabel):
    collection = PolyCollection(
        vertices,
        array=values,
        cmap="viridis",
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


def get_errors(data_file):
    zz, yy, xx = data_file.GetVolumeLocations(flatten=False)
    phi = data_file.Get("gravity.phi", flatten=False)
    gravity = data_file.Get("gravity", flatten=False)
    gx = gravity[:, 0, :, :, :]

    rho0 = 1.0
    r0 = 0.25
    grav_g = 1.0

    radius = np.sqrt(xx**2 + yy**2 + zz**2)
    phi_exact = analytic_phi(radius, rho0=rho0, r0=r0, grav_g=grav_g)
    phi_error = phi - phi_exact

    g_exact = analytic_g(radius, rho0=rho0, r0=r0, grav_g=grav_g)
    gx_exact = np.zeros_like(g_exact)
    nonzero_radius = radius > 0.0
    gx_exact[nonzero_radius] = (
        g_exact[nonzero_radius] * xx[nonzero_radius] / radius[nonzero_radius]
    )
    gx_error = gx - gx_exact

    return phi_error, gx_error


def l2_norm(values):
    return np.sqrt(np.mean(values**2))


def plot_residual(data_file, phi_error, gx_error, output_path, config):
    solver = config["solver"]
    resolution = config["resolution"]
    suffix = " with SMR" if config["refined"] else ""

    phi_vertices, phi_values, horizontal, vertical, normal = slice_cells(
        data_file, np.abs(phi_error)
    )
    gx_vertices, gx_values, *_ = slice_cells(data_file, np.abs(gx_error))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    image = add_cell_plot(
        axes[0],
        phi_vertices,
        phi_values,
        rf"$|\phi - \phi_\mathrm{{analytic}}|$ through {normal}=0",
        horizontal,
        vertical,
    )
    fig.colorbar(image, ax=axes[0], label=r"$\Delta\phi$")

    gx_image = add_cell_plot(
        axes[1],
        gx_vertices,
        gx_values,
        rf"$|g_x - g_{{x,\mathrm{{analytic}}}}|$ through {normal}=0",
        horizontal,
        vertical,
    )
    fig.colorbar(gx_image, ax=axes[1], label=r"$\Delta g_x$")

    fig.suptitle(
        (
            f"{solver}, {resolution}^3{suffix}: "
            f"phi L2={l2_norm(phi_error):.3e}, gx L2={l2_norm(gx_error):.3e}"
        ),
        fontsize=11,
    )

    fig.savefig(
        os.path.join(output_path, f"gravity_residual_{config['tag']}.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


def plot_convergence(errors, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    markers = {"poisson": "o", "monopole": "s"}

    for ax, quantity, ylabel in [
        (axes[0], "phi_l2", r"$||\Delta\phi||_2$"),
        (axes[1], "gx_l2", r"$||\Delta g_x||_2$"),
    ]:
        for solver in SOLVERS:
            rows = [row for row in errors if row["solver"] == solver]
            rows = sorted(rows, key=lambda row: row["resolution"])
            res = np.array([row["resolution"] for row in rows], dtype=float)
            err = np.array([row[quantity] for row in rows], dtype=float)
            ax.plot(res, err, marker=markers[solver], label=solver)

        reference = sorted(errors, key=lambda row: row[quantity])[-1][quantity]
        res_ref = np.array([RESOLUTIONS[0], RESOLUTIONS[-1]], dtype=float)
        err_ref = reference * (res_ref / RESOLUTIONS[0]) ** -2
        ax.plot(res_ref, err_ref, "k--", label="second order")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Base grid resolution")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")

    axes[0].legend()
    fig.savefig(
        os.path.join(output_path, "gravity_l2_convergence.png"),
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        config = ALL_CFGS[step - 1]
        solver = config["solver"]
        resolution = config["resolution"]
        meshblock = config["meshblock"]

        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={resolution}",
            f"parthenon/mesh/nx2={resolution}",
            f"parthenon/mesh/nx3={resolution}",
            f"parthenon/meshblock/nx1={meshblock}",
            f"parthenon/meshblock/nx2={meshblock}",
            f"parthenon/meshblock/nx3={meshblock}",
            f"gravity/type={solver}",
            f"parthenon/output0/id={step}",
            "parthenon/output0/variables=gravity",
            "parthenon/output0/dt=1.0",
            "parthenon/time/nlim=1",
        ]
        if config["refined"]:
            parameters.driver_cmd_line_args += [
                "parthenon/mesh/refinement=static",
                "parthenon/static_refinement1/level=2",
            ]
        else:
            parameters.driver_cmd_line_args += [
                "parthenon/mesh/refinement=none",
                "parthenon/static_refinement1/level=1",
            ]
        if solver == "monopole":
            parameters.driver_cmd_line_args.append(f"gravity/nrbin={4 * resolution}")

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

        errors = []
        test_success = True

        for step, config in enumerate(ALL_CFGS, start=1):
            solver = config["solver"]
            resolution = config["resolution"]
            data_filename = os.path.join(
                parameters.output_path, f"parthenon.{step}.final.phdf"
            )
            if not os.path.exists(data_filename):
                print(f"Missing output file {data_filename}.")
                test_success = False
                continue

            data_file = phdf.phdf(data_filename)
            phi_error, gx_error = get_errors(data_file)
            phi_l2 = l2_norm(phi_error)
            gx_l2 = l2_norm(gx_error)
            phi_max = np.max(np.abs(phi_error))
            gx_max = np.max(np.abs(gx_error))

            if not np.isfinite(phi_l2) or not np.isfinite(gx_l2):
                print(f"Non-finite gravity error for {solver} at {resolution}^3.")
                test_success = False
            if config["refined"]:
                for quantity, value in [
                    ("phi_l2", phi_l2),
                    ("gx_l2", gx_l2),
                    ("phi_max", phi_max),
                    ("gx_max", gx_max),
                ]:
                    if value > REFINED_LIMITS[quantity]:
                        print(
                            f"{solver} SMR {quantity}={value:.6e} exceeds "
                            f"{REFINED_LIMITS[quantity]:.6e}."
                        )
                        test_success = False

            errors.append(
                {
                    "tag": config["tag"],
                    "solver": solver,
                    "resolution": resolution,
                    "refined": config["refined"],
                    "phi_l2": phi_l2,
                    "gx_l2": gx_l2,
                    "phi_max": phi_max,
                    "gx_max": gx_max,
                }
            )
            plot_residual(data_file, phi_error, gx_error, parameters.output_path, config)

        if len(errors) != len(ALL_CFGS):
            return False

        with open(os.path.join(parameters.output_path, "gravity-errors.dat"), "w") as fp:
            fp.write("# tag solver resolution refined phi_l2 gx_l2 phi_max gx_max\n")
            for row in errors:
                fp.write(
                    f"{row['tag']} {row['solver']} {row['resolution']} "
                    f"{int(row['refined'])} {row['phi_l2']:.16e} "
                    f"{row['gx_l2']:.16e} {row['phi_max']:.16e} "
                    f"{row['gx_max']:.16e}\n"
                )

        uniform_errors = [row for row in errors if not row["refined"]]
        plot_convergence(uniform_errors, parameters.output_path)

        for solver in SOLVERS:
            rows = sorted(
                [row for row in uniform_errors if row["solver"] == solver],
                key=lambda row: row["resolution"],
            )
            phi_errors = np.array([row["phi_l2"] for row in rows])
            gx_errors = np.array([row["gx_l2"] for row in rows])
            if np.any(phi_errors[1:] >= phi_errors[:-1]):
                print(f"{solver} phi L2 error does not decrease monotonically.")
                test_success = False
            if np.any(gx_errors[1:] >= gx_errors[:-1]):
                print(f"{solver} gx L2 error does not decrease monotonically.")
                test_success = False

        return test_success
