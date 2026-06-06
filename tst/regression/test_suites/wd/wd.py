import os
import sys

import matplotlib
import h5py
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

import utils.test_case

sys.dont_write_bytecode = True

REFERENCE_FILE = "massrad_one_wd.txt"
THERMO_REFERENCE_FILE = "wd_one_wd.h5"
SUBSAMPLES_PER_DIM = 3
COMPARE_MASS_FRACTION_MIN = 1.0e-2
COMPARE_MASS_FRACTION_MAX = 9.95e-1

MASS_REL_TOL = 1.0e-4
RADIUS_MEAN_REL_TOL = 8.0e-3
RADIUS_P95_REL_TOL = 1.5e-2
RADIUS_MAX_REL_TOL = 7.5e-2

THERMO_DENSITY_MEAN_REL_TOL = 1.0e-8
THERMO_DENSITY_P95_REL_TOL = 1.0e-7
THERMO_DENSITY_MAX_REL_TOL = 1.0e-6
THERMO_TEMPERATURE_MAX_REL_TOL = 1.0e-8
THERMO_COMPOSITION_MAX_ABS_TOL = 1.0e-14
THERMO_YE_MAX_ABS_TOL = 1.0e-14

IDN = 0
NHYDRO = 5
O16 = NHYDRO + 2
NE20 = NHYDRO + 3
YE = NHYDRO + 6
LTEMP = 2


def source_root(parameters):
    return os.path.abspath(
        os.path.join(os.path.dirname(parameters.driver_input_path), "..")
    )


def load_reference(path):
    total_mass = None
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith("# total_mass_g"):
                total_mass = float(line.split("=", 1)[1])
                break

    if total_mass is None:
        raise RuntimeError(f"Reference file {path} does not contain total_mass_g.")

    data = np.loadtxt(path)
    return {
        "total_mass": total_mass,
        "mass_fraction": data[:, 0],
        "radius": data[:, 1],
    }


def load_thermo_reference(path):
    with h5py.File(path, "r") as h5:
        species = [
            s.decode("utf-8") if isinstance(s, bytes) else s
            for s in h5["composition_mass_fraction"].attrs["species"]
        ]
        composition = h5["composition_mass_fraction"][:]

        return {
            "radius": h5["radius_cm"][:],
            "density": h5["density_g_cm3"][:],
            "temperature": h5["temperature_K"][:],
            "o16": composition[:, species.index("O16")],
            "ne20": composition[:, species.index("Ne20")],
            "ye": h5["ye"][:],
        }


def grid_mass_radius(data_file):
    rho = data_file.Get("cons", flatten=False)[:, 0]
    nblock, nz, ny, nx = rho.shape
    nsub = SUBSAMPLES_PER_DIM
    npoint = nblock * nz * ny * nx * nsub**3

    radii = np.empty(npoint)
    masses = np.empty(npoint)
    offsets = (np.arange(nsub) + 0.5) / nsub

    out = 0
    for block in range(nblock):
        dx = np.diff(data_file.xf[block])
        dy = np.diff(data_file.yf[block])
        dz = np.diff(data_file.zf[block])

        x_sub = (
            data_file.xf[block, :-1, None] + offsets[None, :] * dx[:, None]
        ).ravel()
        y_sub = (
            data_file.yf[block, :-1, None] + offsets[None, :] * dy[:, None]
        ).ravel()
        z_sub = (
            data_file.zf[block, :-1, None] + offsets[None, :] * dz[:, None]
        ).ravel()

        zz, yy, xx = np.meshgrid(z_sub, y_sub, x_sub, indexing="ij")
        nlocal = xx.size
        radii[out : out + nlocal] = np.sqrt(
            xx.ravel() ** 2 + yy.ravel() ** 2 + zz.ravel() ** 2
        )

        cell_vol = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
        sub_vol = np.repeat(
            np.repeat(np.repeat(cell_vol / nsub**3, nsub, axis=0), nsub, axis=1),
            nsub,
            axis=2,
        )
        rho_sub = np.repeat(
            np.repeat(np.repeat(rho[block], nsub, axis=0), nsub, axis=1), nsub, axis=2
        )
        masses[out : out + nlocal] = (rho_sub * sub_vol).ravel()
        out += nlocal

    order = np.argsort(radii)
    radii = radii[order]
    cumulative_mass = np.cumsum(masses[order])

    return {
        "total_mass": cumulative_mass[-1],
        "radius": radii,
        "cumulative_mass": cumulative_mass,
    }


def grid_cell_thermo(data_file):
    cons = data_file.Get("cons", flatten=False)
    eos_lambda = data_file.Get("eos_lambda", flatten=False)
    density = cons[:, IDN]
    nblock, nz, ny, nx = density.shape
    npoint = nblock * nz * ny * nx

    radii = np.empty(npoint)
    grid_density = np.empty(npoint)
    temperature = np.empty(npoint)
    o16 = np.empty(npoint)
    ne20 = np.empty(npoint)
    ye = np.empty(npoint)

    out = 0
    for block in range(nblock):
        zz, yy, xx = np.meshgrid(
            data_file.z[block], data_file.y[block], data_file.x[block], indexing="ij"
        )
        nlocal = xx.size

        rho = density[block].ravel()
        radii[out : out + nlocal] = np.sqrt(
            xx.ravel() ** 2 + yy.ravel() ** 2 + zz.ravel() ** 2
        )
        grid_density[out : out + nlocal] = rho
        temperature[out : out + nlocal] = 10.0 ** eos_lambda[block, LTEMP].ravel()
        o16[out : out + nlocal] = cons[block, O16].ravel() / rho
        ne20[out : out + nlocal] = cons[block, NE20].ravel() / rho
        ye[out : out + nlocal] = cons[block, YE].ravel() / rho
        out += nlocal

    return {
        "radius": radii,
        "density": grid_density,
        "temperature": temperature,
        "o16": o16,
        "ne20": ne20,
        "ye": ye,
    }


def compare_profile(grid, reference):
    compare = (
        (reference["mass_fraction"] >= COMPARE_MASS_FRACTION_MIN)
        & (reference["mass_fraction"] <= COMPARE_MASS_FRACTION_MAX)
        & (reference["radius"] > 0.0)
    )

    grid_radius = np.interp(
        reference["mass_fraction"][compare] * grid["total_mass"],
        grid["cumulative_mass"],
        grid["radius"],
    )
    reference_radius = reference["radius"][compare]
    rel_radius_error = (grid_radius - reference_radius) / reference_radius

    return {
        "mass_rel_error": (grid["total_mass"] - reference["total_mass"])
        / reference["total_mass"],
        "radius_mean_rel_error": np.mean(np.abs(rel_radius_error)),
        "radius_p95_rel_error": np.quantile(np.abs(rel_radius_error), 0.95),
        "radius_max_rel_error": np.max(np.abs(rel_radius_error)),
        "mass_fraction": reference["mass_fraction"][compare],
        "reference_radius": reference_radius,
        "grid_radius": grid_radius,
        "rel_radius_error": rel_radius_error,
    }


def rel_errors(actual, expected):
    return (actual - expected) / np.maximum(np.abs(expected), np.finfo(float).tiny)


def compare_thermo_profile(grid, reference):
    compare = (
        (grid["radius"] >= reference["radius"][1])
        & (grid["radius"] <= reference["radius"][-1])
        & np.isfinite(grid["density"])
    )

    if not np.any(compare):
        raise RuntimeError(
            "No WD grid cells were available for thermodynamic comparison."
        )

    radius = grid["radius"][compare]
    reference_density = np.interp(radius, reference["radius"], reference["density"])
    reference_temperature = np.interp(
        radius, reference["radius"], reference["temperature"]
    )
    reference_o16 = np.interp(radius, reference["radius"], reference["o16"])
    reference_ne20 = np.interp(radius, reference["radius"], reference["ne20"])
    reference_ye = np.interp(radius, reference["radius"], reference["ye"])

    density_rel_error = rel_errors(grid["density"][compare], reference_density)
    temperature_rel_error = rel_errors(
        grid["temperature"][compare], reference_temperature
    )
    o16_abs_error = grid["o16"][compare] - reference_o16
    ne20_abs_error = grid["ne20"][compare] - reference_ne20
    ye_abs_error = grid["ye"][compare] - reference_ye

    return {
        "cell_count": np.count_nonzero(compare),
        "density_mean_rel_error": np.mean(np.abs(density_rel_error)),
        "density_p95_rel_error": np.quantile(np.abs(density_rel_error), 0.95),
        "density_max_rel_error": np.max(np.abs(density_rel_error)),
        "temperature_max_rel_error": np.max(np.abs(temperature_rel_error)),
        "o16_max_abs_error": np.max(np.abs(o16_abs_error)),
        "ne20_max_abs_error": np.max(np.abs(ne20_abs_error)),
        "ye_max_abs_error": np.max(np.abs(ye_abs_error)),
        "radius": radius,
        "reference_density": reference_density,
        "grid_density": grid["density"][compare],
        "density_rel_error": density_rel_error,
        "reference_temperature": reference_temperature,
        "grid_temperature": grid["temperature"][compare],
        "temperature_rel_error": temperature_rel_error,
    }


def write_metrics(metrics, output_path):
    with open(os.path.join(output_path, "wd-profile-errors.dat"), "w") as fp:
        fp.write(
            "# mass_rel_error radius_mean_rel_error "
            "radius_p95_rel_error radius_max_rel_error\n"
        )
        fp.write(
            f"{metrics['mass_rel_error']:.16e} "
            f"{metrics['radius_mean_rel_error']:.16e} "
            f"{metrics['radius_p95_rel_error']:.16e} "
            f"{metrics['radius_max_rel_error']:.16e}\n"
        )


def write_thermo_metrics(metrics, output_path):
    with open(os.path.join(output_path, "wd-thermo-errors.dat"), "w") as fp:
        fp.write(
            "# cell_count density_mean_rel_error density_p95_rel_error "
            "density_max_rel_error temperature_max_rel_error "
            "o16_max_abs_error ne20_max_abs_error ye_max_abs_error\n"
        )
        fp.write(
            f"{metrics['cell_count']} "
            f"{metrics['density_mean_rel_error']:.16e} "
            f"{metrics['density_p95_rel_error']:.16e} "
            f"{metrics['density_max_rel_error']:.16e} "
            f"{metrics['temperature_max_rel_error']:.16e} "
            f"{metrics['o16_max_abs_error']:.16e} "
            f"{metrics['ne20_max_abs_error']:.16e} "
            f"{metrics['ye_max_abs_error']:.16e}\n"
        )


def plot_profile(metrics, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(
        metrics["mass_fraction"], metrics["reference_radius"], label="LEAFS reference"
    )
    axes[0].plot(metrics["mass_fraction"], metrics["grid_radius"], "--", label="Apophis SMR")
    axes[0].set_ylabel("radius [cm]")
    axes[0].legend()

    axes[1].plot(metrics["mass_fraction"], metrics["rel_radius_error"])
    axes[1].set_xlabel("enclosed mass fraction")
    axes[1].set_ylabel("relative radius error")

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "wd-profile.png"), dpi=200)
    plt.close(fig)


def plot_thermo_profile(metrics, output_path):
    order = np.argsort(metrics["radius"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(
        metrics["radius"][order],
        metrics["reference_density"][order],
        label="LEAFS reference",
    )
    axes[0, 0].plot(
        metrics["radius"][order],
        metrics["grid_density"][order],
        ".",
        ms=1,
        label="Apophis SMR cells",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"density [g cm$^{-3}$]")
    axes[0, 0].legend()

    axes[0, 1].plot(
        metrics["radius"][order],
        metrics["reference_temperature"][order],
        label="LEAFS reference",
    )
    axes[0, 1].plot(
        metrics["radius"][order],
        metrics["grid_temperature"][order],
        ".",
        ms=1,
        label="Apophis SMR cells",
    )
    axes[0, 1].set_ylabel("temperature [K]")
    axes[0, 1].legend()

    axes[1, 0].plot(
        metrics["radius"][order], metrics["density_rel_error"][order], ".", ms=1
    )
    axes[1, 0].set_xlabel("radius [cm]")
    axes[1, 0].set_ylabel("relative density error")

    axes[1, 1].plot(
        metrics["radius"][order],
        metrics["temperature_rel_error"][order],
        ".",
        ms=1,
    )
    axes[1, 1].set_xlabel("radius [cm]")
    axes[1, 1].set_ylabel("relative temperature error")

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "wd-thermo.png"), dpi=200)
    plt.close(fig)


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        root = source_root(parameters)
        parameters.driver_cmd_line_args = [
            "problem/rhoc=9.90",
            "problem/temp=5.0e5",
            "problem/ye=0.49337656626870025",
            "problem/ofrac=0.65",
            "problem/dr=1.0e3",
            f"eos/helm_table={os.path.join(root, 'data', 'helm_table.dat')}",
            f"parthenon/output0/id={step}",
            "parthenon/output0/variables=cons,prim,gravity,eos_lambda",
            "parthenon/output0/dt=1.0",
            "parthenon/time/nlim=1",
            "parthenon/time/tlim=0.0",
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

        reference_path = os.path.join(parameters.test_path, "data", REFERENCE_FILE)
        thermo_reference_path = os.path.join(
            parameters.test_path, "data", THERMO_REFERENCE_FILE
        )
        data_filename = os.path.join(parameters.output_path, "parthenon.1.final.phdf")

        if not os.path.exists(reference_path):
            print(f"Missing reference file {reference_path}.")
            return False
        if not os.path.exists(thermo_reference_path):
            print(f"Missing thermodynamic reference file {thermo_reference_path}.")
            return False
        if not os.path.exists(data_filename):
            print(f"Missing output file {data_filename}.")
            return False

        reference = load_reference(reference_path)
        thermo_reference = load_thermo_reference(thermo_reference_path)
        data_file = phdf.phdf(data_filename)
        grid = grid_mass_radius(data_file)
        metrics = compare_profile(grid, reference)
        thermo_grid = grid_cell_thermo(data_file)
        thermo_metrics = compare_thermo_profile(thermo_grid, thermo_reference)

        write_metrics(metrics, parameters.output_path)
        write_thermo_metrics(thermo_metrics, parameters.output_path)
        plot_profile(metrics, parameters.output_path)
        plot_thermo_profile(thermo_metrics, parameters.output_path)

        test_success = True
        checks = [
            ("mass_rel_error", abs(metrics["mass_rel_error"]), MASS_REL_TOL),
            (
                "radius_mean_rel_error",
                metrics["radius_mean_rel_error"],
                RADIUS_MEAN_REL_TOL,
            ),
            ("radius_p95_rel_error", metrics["radius_p95_rel_error"], RADIUS_P95_REL_TOL),
            ("radius_max_rel_error", metrics["radius_max_rel_error"], RADIUS_MAX_REL_TOL),
            (
                "density_mean_rel_error",
                thermo_metrics["density_mean_rel_error"],
                THERMO_DENSITY_MEAN_REL_TOL,
            ),
            (
                "density_p95_rel_error",
                thermo_metrics["density_p95_rel_error"],
                THERMO_DENSITY_P95_REL_TOL,
            ),
            (
                "density_max_rel_error",
                thermo_metrics["density_max_rel_error"],
                THERMO_DENSITY_MAX_REL_TOL,
            ),
            (
                "temperature_max_rel_error",
                thermo_metrics["temperature_max_rel_error"],
                THERMO_TEMPERATURE_MAX_REL_TOL,
            ),
            (
                "o16_max_abs_error",
                thermo_metrics["o16_max_abs_error"],
                THERMO_COMPOSITION_MAX_ABS_TOL,
            ),
            (
                "ne20_max_abs_error",
                thermo_metrics["ne20_max_abs_error"],
                THERMO_COMPOSITION_MAX_ABS_TOL,
            ),
            (
                "ye_max_abs_error",
                thermo_metrics["ye_max_abs_error"],
                THERMO_YE_MAX_ABS_TOL,
            ),
        ]
        for name, value, limit in checks:
            if not np.isfinite(value) or value > limit:
                print(f"WD profile {name}={value:.6e} exceeds {limit:.6e}.")
                test_success = False

        return test_success
