#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../constants.hpp"
#include "../hydro/hydro.hpp"
#include "../main.hpp"

namespace wd {
using namespace parthenon::driver::prelude;

namespace {

constexpr int WD_MAX_ZONES = 30000000;
constexpr Real PI = 3.141592653589793238462643383279502884;

struct WDProfile {
  std::vector<Real> radius;
  std::vector<Real> density;
  std::vector<Real> sie;
  std::vector<Real> mass;
  std::vector<Real> phi;
  Real total_mass = 0.0;
  Real surface_radius = 0.0;
};

struct WDProfileCache {
  bool initialized = false;
  WDProfile profile;
};

WDProfileCache &ProfileCache() {
  static WDProfileCache cache;
  return cache;
}

Real GetProblemReal(ParameterInput *pin, const std::string &primary,
                    const std::string &fallback, const Real default_value) {
  if (pin->DoesParameterExist("problem", primary)) {
    return pin->GetReal("problem", primary);
  }
  if (!fallback.empty() && pin->DoesParameterExist("problem", fallback)) {
    return pin->GetReal("problem", fallback);
  }
  return pin->GetOrAddReal("problem", primary, default_value);
}

Real MeanAbar(const Real ofrac) {
  return 1.0 / (ofrac / 16.0 + (1.0 - ofrac) / 20.0);
}

Real PressureFromDensityTemperature(const Apophis::EOS_t &eos, const Real rho,
                                    const Real temp, const Real abar,
                                    const Real zbar, const Real ltemp) {
  Real lambda[3] = {abar, zbar, ltemp};
  return eos.PressureFromDensityTemperature(rho, temp, lambda);
}

Real InternalEnergyFromDensityTemperature(const Apophis::EOS_t &eos, const Real rho,
                                          const Real temp, const Real abar,
                                          const Real zbar, const Real ltemp) {
  Real lambda[3] = {abar, zbar, ltemp};
  return eos.InternalEnergyFromDensityTemperature(rho, temp, lambda);
}

Real SolveDensityForPressure(const Apophis::EOS_t &eos, const Real pressure,
                             const Real rho_guess, const Real temp, const Real abar,
                             const Real zbar, const Real ltemp,
                             const Real rho_floor) {
  constexpr int max_iter = 100;
  constexpr Real tol = 1.0e-12;
  constexpr Real eps = 1.0e-4;

  const Real low_floor = std::max(rho_floor, static_cast<Real>(1.0e-99));
  const Real p_low =
      PressureFromDensityTemperature(eos, low_floor, temp, abar, zbar, ltemp);
  if (pressure <= p_low) {
    return low_floor;
  }

  Real low = low_floor;
  Real high = std::max(rho_guess, 2.0 * low_floor);
  Real p_high = PressureFromDensityTemperature(eos, high, temp, abar, zbar, ltemp);
  int expand_count = 0;
  while (p_high < pressure && expand_count < 100) {
    low = high;
    high *= 2.0;
    p_high = PressureFromDensityTemperature(eos, high, temp, abar, zbar, ltemp);
    expand_count++;
  }

  if (p_high < pressure || !std::isfinite(p_high)) {
    PARTHENON_FAIL("[Apophis]: Could not bracket density for WD pressure solve.");
  }

  Real rho = std::min(std::max(rho_guess, low), high);
  for (int n = 0; n < max_iter; n++) {
    const Real p = PressureFromDensityTemperature(eos, rho, temp, abar, zbar, ltemp);
    if (!std::isfinite(p)) {
      rho = 0.5 * (low + high);
      continue;
    }
    const Real resid = std::abs(p - pressure) / pressure;
    if (resid < tol) {
      return rho;
    }

    if (p > pressure) {
      high = rho;
    } else {
      low = rho;
    }

    const Real rho2 = std::max(low_floor, rho * (1.0 - eps));
    const Real p2 = PressureFromDensityTemperature(eos, rho2, temp, abar, zbar, ltemp);
    const Real dpdrho = (rho > rho2) ? (p - p2) / (rho - rho2) : 0.0;

    Real rho_next = 0.5 * (low + high);
    if (dpdrho > 0.0 && std::isfinite(dpdrho)) {
      const Real rho_newton = rho - (p - pressure) / dpdrho;
      if (rho_newton > low && rho_newton < high && std::isfinite(rho_newton)) {
        rho_next = rho_newton;
      }
    }
    rho = rho_next;
  }

  return rho;
}

WDProfile GenerateWDMassRadius(const Apophis::EOS_t &eos, const Real rhoc,
                               const Real tempc, const Real ye, const Real ofrac,
                               const Real gravity_g, const Real dr,
                               const Real rho_cutoff, const int max_zones) {
  if (rhoc <= 0.0) {
    PARTHENON_FAIL("[Apophis]: problem/rhoc must imply a positive central density.");
  }
  if (tempc <= 0.0) {
    PARTHENON_FAIL("[Apophis]: problem/temp must be positive.");
  }
  if (ofrac < 0.0 || ofrac > 1.0) {
    PARTHENON_FAIL("[Apophis]: problem/ofrac must be in [0, 1].");
  }
  if (ye <= 0.0) {
    PARTHENON_FAIL("[Apophis]: problem/ye must be positive.");
  }
  if (dr <= 0.0) {
    PARTHENON_FAIL("[Apophis]: problem/dr must be positive.");
  }
  if (rho_cutoff <= 0.0) {
    PARTHENON_FAIL("[Apophis]: problem/rho_cutoff must be positive.");
  }
  if (max_zones < 2) {
    PARTHENON_FAIL("[Apophis]: problem/max_zones must be at least 2.");
  }

  const Real abar = MeanAbar(ofrac);
  const Real zbar = abar * ye;
  const Real ltemp = std::log10(tempc);
  const Real rho_floor_solve = std::max(static_cast<Real>(1.0e-99), rho_cutoff * 1.0e-12);

  WDProfile profile;
  profile.radius.reserve(1024);
  profile.density.reserve(1024);
  profile.sie.reserve(1024);
  profile.mass.reserve(1024);

  profile.radius.push_back(0.0);
  profile.density.push_back(rhoc);
  profile.sie.push_back(
      InternalEnergyFromDensityTemperature(eos, rhoc, tempc, abar, zbar, ltemp));
  profile.mass.push_back(0.0);

  Real pressure =
      PressureFromDensityTemperature(eos, profile.density.back(), tempc, abar, zbar, ltemp);
  Real radius = 0.0;
  Real mass = 0.0;

  for (int iz = 1; iz < max_zones; iz++) {
    const Real rho_prev = profile.density.back();
    const Real dpdr = -gravity_g * mass * rho_prev / std::max(radius * radius, 0.1);
    const Real pressure_next = pressure + dr * dpdr;
    if (pressure_next <= 0.0 || !std::isfinite(pressure_next)) {
      break;
    }

    radius += dr;
    const Real rho_next = SolveDensityForPressure(eos, pressure_next, rho_prev, tempc,
                                                  abar, zbar, ltemp, rho_floor_solve);
    const Real shell_radius = radius - 0.5 * dr;
    mass += 0.5 * (rho_prev + rho_next) * 4.0 * PI * shell_radius * shell_radius * dr;

    profile.radius.push_back(radius);
    profile.density.push_back(rho_next);
    profile.sie.push_back(
        InternalEnergyFromDensityTemperature(eos, rho_next, tempc, abar, zbar, ltemp));
    profile.mass.push_back(mass);

    pressure = pressure_next;
    if (rho_next < rho_cutoff) {
      break;
    }
  }

  if (profile.radius.size() < 2) {
    PARTHENON_FAIL("[Apophis]: WD profile generation did not produce enough zones.");
  }
  if (profile.density.back() >= rho_cutoff &&
      static_cast<int>(profile.radius.size()) >= max_zones) {
    PARTHENON_FAIL("[Apophis]: WD profile hit problem/max_zones before reaching "
                   "problem/rho_cutoff.");
  }

  profile.total_mass = profile.mass.back();
  profile.surface_radius = profile.radius.back();
  profile.phi.resize(profile.radius.size(), 0.0);

  const int nprofile = static_cast<int>(profile.radius.size());
  profile.phi[nprofile - 1] =
      -gravity_g * profile.total_mass / std::max(profile.surface_radius, dr);
  for (int iz = nprofile - 2; iz >= 0; iz--) {
    const Real r_inner = profile.radius[iz];
    const Real r_outer = profile.radius[iz + 1];
    const Real m_mid = 0.5 * (profile.mass[iz] + profile.mass[iz + 1]);
    Real dphi = 0.0;
    if (r_inner > 0.0) {
      dphi = gravity_g * m_mid * (1.0 / r_inner - 1.0 / r_outer);
    } else {
      const Real r_mid = 0.5 * (r_inner + r_outer);
      dphi = gravity_g * m_mid * dr / std::max(r_mid * r_mid, 0.1);
    }
    profile.phi[iz] = profile.phi[iz + 1] - dphi;
  }

  return profile;
}

Kokkos::View<Real *> CopyToDevice(const std::string &label,
                                  const std::vector<Real> &values) {
  Kokkos::View<Real *> view(label, values.size());
  auto host = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < values.size(); i++) {
    host(i) = values[i];
  }
  Kokkos::deep_copy(view, host);
  return view;
}

template <typename RadiusView, typename ValueView>
KOKKOS_INLINE_FUNCTION Real InterpolateProfile(const RadiusView &radius,
                                               const ValueView &value,
                                               const int nprofile, const Real r) {
  if (r <= radius(0)) {
    return value(0);
  }
  if (r >= radius(nprofile - 1)) {
    return value(nprofile - 1);
  }

  int lo = 0;
  int hi = nprofile - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (radius(mid) > r) {
      hi = mid;
    } else {
      lo = mid;
    }
  }

  const Real denom = radius(hi) - radius(lo);
  const Real frac = (denom > 0.0) ? (r - radius(lo)) / denom : 0.0;
  return (1.0 - frac) * value(lo) + frac * value(hi);
}

} // namespace

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const Real log_rhoc = GetProblemReal(pin, "rhoc", "log_rho_c", 9.0);
  const Real rhoc = pin->DoesParameterExist("problem", "rho_c")
                        ? pin->GetReal("problem", "rho_c")
                        : std::pow(10.0, log_rhoc);
  const Real tempc = GetProblemReal(pin, "temp", "temp_c", 5.0e5);
  const Real ye = pin->GetOrAddReal("problem", "ye", 0.5);
  const Real ofrac = GetProblemReal(pin, "ofrac", "o_frac", 0.65);
  const Real dr = pin->GetOrAddReal("problem", "dr", 1.0e3);
  const Real rho_cutoff = pin->GetOrAddReal("problem", "rho_cutoff", 1.0e-1);
  const int max_zones = pin->GetOrAddInteger("problem", "max_zones", WD_MAX_ZONES);

  const Real x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  const Real y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  const Real z0 = pin->GetOrAddReal("problem", "z0", 0.0);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &eos = hydro_pkg->Param<Apophis::EOS_t>("eos_host");
  const int ncomp = hydro_pkg->Param<int>("ncomp");
  const int nscalars = hydro_pkg->Param<int>("nscalars");
  const int nlset = hydro_pkg->Param<int>("nlset");
  const Real density_floor = hydro_pkg->Param<Real>("hydro/density_floor");
  const Real gravity_g = pin->GetOrAddReal("gravity", "gravity_constant", GRAVITY_G);

  if (ncomp != 0 && ncomp != 6) {
    PARTHENON_FAIL("[Apophis]: WD problem currently supports the ONe composition layout "
                   "or no composition.");
  }

  const Real atmosphere_density =
      pin->GetOrAddReal("problem", "rho_ambient", std::max(density_floor, 1.0e-5));
  const Real atmosphere_temp = pin->GetOrAddReal("problem", "temp_ambient", tempc);
  const Real ignition_radius = pin->GetOrAddReal("problem", "ignition_radius", -1.0);
  const Real ignition_x = pin->GetOrAddReal("problem", "ignition_x", x0);
  const Real ignition_y = pin->GetOrAddReal("problem", "ignition_y", y0);
  const Real ignition_z = pin->GetOrAddReal("problem", "ignition_z", z0);

  const Real abar = MeanAbar(ofrac);
  const Real zbar = abar * ye;
  const Real ltemp_c = std::log10(tempc);
  const Real ltemp_atm = std::log10(atmosphere_temp);
  const Real atmosphere_sie = InternalEnergyFromDensityTemperature(
      eos, atmosphere_density, atmosphere_temp, abar, zbar, ltemp_atm);

  WDProfileCache &cache = ProfileCache();
  if (!cache.initialized) {
    cache.profile =
        GenerateWDMassRadius(eos, rhoc, tempc, ye, ofrac, gravity_g, dr, rho_cutoff,
                             max_zones);

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "[Apophis]: WD profile generated with "
                << cache.profile.radius.size()
                << " radial zones, R = " << cache.profile.surface_radius << " cm, M = "
                << cache.profile.total_mass / 1.989e33 << " Msun." << std::endl;
    }

    cache.initialized = true;
  }

  const WDProfile &profile = cache.profile;
  const int nprofile = static_cast<int>(profile.radius.size());
  const Real surface_radius = profile.surface_radius;
  const Real total_mass = profile.total_mass;

  auto radius_view = CopyToDevice("wd_radius", profile.radius);
  auto density_view = CopyToDevice("wd_density", profile.density);
  auto sie_view = CopyToDevice("wd_sie", profile.sie);
  auto mass_view = CopyToDevice("wd_mass", profile.mass);
  auto phi_view = CopyToDevice("wd_phi", profile.phi);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto pmb_data = pmb->meshblock_data.Get();
  auto &u = pmb_data->Get("cons").data;
  auto &eos_lambda = pmb_data->Get("eos_lambda").data;
  auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "ProblemGenerator WD", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i) - x0;
        const Real y = (ndim > 1) ? coords.Xc<2>(j) - y0 : 0.0;
        const Real z = (ndim > 2) ? coords.Xc<3>(k) - z0 : 0.0;
        const Real r = std::sqrt(x * x + y * y + z * z);
        const bool inside_star = (r <= surface_radius);

        const Real rho = inside_star
                             ? InterpolateProfile(radius_view, density_view, nprofile, r)
                             : atmosphere_density;
        const Real sie =
            inside_star ? InterpolateProfile(radius_view, sie_view, nprofile, r)
                        : atmosphere_sie;
        const Real ltemp = inside_star ? ltemp_c : ltemp_atm;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = rho * sie;

        for (int n = NHYDRO; n < NHYDRO + nscalars; n++) {
          u(n, k, j, i) = 0.0;
        }

        if (ncomp == 6) {
          u(NHYDRO + 2, k, j, i) = ofrac * rho;
          u(NHYDRO + 3, k, j, i) = (1.0 - ofrac) * rho;
          u(NHYDRO + 6, k, j, i) = ye * rho;
        }

        const int first_lset = NHYDRO + ncomp + ((ncomp > 0) ? 1 : 0);
        for (int lset_id = 0; lset_id < nlset; lset_id++) {
          const int lset_idx = first_lset + 2 * lset_id;
          const int xfuel_idx = lset_idx + 1;
          const Real dx_ign = coords.Xc<1>(i) - ignition_x;
          const Real dy_ign = (ndim > 1) ? coords.Xc<2>(j) - ignition_y : 0.0;
          const Real dz_ign = (ndim > 2) ? coords.Xc<3>(k) - ignition_z : 0.0;
          const Real r_ign =
              std::sqrt(dx_ign * dx_ign + dy_ign * dy_ign + dz_ign * dz_ign);
          const Real lset = (ignition_radius > 0.0) ? ignition_radius - r_ign
                                                    : -surface_radius;
          const Real xfuel = (inside_star && lset <= 0.0) ? 1.0 : 0.0;
          u(lset_idx, k, j, i) = rho * lset;
          u(xfuel_idx, k, j, i) = rho * xfuel;
        }

        eos_lambda(0, k, j, i) = abar;
        eos_lambda(1, k, j, i) = zbar;
        eos_lambda(2, k, j, i) = ltemp;
      });

  if (pmb_data->HasVariable("gravity")) {
    auto &gravity = pmb_data->Get("gravity").data;
    pmb->par_for(
        "ProblemGenerator WD gravity", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x = coords.Xc<1>(i) - x0;
          const Real y = (ndim > 1) ? coords.Xc<2>(j) - y0 : 0.0;
          const Real z = (ndim > 2) ? coords.Xc<3>(k) - z0 : 0.0;
          const Real r2 = x * x + y * y + z * z;
          const Real r = std::sqrt(r2);
          if (r <= 0.0) {
            gravity(0, k, j, i) = 0.0;
            gravity(1, k, j, i) = 0.0;
            gravity(2, k, j, i) = 0.0;
            return;
          }

          const Real m_enclosed =
              (r <= surface_radius) ? InterpolateProfile(radius_view, mass_view, nprofile, r)
                                    : total_mass;
          const Real g_r = -gravity_g * m_enclosed / r2;
          const Real rinv = 1.0 / r;
          gravity(0, k, j, i) = g_r * x * rinv;
          gravity(1, k, j, i) = (ndim > 1) ? g_r * y * rinv : 0.0;
          gravity(2, k, j, i) = (ndim > 2) ? g_r * z * rinv : 0.0;
        });
  }

  if (pmb_data->HasVariable("gravity.phi")) {
    auto &phi = pmb_data->Get("gravity.phi").data;
    pmb->par_for(
        "ProblemGenerator WD phi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x = coords.Xc<1>(i) - x0;
          const Real y = (ndim > 1) ? coords.Xc<2>(j) - y0 : 0.0;
          const Real z = (ndim > 2) ? coords.Xc<3>(k) - z0 : 0.0;
          const Real r = std::sqrt(x * x + y * y + z * z);
          phi(0, k, j, i) =
              (r <= surface_radius)
                  ? InterpolateProfile(radius_view, phi_view, nprofile, r)
                  : -gravity_g * total_mass / std::max(r, surface_radius);
        });
  }
}

} // namespace wd
