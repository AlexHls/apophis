#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../constants.hpp"
#include "../main.hpp"

namespace grav_test {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto pid = pin->GetInteger("problem", "pid");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

  if (pid == 1) {

    // Accuracy Test for a 3D model
    // Following Section 4.1 of Mandal, Mukherjee, & Mignone 2023

    auto rho0 = pin->GetReal("problem", "rho0");
    auto r0 = pin->GetReal("problem", "r0");
    auto x0 = pin->GetOrAddReal("problem", "x0", 0.0);
    auto y0 = pin->GetOrAddReal("problem", "y0", 0.0);
    auto z0 = pin->GetOrAddReal("problem", "z0", 0.0);

    const Real r0sqr_inv = 1.0 / (r0 * r0);

    pmb->par_for(
        "ProblemGenerator GravityTest", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real x = coords.Xc<1>(i);
          Real y = coords.Xc<2>(j);
          Real z = coords.Xc<3>(k);
          Real r = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          Real rho = 0.0;

          if (r <= r0) {
            rho = rho0 * SQR(1.0 - r * r * r0sqr_inv);
          }

          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 1.0; // Set to some arbitrary background value
        });
  } else if (pid == 2) {
    // Binary potential test
    // Following Section 4.2 of Tomida & Stone 2023
    //
    // Two uniform-density spheres (default values):
    //   M1 = 2 at ( 6/1024, 0, 0 )
    //   M2 = 1 at ( -12/1024, 0, 0 )
    //   R = 6/1024
    auto m1 = pin->GetOrAddReal("problem", "m1", 2.0);
    auto m2 = pin->GetOrAddReal("problem", "m2", 1.0);

    auto rstar = pin->GetOrAddReal("problem", "rstar", 6.0 / 1024.0);

    auto x1 = pin->GetOrAddReal("problem", "x1", 6.0 / 1024.0);
    auto y1 = pin->GetOrAddReal("problem", "y1", 0.0);
    auto z1 = pin->GetOrAddReal("problem", "z1", 0.0);

    auto x2 = pin->GetOrAddReal("problem", "x2", -12.0 / 1024.0);
    auto y2 = pin->GetOrAddReal("problem", "y2", 0.0);
    auto z2 = pin->GetOrAddReal("problem", "z2", 0.0);

    auto nsub = pin->GetOrAddInteger("problem", "nsub", 10);

    const Real vol_sphere = 4.0 / 3.0 * M_PI * std::pow(rstar, 3);
    const Real rho1 = m1 / vol_sphere;
    const Real rho2 = m2 / vol_sphere;
    const Real density_floor = pin->GetOrAddReal("problem", "density_floor", 1.0e-300);

    IndexRange iib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange ijb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange ikb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for(
        "ProblemGenerator BinaryPotentialTest", ikb.s, ikb.e, ijb.s, ijb.e, iib.s, iib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);

          const Real r1 = std::sqrt(SQR(x - x1) + SQR(y - y1) + SQR(z - z1));
          const Real r2 = std::sqrt(SQR(x - x2) + SQR(y - y2) + SQR(z - z2));

          const Real dx = coords.Xf<1>(i + 1) - coords.Xf<1>(i);
          const Real dd = dx / static_cast<Real>(nsub);
          const Real dv = 1.0 / static_cast<Real>(nsub * nsub * nsub);
          const Real dr = 0.6 * std::sqrt(3.0) * dx;

          Real rho = density_floor;

          if (r1 < rstar + dr) {
            if (r1 < rstar - dr) {
              rho = rho1;
            } else {
              const Real xf = coords.Xf<1>(i);
              const Real yf = coords.Xf<2>(j);
              const Real zf = coords.Xf<3>(k);
              for (int kk = 0; kk < nsub; ++kk) {
                const Real zs = zf + (kk + 0.5) * dd;
                for (int jj = 0; jj < nsub; ++jj) {
                  const Real ys = yf + (jj + 0.5) * dd;
                  for (int ii = 0; ii < nsub; ++ii) {
                    const Real xs = xf + (ii + 0.5) * dd;
                    const Real rs = std::sqrt(SQR(xs - x1) + SQR(ys - y1) + SQR(zs - z1));
                    if (rs < rstar) {
                      rho += dv * rho1;
                    }
                  }
                }
              }
            }
          }

          if (r2 < rstar + dr) {
            if (r2 < rstar - dr) {
              rho = rho2;
            } else {
              const Real xf = coords.Xf<1>(i);
              const Real yf = coords.Xf<2>(j);
              const Real zf = coords.Xf<3>(k);
              for (int kk = 0; kk < nsub; ++kk) {
                const Real zs = zf + (kk + 0.5) * dd;
                for (int jj = 0; jj < nsub; ++jj) {
                  const Real ys = yf + (jj + 0.5) * dd;
                  for (int ii = 0; ii < nsub; ++ii) {
                    const Real xs = xf + (ii + 0.5) * dd;
                    const Real rs = std::sqrt(SQR(xs - x2) + SQR(ys - y2) + SQR(zs - z2));
                    if (rs < rstar) {
                      rho += dv * rho2;
                    }
                  }
                }
              }
            }
          }

          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = rho;
        });

  } else {
    PARTHENON_FAIL("Invalid problem id");
  };
}

void PostInitialization(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto pid = pin->GetInteger("problem", "pid");
  if (pid != 2 || md->partition != 0) return;

  const Real m1 = pin->GetOrAddReal("problem", "m1", 2.0);
  const Real m2 = pin->GetOrAddReal("problem", "m2", 1.0);
  const Real target_mass = m1 + m2;

  Real mass = 0.0;
  for (auto &block : pm->block_list) {
    auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);

    auto &u = block->meshblock_data.Get()->Get("cons").data;
    auto &coords = block->coords;

    Real block_mass = 0.0;
    block->par_reduce(
        "BinaryPotentialMass", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmass) {
          lmass += u(IDN, k, j, i) * coords.CellVolume(k, j, i);
        },
        Kokkos::Sum<Real>(block_mass));
    mass += block_mass;
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif

  if (mass <= 0.0) {
    PARTHENON_FAIL("[Apophis]: Binary potential initialized with non-positive mass.");
  }

  if ((mass < 0.7 * target_mass || mass > 1.3 * target_mass) &&
      parthenon::Globals::my_rank == 0) {
    PARTHENON_WARN("Too much or too little mass. Resolution is too low.");
  }

  const Real fac = target_mass / mass;
  for (auto &block : pm->block_list) {
    auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);

    auto &u = block->meshblock_data.Get()->Get("cons").data;

    block->par_for(
        "BinaryPotentialMassRescale", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) { u(IDN, k, j, i) *= fac; });
  }
}

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  static_cast<void>(tm);

  auto pid = pin->GetInteger("problem", "pid");
  if (pid != 2) return;

  const Real m1 = pin->GetOrAddReal("problem", "m1", 2.0);
  const Real m2 = pin->GetOrAddReal("problem", "m2", 1.0);
  const Real rstar = pin->GetOrAddReal("problem", "rstar", 6.0 / 1024.0);

  const Real x1 = pin->GetOrAddReal("problem", "x1", 6.0 / 1024.0);
  const Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  const Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);

  const Real x2 = pin->GetOrAddReal("problem", "x2", -12.0 / 1024.0);
  const Real y2 = pin->GetOrAddReal("problem", "y2", 0.0);
  const Real z2 = pin->GetOrAddReal("problem", "z2", 0.0);

  const Real gravity_g = pin->GetOrAddReal("gravity", "gravity_constant", GRAVITY_G);
  const Real pi = M_PI;
  const Real vol_sphere = 4.0 * pi / 3.0 * std::pow(rstar, 3);
  const Real den1 = m1 / vol_sphere;
  const Real den2 = m2 / vol_sphere;

  Real potential_l2 = 0.0;
  Real acceleration_l2 = 0.0;
  Real max_true_error = 0.0;
  Real max_residual = 0.0;

  for (auto &block : mesh->block_list) {
    auto &pmb_data = block->meshblock_data.Get();
    if (!pmb_data->HasVariable("gravity.phi") || !pmb_data->HasVariable("gravity.rhs")) {
      continue;
    }

    auto ib = block->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = block->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = block->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto phi = pmb_data->Get("gravity.phi").data.GetHostMirrorAndCopy();
    auto &rhs_dev = pmb_data->Get("gravity.rhs").data;
    auto rhs = rhs_dev.GetHostMirrorAndCopy();
    auto &coords = block->coords;

    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          const Real dx = coords.Dxc<1>(i);
          const Real dy = coords.Dxc<2>(j);
          const Real dz = coords.Dxc<3>(k);

          const Real r1 = std::sqrt(SQR(x - x1) + SQR(y - y1) + SQR(z - z1));
          const Real r2 = std::sqrt(SQR(x - x2) + SQR(y - y2) + SQR(z - z2));

          Real p1, p2, ax1, ay1, az1, ax2, ay2, az2;
          if (r1 > rstar) {
            p1 = -gravity_g * m1 / r1;
            ax1 = -gravity_g * m1 / (r1 * r1 * r1) * (x - x1);
            ay1 = -gravity_g * m1 / (r1 * r1 * r1) * (y - y1);
            az1 = -gravity_g * m1 / (r1 * r1 * r1) * (z - z1);
          } else {
            p1 = -gravity_g * pi * 2.0 / 3.0 * den1 * (3.0 * rstar * rstar - r1 * r1);
            ax1 = -gravity_g * pi * 4.0 / 3.0 * den1 * (x - x1);
            ay1 = -gravity_g * pi * 4.0 / 3.0 * den1 * (y - y1);
            az1 = -gravity_g * pi * 4.0 / 3.0 * den1 * (z - z1);
          }

          if (r2 > rstar) {
            p2 = -gravity_g * m2 / r2;
            ax2 = -gravity_g * m2 / (r2 * r2 * r2) * (x - x2);
            ay2 = -gravity_g * m2 / (r2 * r2 * r2) * (y - y2);
            az2 = -gravity_g * m2 / (r2 * r2 * r2) * (z - z2);
          } else {
            p2 = -gravity_g * pi * 2.0 / 3.0 * den2 * (3.0 * rstar * rstar - r2 * r2);
            ax2 = -gravity_g * pi * 4.0 / 3.0 * den2 * (x - x2);
            ay2 = -gravity_g * pi * 4.0 / 3.0 * den2 * (y - y2);
            az2 = -gravity_g * pi * 4.0 / 3.0 * den2 * (z - z2);
          }

          const Real pot0 = p1 + p2;
          const Real ax0 = ax1 + ax2;
          const Real ay0 = ay1 + ay2;
          const Real az0 = az1 + az2;

          const Real ax = -(phi(0, k, j, i + 1) - phi(0, k, j, i - 1)) / (2.0 * dx);
          const Real ay = -(phi(0, k, j + 1, i) - phi(0, k, j - 1, i)) / (2.0 * dy);
          const Real az = -(phi(0, k + 1, j, i) - phi(0, k - 1, j, i)) / (2.0 * dz);

          const Real perr = (pot0 - phi(0, k, j, i)) / pot0;
          const Real aerr_denom = SQR(ax0) + SQR(ay0) + SQR(az0);
          const Real aerr =
              (aerr_denom > 0.0)
                  ? std::sqrt((SQR(ax - ax0) + SQR(ay - ay0) + SQR(az - az0)) /
                              aerr_denom)
                  : 0.0;
          const Real vol = coords.CellVolume(k, j, i);

          rhs(0, k, j, i) = perr;
          potential_l2 += std::abs(perr) * vol;
          acceleration_l2 += aerr * vol;
          max_residual = std::max(max_residual, static_cast<Real>(std::abs(perr)));
          if (std::abs(z) < dz) {
            max_true_error = std::max(max_true_error, static_cast<Real>(std::abs(perr)));
          }
        }
      }
    }

    rhs_dev.DeepCopy(rhs);
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &potential_l2, 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &acceleration_l2, 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &max_true_error, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &max_residual, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif

  const auto mesh_size = mesh->mesh_size;
  const Real total_volume =
      (mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR)) *
      (mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR)) *
      (mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR));
  potential_l2 = std::sqrt(potential_l2 / total_volume);
  acceleration_l2 = std::sqrt(acceleration_l2 / total_volume);

  if (parthenon::Globals::my_rank == 0) {
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1);
    std::cout << "=====================================================" << std::endl;
    std::cout << "Potential    L2       : " << potential_l2 << std::endl;
    std::cout << "Acceleration L2       : " << acceleration_l2 << std::endl;
    std::cout << "Max True Error        : " << max_true_error << std::endl;
    std::cout << "Max Stored Residual   : " << max_residual << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
  }
}

} // namespace grav_test
