#include <cmath>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

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
    const Real rstar_sqr = rstar * rstar;

    const Real nsub_tot = static_cast<Real>(nsub * nsub * nsub);

    pmb->par_for(
        "ProblemGenerator BinaryPotentialTest", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real rho_sum = 0.0;

          const Real xl = coords.Xf<1>(i);
          const Real xr = coords.Xf<1>(i + 1);
          const Real yl = coords.Xf<2>(j);
          const Real yr = coords.Xf<2>(j + 1);
          const Real zl = coords.Xf<3>(k);
          const Real zr = coords.Xf<3>(k + 1);

          const Real nsub_inv = 1.0 / static_cast<Real>(nsub);

          const Real dx = (xr - xl) * nsub_inv;
          const Real dy = (yr - yl) * nsub_inv;
          const Real dz = (zr - zl) * nsub_inv;

          for (int kk = 0; kk < nsub; kk++) {
            const Real zs = zl + (kk + 0.5) * dz;
            for (int jj = 0; jj < nsub; jj++) {
              const Real ys = yl + (jj + 0.5) * dy;
              for (int ii = 0; ii < nsub; ii++) {
                const Real xs = xl + (ii + 0.5) * dx;

                const Real r1sqr = SQR(xs - x1) + SQR(ys - y1) + SQR(zs - z1);
                const Real r2sqr = SQR(xs - x2) + SQR(ys - y2) + SQR(zs - z2);

                if (r1sqr <= rstar_sqr) {
                  rho_sum += rho1;
                }

                if (r2sqr <= rstar_sqr) {
                  rho_sum += rho2;
                }
              }
            }
          }

          Real rho = rho_sum / nsub_tot;

          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 1.0; // Set to some arbitrary background value
        });

  } else {
    PARTHENON_FAIL("Invalid problem id");
  };
}
} // namespace grav_test
