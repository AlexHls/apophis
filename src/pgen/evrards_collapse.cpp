// Evrard's collapse test

#include <cmath>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"

namespace ec {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto mass = pin->GetReal("problem", "mass");
  auto radius = pin->GetReal("problem", "radius");
  auto u_therm = pin->GetReal("problem", "u_therm");
  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator Evrard", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);
        Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        Real rho = 1e-5;

        if (r < radius) {
          rho = mass / (2.0 * M_PI * SQR(radius) * r);
        }

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = u_therm * rho;
      });
}
} // namespace ec