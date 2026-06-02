// Accuracy Test for a 3D model
// Following Section 4.1 of Mandal, Mukherjee, & Mignone 2023

#include <cmath>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"

namespace grav_test {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto rho0 = pin->GetReal("problem", "rho0");
  auto r0 = pin->GetReal("problem", "r0");
  auto x0 = pin->GetOrAddReal("problem", "x0", 0.0);
  auto y0 = pin->GetOrAddReal("problem", "y0", 0.0);
  auto z0 = pin->GetOrAddReal("problem", "z0", 0.0);

  const Real r0sqr_inv = 1.0 / (r0 * r0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

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
}
} // namespace grav_test
