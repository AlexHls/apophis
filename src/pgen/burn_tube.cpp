#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "parthenon/prelude.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"
#include "singularity-eos/eos/eos.hpp"

using namespace parthenon::package::prelude;

namespace burn_tube {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rho_u = pin->GetOrAddReal("problem", "rho_u", 5.0e8);
  Real temp_u = pin->GetOrAddReal("problem", "temp_u", 1.0e9);
  Real temp_b = pin->GetOrAddReal("problem", "temp_b", 9.0e9);
  Real x1 = pin->GetOrAddReal("problem", "x1", 0.0);
  Real x2 = pin->GetOrAddReal("problem", "x2", 0.0);
  Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  Real y2 = pin->GetOrAddReal("problem", "y2", 0.0);
  Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  Real z2 = pin->GetOrAddReal("problem", "z2", 0.0);
  Real radius = pin->GetOrAddReal("problem", "radius", 1e7);
  Real gm1 = pin->GetOrAddReal("eos", "gm1", 0.6666667);
  Real cv = pin->GetOrAddReal("eos", "cv", 1.5);
  singularity::EOS eos_host = singularity::IdealGas(gm1, cv);
  singularity::EOS eos = eos_host.GetOnDevice();
  Real ofrac = pin->GetOrAddReal("problem", "ofrac", 0.1);
  Real ye = pin->GetOrAddReal("problem", "ye", 0.5);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  // setup uniform ambient medium with spherical over-pressured region
  pmb->par_for(
      "ProblemGenerator Blast", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);

        Real rad1 = std::sqrt(SQR(x - x1) + SQR(y - y1) + SQR(z - z1));
        Real rad2 = std::sqrt(SQR(x - x2) + SQR(y - y2) + SQR(z - z2));

        u(IDN, k, j, i) = rho_u;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) =
            eos.InternalEnergyFromDensityTemperature(rho_u, temp_u) * rho_u;

        for (int n = NHYDRO; n < NHYDRO + 7; n++) {
          u(n, k, j, i) = 0.0;
        }
        u(NHYDRO + 2, k, j, i) = ofrac * rho_u;         // O16
        u(NHYDRO + 3, k, j, i) = (1.0 - ofrac) * rho_u; // Ne20
        u(NHYDRO + 6, k, j, i) = ye * rho_u;

        // TODO(alexhls): Make this more flexible for different
        // ignition bubble setups
        u(NHYDRO + 7, k, j, i) = -1.0e-12; // lset1
        u(NHYDRO + 7, k, j, i) = std::max(
            u(NHYDRO + 7, k, j, i),
            radius - std::sqrt(SQR(x - x1) + SQR(y - y1) + SQR(z - z1)));
        u(NHYDRO + 7, k, j, i) = std::max(
            u(NHYDRO + 7, k, j, i),
            radius - std::sqrt(SQR(x - x2) + SQR(y - y2) + SQR(z - z2)));
        u(NHYDRO + 7, k, j, i) = u(NHYDRO + 7, k, j, i) * rho_u;

        if (rad1 < radius || rad2 < radius) {
          u(IEN, k, j, i) =
              eos.InternalEnergyFromDensityTemperature(rho_u, temp_b) * rho_u;
          u(NHYDRO + 2, k, j, i) = 0.0;
          u(NHYDRO + 3, k, j, i) = 0.0;
          u(NHYDRO + 5, k, j, i) = 1.0 * rho_u;
          ;
        }
      });
}

} // namespace burn_tube
