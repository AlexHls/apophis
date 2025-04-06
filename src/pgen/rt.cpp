//! Sets up several different problems:
//!   - iprob=1: 2D Single mode perturbation
//!   - iprob=2: 2D Multimode perturbation
//! See https://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
#include <cmath>
#include <cstring>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>

#include "../main.hpp"

namespace rt {
using namespace parthenon::driver::prelude;


void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto iprob = pin->GetInteger("problem/rt", "iprob");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u = rc->Get("cons").data;
  auto &coords = pmb->coords;

  if (iprob == 1) {
    // Read problem parameters
    Real rho_lower = pin->GetReal("problem/rt", "rho_lower");
    Real rho_upper = pin->GetReal("problem/rt", "rho_upper");
    Real en_ini = pin->GetReal("problem/rt", "en_ini");
    Real perturb_strength = pin->GetReal("problem/rt", "perturb_strength");
    pmb->par_for(
        "ProblemGenerator", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          u(IM1, k, j, i) = perturb_strength *
                            (1.0 + std::cos(4.0 * M_PI * coords.Xc<1>(i))) *
                            (1.0 + std::cos(3.0 * M_PI * coords.Xc<2>(j))) /
                            4.0;
          if (coords.Xc<2>(j) < 0.0) {
            u(IDN, k, j, i) = rho_lower;
            u(IM1, k, j, i) = rho_lower * u(IM1, k, j, i);
            u(IEN, k, j, i) = en_ini * rho_lower;
          } else {
            u(IDN, k, j, i) = rho_upper;
            u(IM1, k, j, i) = rho_upper * u(IM1, k, j, i);
            u(IEN, k, j, i) = en_ini * rho_upper;
          }
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
        });
  }

  if (iprob == 2) {
    // TODO: implement this
  }
}

} // namespace rt
