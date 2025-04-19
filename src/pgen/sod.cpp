#include "../main.hpp"

#include "Kokkos_Macros.hpp"
#include "parthenon/driver.hpp"
#include "parthenon/prelude.hpp"

using namespace parthenon::driver::prelude;

namespace sod {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rho_l = pin->GetOrAddReal("problem/sod", "rho_l", 1.0);
  Real press_l = pin->GetOrAddReal("problem/sod", "pres_l", 1.0);
  Real u_l = pin->GetOrAddReal("problem/sod", "u_l", 0.0);
  Real rho_r = pin->GetOrAddReal("problem/sod", "rho_r", 0.125);
  Real press_r = pin->GetOrAddReal("problem/sod", "pres_r", 0.1);
  Real u_r = pin->GetOrAddReal("problem/sod", "u_r", 0.0);
  Real x_discont = pin->GetOrAddReal("problem/sod", "x_discont", 0.5);

  Real gamma = pin->GetOrAddReal("hydro", "gamma", 1.4);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (coords.Xc<1>(k, j, i) < x_discont) {
          u(IDN, k, j, i) = rho_l;
          u(IM1, k, j, i) = rho_l * u_l;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 0.5 * rho_l * u_l * u_l + press_l / (gamma - 1.0);
        } else {
          u(IDN, k, j, i) = rho_r;
          u(IM1, k, j, i) = rho_r * u_r;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 0.5 * rho_r * u_r * u_r + press_r / (gamma - 1.0);
        }
      });
}

} // namespace sod
