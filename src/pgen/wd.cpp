#include <cmath>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../constants.hpp"
#include "../main.hpp"

#define _WD_NZN_ 30000000

namespace wd {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto rhoc = pin->GetReal("problem", "rhoc"); // Log central density
  auto ye = pin->GetReal("problem", "ye");
  auto ofrac = pin->GetReal("problem", "ofrac");
  auto tempc = pin->GetReal("problem", "temp");

  rhoc = std::pow(10.0, rhoc);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

  auto pmb_data = pmb->meshblock_data.Get();
  auto &phi = pmb_data->Get("gravity.phi").data;
  auto gravity_g = pin->GetOrAddReal("gravity", "gravity_constant", GRAVITY_G);

  pmb->par_for(
      "ProblemGenerator WD", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);
        Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        Real rho = 1e-5;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = rho;
      });
}

std::array<Real, _WD_NZN_> GenerateWDMassRadius(const Real rhoc, const Real tempc,
                                                const Real ye, const Real ofrac,
                                                const Real gravity_g) {
  // Hard coded abar/ zbar values for a ONe WD for now
  const Real abar = 16.0 * ofrac + 20.0 * (1.0 - ofrac);
  const Real zbar = ye * abar;

  Real lambda[3] = {abar, zbar, std::log10(tempc)};

  std::array<Real, _WD_NZN_> rho;
  std::array<Real, _WD_NZN_> temp;
  std::array<Real, _WD_NZN_> u;
  std::array<Real, _WD_NZN_> pres;

  Real rad = 0.0;
  constexpr Real dr = 1e3;

  for (int i = 0; i < _WD_NZN_; i++) {
    const Real dpdr = -gravity_g * rho[i - 1] / std::max(SQR(rad), 0.1);
    rad += dr;
  }
};
} // namespace wd
