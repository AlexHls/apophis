#include "burner.hpp"
#include "../main.hpp"

#include "config.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon/parthenon.hpp"
#include <cmath>

using namespace parthenon::package::prelude;

namespace Apophis {

TaskStatus Burn(std::shared_ptr<MeshData<Real>> &md, const int lset_id,
                const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const auto ncomp = hydro_pkg->Param<int>("ncomp");
  const int lset_idx = NHYDRO + ncomp + 1 + lset_id; // 1 for ye

  auto &lset_pack = md->PackVariables(
      std::vector<std::string>{"lset" + std::to_string(lset_id)});

  const auto ndim_ = cons_pack.GetNdim();
  pmb->par_for(
      "Levelset gradient", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = prim_pack.GetCoords(b);

        const auto &ndim = ndim_;

        Real h = 0.0;
        Real lgrad = 0.0;
        Real rgrad = 0.0;
        Real gradupw = 0.0;

        // x-dir
        Real &u_d = cons(IDN, k, j, i);
        Real &u_dp1 = cons(IDN, k, j, i + 1);
        Real &u_dm1 = cons(IDN, k, j, i - 1);

        Real &u_lset = cons(lset_idx, k, j, i);
        Real p_lset = u_lset / u_d;
        Real p_lset_p1 = cons(lset_idx, k, j, i + 1) / u_dp1;
        Real p_lset_m1 = cons(lset_idx, k, j, i - 1) / u_dm1;

        lgrad = (p_lset - p_lset_m1) /
                (coords.Xc<X1DIR>(k, j, i) - coords.Xc<X1DIR>(k, j, i - 1));
        rgrad = (p_lset_p1 - p_lset) /
                (coords.Xc<X1DIR>(k, j, i + 1) - coords.Xc<X1DIR>(k, j, i));
        gradupw = std::min(lgrad, 0.0) + std::max(rgrad, 0.0);
        if ((lgrad < 0.0) && (rgrad > 0.0)) {
          gradupw = 0.5 * (std::abs(lgrad) + std::abs(rgrad));
        }
        h += gradupw * gradupw;

        // y-dir
        if (ndim > 1) {
          Real &u_dp1 = cons(IDN, k, j + 1, i);
          Real &u_dm1 = cons(IDN, k, j - 1, i);

          p_lset_p1 = cons(lset_idx, k, j + 1, i) / u_dp1;
          p_lset_m1 = cons(lset_idx, k, j - 1, i) / u_dm1;

          lgrad = (p_lset - p_lset_m1) /
                  (coords.Xc<X2DIR>(k, j, i) - coords.Xc<X2DIR>(k, j - 1, i));
          rgrad = (p_lset_p1 - p_lset) /
                  (coords.Xc<X2DIR>(k, j + 1, i) - coords.Xc<X2DIR>(k, j, i));
          gradupw = std::min(lgrad, 0.0) + std::max(rgrad, 0.0);
          if ((lgrad < 0.0) && (rgrad > 0.0)) {
            gradupw = 0.5 * (std::abs(lgrad) + std::abs(rgrad));
          }
          h += gradupw * gradupw;
        }

        // z-dir
        if (ndim > 2) {
          Real &u_dp1 = cons(IDN, k + 1, j, i);
          Real &u_dm1 = cons(IDN, k - 1, j, i);

          p_lset_p1 = cons(lset_idx, k + 1, j, i) / u_dp1;
          p_lset_m1 = cons(lset_idx, k - 1, j, i) / u_dm1;

          lgrad = (p_lset - p_lset_m1) /
                  (coords.Xc<X3DIR>(k, j, i) - coords.Xc<X3DIR>(k - 1, j, i));
          rgrad = (p_lset_p1 - p_lset) /
                  (coords.Xc<X3DIR>(k + 1, j, i) - coords.Xc<X3DIR>(k, j, i));
          gradupw = std::min(lgrad, 0.0) + std::max(rgrad, 0.0);
          if ((lgrad < 0.0) && (rgrad > 0.0)) {
            gradupw = 0.5 * (std::abs(lgrad) + std::abs(rgrad));
          }
          h += gradupw * gradupw;
        }

        // Advect the levelset with the flame speed
        Real &u_ye = cons(NHYDRO + ncomp, k, j, i);
        Real ye = u_ye / u_d;
        Real vburn =
            16.0e5 * std::pow((u_d / 1.0e9), 0.813) * (1.0 + 96.8 * (0.5 - ye));
        p_lset = p_lset + 0.5 * std::sqrt(h) * vburn * dt;
        u_lset = p_lset * u_d;
      });
  return TaskStatus::complete;
}
} // namespace Apophis
