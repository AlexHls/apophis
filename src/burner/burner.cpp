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

  int jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  if (pmb->block_size.nx(X2DIR) > 1) {
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    else // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
  }

  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const auto ncomp = hydro_pkg->Param<int>("ncomp");
  const int lset_idx = NHYDRO + ncomp + 1 + lset_id; // 1 for ye

  auto &lset_pack = md->PackVariables(
      std::vector<std::string>{"lset" + std::to_string(lset_id)});

  const auto ndim_ = cons_pack.GetNdim();
  pmb->par_for(
      "Levelset advection", 0, cons_pack.GetDim(5) - 1, kl, ku, jl, ju,
      ib.s - 1, ib.e + 1,
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

  pmb->par_for(
      "Burn", 0, cons_pack.GetDim(5) - 1, kl, ku, jl, ju, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);

        Real &u_d = cons(IDN, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        Real di = 1.0 / u_d;

        // TODO(alexhls): Make this more flexible for other fuel types
        Real &u_xo16 = cons(NHYDRO + 2, k, j, i);
        Real &u_xneon = cons(NHYDRO + 3, k, j, i);
        Real &u_ime = cons(NHYDRO + 4, k, j, i);
        Real &u_xni56 = cons(NHYDRO + 5, k, j, i);

        Real xo15_ini = u_xo16 * di;
        Real neon_ini = u_xneon * di;

        Real ni56_ini = u_xni56 * di;

        Real &u_xfuel = cons(lset_idx + 1, k, j, i);
        Real xfuel = u_xfuel * di;
        bool conv_flag = false;

        Real oldenergy = 0.0;
        for (int n = NHYDRO; n < NHYDRO + ncomp; n++) {
          Real ebind = GetEBind<Fluid::euler>(n - NHYDRO);
          oldenergy += cons(n, k, j, i) * ebind;
        }

        // TODO(alexhls): Get the actual alpha value
        Real alpha = 2.0;
        if (cons(lset_idx, k, j, i) > 0.0) {
          alpha = 0.0;
        }

        // TODO(alexhls): Replace this with a burndens check as is
        // done in LEAFS. Also get threshold from table.
        if (u_d > DENS_THRESH) {
          Real conversion = std::max(0.0, xfuel - alpha);
          xfuel = std::min(1.0, std::max(0.0, xfuel - conversion));
          if (conversion > 0.0) {
            conv_flag = true;
            //  TODO: Make this more accurate to account for partial burning
            //  of a cell
            u_xo16 = 0.0;
            u_xneon = 0.0;
            u_xni56 = 1.0 * u_d;
          }
          u_xfuel = xfuel * u_d;
        }

        // Release energy
        if (conv_flag) {
          Real xsum = 0.0;
          Real newenergy = 0.0;
          for (int n = NHYDRO; n < NHYDRO + ncomp; n++) {
            xsum += cons(n, k, j, i) / u_d;
          }
          // Note: This doesn't do anything yet since we burn everything
          // to Ni56.
          u_ime = std::max(0.0, 1.0 - xsum) * u_d;
          for (int n = NHYDRO; n < NHYDRO + ncomp; n++) {
            Real ebind = GetEBind<Fluid::euler>(n - NHYDRO);
            newenergy += cons(n, k, j, i) * ebind;
          }
          u_e += (newenergy - oldenergy);
        }
      });

  return TaskStatus::complete;
}
} // namespace Apophis
