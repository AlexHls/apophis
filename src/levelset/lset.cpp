#include "lset.hpp"
#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::package::prelude;

namespace Apophis {

TaskStatus ReinitializeLset(std::shared_ptr<MeshData<Real>> &md, const int lset_id) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  int il, iu, jl, ju, kl, ku;
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

  auto &lset_pack =
      md->PackVariables(std::vector<std::string>{"lset" + std::to_string(lset_id)});

  const auto ndim_ = cons_pack.GetNdim();
  pmb->par_for(
      "ReinitializeLevelset", 0, cons_pack.GetDim(5) - 1, kl, ku, jl, ju, ib.s - 1,
      ib.e + 1, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &lset = lset_pack(b);
        const auto &coords = prim_pack.GetCoords(b);

        const auto &ndim = ndim_;

        int nxdim = cons.GetDim(X1DIR);
        int nydim = cons.GetDim(X2DIR);
        int nzdim = cons.GetDim(X3DIR);

        Real &dist = lset(LIDST, k, j, i);

        Real &u_d = cons(IDN, k, j, i);
        Real &u_lset = cons(lset_idx, k, j, i);

        Real di = 1.0 / u_d;

        Real p_lset = u_lset * di;

        // TODO(alexhls): Make sure this matches the values from the
        // LEAFS implementation. This should refer to the minimum cell size.
        Real ref_len =
            std::max(coords.Dxc<X1DIR>(k, j, i),
                     std::max(coords.Dxc<X2DIR>(k, j, i), coords.Dxc<X3DIR>(k, j, i)));
        Real deldist = 1.5 * ref_len;
        Real spread = 0.7 * ref_len;
        Real dt_p = 0.1 * ref_len;

        dist = (20 * 4 *
                std::max(ref_len, std::max(coords.Dxc<X1DIR>(k, j, i),
                                           std::max(coords.Dxc<X2DIR>(k, j, i),
                                                    coords.Dxc<X3DIR>(k, j, i)))));

        // x-dir
        Real cr;
        Real p_lset_p1 = cons(lset_idx, k, j, i + 1) / cons(IDN, k, j, i + 1);
        if (p_lset * p_lset_p1 < 0) {
          cr = coords.Xc<X1DIR>(k, j, i) +
               std::abs(p_lset) / (std::abs(p_lset_p1) - p_lset) *
                   (coords.Xc<X1DIR>(k, j, i + 1) - coords.Xc<X1DIR>(k, j, i));
          for (int ii = std::max(0, i - LSET_BORDER);
               ii < std::min(nxdim, i + LSET_BORDER); ii++) {
            for (int jj = std::max(0, j - LSET_BORDER);
                 jj < std::min(nydim, j + LSET_BORDER); jj++) {
              for (int kk = std::max(0, k - LSET_BORDER);
                   kk < std::min(nzdim, k + LSET_BORDER); kk++) {
                dist = std::min(dist, SQR(coords.Xc<X1DIR>(kk, jj, ii) - cr) +
                                          SQR(coords.Xc<X2DIR>(kk, jj, ii) - cr) +
                                          SQR(coords.Xc<X3DIR>(kk, jj, ii) - cr));
              }
            }
          }
        }

        // y-dir
        if (ndim > 1) {
          Real p_lset_p1 = cons(lset_idx, k, j + 1, i) / cons(IDN, k, j + 1, i);
          if (p_lset * p_lset_p1 < 0) {
            cr = coords.Xc<X2DIR>(k, j, i) +
                 std::abs(p_lset) / (std::abs(p_lset_p1) - p_lset) *
                     (coords.Xc<X2DIR>(k, j + 1, i) - coords.Xc<X2DIR>(k, j, i));
            for (int ii = std::max(0, i - LSET_BORDER);
                 ii < std::min(nxdim, i + LSET_BORDER); ii++) {
              for (int jj = std::max(0, j - LSET_BORDER);
                   jj < std::min(nydim, j + LSET_BORDER); jj++) {
                for (int kk = std::max(0, k - LSET_BORDER);
                     kk < std::min(nzdim, k + LSET_BORDER); kk++) {
                  dist = std::min(dist, SQR(coords.Xc<X1DIR>(kk, jj, ii) - cr) +
                                            SQR(coords.Xc<X2DIR>(kk, jj, ii) - cr) +
                                            SQR(coords.Xc<X3DIR>(kk, jj, ii) - cr));
                }
              }
            }
          }
        }

        // z-dir
        if (ndim > 2) {
          Real p_lset_p1 = cons(lset_idx, k + 1, j, i) / cons(IDN, k + 1, j, i);
          if (p_lset * p_lset_p1 < 0) {
            cr = coords.Xc<X3DIR>(k, j, i) +
                 std::abs(p_lset) / (std::abs(p_lset_p1) - p_lset) *
                     (coords.Xc<X3DIR>(k + 1, j, i) - coords.Xc<X3DIR>(k, j, i));
            for (int ii = std::max(0, i - LSET_BORDER);
                 ii < std::min(nxdim, i + LSET_BORDER); ii++) {
              for (int jj = std::max(0, j - LSET_BORDER);
                   jj < std::min(nydim, j + LSET_BORDER); jj++) {
                for (int kk = std::max(0, k - LSET_BORDER);
                     kk < std::min(nzdim, k + LSET_BORDER); kk++) {
                  dist = std::min(dist, SQR(coords.Xc<X1DIR>(kk, jj, ii) - cr) +
                                            SQR(coords.Xc<X2DIR>(kk, jj, ii) - cr) +
                                            SQR(coords.Xc<X3DIR>(kk, jj, ii) - cr));
                }
              }
            }
          }
        }

        dist = std::sqrt(dist);
      });
  return TaskStatus::complete;
}

} // namespace Apophis
