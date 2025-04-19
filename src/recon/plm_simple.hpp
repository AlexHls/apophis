#ifndef RECON_PLM_SIMPLE_HPP_
#define RECON_PLM_SIMPLE_HPP_

#include "../main.hpp"

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include <parthenon/parthenon.hpp>

using parthenon::Real;
using parthenon::ScratchPad2D;

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
         Real &qr_i) {
  // L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // Limiter
  Real dq2 = dql * dqr;
  Real dqm = 0.0;
  if (dq2 > 0.0) {
    dqm = dq2 / (dql + dqr);
  }

  // L/R states
  ql_ip1 = q_i + dqm;
  qr_i = q_i - dqm;
}

template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<recon == Reconstruction::plm, void>::type
    Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j,
                const int il, const int iu,
                const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
                ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1),
            qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i),
            qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i),
            qr(n, i));
      }
    });
  }
}

#endif // RECON_PLM_SIMPLE_HPP_
