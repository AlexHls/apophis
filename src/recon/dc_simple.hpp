#ifndef RECON_DC_SIMPLE_HPP_
#define RECON_DC_SIMPLE_HPP_

#include "../main.hpp"

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include <parthenon/parthenon.hpp>

using parthenon::Real;
using parthenon::ScratchPad2D;

template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::dc, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        ql(n, i + 1) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      }
    });
  }
}

#endif // RECON_DC_SIMPLE_HPP_
