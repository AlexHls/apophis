#ifndef RSOLVERS_HYDRO_NONE_HPP_
#define RSOLVERS_HYDRO_NONE_HPP_

#include "../../main.hpp"
#include "rsolvers.hpp"

#include "parthenon/parthenon.hpp"

using parthenon::ParArray4D;
using parthenon::Real;
using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//  \brief None hydro solver. Dummy function to effectively disable the hydro scheme.
template <>
struct Riemann<Fluid::euler, RiemannSolver::none> {
  KOKKOS_FORCEINLINE_FUNCTION void static Solve(
      parthenon::team_mbr_t const &member, const int k, const int j, const int il,
      const int iu, const int ivx, const ScratchPad2D<Real> &wl,
      const ScratchPad2D<Real> &wr, parthenon::VariableFluxPack<Real> &cons,
      const ScratchPad2D<Real> &ifl, const ScratchPad2D<Real> &ifr, const EOS_t &eos,
      parthenon::VariablePack<Real> &eos_lambda) {
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      cons.flux(ivx, IDN, k, j, i) = 0.0;
      cons.flux(ivx, ivx, k, j, i) = 0.0;
      cons.flux(ivx, ivy, k, j, i) = 0.0;
      cons.flux(ivx, ivz, k, j, i) = 0.0;
      cons.flux(ivx, IEN, k, j, i) = 0.0;
    });
  }
};

#endif // RSOLVERS_HYDRO_NONE_HPP_
