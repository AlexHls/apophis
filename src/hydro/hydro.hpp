#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeHydro(ParameterInput *pin);

template <Fluid fluid>
  Real EstimateTimestep(MeshData<Real> *md);

template <class T>
  void ConsToPrim(MeshData<Real> *md);

TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md);

template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md);
using FluxFun_t = decltype(CalculateFluxes<Fluid::euler, Reconstruction::plm,
                                           RiemannSolver::hllc>);
using FluxFunKey_t = std::tuple<Fluid, Reconstruction, RiemannSolver>;

// Add flux function pointer to map containing all compiled in flux functions
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
void add_flux_fun(std::map<FluxFunKey_t, FluxFun_t *> &flux_functions) {
  flux_functions[std::make_tuple(fluid, recon, rsolver)] =
      Apophis::CalculateFluxes<fluid, recon, rsolver>;
}

template <Fluid fluid>
  constexpr size_t GetNVars();

template <>
  constexpr size_t GetNVars<Fluid::euler>() {
    return 5;
  }

} // namespace Apophis

#endif // HYDRO_HYDRO_HPP_
