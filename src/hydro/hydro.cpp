#include "hydro.hpp"
#include "../main.hpp"
#include "../recon/dc_simple.hpp"
#include "../recon/plm_simple.hpp"

#include "interface/state_descriptor.hpp"
#include "parthenon/parthenon.hpp"
#include "singularity-eos/eos/eos.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeHydro(ParameterInput *pin) {
  auto pkg = std::make_shared<parthenon::StateDescriptor>("Hydro");
  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);

  pkg->AddParam<>("cfl", cfl);

  // Fluid
  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "euler");
  auto fluid = Fluid::undefined;
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = GetNVars<Fluid::euler>();
  } else {
    PARTHENON_FAIL("[Apophis]: Fluid not recognized. Exiting.");
  }

  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);

  // Reconstruction
  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_ghost = 3;
  auto recon = Reconstruction::undefined;

  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    recon_need_ghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    recon_need_ghost = 2;
  } else {
    PARTHENON_FAIL("[Apophis]: Reconstruction not recognized. Exiting.");
  }

  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < recon_need_ghost) {
    PARTHENON_FAIL(
        "[Apophis]: Not enough ghost zones for reconstruction. Exiting.");
  }

  pkg->AddParam<>("reconstruction", recon);

  // Equation of state
  const auto eos_str = pin->GetOrAddString("eos", "type", "ideal");
  if (eos_str == "ideal") {
    const Real gm1_in = pin->GetOrAddReal("eos", "gm1", 0.6666667);
    const Real cv_in = pin->GetOrAddReal("eos", "cv", 1.5);
    singularity::EOS eos = singularity::IdealGas(gm1_in, cv_in);
    singularity::EOS eos_device = eos.GetOnDevice();
    pkg->AddParam<>("eos", eos_device);
    pkg->AddParam<>("eos_host", eos);
    pkg->AddParam<>("update_lambda", false);
  } else {
    PARTHENON_FAIL("[Apophis]: EOS not recognized. Exiting.");
  }

  // Riemann solver
  const auto riemann_str = pin->GetString("hydro", "riemann");
  auto riemann = RiemannSolver::undefined;

  if (riemann_str == "hllc") {
    riemann = RiemannSolver::hllc;
  } else {
    PARTHENON_FAIL("[Apophis]: Riemann solver not recognized. Exiting.");
  }

  pkg->AddParam<>("riemann_solver", riemann);

  // Add flux functions
  std::map<std::tuple<Fluid, Reconstruction, RiemannSolver>, FluxFun_t *>
      flux_functions{};
  add_flux_fun<Fluid::euler, Reconstruction::dc, RiemannSolver::hllc>(
      flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hllc>(
      flux_functions);

  FluxFun_t *flux_other_stage = nullptr;
  flux_other_stage = flux_functions.at(std::make_tuple(fluid, recon, riemann));

  // Integrator
  const auto integrator_str = pin->GetString("parthenon/time", "integrator");
  auto integrator = Integrator::undefined;
  FluxFun_t *flux_first_stage = flux_other_stage;

  if (integrator_str == "rk1") {
    integrator = Integrator::rk1;
  } else if (integrator_str == "rk2") {
    integrator = Integrator::rk2;
  } else if (integrator_str == "rk3") {
    integrator = Integrator::rk3;
  } else {
    PARTHENON_FAIL("[Apophis]: Integrator not recognized. Exiting.");
  }

  pkg->AddParam<>("integrator", integrator);
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  // Add fields
  std::string field_name = "cons";
  std::vector<std::string> cons_labels(NHYDRO);
  cons_labels[IDN] = "Density";
  cons_labels[IM1] = "MomentumDensity1";
  cons_labels[IM2] = "MomentumDensity2";
  cons_labels[IM3] = "MomentumDensity3";
  cons_labels[IEN] = "TotalEnergyDensity";
  parthenon::Metadata m(
      {parthenon::Metadata::Cell, parthenon::Metadata::Independent,
       parthenon::Metadata::FillGhost, parthenon::Metadata::WithFluxes},
      std::vector<int>({NHYDRO}), cons_labels);
  pkg->AddField(field_name, m);

  field_name = "prim";
  std::vector<std::string> prim_labels(NHYDRO);
  prim_labels[IDN] = "Density";
  prim_labels[IV1] = "Velocity1";
  prim_labels[IV2] = "Velocity2";
  prim_labels[IV3] = "Velocity3";
  prim_labels[IPR] = "Pressure";
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::Derived},
      std::vector<int>({NHYDRO}), prim_labels);
  pkg->AddField(field_name, m);

  return pkg;
}

template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
  return TaskStatus::complete;
}

} // namespace Apophis
