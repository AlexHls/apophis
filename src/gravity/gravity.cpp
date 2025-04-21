#include "gravity.hpp"
#include "../constants.hpp"
#include "../main.hpp"
#include "KokkosCore_Config_SetupBackend.hpp"
#include "gsolvers/constant_gravity.hpp"
#include "gsolvers/none_gravity.hpp"
#include "gsolvers/poisson_gravity.hpp"
#include "interface/state_descriptor.hpp"

#include "parthenon/package.hpp"
#include "parthenon/parthenon.hpp"
#include <solvers/bicgstab_solver.hpp>
#include <solvers/solver_utils.hpp>

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor> InitializeGravity(ParameterInput *pin) {
  auto pkg = std::make_shared<parthenon::StateDescriptor>("Gravity");
  const auto gravity_str = pin->GetOrAddString("gravity", "type", "none");
  auto gravity = Gravity::undefined;
  std::shared_ptr<GravitySolver> solver = nullptr;

  const Real gravity_g = pin->GetOrAddReal("gravity", "gravity_constant", GRAVITY_G);
  pkg->AddParam<>("gravity_constant", gravity_g);

  if (gravity_str == "none") {
    gravity = Gravity::none;
    solver = std::make_shared<NoneGravitySolver>(pin);
  } else if (gravity_str == "constant") {
    gravity = Gravity::constant;
    solver = std::make_shared<ConstantGravitySolver>(pin);
  } else if (gravity_str == "poisson") {
    gravity = Gravity::poisson;
    solver = std::make_shared<PoissonGravitySolver>(pin);
  } else {
    PARTHENON_FAIL("[Apophis]: Gravity not recognized. Exiting.");
  }

  pkg->AddParam<>("gravity", gravity);
  pkg->AddParam<>("gravity_solver", solver);

  // Add the gravity field
  std::string field_name = "gravity";
  std::vector<std::string> labels(3);
  labels[0] = "gx";
  labels[1] = "gy";
  labels[2] = "gz";
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived},
                               std::vector<int>({3}), labels);

  pkg->AddField(field_name, m);

  // If poisson solver, add the potential field and set solver settings
  if (gravity == Gravity::poisson) {
    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Independent,
                             parthenon::Metadata::FillGhost,
                             parthenon::Metadata::GMGRestrict});
    pkg->AddField(phi::name(), m);

    m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Independent,
                             parthenon::Metadata::OneCopy});
    pkg->AddField(laplace::name(), m);

    auto m_no_ghost =
        parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                             parthenon::Metadata::OneCopy});
    pkg->AddField(rhs::name(), m_no_ghost);

    int max_poisson_iterations = pin->GetOrAddInteger("gravity", "max_iterations", 10000);
    pkg->AddParam<>("max_iterations", max_poisson_iterations);

    std::string solver = pin->GetOrAddString("gravity", "solver", "MGBiCGSTAB");
    pkg->AddParam<>("solver", solver);

    Real err_tol = pin->GetOrAddReal("gravity", "err_tol", 1.0e-8);
    pkg->AddParam<>("err_tol", err_tol);

    PoissonOp eq;

    if (solver == "MGBiCGSTAB") {
      parthenon::solvers::BiCGSTABParams bicgstab_params(pin, "gravity/solver_params");
      parthenon::solvers::BiCGSTABSolver<phi, rhs, PoissonOp> bicg_solver(
          pkg.get(), bicgstab_params, eq);
      pkg->AddParam<>("MGBiCGSTABsolver", bicg_solver,
                      parthenon::Params::Mutability::Mutable);
    } else {
      PARTHENON_FAIL("[Apophis]: Gravity solver not recognized. Exiting.");
    }
  }

  return pkg;
}

TaskStatus ApplyGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto w = prim_pack(b);
        const auto &grav = grav_pack(b);

        cons(IM1, k, j, i) += grav(0, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IM2, k, j, i) += grav(1, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IM3, k, j, i) += grav(2, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IEN, k, j, i) +=
            (grav(0, k, j, i) * w(IV1, k, j, i) + grav(1, k, j, i) * w(IV2, k, j, i) +
             grav(2, k, j, i) * w(IV3, k, j, i)) *
            w(IDN, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

} // namespace Apophis
