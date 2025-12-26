#include "gravity.hpp"
#include "../constants.hpp"
#include "../main.hpp"
#include "KokkosCore_Config_SetupBackend.hpp"
#include "gsolvers/constant_gravity.hpp"
#include "gsolvers/monopole_gravity.hpp"
#include "gsolvers/none_gravity.hpp"
#include "gsolvers/poisson_gravity.hpp"
#include "interface/state_descriptor.hpp"

#include "parthenon/package.hpp"
#include "parthenon/parthenon.hpp"
#include <bvals/boundary_conditions_generic.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/solver_utils.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::BoundaryFunction;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace Apophis {

struct any_gravity : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_gravity(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "gravity[.].*"; }
};

template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_gravity>(rc, coarse, 0.0);
  };
}

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
  } else if (gravity_str == "monopole") {
    gravity = Gravity::monopole;
    solver = std::make_shared<MonopoleGravitySolver>(pin);
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

  // If multipole solver, add solver settings
  if (gravity == Gravity::monopole) {
    // nrbin and arrays are now handled in MonopoleGravitySolver constructor
    const Real rmax = pin->GetInteger("gravity", "rmax");
    pkg->AddParam<Real>("rmax", rmax);
  }

  // If poisson solver, add the potential field and set solver settings
  if (gravity == Gravity::poisson) {
    // Special boundary conditions for the potential field
    using BF = parthenon::BoundaryFace;
    pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
    pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
    pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
    pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

    // Poisson specific settings
    int max_poisson_iterations = pin->GetOrAddInteger("gravity", "max_iterations", 10000);
    pkg->AddParam<>("max_iterations", max_poisson_iterations);

    Real diagonal_alpha = pin->GetOrAddReal("gravity", "diagonal_alpha", 0.0);
    pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

    std::string solver = pin->GetOrAddString("gravity", "solver", "MG");
    pkg->AddParam<>("solver", solver);

    bool flux_correct = pin->GetOrAddBoolean("gravity", "flux_correct", false);
    pkg->AddParam<>("flux_correct", flux_correct);

    Real err_tol = pin->GetOrAddReal("gravity", "err_tol", 1.0e-8);
    pkg->AddParam<>("err_tol", err_tol);

    PoissonEquation eq;
    eq.do_flux_cor = flux_correct;

    parthenon::solvers::MGParams mg_params(pin, "gravity/solver_params");
    parthenon::solvers::MGSolver<phi, rhs, PoissonEquation> mg_solver(pkg.get(),
                                                                      mg_params, eq);
    pkg->AddParam<>("MGsolver", mg_solver, parthenon::Params::Mutability::Mutable);

    parthenon::solvers::BiCGSTABParams bicgstab_params(pin, "gravity/solver_params");
    parthenon::solvers::BiCGSTABSolver<phi, rhs, PoissonEquation> bicg_solver(
        pkg.get(), bicgstab_params, eq);
    pkg->AddParam<>("MGBiCGSTABsolver", bicg_solver,
                    parthenon::Params::Mutability::Mutable);

    // Add fields needed for the Poisson solver
    using namespace parthenon::refinement_ops;
    auto mflux_comm = parthenon::Metadata(
        {parthenon::Metadata::Cell, parthenon::Metadata::Independent,
         parthenon::Metadata::FillGhost, parthenon::Metadata::WithFluxes,
         parthenon::Metadata::GMGRestrict});
    mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(phi::name(), mflux_comm);

    auto m_no_ghost =
        parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                             parthenon::Metadata::OneCopy});
    pkg->AddField(rhs::name(), m_no_ghost);
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

  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto w = prim_pack(b);
        const auto &grav = grav_pack(b);

        cons(IM1, k, j, i) += grav(0, k, j, i) * w(IDN, k, j, i) * dt;
        Real de = grav(0, k, j, i) * w(IV1, k, j, i);

        if (ndim >= 2) {
          cons(IM2, k, j, i) += grav(1, k, j, i) * w(IDN, k, j, i) * dt;
          de += grav(1, k, j, i) * w(IV2, k, j, i);
        }

        if (ndim >= 3) {
          cons(IM3, k, j, i) += grav(2, k, j, i) * w(IDN, k, j, i) * dt;
          de += grav(2, k, j, i) * w(IV3, k, j, i);
        }

        cons(IEN, k, j, i) += de * w(IDN, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

} // namespace Apophis
