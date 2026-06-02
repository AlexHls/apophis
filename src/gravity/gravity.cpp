#include "gravity.hpp"
#include "../constants.hpp"
#include "../main.hpp"
#include "KokkosCore_Config_SetupBackend.hpp"
#include "basic_types.hpp"
#include "gsolvers/constant_gravity.hpp"
#include "gsolvers/monopole_gravity.hpp"
#include "gsolvers/none_gravity.hpp"
#include "gsolvers/poisson_gravity.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"

#include "kokkos_abstraction.hpp"
#include "parthenon/package.hpp"
#include "parthenon/parthenon.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "solvers/internal_prolongation.hpp"
#include "utils/error_checking.hpp"
#include "utils/reductions.hpp"
#include <bvals/boundary_conditions_generic.hpp>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/solver_utils.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::BoundaryFunction;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace Apophis {

template <parthenon::CoordinateDirection DIR, BCSide SIDE, class... var_ts>
void MultipoleGravityBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
                        parthenon::TopologicalElement el) {
  using namespace parthenon;

  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");
  //
  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto ttFlag = [el] {
    const auto tt = GetTopologicalType(el);
    switch (tt) {
    case (TopologicalType::Cell):
      return Metadata::Cell;
    case (TopologicalType::Face):
      return Metadata::Face;
    case (TopologicalType::Edge):
      return Metadata::Edge;
    case (TopologicalType::Node):
      return Metadata::Node;
    default:
      PARTHENON_FAIL("Unknown topological type")
    }
  }();

  for (auto fine : {false, true}) {
    std::vector<MetadataFlag> flags{Metadata::FillGhost, ttFlag};
    if (fine) flags.push_back(Metadata::Fine);
    std::set<PDOpt> opts = coarse ? std::set<PDOpt>{PDOpt::Coarse} : std::set<PDOpt>{};
    const auto desc = MakePackDescriptor<var_ts...>(rc.get(), flags, opts);
    auto q = desc.GetPack(rc.get());
    const int b = 0;
    const int lstart = q.GetLowerBoundHost(b);
    const int lend = q.GetUpperBoundHost(b);
    if (lend < lstart) return;
    auto nb = IndexRange{lstart, lend};

    MeshBlock *pmb = rc->GetBlockPointer();
    const auto &bounds = fine ? (coarse ? pmb->cellbounds : pmb->f_cellbounds)
                              : (coarse ? pmb->c_cellbounds : pmb->cellbounds);

    const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior, el)
                           : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el)
                                 : bounds.GetBoundsK(IndexDomain::interior, el));
    const int ref = INNER ? range.s : range.e;

    std::string label = "MultipoleGravity";
    label += (INNER ? "Inner" : "Outer");
    label += "X" + std::to_string(DIR);

    constexpr IndexDomain domain =
        INNER ? (X1 ? IndexDomain::inner_x1
                    : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
              : (X1 ? IndexDomain::outer_x1
                    : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

    // used for reflections
    const int offset = 2 * ref + (INNER ? -1 : 1);

    // Compute phi0
    auto pkg = pmb->packages.Get("Gravity");
    const auto mpcoeff =
        pkg->Param<parthenon::AllReduce<parthenon::HostArray1D<Real>>>("mpcoeff");

    pmb->par_for_bndry(
        PARTHENON_AUTO_LABEL, nb, domain, el, coarse, fine,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          const auto &coords = q.GetCoordinates(b);
          const int face_i = INNER ? ref : ref + 1;

          const Real x = X1 ? coords.template Xf<1>(face_i) : coords.template Xc<1>(i);
          const Real y = X2 ? coords.template Xf<2>(face_i) : coords.template Xc<2>(j);
          const Real z = X3 ? coords.template Xf<3>(face_i) : coords.template Xc<3>(k);

          const Real phi0 = MultipolePotential(mpcoeff.val, x, y, z);

          q(b, el, l, k, j, i) = 2.0 * phi0 - q(b, el, l, X3 ? offset - k : k,
                                                X2 ? offset - j : j, X1 ? offset - i : i);
        });
  }
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, class... var_ts>
void MultipoleGravityBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = parthenon::TopologicalElement;
  // for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3})
    MultipoleGravityBC<DIR, SIDE, var_ts...>(rc, coarse, el);
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    // GenericBC<DIR, SIDE, BCType::FixedFace, any_gravity>(rc, coarse, 0.0);
    MultipoleGravityBC<DIR, SIDE, any_gravity>(rc, coarse);
  };
}

struct ProlongateSharedQuadratic {
  static constexpr bool OperationRequired(parthenon::TopologicalElement fel,
                                          parthenon::TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM, parthenon::TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION static void
  QuadraticWeights(const parthenon::Coordinates_t &coords,
                   const parthenon::Coordinates_t &coarse_coords, const int ci,
                   const int fi, Real *w) {
    const Real xm = coarse_coords.X<DIM, el>(ci - 1);
    const Real xc = coarse_coords.X<DIM, el>(ci);
    const Real xp = coarse_coords.X<DIM, el>(ci + 1);
    const Real xf = coords.X<DIM, el>(fi);

    w[0] = (xf - xc) * (xf - xp) / ((xm - xc) * (xm - xp));
    w[1] = (xf - xm) * (xf - xp) / ((xc - xm) * (xc - xp));
    w[2] = (xf - xm) * (xf - xc) / ((xp - xm) * (xp - xc));
  }

  template <int DIM, parthenon::TopologicalElement el = parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement /*cel*/ = parthenon::TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t &coarse_coords,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pcoarse,
     const parthenon::ParArrayND<Real, parthenon::VariableState> *pfine) {
    using TE = parthenon::TopologicalElement;
    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;
    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);

    const int fi = INCLUDE_X1 ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = INCLUDE_X2 ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = INCLUDE_X3 ? (k - ckb.s) * 2 + kb.s : kb.s;

    for (int ok_child = 0; ok_child < 1 + INCLUDE_X3; ++ok_child) {
      Real wk[3]{0.0, 1.0, 0.0};
      if constexpr (INCLUDE_X3) {
        QuadraticWeights<3, el>(coords, coarse_coords, k, fk + ok_child, wk);
      }

      for (int oj_child = 0; oj_child < 1 + INCLUDE_X2; ++oj_child) {
        Real wj[3]{0.0, 1.0, 0.0};
        if constexpr (INCLUDE_X2) {
          QuadraticWeights<2, el>(coords, coarse_coords, j, fj + oj_child, wj);
        }

        for (int oi_child = 0; oi_child < 1 + INCLUDE_X1; ++oi_child) {
          Real wi[3]{0.0, 1.0, 0.0};
          if constexpr (INCLUDE_X1) {
            QuadraticWeights<1, el>(coords, coarse_coords, i, fi + oi_child, wi);
          }

          Real value = 0.0;
          for (int ok = -INCLUDE_X3; ok <= INCLUDE_X3; ++ok) {
            for (int oj = -INCLUDE_X2; oj <= INCLUDE_X2; ++oj) {
              for (int oi = -INCLUDE_X1; oi <= INCLUDE_X1; ++oi) {
                value += wk[ok + 1] * wj[oj + 1] * wi[oi + 1] *
                         coarse(element_idx, l, m, n, k + ok, j + oj, i + oi);
              }
            }
          }

          fine(element_idx, l, m, n, fk + ok_child, fj + oj_child,
               fi + oi_child) = value;
        }
      }
    }
  }
};

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
    const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
    const Real x2min = pin->GetOrAddReal("parthenon/mesh", "x2min", 0.0);
    const Real x2max = pin->GetOrAddReal("parthenon/mesh", "x2max", 0.0);
    const Real x3min = pin->GetOrAddReal("parthenon/mesh", "x3min", 0.0);
    const Real x3max = pin->GetOrAddReal("parthenon/mesh", "x3max", 0.0);
    const Real x1edge = std::max(std::abs(x1min), std::abs(x1max));
    const Real x2edge = std::max(std::abs(x2min), std::abs(x2max));
    const Real x3edge = std::max(std::abs(x3min), std::abs(x3max));
    const Real default_rmax =
        std::sqrt(x1edge * x1edge + x2edge * x2edge + x3edge * x3edge);
    const Real rmax = pin->GetOrAddReal("gravity", "rmax", default_rmax);
    if (rmax <= 0.0) {
      PARTHENON_FAIL("[Apophis]: gravity/rmax must be positive for monopole gravity.");
    }
    pkg->AddParam<Real>("rmax", rmax);

    auto m_phi =
        parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                             parthenon::Metadata::OneCopy});
    pkg->AddField(phi::name(), m_phi);
  }

  // If poisson solver, add the potential field and set solver settings
  if (gravity == Gravity::poisson) {
    // Multipole coefficients for boundary conditions
    parthenon::AllReduce<parthenon::HostArray1D<Real>> mpcoeff;
    mpcoeff.val = parthenon::HostArray1D<Real>("mpcoeff", 9);
    pkg->AddParam("mpcoeff", mpcoeff, true);

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

    std::string prolong =
        pin->GetOrAddString("gravity", "boundary_prolongation", "Linear");

    Real err_tol = pin->GetOrAddReal("gravity", "err_tol", 1.0e-8);
    pkg->AddParam<>("err_tol", err_tol);

    using PoissEq = PoissonEquation<phi, D>;
    PoissEq eq(pin, "gravity");
    pkg->AddParam<>("poisson_equation", eq, parthenon::Params::Mutability::Mutable);

    std::shared_ptr<parthenon::solvers::SolverBase> psolver;
    using prolongator_t = parthenon::solvers::ProlongationBlockInteriorDefault;
    using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;

    if (solver == "MG") {
      psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
          "base", "phi", "rhs", pin, "gravity/solver_params", PoissEq(pin, "gravity"));
    } else if (solver == "MGBiCGSTAB") {
      psolver =
          std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
              "base", "phi", "rhs", pin, "gravity/solver_params",
              PoissEq(pin, "gravity"));
    } else {
      PARTHENON_FAIL("Unknown solver type " + solver + ".");
    }
    pkg->AddParam("solver_pointer", psolver);

    // Add fields needed for the Poisson solver
    using namespace parthenon::refinement_ops;
    auto mD = parthenon::Metadata(
        {parthenon::Metadata::Independent, parthenon::Metadata::OneCopy,
         parthenon::Metadata::Face, parthenon::Metadata::GMGRestrict});
    mD.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(D::name(), mD);

    auto mflux_comm = parthenon::Metadata(
        {parthenon::Metadata::Cell, parthenon::Metadata::Independent,
         parthenon::Metadata::FillGhost, parthenon::Metadata::WithFluxes,
         parthenon::Metadata::GMGRestrict, parthenon::Metadata::GMGProlongate});

    if (prolong == "Linear") {
      mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    } else if (prolong == "Quadratic") {
      mflux_comm.RegisterRefinementOps<ProlongateSharedQuadratic, RestrictAverage>();
    } else if (prolong == "Constant") {
      mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
    } else {
      PARTHENON_FAIL("Unknown prolongation method for gravity boundaries.");
    }
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
