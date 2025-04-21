#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"
#include "solvers/bicgstab_solver.hpp"

#include "../../main.hpp"

namespace Apophis {

// 7-point stencil for the Laplacian operator using second central differences
struct PoissonOp {
  template <class x_t, class out_t, class TL_t>
  TaskID Ax(TL_t &tl, TaskID dep, std::shared_ptr<MeshData<Real>> &md) {

    return tl.AddTask(
        dep, "PoissonOp::Ax",
        [](std::shared_ptr<parthenon::MeshData<Real>> &md) {
          using TE = parthenon::TopologicalElement;
          TE te = TE::CC;
          auto pkg = md->GetMeshPointer()->packages.Get("Gravity");
          const int ndim = md->GetMeshPointer()->ndim;

          int nblocks = md->NumBlocks();
          std::vector<bool> include_block(nblocks, true);
          auto desc = parthenon::MakePackDescriptor<x_t, out_t>(md.get(), {}, {});
          auto pack = desc.GetPack(md.get(), include_block);

          IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
          IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
          IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

          parthenon::par_for(
              "Laplace7pt", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              KOKKOS_LAMBDA(int b, int k, int j, int i) {
                const auto &coords = pack.GetCoordinates(b);
                Real idx2 = 1 / (coords.Dxc<1>(i) * coords.Dxc<1>(i));
                pack(b, te, out_t(), k, j, i) =
                    (pack(b, te, x_t(), k, j, i - 1) + pack(b, te, x_t(), k, j, i + 1) -
                     2 * pack(b, te, x_t(), k, j, i)) *
                    idx2;
                if (ndim > 1) {
                  Real idy2 = 1 / (coords.Dxc<2>(j) * coords.Dxc<2>(j));
                  pack(b, te, out_t(), k, j, i) +=
                      (pack(b, te, x_t(), k, j - 1, i) + pack(b, te, x_t(), k, j + 1, i) -
                       2 * pack(b, te, x_t(), k, j, i)) *
                      idy2;
                }
                if (ndim > 2) {
                  Real idz2 = 1 / (coords.Dxc<3>(k) * coords.Dxc<3>(k));
                  pack(b, te, out_t(), k, j, i) +=
                      (pack(b, te, x_t(), k - 1, j, i) + pack(b, te, x_t(), k + 1, j, i) -
                       2 * pack(b, te, x_t(), k, j, i)) *
                      idz2;
                }
              });
          return TaskStatus::complete;
        },
        md);
  }

  //-----------------------------------------------------------------------------
  // Build the diagonal of the 7‑point Laplace operator:
  //   diag(i,j,k) = −2*(1/Δx² + 1/Δy² + 1/Δz²)
  //-----------------------------------------------------------------------------
  template <class diag_t>
  static parthenon::TaskStatus
  SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md->GetMeshPointer()->packages.Get("Gravity");

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc = parthenon::MakePackDescriptor<diag_t, D>(md.get());
    auto pack = desc.GetPack(md.get(), include_block);

    auto diag = md->PackVariables({diag_t::name()}).Get(0)->data;
    auto pmb = md->GetBlockData(0)->GetBlockPointer();

    // interior cell ranges
    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // constant spacings (uniform grid assumed)
    Real idx2 = 1.0 / (pmb->coords.Dx<0>(0) * pmb->coords.Dx<0>(0));
    Real idy2 = 1.0 / (pmb->coords.Dx<1>(0) * pmb->coords.Dx<1>(0));
    Real idz2 = 1.0 / (pmb->coords.Dx<2>(0) * pmb->coords.Dx<2>(0));

    pmb->par_for(
        "PoissonOp::SetDiagonal", 0, diag.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(int b, int k, int j, int i) {
          diag(0, k, j, i) = -2.0 * (idx2 + idy2 + idz2);
        });

    return TaskStatus::complete;
  }
};

// Poisson gravity solver
struct PoissonGravitySolver : GravitySolver {
  explicit PoissonGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ComputeRhs(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md);
};

TaskID PoissonGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                      const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto build_rhs = tl.AddTask(dep, &PoissonGravitySolver::ComputeRhs, this, md0);

  auto pkg = pmesh->packages.Get<parthenon::StateDescriptor>("Gravity");
  auto *bicg = pkg->MutableParam<parthenon::solvers::BiCGSTABSolver<phi, rhs, PoissonOp>>(
      "MGBiCGSTABsolver");

  auto setup = bicg->AddSetupTasks(tl, build_rhs, partition, pmesh);
  auto solve = bicg->AddTasks(tl, setup, pmesh, partition);

  auto comp_grav =
      tl.AddTask(solve, &PoissonGravitySolver::ComputeGravityVector, this, md0);

  return comp_grav;
}

TaskStatus PoissonGravitySolver::ComputeRhs(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto rhs_pack = md->PackVariables(std::vector<std::string>{"rhs"});
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const Real gravity_g = pmb->packages.Get("Gravity")->Param<Real>("gravity_constant");
  const Real four_pi_g = 4.0 * M_PI * gravity_g;

  pmb->par_for(
      "BuildRhs", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &rhs = rhs_pack(b);

        rhs(0, k, j, i) = four_pi_g * cons(IDN, k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus
PoissonGravitySolver::ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  const auto phi_pack = md->PackVariables(std::vector<std::string>{"potential"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Gravity");
  const auto gravity_g = pkg->Param<Real>("gravity_constant");

  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &phi = phi_pack(b);
        auto &grav = grav_pack(b);
        const auto &coords = phi_pack.GetCoords(b);

        Real dx = coords.Dxc<1>(i);
        Real dy = coords.Dxc<2>(j);
        Real dz = coords.Dxc<3>(k);

        Real dphi_dx = (phi(0, k, j, i + 1) - phi(0, k, j, i - 1)) / (2.0 * dx);
        grav(0, k, j, i) = -dphi_dx;

        if (ndim >= 2) {
          Real dphi_dy = (phi(0, k, j + 1, i) - phi(0, k, j - 1, i)) / (2.0 * dy);
          grav(1, k, j, i) = -dphi_dy;
        }
        if (ndim >= 3) {
          Real dphi_dz = (phi(0, k + 1, j, i) - phi(0, k - 1, j, i)) / (2.0 * dz);
          grav(2, k, j, i) = -dphi_dz;
        }
      });
  return TaskStatus::complete;
}

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_