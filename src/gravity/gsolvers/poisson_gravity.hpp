#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"
#include "solvers/bicgstab_solver.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_utils.hpp"

#include "../../main.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

// 7-point stencil for the Laplacian operator using second central differences
struct PoissonEquation {
  bool do_flux_cor = false;

  template <class x_t, class out_t, class TL_t>
  TaskID Ax(TL_t &tl, TaskID dep, std::shared_ptr<MeshData<Real>> &md) {
    auto flux_res = tl.AddTask(dep, CalculateFluxes<x_t>, md);
    if (do_flux_cor && !(md->grid.type == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md);
      auto send_flxcor = tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md);
      auto recv_flxcor = tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix<x_t, out_t>, md);
  }

  template <class var_t>
  static TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    using TE = parthenon::TopologicalElement;

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        "CalculateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i)) / dx1;
          if (i == ib.e)
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1)) / dx1;

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) / dx2;
            if (j == jb.e)
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) /
                  dx2;
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) / dx3;
            if (k == kb.e)
              pack.flux(b, X2DIR, var_t(), k + 1, j, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) /
                  dx3;
          }
        });
    return TaskStatus::complete;
  }

  template <class in_t, class out_t>
  static TaskStatus FluxMultiplyMatrix(std::shared_ptr<MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md->GetMeshPointer()->packages.Get("Gravity");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc =
        parthenon::MakePackDescriptor<in_t, out_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack(b, te, out_t(), k, j, i) = -alpha * pack(b, te, in_t(), k, j, i);
          pack(b, te, out_t(), k, j, i) += (pack.flux(b, X1DIR, in_t(), k, j, i) -
                                            pack.flux(b, X1DIR, in_t(), k, j, i + 1)) /
                                           dx1;

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack(b, te, out_t(), k, j, i) += (pack.flux(b, X2DIR, in_t(), k, j, i) -
                                              pack.flux(b, X2DIR, in_t(), k, j + 1, i)) /
                                             dx2;
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack(b, te, out_t(), k, j, i) += (pack.flux(b, X3DIR, in_t(), k, j, i) -
                                              pack.flux(b, X3DIR, in_t(), k + 1, j, i)) /
                                             dx3;
          }
        });
    return TaskStatus::complete;
  }

  template <class diag_t>
  TaskStatus SetDiagonal(std::shared_ptr<MeshData<Real>> &md) {
    /*
     * Set the diagonal of the matrix to be -2/dx^2 - alpha
     * where dx is the grid spacing in each direction and alpha is a user-defined
     * parameter. ATTENTION: When comparing to the Parthenon example for the Poisson
     * solver, we fix D = 1.0, i.e. this isn't as flexible.
     */
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md->GetMeshPointer()->packages.Get("Gravity");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc = parthenon::MakePackDescriptor<diag_t>(md.get());
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        "StoreDiagonal", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);

          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          Real diag_elem = -2.0 / (dx1 * dx1) - alpha;

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            diag_elem -= 2.0 / (dx2 * dx2);
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            diag_elem -= 2.0 / (dx3 * dx3);
          }
          pack(b, te, diag_t(), k, j, i) = diag_elem;
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
  auto solver = pkg->Param<std::string>("solver");
  auto *mg_solver =
      pkg->MutableParam<parthenon::solvers::MGSolver<phi, rhs, PoissonEquation>>(
          "MGsolver");
  auto *bicgstab_solver =
      pkg->MutableParam<parthenon::solvers::BiCGSTABSolver<phi, rhs, PoissonEquation>>(
          "MGBiCGSTABsolver");

  auto zero_phi =
      tl.AddTask(build_rhs, TF(parthenon::solvers::utils::SetToZero<phi>), md0);

  auto solve = zero_phi;
  if (solver == "BiCGSTAB") {
    auto setup = bicgstab_solver->AddSetupTasks(tl, zero_phi, partition, pmesh);
    solve = bicgstab_solver->AddTasks(tl, setup, pmesh, partition);
  } else if (solver == "MG") {
    auto setup = mg_solver->AddSetupTasks(tl, zero_phi, partition, pmesh);
    solve = mg_solver->AddTasks(tl, setup, pmesh, partition);
  } else {
    PARTHENON_FAIL("Unknown gravity solver type.");
  }

  // Important, otherwise ghost cells of phi will not be updated
  auto bnd_exchg = parthenon::AddBoundaryExchangeTasks(solve, tl, md0, pmesh->multilevel);

  auto comp_grav =
      tl.AddTask(bnd_exchg, &PoissonGravitySolver::ComputeGravityVector, this, md0);

  return comp_grav;
}

TaskStatus PoissonGravitySolver::ComputeRhs(std::shared_ptr<MeshData<Real>> &md) {
  using namespace parthenon;
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire, te);

  auto pkg = md->GetMeshPointer()->packages.Get("Gravity");

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);

  auto desc = parthenon::MakePackDescriptor<rhs>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);

  // TODO(alexhls): Include this in the pack descriptor
  const auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  const Real gravity_g = pkg->Param<Real>("gravity_constant");
  const Real four_pi_g = 4.0 * M_PI * gravity_g;

  parthenon::par_for(
      "BuildRhs", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);

        pack(b, te, rhs(), k, j, i) = four_pi_g * cons(IDN, k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus
PoissonGravitySolver::ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  const auto phi_pack = md->PackVariables(std::vector<std::string>{"gravity.phi"});

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
