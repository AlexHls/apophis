#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"
#include "basic_types.hpp"
#include "solvers/bicgstab_solver.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_base.hpp"
#include "solvers/solver_utils.hpp"

#include "../../main.hpp"
#include "utils/type_list.hpp"
#include <memory>

using namespace parthenon::driver::prelude;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace Apophis {

// 7-point stencil for the Laplacian operator using second central differences
template <class var_t, class D_t>
struct PoissonEquation {
  bool do_flux_cor = false;
  bool set_flux_boundary = false;
  bool include_flux_dx = false;

  using IndependentVars = parthenon::TypeList<var_t>;

  PoissonEquation(parthenon::ParameterInput *pin, const std::string &label) {
    do_flux_cor = pin->GetOrAddBoolean(label, "flux_correct", false);
    set_flux_boundary = pin->GetOrAddBoolean(label, "set_flux_boundary", false);
    include_flux_dx =
        (pin->GetOrAddString(label, "boundary_prolongation", "Linear") == "Constant");
  }

  TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
            std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
            std::shared_ptr<parthenon::MeshData<Real>> &md_in,
            std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
    if (set_flux_boundary) {
      flux_res = tl.AddTask(flux_res, SetFluxBoundaries, md_mat, md_in, include_flux_dx);
    }
    if (do_flux_cor && !(md_mat->grid.type == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
      auto send_flxcor =
          tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
      auto recv_flxcor =
          tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
  }

  template <parthenon::CoordinateDirection dir, class coords_t>
  KOKKOS_INLINE_FUNCTION auto GetEffectiveInverseDx2(const coords_t &coords, const int k,
                                                     const int j, const int i) {
    using TE = parthenon::TopologicalElement;
    constexpr TE te = dir == X1DIR ? TE::F1 : (dir == X2DIR ? TE::F2 : TE::F3);
    constexpr int ioff = (dir == X1DIR);
    constexpr int joff = (dir == X2DIR);
    constexpr int koff = (dir == X3DIR);

    const Real xp = coords.template Xc<dir>(k + koff, j + joff, i + ioff);
    const Real xc = coords.template Xc<dir>(k, j, i);
    const Real xm = coords.template Xc<dir>(k - koff, j - joff, i - ioff);

    const Real dxp = xp - xc;
    const Real dxm = xc - xm;
    const Real Ap = coords.template Volume<te>(k + koff, j + joff, i + ioff);
    const Real Am = coords.template Volume<te>(k, j, i);
    const Real Vol = coords.template Volume<TE::CC>(k, j, i);
    return std::make_pair(Ap / (dxp * Vol), Am / (dxm * Vol));
  }

  TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                         std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    const int ndim = md_mat->GetMeshPointer()->ndim;
    IndexRange ib = md_mat->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_mat->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_mat->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md_mat->GetMeshPointer()->packages.Get("Gravity");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    auto desc_mat = parthenon::MakePackDescriptor<D_t>(md_mat.get());
    auto desc_diag = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack_mat = desc_mat.GetPack(md_mat.get());
    auto pack_diag = desc_diag.GetPack(md_diag.get());
    using TE = parthenon::TopologicalElement;
    parthenon::par_for(
        "StoreDiagonal", 0, pack_mat.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_mat.GetCoordinates(b);
          // Build the unigrid diagonal of the matrix
          Real diag_elem = -alpha;
          {
            auto [idx2p, idx2m] = GetEffectiveInverseDx2<X1DIR>(coords, k, j, i);
            diag_elem -= (pack_mat(b, TE::F1, D_t(), k, j, i) * idx2m +
                          pack_mat(b, TE::F1, D_t(), k, j, i + 1) * idx2p);
          }
          if (ndim > 1) {
            auto [idx2p, idx2m] = GetEffectiveInverseDx2<X2DIR>(coords, k, j, i);
            diag_elem -= (pack_mat(b, TE::F2, D_t(), k, j, i) * idx2m +
                          pack_mat(b, TE::F2, D_t(), k, j + 1, i) * idx2p);
          }
          if (ndim > 2) {
            auto [idx2p, idx2m] = GetEffectiveInverseDx2<X3DIR>(coords, k, j, i);
            diag_elem -= (pack_mat(b, TE::F3, D_t(), k, j, i) * idx2m +
                          pack_mat(b, TE::F3, D_t(), k + 1, j, i) * idx2p);
          }
          pack_diag(b, te, var_t(), k, j, i) = diag_elem;
        });
    return TaskStatus::complete;
  }

  static TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    using TE = parthenon::TopologicalElement;

    int nblocks = md->NumBlocks();

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get());
    auto desc_mat = parthenon::MakePackDescriptor<D_t>(md_mat.get(), {});
    auto pack_mat = desc_mat.GetPack(md_mat.get());
    parthenon::par_for(
        "CaclulateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              pack_mat(b, TE::F1, D_t(), k, j, i) / coords.template Dxc<X1DIR>(k, j, i) *
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i));
          if (i == ib.e)
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                pack_mat(b, TE::F1, D_t(), k, j, i + 1) /
                coords.template Dxc<X1DIR>(k, j, i + 1) *
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1));

          if (ndim > 1) {
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                pack_mat(b, TE::F2, D_t(), k, j, i) *
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) /
                coords.template Dxc<X2DIR>(k, j, i);
            if (j == jb.e)
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  pack_mat(b, TE::F2, D_t(), k, j + 1, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) /
                  coords.template Dxc<X2DIR>(k, j + 1, i);
          }

          if (ndim > 2) {
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                pack_mat(b, TE::F3, D_t(), k, j, i) *
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) /
                coords.template Dxc<X3DIR>(k, j, i);
            if (k == kb.e)
              pack.flux(b, X3DIR, var_t(), k + 1, j, i) =
                  pack_mat(b, TE::F3, D_t(), k + 1, j, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) /
                  coords.template Dxc<X3DIR>(k + 1, j, i);
          }
        });
    return TaskStatus::complete;
  }

  static TaskStatus SetFluxBoundaries(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                      std::shared_ptr<parthenon::MeshData<Real>> &md,
                                      bool do_flux_dx) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);

    using TE = parthenon::TopologicalElement;

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto desc_mat = parthenon::MakePackDescriptor<D_t>(md.get());
    auto pack = desc.GetPack(md.get());
    auto pack_mat = desc_mat.GetPack(md_mat.get());
    const std::size_t scratch_size_in_bytes = 0;
    const std::size_t scratch_level = 1;

    const parthenon::Indexer3D idxers[6]{
        parthenon::Indexer3D(kb, jb, {ib.s, ib.s}),
        parthenon::Indexer3D(kb, jb, {ib.e + 1, ib.e + 1}),
        parthenon::Indexer3D(kb, {jb.s, jb.s}, ib),
        parthenon::Indexer3D(kb, {jb.e + 1, jb.e + 1}, ib),
        parthenon::Indexer3D({kb.s, kb.s}, jb, ib),
        parthenon::Indexer3D({kb.e + 1, kb.e + 1}, jb, ib)};
    constexpr int x1off[6]{-1, 1, 0, 0, 0, 0};
    constexpr int x2off[6]{0, 0, -1, 1, 0, 0};
    constexpr int x3off[6]{0, 0, 0, 0, -1, 1};
    constexpr TE tes[6]{TE::F1, TE::F1, TE::F2, TE::F2, TE::F3, TE::F3};
    constexpr int dirs[6]{X1DIR, X1DIR, X2DIR, X2DIR, X3DIR, X3DIR};
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "SetFluxBoundaries", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          const int gid = pack.GetGID(b);
          const int level = pack.GetLevel(b, 0, 0, 0);
          for (int face = 0; face < ndim * 2; ++face) {
            const auto &idxer = idxers[face];
            const auto dir = dirs[face];
            const auto te = tes[face];
            // Impose the zero Dirichlet boundary condition at the actual boundary
            if (pack.IsPhysicalBoundary(b, x3off[face], x2off[face], x1off[face])) {
              const int koff = x3off[face] > 0 ? -1 : 0;
              const int joff = x2off[face] > 0 ? -1 : 0;
              const int ioff = x1off[face] > 0 ? -1 : 0;
              const int sign = x1off[face] + x2off[face] + x3off[face];
              parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                       idxer.size() - 1, [&](const int idx) {
                                         const auto [k, j, i] = idxer(idx);
                                         pack.flux(b, dir, var_t(), k, j, i) =
                                             sign * pack_mat(b, te, D_t(), k, j, i) *
                                             pack(b, var_t(), k + koff, j + joff,
                                                  i + ioff) /
                                             (0.5 * coords.Dxc(dir, k, j, i));
                                       });
            }
            // Correct for size of neighboring zone at fine-coarse boundary when using
            // constant prolongation
            if (do_flux_dx &&
                pack.GetLevel(b, x3off[face], x2off[face], x1off[face]) == level - 1) {
              parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                       idxer.size() - 1, [&](const int idx) {
                                         const auto [k, j, i] = idxer(idx);
                                         pack.flux(b, dir, var_t(), k, j, i) /= 1.5;
                                       });
            }
          }
        });
    return TaskStatus::complete;
  }

  static TaskStatus
  FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md->GetMeshPointer()->packages.Get("Gravity");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    static auto desc =
        parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    static auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
    auto pack = desc.GetPack(md.get());
    auto pack_out = desc_out.GetPack(md_out.get());
    parthenon::par_for(
        "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack_out(b, te, var_t(), k, j, i) = -alpha * pack(b, te, var_t(), k, j, i);
          pack_out(b, te, var_t(), k, j, i) +=
              (pack.flux(b, X1DIR, var_t(), k, j, i) *
                   coords.template Volume<TE::F1>(k, j, i) -
               pack.flux(b, X1DIR, var_t(), k, j, i + 1) *
                   coords.template Volume<TE::F1>(k, j, i + 1)) /
              coords.template Volume<TE::CC>(k, j, i);

          if (ndim > 1) {
            pack_out(b, te, var_t(), k, j, i) +=
                (pack.flux(b, X2DIR, var_t(), k, j, i) *
                     coords.template Volume<TE::F2>(k, j, i) -
                 pack.flux(b, X2DIR, var_t(), k, j + 1, i) *
                     coords.template Volume<TE::F2>(k, j + 1, i)) /
                coords.template Volume<TE::CC>(k, j, i);
          }

          if (ndim > 2) {
            pack_out(b, te, var_t(), k, j, i) +=
                (pack.flux(b, X3DIR, var_t(), k, j, i) *
                     coords.template Volume<TE::F3>(k, j, i) -
                 pack.flux(b, X3DIR, var_t(), k + 1, j, i) *
                     coords.template Volume<TE::F3>(k + 1, j, i)) /
                coords.template Volume<TE::CC>(k, j, i);
          }
        });
    return TaskStatus::complete;
  }
};

// Poisson gravity solver
struct PoissonGravitySolver : GravitySolver {
  explicit PoissonGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}
  TaskID PreComputeTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                         const int partition) override;
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ComputeMPCoeff(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ScaleMPCoeff(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ComputeRhs(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md);
};

TaskID PoissonGravitySolver::PreComputeTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                             const int partition) {

  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto comp_mpcoeff = tl.AddTask(dep, &PoissonGravitySolver::ComputeMPCoeff, this, md0);

  auto pkg = pmesh->packages.Get<parthenon::StateDescriptor>("Gravity");
  AllReduce<parthenon::HostArray1D<Real>> *mpcoeff =
      pkg->MutableParam<AllReduce<parthenon::HostArray1D<Real>>>("mpcoeff");

  auto start_reduce =
      tl.AddTask(TaskQualifier::once_per_region, comp_mpcoeff,
                 &AllReduce<parthenon::HostArray1D<Real>>::StartReduce, mpcoeff, MPI_SUM);
  // test the reduction until it completes
  auto finish_reduce =
      tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync, start_reduce,
                 &AllReduce<parthenon::HostArray1D<Real>>::CheckReduce, mpcoeff);

  auto scale = tl.AddTask(finish_reduce, &PoissonGravitySolver::ScaleMPCoeff, this, md0);

  return scale;
}

TaskID PoissonGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                      const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto build_rhs = tl.AddTask(dep, &PoissonGravitySolver::ComputeRhs, this, md0);

  auto pkg = pmesh->packages.Get<parthenon::StateDescriptor>("Gravity");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto &md_phi = pmesh->mesh_data.Add("phi", md0, {phi::name()});
  auto &md_rhs = pmesh->mesh_data.Add("rhs", md0, {phi::name()});

  auto copy_rhs = tl.AddTask(
      build_rhs, TF(parthenon::solvers::utils::between_fields::CopyData<rhs, phi>), md0);
  copy_rhs = tl.AddTask(copy_rhs,
                        TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<phi>>),
                        md0, md_rhs);
  auto zero_phi =
      tl.AddTask(copy_rhs, TF(parthenon::solvers::utils::SetToZero<phi>), md_phi);
  auto setup = psolver->AddSetupTasks(tl, zero_phi, partition, pmesh);

  auto solve = psolver->AddTasks(tl, setup, partition, pmesh);

  auto copy_back =
      tl.AddTask(solve, TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<phi>>),
                 md_phi, md0);

  // Important, otherwise ghost cells of phi will not be updated
  auto bnd_exchg = parthenon::AddBoundaryExchangeTasks(solve, tl, md0, pmesh->multilevel);

  auto comp_grav =
      tl.AddTask(copy_back, &PoissonGravitySolver::ComputeGravityVector, this, md0);

  return comp_grav;
}

TaskStatus PoissonGravitySolver::ComputeMPCoeff(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Gravity");
  AllReduce<parthenon::HostArray1D<Real>> *mpcoeff =
      pkg->MutableParam<AllReduce<parthenon::HostArray1D<Real>>>("mpcoeff");

  // Reset to zero
  for (int n = 0; n < 10; n++) {
    mpcoeff->val(n) = 0.0;
  }

  pmb->par_for(
      "ComputeMPCoeff", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        const Real vol = coords.CellVolume(k, j, i);

        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);
        Real r2 = x * x + y * y + z * z;

        Real s = prim_pack(b, IDN, k, j, i) * vol;

        Real m0 = s;
        Real m1 = s * y;
        Real m2 = s * z;
        Real m3 = s * x;
        Real m4 = s * x * y;
        Real m5 = s * y * z;
        Real m6 = s * (3.0 * z * z - r2);
        Real m7 = s * z * x;
        Real m8 = s * 0.5 * (x * x - y * y);

        Kokkos::atomic_add(&mpcoeff->val(0), m0);
        Kokkos::atomic_add(&mpcoeff->val(1), m1);
        Kokkos::atomic_add(&mpcoeff->val(2), m2);
        Kokkos::atomic_add(&mpcoeff->val(3), m3);
        Kokkos::atomic_add(&mpcoeff->val(4), m4);
        Kokkos::atomic_add(&mpcoeff->val(5), m5);
        Kokkos::atomic_add(&mpcoeff->val(6), m6);
        Kokkos::atomic_add(&mpcoeff->val(7), m7);
        Kokkos::atomic_add(&mpcoeff->val(8), m8);
      });

  return TaskStatus::complete;
}

TaskStatus PoissonGravitySolver::ScaleMPCoeff(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  auto pkg = pmb->packages.Get("Gravity");
  AllReduce<parthenon::HostArray1D<Real>> *mpcoeff =
      pkg->MutableParam<AllReduce<parthenon::HostArray1D<Real>>>("mpcoeff");
  const Real gravity_g = pkg->Param<Real>("gravity_constant");

  // constants for multipole expansion
  constexpr Real c0 = -0.25 / M_PI;
  constexpr Real c1 = -0.25 / M_PI;
  constexpr Real c2 = -0.0625 / M_PI;
  constexpr Real c2a = -0.75 / M_PI;

  mpcoeff->val(0) *= c0 * gravity_g;
  mpcoeff->val(1) *= c1 * gravity_g;
  mpcoeff->val(2) *= c1 * gravity_g;
  mpcoeff->val(3) *= c1 * gravity_g;
  mpcoeff->val(4) *= c2a * gravity_g;
  mpcoeff->val(5) *= c2a * gravity_g;
  mpcoeff->val(6) *= c2 * gravity_g;
  mpcoeff->val(7) *= c2a * gravity_g;
  mpcoeff->val(8) *= c2a * gravity_g;

  return TaskStatus::complete;
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

  auto desc = parthenon::MakePackDescriptor<rhs, D>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);

  // TODO(alexhls): Include this in the pack descriptor
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const Real gravity_g = pkg->Param<Real>("gravity_constant");
  const Real four_pi_g = 4.0 * M_PI * gravity_g;

  parthenon::par_for(
      "BuildRhs", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &prim = prim_pack(b);

        pack(b, te, rhs(), k, j, i) = four_pi_g * prim(IDN, k, j, i);

        // Init D to 1.0 for standard PoissonEquation
        // TODO Move thsi somewhere else to avoid recomputation
        pack(b, TE::F1, D(), k, j, i) = 1.0;
        pack(b, TE::F2, D(), k, j, i) = 1.0;
        pack(b, TE::F3, D(), k, j, i) = 1.0;
      });
  return TaskStatus::complete;
}

TaskStatus
PoissonGravitySolver::ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
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
      "UpdateGravity", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
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
