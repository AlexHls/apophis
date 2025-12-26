#ifndef GRAVITY_GSOLVERS_MONOPOLE_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_MONOPOLE_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"
#include "tasks/tasks.hpp"
#include <vector>

using namespace parthenon::driver::prelude;

namespace Apophis {

// Monopole gravity solver
struct MonopoleGravitySolver : GravitySolver {
  explicit MonopoleGravitySolver(ParameterInput *pin);
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ZeroRadialBins(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus AccumulateMass(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus TransferMassToEnclosed(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus PrefixSum(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ComputeRadialPhi(std::shared_ptr<MeshData<Real>> &md);

  TaskStatus ComputeGravityVectorDirect(std::shared_ptr<MeshData<Real>> &md);

 private:
  int nrbin_;
  Kokkos::View<Real *> m_local_;
  Kokkos::View<Real *> m_enc_;
  Kokkos::View<Real *> phi_rad_;

  parthenon::AllReduce<std::vector<Real>> reduce_mass_;
};

TaskID MonopoleGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                       const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto zero = tl.AddTask(dep, &MonopoleGravitySolver::ZeroRadialBins, this, md0);
  auto accum = tl.AddTask(zero, &MonopoleGravitySolver::AccumulateMass, this, md0);

  // Fill the reduction vector with accumulated mass from m_local
  auto fill_reduce = tl.AddTask(parthenon::TaskQualifier::local_sync, accum, [this]() {
    auto m_local_host = Kokkos::create_mirror_view(m_local_);
    Kokkos::deep_copy(m_local_host, m_local_);
    for (int i = 0; i < nrbin_; i++) {
      reduce_mass_.val[i] = m_local_host(i);
    }
    return TaskStatus::complete;
  });

  // Global reduction: sum m_local across all ranks
  auto start_reduce =
      tl.AddTask(parthenon::TaskQualifier::local_sync, fill_reduce,
                 &AllReduce<std::vector<Real>>::StartReduce, &reduce_mass_, MPI_SUM);

  auto check_reduce = tl.AddTask(
      parthenon::TaskQualifier::once_per_region | parthenon::TaskQualifier::local_sync,
      start_reduce, &AllReduce<std::vector<Real>>::CheckReduce, &reduce_mass_);

  auto transfer =
      tl.AddTask(check_reduce, &MonopoleGravitySolver::TransferMassToEnclosed, this, md0);
  auto prefix = tl.AddTask(transfer, &MonopoleGravitySolver::PrefixSum, this, md0);

  auto comp_grav =
      tl.AddTask(prefix, &MonopoleGravitySolver::ComputeGravityVectorDirect, this, md0);

  return comp_grav;
}

MonopoleGravitySolver::MonopoleGravitySolver(ParameterInput *pin) : GravitySolver(pin) {
  // If nrbin set to -1, nrbin=nx
  nrbin_ = pin->GetOrAddInteger("gravity", "nrbin", -1);
  if (nrbin_ <= 0) {
    const int nx = pin->GetInteger("parthenon/mesh", "nx");
    nrbin_ = nx;
  }

  // Allocate Kokkos views
  m_local_ = Kokkos::View<Real *>("m_local", nrbin_);
  m_enc_ = Kokkos::View<Real *>("m_enc", nrbin_);

  // Initialize to zero
  Kokkos::deep_copy(m_local_, 0.0);
  Kokkos::deep_copy(m_enc_, 0.0);

  // Initialize the reduction vector
  reduce_mass_.val.resize(nrbin_, 0.0);
}

TaskStatus MonopoleGravitySolver::ZeroRadialBins(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::deep_copy(m_local_, 0.0);
  Kokkos::deep_copy(m_enc_, 0.0);
  return TaskStatus::complete;
}

TaskStatus MonopoleGravitySolver::AccumulateMass(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Gravity");
  const Real rmax = pkg->Param<Real>("rmax");

  const Real dr = rmax / nrbin_;

  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real *> m_local_dev("m_local_accum", nrbin_);

  pmb->par_for(
      "AccumulateMonopoleMass", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto w = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real r = std::sqrt(x * x + y * y + z * z);

        if (r >= rmax) return;

        const int bin = static_cast<int>(r / dr);
        const Real dm = w(IDN, k, j, i) * coords.CellVolume(k, j, i);

        Kokkos::atomic_add(&m_local_dev(bin), dm);
      });

  // Copy results to m_local_ for reduction
  Kokkos::deep_copy(m_local_, m_local_dev);
  return TaskStatus::complete;
}

TaskStatus MonopoleGravitySolver::PrefixSum(std::shared_ptr<MeshData<Real>> &md) {
  // Copy to host to perform serial prefix sum
  auto m_enc_host = Kokkos::create_mirror_view(m_enc_);
  Kokkos::deep_copy(m_enc_host, m_enc_);

  // Serial prefix sum on host
  for (int i = 1; i < nrbin_; i++) {
    m_enc_host(i) += m_enc_host(i - 1);
  }

  // Copy back to device
  Kokkos::deep_copy(m_enc_, m_enc_host);
  return TaskStatus::complete;
}

TaskStatus
MonopoleGravitySolver::TransferMassToEnclosed(std::shared_ptr<MeshData<Real>> &md) {
  // Copy reduced m_local from the AllReduce result to m_enc
  auto m_enc_host = Kokkos::create_mirror_view(m_enc_);
  for (int i = 0; i < nrbin_; i++) {
    m_enc_host(i) = reduce_mass_.val[i];
  }
  Kokkos::deep_copy(m_enc_, m_enc_host);
  return TaskStatus::complete;
}

TaskStatus
MonopoleGravitySolver::ComputeGravityVectorDirect(std::shared_ptr<MeshData<Real>> &md) {
  // Compute gravity directly from enclosed mass without computing phi
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Gravity");
  const Real rmax = pkg->Param<Real>("rmax");
  const Real gravity_g = pkg->Param<Real>("gravity_constant");
  const Real dr = rmax / nrbin_;

  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "UpdateGravityDirect", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real dx = coords.Dxc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real r = std::sqrt(x * x + y * y + z * z);

        auto &grav = grav_pack(b);

        if (r < dx) {
          // Avoid singularity at origin
          grav(0, k, j, i) = 0.0;
          if (ndim >= 2) grav(1, k, j, i) = 0.0;
          if (ndim >= 3) grav(2, k, j, i) = 0.0;
          return;
        }

        // Determine which radial bin this point is in
        Real m_enclosed;
        if (r >= rmax) {
          m_enclosed = m_enc_(nrbin_ - 1);
        } else {
          int bin = static_cast<int>(r / dr);
          // Clamp bin to valid range
          if (bin >= nrbin_) bin = nrbin_ - 1;
          if (bin < 0) bin = 0;
          m_enclosed = m_enc_(bin);
        }

        // Gravitational acceleration: g = -G*M_enc/r^2 in radial direction
        const Real g_r = -gravity_g * m_enclosed / (r * r);

        // Convert to Cartesian components
        grav(0, k, j, i) = g_r * x / r;
        if (ndim >= 2) grav(1, k, j, i) = g_r * y / r;
        if (ndim >= 3) grav(2, k, j, i) = g_r * z / r;
      });
  return TaskStatus::complete;
}

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_MONOPOLE_GRAVITY_HPP_
