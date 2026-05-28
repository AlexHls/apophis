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
  TaskID PreComputeTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                         const int partition) override;
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ZeroRadialBins(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus AccumulateMass(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus BuildRadialTables(std::shared_ptr<MeshData<Real>> &md);

  TaskStatus ComputeGravityVectorDirect(std::shared_ptr<MeshData<Real>> &md);

 private:
  int nrbin_;
  Kokkos::View<Real *> m_local_;
  Kokkos::View<Real *> m_enc_;
  Kokkos::View<Real *> phi_rad_;

  parthenon::AllReduce<std::vector<Real>> reduce_mass_;
};

TaskID MonopoleGravitySolver::PreComputeTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                              const int partition) {
  // TODO
  // Nothing to be done, in the future compbine with monopole precompute of poisson
  // gravity
  return dep;
}

TaskID MonopoleGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                       const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto zero = tl.AddTask(dep, &MonopoleGravitySolver::ZeroRadialBins, this, md0);
  auto accum = tl.AddTask(zero, &MonopoleGravitySolver::AccumulateMass, this, md0);

  auto fill_reduce = tl.AddTask(parthenon::TaskQualifier::local_sync, accum, [this]() {
    auto m_local_host = Kokkos::create_mirror_view(m_local_);
    Kokkos::deep_copy(m_local_host, m_local_);
    for (int i = 0; i < nrbin_; i++) {
      reduce_mass_.val[i] = m_local_host(i);
    }
    return TaskStatus::complete;
  });

  auto start_reduce =
      tl.AddTask(parthenon::TaskQualifier::once_per_region, fill_reduce,
                 &AllReduce<std::vector<Real>>::StartReduce, &reduce_mass_, MPI_SUM);

  auto check_reduce = tl.AddTask(
      parthenon::TaskQualifier::once_per_region | parthenon::TaskQualifier::local_sync,
      start_reduce, &AllReduce<std::vector<Real>>::CheckReduce, &reduce_mass_);

  auto tables =
      tl.AddTask(check_reduce, &MonopoleGravitySolver::BuildRadialTables, this, md0);

  auto comp_grav =
      tl.AddTask(tables, &MonopoleGravitySolver::ComputeGravityVectorDirect, this, md0);

  return comp_grav;
}

MonopoleGravitySolver::MonopoleGravitySolver(ParameterInput *pin) : GravitySolver(pin) {
  nrbin_ = pin->GetOrAddInteger("gravity", "nrbin", -1);
  if (nrbin_ <= 0) {
    const int nx1 = pin->GetInteger("parthenon/mesh", "nx1");
    nrbin_ = nx1;
  }

  m_local_ = Kokkos::View<Real *>("m_local", nrbin_);
  m_enc_ = Kokkos::View<Real *>("m_enc", nrbin_ + 1);
  phi_rad_ = Kokkos::View<Real *>("phi_rad", nrbin_ + 1);

  Kokkos::deep_copy(m_local_, 0.0);
  Kokkos::deep_copy(m_enc_, 0.0);
  Kokkos::deep_copy(phi_rad_, 0.0);

  reduce_mass_.val.resize(nrbin_, 0.0);
}

TaskStatus MonopoleGravitySolver::ZeroRadialBins(std::shared_ptr<MeshData<Real>> &md) {
  Kokkos::deep_copy(m_local_, 0.0);
  Kokkos::deep_copy(m_enc_, 0.0);
  Kokkos::deep_copy(phi_rad_, 0.0);
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
  Kokkos::deep_copy(m_local_dev, 0.0);

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

        int bin = static_cast<int>(r / dr);
        if (bin < 0) bin = 0;
        if (bin >= nrbin_) bin = nrbin_ - 1;
        const Real dm = w(IDN, k, j, i) * coords.CellVolume(k, j, i);

        Kokkos::atomic_add(&m_local_dev(bin), dm);
      });

  Kokkos::deep_copy(m_local_, m_local_dev);
  return TaskStatus::complete;
}

TaskStatus MonopoleGravitySolver::BuildRadialTables(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Gravity");
  const Real rmax = pkg->Param<Real>("rmax");
  const Real gravity_g = pkg->Param<Real>("gravity_constant");
  const Real dr = rmax / nrbin_;
  const int ndim = pmb->pmy_mesh->ndim;

  auto m_enc_host = Kokkos::create_mirror_view(m_enc_);
  auto phi_rad_host = Kokkos::create_mirror_view(phi_rad_);

  m_enc_host(0) = 0.0;
  for (int i = 0; i < nrbin_; i++) {
    m_enc_host(i + 1) = m_enc_host(i) + reduce_mass_.val[i];
  }

  const auto radial_gravity = [gravity_g, ndim](const Real m, const Real r) {
    if (r <= 0.0) return 0.0;

    if (ndim == 1) {
      return -gravity_g * m;
    } else if (ndim == 2) {
      return -2.0 * gravity_g * m / r;
    } else {
      return -gravity_g * m / (r * r);
    }
  };

  const Real mtot = m_enc_host(nrbin_);
  phi_rad_host(nrbin_) = (rmax > 0.0 && ndim == 3) ? -gravity_g * mtot / rmax : 0.0;

  for (int i = nrbin_ - 1; i >= 0; i--) {
    const Real r_inner = i * dr;
    const Real r_outer = (i + 1) * dr;
    Real dphi = 0.0;
    if (ndim == 3) {
      const Real m_inner = m_enc_host(i);
      const Real shell_mass = m_enc_host(i + 1) - m_inner;
      if (r_inner == 0.0) {
        dphi = -gravity_g * shell_mass / (2.0 * r_outer);
      } else {
        const Real denom = r_outer * r_outer * r_outer -
                           r_inner * r_inner * r_inner;
        dphi = -gravity_g *
               (m_inner * (1.0 / r_inner - 1.0 / r_outer) +
                shell_mass / denom *
                    (0.5 * (r_outer * r_outer - r_inner * r_inner) +
                     r_inner * r_inner * r_inner *
                         (1.0 / r_outer - 1.0 / r_inner)));
      }
    } else {
      const Real r_mid = 0.5 * (r_inner + r_outer);
      const Real m_mid = 0.5 * (m_enc_host(i) + m_enc_host(i + 1));
      dphi = radial_gravity(m_mid, r_mid) * dr;
    }
    phi_rad_host(i) = phi_rad_host(i + 1) + dphi;
  }

  Kokkos::deep_copy(m_enc_, m_enc_host);
  Kokkos::deep_copy(phi_rad_, phi_rad_host);
  return TaskStatus::complete;
}

TaskStatus
MonopoleGravitySolver::ComputeGravityVectorDirect(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  auto phi_pack = md->PackVariables(std::vector<std::string>{"gravity.phi"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Gravity");
  const Real rmax = pkg->Param<Real>("rmax");
  const Real gravity_g = pkg->Param<Real>("gravity_constant");
  const Real dr = rmax / nrbin_;
  const auto m_enc = m_enc_;
  const auto phi_rad = phi_rad_;

  const int ndim = pmb->pmy_mesh->ndim;

  pmb->par_for(
      "UpdateGravityDirect", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real y = (ndim >= 2) ? coords.Xc<2>(j) : 0.0;
        const Real z = (ndim >= 3) ? coords.Xc<3>(k) : 0.0;

        const Real r2 = x * x + y * y + z * z;
        const Real r = std::sqrt(r2);

        auto &grav = grav_pack(b);
        auto &phi = phi_pack(b);

        if (r < 1e-10) {
          grav(0, k, j, i) = 0.0;
          grav(1, k, j, i) = 0.0;
          grav(2, k, j, i) = 0.0;
          phi(0, k, j, i) = phi_rad(0);
          return;
        }

        Real m_enclosed;
        Real phi_val;
        if (r >= rmax) {
          m_enclosed = m_enc(nrbin_);
          phi_val = (ndim == 3) ? -gravity_g * m_enclosed / r : phi_rad(nrbin_);
        } else {
          const Real rf = r / dr;
          int bin = static_cast<int>(rf);

          if (bin < 0) bin = 0;
          if (bin >= nrbin_) bin = nrbin_ - 1;

          const Real r_inner = bin * dr;
          const Real r_outer = (bin + 1) * dr;
          const Real shell_mass = m_enc(bin + 1) - m_enc(bin);
          Real mass_frac;
          if (ndim == 1) {
            mass_frac = (r - r_inner) / dr;
          } else if (ndim == 2) {
            const Real denom = r_outer * r_outer - r_inner * r_inner;
            mass_frac = (r * r - r_inner * r_inner) / denom;
          } else {
            const Real r_inner3 = r_inner * r_inner * r_inner;
            const Real r_outer3 = r_outer * r_outer * r_outer;
            mass_frac = (r * r * r - r_inner3) / (r_outer3 - r_inner3);
          }
          if (mass_frac < 0.0) mass_frac = 0.0;
          if (mass_frac > 1.0) mass_frac = 1.0;
          m_enclosed = m_enc(bin) + shell_mass * mass_frac;

          const Real frac = rf - bin;
          phi_val = (1.0 - frac) * phi_rad(bin) + frac * phi_rad(bin + 1);
        }

        Real g_r;
        if (ndim == 1) {
          g_r = -gravity_g * m_enclosed;
        } else if (ndim == 2) {
          g_r = -2.0 * gravity_g * m_enclosed / r;
        } else {
          g_r = -gravity_g * m_enclosed / r2;
        }

        const Real rinv = 1.0 / r;
        grav(0, k, j, i) = g_r * x * rinv;
        grav(1, k, j, i) = (ndim >= 2) ? g_r * y * rinv : 0.0;
        grav(2, k, j, i) = (ndim >= 3) ? g_r * z * rinv : 0.0;
        phi(0, k, j, i) = phi_val;
      });
  return TaskStatus::complete;
}

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_MONOPOLE_GRAVITY_HPP_
