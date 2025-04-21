#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"

namespace Apophis {

// Poisson gravity solver
struct PoissonGravitySolver : GravitySolver {
  explicit PoissonGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ComputePhi(std::shared_ptr<MeshData<Real>> &md);
  TaskStatus ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md);

  template <class in_t, class out_t>
  static parthenon::TaskStatus PhiMultiplyMatrix(std::shared_ptr<MeshData<Real>> &md);
};

TaskID PoissonGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                      const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto id1 = tl.AddTask(dep, &PoissonGravitySolver::ComputePhi, this, md0);
  auto id2 = tl.AddTask(id1, &PoissonGravitySolver::ComputeGravityVector, this, md0);
  return id2;
}

TaskStatus PoissonGravitySolver::ComputePhi(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto phi_pack = md->PackVariables(std::vector<std::string>{"potential"});
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "UpdatePhi", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &phi = phi_pack(b);

        Real r_sqr = SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k));

        phi(0, k, j, i) = -0.1 / (r_sqr + 1.0e-1);
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