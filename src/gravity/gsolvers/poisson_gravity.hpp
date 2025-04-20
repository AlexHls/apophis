#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"

namespace Apophis {

// Poisson gravity solver
struct PoissonGravitySolver : GravitySolver {
  explicit PoissonGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

  TaskStatus ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md);

  template <class in_t, class out_t>
  static parthenon::TaskStatus PhiMultiplyMatrix(std::shared_ptr<MeshData<Real>> &md);
};

TaskID PoissonGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                      const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  auto id2 = tl.AddTask(dep, &PoissonGravitySolver::ComputeGravityVector, this, md0);
  return id2;
}

TaskStatus
PoissonGravitySolver::ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Gravity");
  const auto gravity_g = pkg->Param<Real>("gravity_constant");

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        auto &grav = grav_pack(b);

        // Gravity of form G = -GM/r^2 * rhat

        Real r_sqr = SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k));

        grav(0, k, j, i) = -gravity_g * 1.0 / (r_sqr + 1.0e-10) *
                           (coords.Xc<1>(i) / abs(coords.Xc<1>(i) + 1.0e-10));
        grav(1, k, j, i) = -gravity_g * 1.0 / (r_sqr + 1.0e-10) *
                           (coords.Xc<2>(j) / abs(coords.Xc<2>(j) + 1.0e-10));
        grav(2, k, j, i) = -gravity_g * 1.0 / (r_sqr + 1.0e-10) *
                           (coords.Xc<3>(k) / abs(coords.Xc<3>(k) + 1.0e-10));
      });
  return TaskStatus::complete;
}

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_