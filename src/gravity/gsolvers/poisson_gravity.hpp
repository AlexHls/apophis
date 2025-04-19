#ifndef GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"

namespace Apophis {

// Poisson gravity solver
struct PoissonGravitySolver : GravitySolver {
  explicit PoissonGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}
  TaskID AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh, const int partition) override;

 private:
  // 1) build the RHS = 4πGρ and solve for φ
  TaskStatus BuildPoissonRHS(std::shared_ptr<MeshData<Real>> &md);
  // 2) compute gravity = –∇φ
  TaskStatus ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md);
};

TaskID PoissonGravitySolver::AddTasks(TaskList &tl, TaskID dep, Mesh *pmesh,
                                      const int partition) {
  auto &md0 = pmesh->mesh_data.GetOrAdd("base", partition);

  // 1) build & solve φ
  auto id1 = tl.AddTask(dep, &PoissonGravitySolver::BuildPoissonRHS, this, md0);
  // 2) compute gravity = –∇φ
  auto id2 = tl.AddTask(id1, &PoissonGravitySolver::ComputeGravityVector, this, md0);
  return id2;
}

TaskStatus PoissonGravitySolver::BuildPoissonRHS(std::shared_ptr<MeshData<Real>> &md) {
  return TaskStatus::complete;
}

TaskStatus
PoissonGravitySolver::ComputeGravityVector(std::shared_ptr<MeshData<Real>> &md) {
  return TaskStatus::complete;
}

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_POISSON_GRAVITY_HPP_