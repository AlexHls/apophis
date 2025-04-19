#ifndef GRAVITY_GSOLVERS_CONSTANT_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_CONSTANT_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"

namespace Apophis {

// Constant gravity solver
struct ConstantGravitySolver : GravitySolver {

  explicit ConstantGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}

  TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh) override {
    // Nothing to do here, gravity is constant
    // and does not need to be updated
    return dependence;
  }
};

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_CONSTANT_GRAVITY_HPP_