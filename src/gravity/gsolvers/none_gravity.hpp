#ifndef GRAVITY_GSOLVERS_NONE_GRAVITY_HPP_
#define GRAVITY_GSOLVERS_NONE_GRAVITY_HPP_

#include "../gravity.hpp"

#include "../../main.hpp"

namespace Apophis {

// None gravity solver
struct NoneGravitySolver : GravitySolver {

  explicit NoneGravitySolver(ParameterInput *pin) : GravitySolver(pin) {}

  TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh) override {
    // No tasks to add for None gravity
    return dependence;
  }
};

} // namespace Apophis

#endif // GRAVITY_GSOLVERS_NONE_GRAVITY_HPP_