#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

// Abstract interface for gravity object
struct GravitySolver {
  explicit GravitySolver(ParameterInput *pin) : pin_(pin) {}
  virtual ~GravitySolver() = default;
  virtual TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh,
                          const int partition) = 0;

 protected:
  ParameterInput *pin_;
};

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

VARIABLE(gravity, phi);
VARIABLE(gravity, rhs);
VARIABLE(gravity, D);

std::shared_ptr<parthenon::StateDescriptor> InitializeGravity(ParameterInput *pin);

TaskStatus ApplyGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt);

} // namespace Apophis

#endif // GRAVITY_GRAVITY_HPP_
