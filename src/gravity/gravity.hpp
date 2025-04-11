#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeGravity(ParameterInput *pin);

template <Fluid fluid, Gravity gravity>
TaskStatus CalculateGravity(std::shared_ptr<MeshData<Real>> &md);
using GravityFun_t = decltype(CalculateGravity<Fluid::euler, Gravity::none>);
using GravityFunKey_t = std::tuple<Fluid, Gravity>;

// Add gravity function pointer to map containing all compiled in gravity functions
template <Fluid fluid, Gravity gravity>
void add_gravity_fun(std::map<GravityFunKey_t, GravityFun_t *> &gravity_functions) {
  gravity_functions[std::make_tuple(fluid, gravity)] =
      Apophis::CalculateGravity<fluid, gravity>;
}

template <Fluid fluid, Gravity gravity>
TaskStatus CalculateGravity(std::shared_ptr<MeshData<Real>> &md);

TaskStatus ApplyGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt);

} // namespace Apophis

#endif // GRAVITY_GRAVITY_HPP_
