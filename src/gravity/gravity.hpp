#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeGravity(ParameterInput *pin);

TaskStatus UpdateGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt);

} // namespace Apophis

#endif // GRAVITY_GRAVITY_HPP_
