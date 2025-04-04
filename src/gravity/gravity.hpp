#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeGravity(ParameterInput *pin);

} // namespace Apophis

#endif // GRAVITY_GRAVITY_HPP_
