#include "gravity.hpp"
#include "../main.hpp"
#include "interface/state_descriptor.hpp"

#include "parthenon/package.hpp"
#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeGravity(ParameterInput *pin) {
  auto pkg = std::make_shared<parthenon::StateDescriptor>("Gravity");
  const auto gravity_str = pin->GetOrAddString("gravity", "type", "none");
  auto gravity = Gravity::undefined;

  if (gravity_str == "none") {
    gravity = Gravity::none;
  } else if (gravity_str == "constant") {
    gravity = Gravity::constant;
  } else {
    PARTHENON_FAIL("[Apophis]: Gravity not recognized. Exiting.");
  }

  pkg->AddParam<>("gravity", gravity);
  return pkg;
}

} // namespace Apophis
