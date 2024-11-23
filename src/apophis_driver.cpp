#include "apophis_driver.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

parthenon::Packages_t
ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin) {
  parthenon::Packages_t packages;
  return packages;
}

ApophisDriver::ApophisDriver(ParameterInput *pin, ApplicationInput *app_in,
                             Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {}

TaskCollection ApophisDriver::MakeTaskCollection(BlockList_t &blocks,
                                                 int stage) {
  TaskCollection tc;

  return tc;
}

} // namespace Apophis
