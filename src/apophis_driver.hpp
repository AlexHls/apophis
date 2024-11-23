#ifndef APOPHIS_DRIVER_HPP_
#define APOPHIS_DRIVER_HPP_

#include "main.hpp"

// Parthenon headers
#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

  parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

  class ApophisDriver : public parthenon::MultiStageDriver{
    public:
      ApophisDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);

      auto MakeTaskCollection(BlockList_t &blocks, int stage) -> TaskCollection;
  };

} // namespace Apophis

#endif // APHOPHIS_DRIVER_HPP_
