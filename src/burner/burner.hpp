#ifndef BURNER_BURNER_HPP_
#define BURNER_BURNER_HPP_

#define DENS_THRESH 1.0e5

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::package::prelude;

namespace Apophis {

TaskStatus Burn(std::shared_ptr<MeshData<Real>> &md, const int nlset, const Real dt);

template <Fluid fluid>
constexpr Real GetEBind(int i);

template <>
constexpr Real GetEBind<Fluid::euler>(int i) {
  switch (i) {
  case 0:
    return 6.8266e+18;
  case 1:
    return 7.41121e+18;
  case 2:
    return 7.69691e+18;
  case 3:
    return 7.74994e+18;
  case 4:
    return 8.17906e+18;
  case 5:
    return 8.34e+18;
  default:
    return 0.0;
  }
}

} // namespace Apophis

#endif // BURNER_BURNER_HPP_
