#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_

#include "parthenon/driver.hpp"

namespace sod {
  using namespace parthenon::driver::prelude;

  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace sod

#endif // PGEN_PGEN_HPP_
