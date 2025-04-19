#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_

#include "parthenon/driver.hpp"

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace sod

namespace linear_wave {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace blast

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace kh

namespace burn_tube {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace burn_tube

namespace rt {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin);
} // namespace rt
#endif // PGEN_PGEN_HPP_
