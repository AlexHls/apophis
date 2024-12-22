#ifndef BURNER_BURNER_HPP_
#define BURNER_BURNER_HPP_

#define DENS_THRESH 1.0e5

#include "parthenon/parthenon.hpp"

using namespace parthenon::package::prelude;

namespace Apophis {

  TaskStatus Burn(std::shared_ptr<MeshData<Real>> &md, const int nlset,
                  const Real dt);

} // namespace Apophis

#endif // BURNER_BURNER_HPP_
