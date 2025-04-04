#ifndef LEVELSET_LSET_HPP_
#define LEVELSET_LSET_HPP_

#define LSET_BORDER 4
#define BELT 4

#include "parthenon/parthenon.hpp"

using namespace parthenon::package::prelude;

namespace Apophis {

TaskStatus ReinitializeLset(std::shared_ptr<MeshData<Real>> &md, const int lset_id);

} // namespace Apophis

#endif // LEVELSET_LSET_HPP_
