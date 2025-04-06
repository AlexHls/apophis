#ifndef BOUNDARIES_APOPHIS_BOUNDARIES_HPP
#define BOUNDARIES_APOPHIS_BOUNDARIES_HPP

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::BoundaryFunction;

namespace Boundaries {

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);
void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse);

void ProcessBoundaryConditions(parthenon::ParthenonManager &pman);

} // namespace Boundaries

#endif // BOUNDARIES_APOPHIS_BOUNDARIES_HPP
