#include "gravity.hpp"
#include "../main.hpp"
#include "KokkosCore_Config_SetupBackend.hpp"
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

TaskStatus UpdateGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto w = prim_pack(b);
        cons(IM2, k, j, i) -= 0.1 * w(IDN, k, j, i) * dt;
        cons(IEN, k, j, i) -= 0.1 * w(IDN, k, j, i) * w(IV2, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

} // namespace Apophis
