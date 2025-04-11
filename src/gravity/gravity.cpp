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

  // Add the gravity field
  std::string field_name = "gravity";
  std::vector<std::string> labels(3);
  labels[0] = "gx";
  labels[1] = "gy";
  labels[2] = "gz";
  auto m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::Derived,
       parthenon::Metadata::Intensive, parthenon::Metadata::FillGhost},
      std::vector<int>({3}), labels);

  pkg->AddField(field_name, m);

  // Add gravity functions
  std::map<std::tuple<Fluid, Gravity>, GravityFun_t *> gravity_functions{};
  add_gravity_fun<Fluid::euler, Gravity::none>(gravity_functions);
  add_gravity_fun<Fluid::euler, Gravity::constant>(gravity_functions);

  GravityFun_t *gravity_fun = nullptr;
  gravity_fun = gravity_functions.at(std::make_tuple(Fluid::euler, gravity));

  pkg->AddParam<GravityFun_t *>("gravity_fun", gravity_fun);

  return pkg;
}

template <>
TaskStatus CalculateGravity<Fluid::euler, Gravity::none>(
    std::shared_ptr<MeshData<Real>> &md) {
  return TaskStatus::complete;
}

template <>
TaskStatus CalculateGravity<Fluid::euler, Gravity::constant>(
    std::shared_ptr<MeshData<Real>> &md) {
  // For constant gravity, also no update is needed
  return TaskStatus::complete;
}

TaskStatus ApplyGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto grav_pack = md->PackVariables(std::vector<std::string>{"gravity"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  pmb->par_for(
      "UpdateGravity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto w = prim_pack(b);
        const auto &grav = grav_pack(b);

        cons(IM1, k, j, i) += grav(0, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IM2, k, j, i) += grav(1, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IM3, k, j, i) += grav(2, k, j, i) * w(IDN, k, j, i) * dt;
        cons(IEN, k, j, i) += (grav(0, k, j, i) * w(IV1, k, j, i) +
                               grav(1, k, j, i) * w(IV2, k, j, i) +
                               grav(2, k, j, i) * w(IV3, k, j, i)) *
                              w(IDN, k, j, i) * dt;
      });

  return TaskStatus::complete;
}

} // namespace Apophis
