#include "apophis_driver.hpp"
#include "burner/burner.hpp"
#include "hydro/hydro.hpp"
#include "levelset/lset.hpp"

#include <amr_criteria/refinement_package.hpp>
#include <parthenon/parthenon.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

using namespace parthenon::driver::prelude;

namespace Apophis {

parthenon::Packages_t
ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin) {
  parthenon::Packages_t packages;
  packages.Add(Apophis::InitializeHydro(pin.get()));
  return packages;
}

ApophisDriver::ApophisDriver(ParameterInput *pin, ApplicationInput *app_in,
                             Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  pin->CheckRequired("parthenon/time", "cfl");
}

TaskCollection ApophisDriver::MakeTaskCollection(BlockList_t &blocks,
                                                 int stage) {
  TaskCollection tc;
  const auto &stage_name = integrator->stage_name;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  TaskID none(0);

  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &async_region_1 =
      tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_1[i];

    auto &u0 = pmb->meshblock_data.Get();

    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);

      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            return TaskStatus::complete;
          },
          u0.get(), u1.get());
    }
  }
  const int num_partitions = pmesh->DefaultNumPartitions();

  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

    const auto flux_str =
        (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    auto send_flx =
        tl.AddTask(calc_flux, parthenon::LoadAndSendFluxCorrections, mu0);
    auto recv_flx =
        tl.AddTask(calc_flux, parthenon::ReceiveFluxCorrections, mu0);
    auto set_flx = tl.AddTask(recv_flx, parthenon::SetFluxCorrections, mu0);

    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);
    auto update = tl.AddTask(
        set_flx, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>,
        mu0.get(), mu1.get(), integrator->gam0[stage - 1],
        integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    const auto nlset = hydro_pkg->Param<int>("nlset");
    for (int lset_id = 0; lset_id < nlset; lset_id++) {
      auto reinit = tl.AddTask(update, ReinitializeLset, mu0, lset_id);
      auto burn = tl.AddTask(reinit, Burn, mu0, lset_id,
                             integrator->beta[stage - 1] * integrator->dt);
    }

    parthenon::AddBoundaryExchangeTasks(update, tl, mu0, pmesh->multilevel);
  }

  TaskRegion &async_region_3 =
      tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto prolongBound = none;

    auto set_bc =
        tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, u0);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived = tl.AddTask(
        none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>,
          mu0.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 =
        tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine = tl.AddTask(
          none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}

} // namespace Apophis
