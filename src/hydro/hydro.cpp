#include "hydro.hpp"
#include "../main.hpp"
#include "../recon/dc_simple.hpp"
#include "../recon/plm_simple.hpp"
#include "rsolvers/hydro_hllc.hpp"

#include "basic_types.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon/parthenon.hpp"
#include "singularity-eos/eos/eos.hpp"
#include <parthenon/package.hpp>

using parthenon::DevExecSpace;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

using namespace parthenon::driver::prelude;

namespace Apophis {

std::shared_ptr<parthenon::StateDescriptor>
InitializeHydro(ParameterInput *pin) {
  auto pkg = std::make_shared<parthenon::StateDescriptor>("Hydro");
  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);

  pkg->AddParam<>("cfl", cfl);

  // Fluid
  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "euler");
  auto fluid = Fluid::undefined;
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = GetNVars<Fluid::euler>();
  } else {
    PARTHENON_FAIL("[Apophis]: Fluid not recognized. Exiting.");
  }

  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);

  // Reconstruction
  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_ghost = 3;
  auto recon = Reconstruction::undefined;

  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    recon_need_ghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    recon_need_ghost = 2;
  } else {
    PARTHENON_FAIL("[Apophis]: Reconstruction not recognized. Exiting.");
  }

  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < recon_need_ghost) {
    PARTHENON_FAIL(
        "[Apophis]: Not enough ghost zones for reconstruction. Exiting.");
  }

  pkg->AddParam<>("reconstruction", recon);

  // Equation of state
  const auto eos_str = pin->GetOrAddString("eos", "type", "ideal");
  if (eos_str == "ideal") {
    const Real gm1_in = pin->GetOrAddReal("eos", "gm1", 0.6666667);
    const Real cv_in = pin->GetOrAddReal("eos", "cv", 1.5);
    singularity::EOS eos = singularity::IdealGas(gm1_in, cv_in);
    singularity::EOS eos_device = eos.GetOnDevice();
    pkg->AddParam<>("eos", eos_device);
    pkg->AddParam<>("eos_host", eos);
    pkg->AddParam<>("update_lambda", false);
  } else {
    PARTHENON_FAIL("[Apophis]: EOS not recognized. Exiting.");
  }

  pkg->FillDerivedMesh = ConsToPrim<singularity::EOS>;
  pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::euler>;

  // Riemann solver
  const auto riemann_str = pin->GetString("hydro", "riemann");
  auto riemann = RiemannSolver::undefined;

  if (riemann_str == "hllc") {
    riemann = RiemannSolver::hllc;
  } else {
    PARTHENON_FAIL("[Apophis]: Riemann solver not recognized. Exiting.");
  }

  pkg->AddParam<>("riemann_solver", riemann);

  // Add flux functions
  std::map<std::tuple<Fluid, Reconstruction, RiemannSolver>, FluxFun_t *>
      flux_functions{};
  add_flux_fun<Fluid::euler, Reconstruction::dc, RiemannSolver::hllc>(
      flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hllc>(
      flux_functions);

  FluxFun_t *flux_other_stage = nullptr;
  flux_other_stage = flux_functions.at(std::make_tuple(fluid, recon, riemann));

  // Integrator
  const auto integrator_str = pin->GetString("parthenon/time", "integrator");
  auto integrator = Integrator::undefined;
  FluxFun_t *flux_first_stage = flux_other_stage;

  if (integrator_str == "rk1") {
    integrator = Integrator::rk1;
  } else if (integrator_str == "rk2") {
    integrator = Integrator::rk2;
  } else if (integrator_str == "rk3") {
    integrator = Integrator::rk3;
  } else {
    PARTHENON_FAIL("[Apophis]: Integrator not recognized. Exiting.");
  }

  pkg->AddParam<>("integrator", integrator);
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  // Add fields
  std::string field_name = "cons";
  std::vector<std::string> cons_labels(NHYDRO);
  cons_labels[IDN] = "Density";
  cons_labels[IM1] = "MomentumDensity1";
  cons_labels[IM2] = "MomentumDensity2";
  cons_labels[IM3] = "MomentumDensity3";
  cons_labels[IEN] = "TotalEnergyDensity";
  parthenon::Metadata m(
      {parthenon::Metadata::Cell, parthenon::Metadata::Independent,
       parthenon::Metadata::FillGhost, parthenon::Metadata::WithFluxes},
      std::vector<int>({NHYDRO}), cons_labels);
  pkg->AddField(field_name, m);

  field_name = "prim";
  std::vector<std::string> prim_labels(NHYDRO);
  prim_labels[IDN] = "Density";
  prim_labels[IV1] = "Velocity1";
  prim_labels[IV2] = "Velocity2";
  prim_labels[IV3] = "Velocity3";
  prim_labels[IPR] = "Pressure";
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::Derived},
      std::vector<int>({NHYDRO}), prim_labels);
  pkg->AddField(field_name, m);

  field_name = "eos_lambda";
  std::vector<std::string> eos_lambda_labels(3);
  eos_lambda_labels[0] = "Abar";
  eos_lambda_labels[1] = "Zbar";
  eos_lambda_labels[2] = "lT";
  m = parthenon::Metadata({parthenon::Metadata::Cell,
                           parthenon::Metadata::Derived,
                           parthenon::Metadata::Intensive},
                          std::vector<int>({3}), eos_lambda_labels);
  pkg->AddField(field_name, m);

  field_name = "gamma";
  std::vector<std::string> gamma_labels(2);
  gamma_labels[0] = "gamma_c";
  gamma_labels[1] = "gamma_e";
  m = parthenon::Metadata(
      {parthenon::Metadata::Cell, parthenon::Metadata::Derived},
      std::vector<int>({2}), gamma_labels);
  pkg->AddField(field_name, m);

  // Misc
  Real dfloor =
      pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * float_min));
  Real pfloor =
      pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024 * float_min));

  pkg->AddParam<Real>("hydro/density_floor", dfloor);
  pkg->AddParam<Real>("hydro/pressure_floor", pfloor);

  auto scratch_level = pin->GetOrAddInteger("hdyro", "scratch_level", 0);
  pkg->AddParam<int>("scratch_level", scratch_level);

  return pkg;
}

template <Fluid fluid> Real EstimateTimestep(MeshData<Real> *md) {
  static constexpr Real C_LIGHT = 2.99792458e10;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &eos_lambda_pack =
      md->PackVariables(std::vector<std::string>{"eos_lambda"});
  const auto &eos_ = hydro_pkg->Param<singularity::EOS>("eos");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  const auto ndim_ = prim_pack.GetNdim();
  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &cons = cons_pack(b);
        auto &eos_lambda = eos_lambda_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        // nvcc thing
        const auto &ndim = ndim_;
        const auto &eos = eos_;

        Real w[(NHYDRO)];
        w[IDN] = prim(IDN, k, j, i);
        w[IV1] = prim(IV1, k, j, i);
        w[IV2] = prim(IV2, k, j, i);
        w[IV3] = prim(IV3, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real lambda_max_x, lambda_max_y, lambda_max_z;
        Real e_internal =
            (cons(IEN, k, j, i) -
             0.5 * w[IDN] * (SQR(w[IV1]) + SQR(w[IV2]) + SQR(w[IV3]))) /
            w[IDN];

        Real &abar = eos_lambda(0, k, j, i);
        Real &zbar = eos_lambda(1, k, j, i);
        Real &lT = eos_lambda(2, k, j, i);
        Real lambda[3] = {abar, zbar, lT};
        Real bulkmod = eos.BulkModulusFromDensityInternalEnergy(
            w[IDN], e_internal, lambda);

        const Real gamma_c = bulkmod / w[IPR];
        lambda_max_x =
            C_LIGHT * std::sqrt(gamma_c / (1 + (e_internal + SQR(C_LIGHT)) *
                                                   w[IDN] / w[IPR]));
        lambda_max_y = lambda_max_x;
        lambda_max_z = lambda_max_x;

        min_dt = fmin(min_dt,
                      coords.Dxc<1>(k, j, i) / (fabs(w[IV1]) + lambda_max_x));
        if (ndim > 1) {
          min_dt = fmin(min_dt,
                        coords.Dxc<2>(k, j, i) / (fabs(w[IV2]) + lambda_max_y));
        }
        if (ndim > 2) {
          min_dt = fmin(min_dt,
                        coords.Dxc<3>(k, j, i) / (fabs(w[IV3]) + lambda_max_z));
        }
      },
      Kokkos::Min<Real>(min_dt_hyperbolic));

  return cfl_hyp * min_dt_hyperbolic;
}

template <class T> void ConsToPrim(MeshData<Real> *md) {
  const auto &eos_ =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<T>(
          "eos");
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto gamma_pack = md->PackVariables(std::vector<std::string>{"gamma"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto density_floor_ =
      pmb->packages.Get("Hydro")->Param<Real>("hydro/density_floor");
  auto pressure_floor_ =
      pmb->packages.Get("Hydro")->Param<Real>("hydro/pressure_floor");
  auto eos_lambda_pack =
      md->PackVariables(std::vector<std::string>{"eos_lambda"});
  bool update_lambda = pmb->packages.Get("Hydro")->Param<bool>("update_lambda");

  // Temperature limits for root finding & initial guess
  static constexpr int ilTMin_ = 3;
  static constexpr int ilTMax_ = 13;
  static constexpr Real lTMin = ilTMin_;
  static constexpr Real lTMax = ilTMax_;

  pmb->par_for(
      "ConservedToPrimitive", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        auto &gamma = gamma_pack(b);
        // nvcc thing
        const auto &eos = eos_;

        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        Real &w_d = prim(IDN, k, j, i);
        Real &w_vx = prim(IV1, k, j, i);
        Real &w_vy = prim(IV2, k, j, i);
        Real &w_vz = prim(IV3, k, j, i);
        Real &w_p = prim(IPR, k, j, i);

        Real &gamma_c = gamma(0, k, j, i);
        Real &gamma_e = gamma(1, k, j, i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ? u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0 / u_d;
        w_vx = u_m1 * di;
        w_vy = u_m2 * di;
        w_vz = u_m3 * di;

        Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));

        Real temp;

        auto &eos_lambda = eos_lambda_pack(b);
        Real &abar = eos_lambda(0, k, j, i);
        Real &zbar = eos_lambda(1, k, j, i);
        Real &lT = eos_lambda(2, k, j, i);
        if (lT == 0.0) {
          lT = 7.0;
        }
        if (update_lambda) {

          Real ab = 0.0;
          Real zb = 0.0;

          Real lambda_tmp[3] = {abar, zbar, lT};

          temp = eos.TemperatureFromDensityInternalEnergy(
              u_d, (u_e - e_k) / u_d, lambda_tmp);
          lT = std::log10(temp);
          // If initial guess is outside of range, set it to default value
          if ((lT < lTMin) || (lT > lTMax)) {
            lT = 7;
          }
        }

        Real lambda[3] = {abar, zbar, lT};
        Real gm1 = eos.GruneisenParamFromDensityInternalEnergy(
            u_d, (u_e - e_k) / u_d, lambda);
        w_p = gm1 * (u_e - e_k);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k);
        w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;

        gamma_c = eos.BulkModulusFromDensityInternalEnergy(
                      u_d, (u_e - e_k) / u_d, lambda) /
                  w_p;
        gamma_e = eos.GruneisenParamFromDensityInternalEnergy(
                      u_d, (u_e - e_k) / u_d, lambda) +
                  1.0;
      });
}

template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (pmb->block_size.nx(X2DIR) > 1) {
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    else // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
  }

  std::vector<parthenon::MetadataFlag> flags_ind(
      {parthenon::Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");

  const auto &eos = pkg->Param<singularity::EOS>("eos");

  auto num_scratch_vars = nhydro;
  auto prim_list = std::vector<std::string>({"prim"});
  auto gamma_list = std::vector<std::string>({"gamma"});

  auto const &prim_pack = md->PackVariables(prim_list);
  auto const &gamma_pack = md->PackVariables(gamma_list);

  const auto &eos_lambda_pack =
      md->PackVariables(std::vector<std::string>{"eos_lambda"});

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 4;

  auto riemann = Riemann<fluid, rsolver>();

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "x1 flux", DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, kl, ku,
      jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                    const int j) {
        const auto &coords = cons_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const auto &gamma = gamma_pack(b);
        auto &eos_lambda = eos_lambda_pack(b);
        auto &cons = cons_pack(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level), 2,
                                          nx1);
        parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level), 2,
                                          nx1);
        // get reconstructed state on faces
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, prim, wl,
                                  wr);
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, gamma, ifl,
                                  ifr);

        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        riemann.Solve(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, ifl, ifr,
                      eos, eos_lambda);
      });
  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 6;
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x2 flux", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, kl,
        ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          const auto &coords = cons_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto &gamma = gamma_pack(b);
          auto &eos_lambda = eos_lambda_pack(b);
          auto &cons = cons_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level),
                                            2, nx1);
          parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level),
                                            2, nx1);
          parthenon::ScratchPad2D<Real> iflb(member.team_scratch(scratch_level),
                                             2, nx1);
          for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, prim, wlb, wr);
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, gamma, iflb, ifr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV2, wl, wr, cons, ifl, ifr,
                            eos, eos_lambda);
              member.team_barrier();
            }

            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
            auto *tmpf = ifl.data();
            ifl.assign_data(iflb.data());
            iflb.assign_data(tmpf);
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 flux", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, jl,
        ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
          const auto &coords = cons_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto &gamma = gamma_pack(b);
          auto &eos_lambda = eos_lambda_pack(b);
          auto &cons = cons_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level),
                                            2, nx1);
          parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level),
                                            2, nx1);
          parthenon::ScratchPad2D<Real> iflb(member.team_scratch(scratch_level),
                                             2, nx1);
          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, prim, wlb, wr);
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, gamma, iflb, ifr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV3, wl, wr, cons, ifl, ifr,
                            eos, eos_lambda);
              member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
            auto *tmpf = ifl.data();
            ifl.assign_data(iflb.data());
            iflb.assign_data(tmpf);
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace Apophis
