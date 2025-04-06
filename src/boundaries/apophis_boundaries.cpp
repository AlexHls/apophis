#include "apophis_boundaries.hpp"
#include "../main.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_manager.hpp"

using namespace parthenon;
using namespace parthenon::BoundaryFunction;

namespace Boundaries {

// Copied from parthenon with slight modifications to account for scalar
// velocities
enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };
template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE, class... var_ts>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
               TopologicalElement el, Real val) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR,
                "DIR must be X[123]DIR");

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);
  constexpr int REFLECT_IDX = (X1 ? IV1 : (X2 ? IV2 : IV3));

  static auto descriptors =
      parthenon::BoundaryFunction::impl::GetPackDescriptorMap<var_ts...>(rc);
  for (auto fine : {false, true}) {
    auto q = descriptors[parthenon::BoundaryFunction::impl::desc_key_t{
                             coarse, fine, GetTopologicalType(el)}]
                 .GetPack(rc.get());
    const int b = 0;
    const int lstart = q.GetLowerBoundHost(b);
    const int lend = q.GetUpperBoundHost(b);
    if (lend < lstart)
      return;
    auto nb = IndexRange{lstart, lend};

    MeshBlock *pmb = rc->GetBlockPointer();
    const auto &bounds = fine ? (coarse ? pmb->cellbounds : pmb->f_cellbounds)
                              : (coarse ? pmb->c_cellbounds : pmb->cellbounds);

    const auto &range =
        X1 ? bounds.GetBoundsI(IndexDomain::interior, el)
           : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el)
                 : bounds.GetBoundsK(IndexDomain::interior, el));
    const int ref = INNER ? range.s : range.e;

    std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
    label += (INNER ? "Inner" : "Outer");
    label += "X" + std::to_string(DIR);

    constexpr IndexDomain domain =
        INNER ? (X1 ? IndexDomain::inner_x1
                    : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
              : (X1 ? IndexDomain::outer_x1
                    : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

    // used for reflections
    const int offset = 2 * ref + (INNER ? -1 : 1);
    pmb->par_for_bndry(
        PARTHENON_AUTO_LABEL, nb, domain, el, coarse, fine,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          if (TYPE == BCType::Reflect) {
            const bool reflect = (l == REFLECT_IDX);
            q(b, el, l, k, j, i) = (reflect ? -1.0 : 1.0) *
                                   q(b, el, l, X3 ? offset - k : k,
                                     X2 ? offset - j : j, X1 ? offset - i : i);
          } else {
            q(b, el, l, k, j, i) =
                q(b, el, l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
          }
        });
  }
}

template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE, class... var_ts>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
               Real val = 0.0) {
  using TE = TopologicalElement;
  for (auto el :
       {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<DIR, SIDE, TYPE, var_ts...>(rc, coarse, el, val);
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X1DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X2DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Inner, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  GenericBC<X3DIR, BCSide::Outer, BCType::Reflect, variable_names::any>(rc,
                                                                        coarse);
}

void ProcessBoundaryConditions(ParthenonManager &pman) {
  using namespace BoundaryFunction;
  const std::string REFLECTING = "reflecting";
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1, REFLECTING,
                                            &ReflectInnerX1);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1, REFLECTING,
                                            &ReflectOuterX1);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x2, REFLECTING,
                                            &ReflectInnerX2);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x2, REFLECTING,
                                            &ReflectOuterX2);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x3, REFLECTING,
                                            &ReflectInnerX3);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x3, REFLECTING,
                                            &ReflectOuterX3);
}

} // namespace Boundaries
