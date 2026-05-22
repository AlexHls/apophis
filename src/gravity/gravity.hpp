#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../main.hpp"

#include "parthenon/parthenon.hpp"

using namespace parthenon::driver::prelude;

namespace Apophis {

// Abstract interface for gravity object
struct GravitySolver {
  explicit GravitySolver(ParameterInput *pin) : pin_(pin) {}
  virtual ~GravitySolver() = default;
  virtual TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh,
                          const int partition) = 0;
  virtual TaskID PreComputeTasks(TaskList &tl, TaskID dependence, Mesh *pmesh,
                                 const int partition) = 0;

 protected:
  ParameterInput *pin_;
};

#define VARIABLE(ns, varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

VARIABLE(gravity, phi);
VARIABLE(gravity, rhs);
VARIABLE(gravity, D);

struct any_gravity : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_gravity(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "gravity[.].*"; }
};

template <class Coeffs>
KOKKOS_INLINE_FUNCTION Real MultipolePotential(const Coeffs &coeffs, const Real x,
                                               const Real y, const Real z) {
  const Real r2 = x * x + y * y + z * z;
  if (r2 == 0.0) return 0.0;

  const Real r = sqrt(r2);
  const Real r3 = r * r2;
  const Real r5 = r3 * r2;

  return coeffs(0) / r + (coeffs(1) * y + coeffs(2) * z + coeffs(3) * x) / r3 +
         (coeffs(4) * x * y + coeffs(5) * y * z + coeffs(6) * (3.0 * z * z - r2) +
          coeffs(7) * z * x + coeffs(8) * 0.5 * (x * x - y * y)) /
             r5;
}

std::shared_ptr<parthenon::StateDescriptor> InitializeGravity(ParameterInput *pin);

TaskStatus ApplyGravity(std::shared_ptr<MeshData<Real>> &md, const Real dt);

} // namespace Apophis

#endif // GRAVITY_GRAVITY_HPP_
