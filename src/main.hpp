#ifndef MAIN_HPP_
#define MAIN_HPP_

#include "basic_types.hpp" // Real

enum {
  IDN = 0,
  IM1 = 1,
  IM2 = 2,
  IM3 = 3,
  IEN = 4,
  NHYDRO = 5,
};

enum {
  IV1 = 1,
  IV2 = 2,
  IV3 = 3,
  IPR = 4,
};

enum {
  LIFL = 0, // fuel
  LIDST = 1, // dist
};

enum class Fluid {
  undefined,
  euler,
};

enum class Reconstruction {
  undefined,
  dc,
  plm,
  ppm,
};

enum class RiemannSolver {
  undefined,
  hlle,
  hllc,
  lhllc,
};

enum class Integrator {
  undefined,
  rk1,
  rk2,
  rk3,
  vl2,
};

enum class Gravity {
  undefined,
  none,
  constant,
};

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

#endif // MAIN_HPP_
