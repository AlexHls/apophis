#ifndef RSOLVERS_RSOLVERS_HPP_
#define RSOLVERS_RSOLVERS_HPP_

#include "../../main.hpp"
#include <singularity-eos/eos/eos_ideal.hpp>
#include <singularity-eos/eos/eos_helmholtz.hpp>
#include <singularity-eos/eos/eos_variant.hpp>

using EOS_t = singularity::Variant<singularity::IdealGas, singularity::Helmholtz>;

// First declare general template
template <Fluid fluid, RiemannSolver rsolver>
struct Riemann;

#endif // RSOLVERS_RSOLVERS_HPP_
