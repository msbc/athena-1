//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal.cpp
//! \brief implements ideal EOS in general EOS framework, mostly for debuging
//======================================================================================

// C headers

// C++ headers
#include <sstream>

// Athena++ headers
#include "../eos.hpp"
namespace {
  Real tnorm = 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.) * egas;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  return pres / (gamma_ - 1.);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  return gamma_ * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  if (NR_RADIATION_ENABLED | IM_RADIATION_ENABLED) {
    if (pin->DoesParameterExist("radiation", "prat")) {
      tnorm = std::pow(1.0 + pin->GetReal("radiation", "prat"), -0.25);
    } else if (pin->DoesParameterExist("radiation", "T_unit")) {
      tnorm = 1.0 / pin->GetReal("radiation", "T_unit");
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "No radiation Prat or T_unit found in input file." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  return;
}

Real EquationOfState::TgasFromRhoEg(Real rho, Real egas) {
  return (gamma_ - 1.) * egas / rho;
}
Real EquationOfState::TgasFromRhoP(Real rho, Real pres) {
  return pres / rho;
}
Real EquationOfState::EgasFromRhoT(Real rho, Real temp) {
  return temp * rho / (gamma_ - 1.);
}
Real EquationOfState::dTdeFromRhoTgas(Real rho, Real temp) {
  return (gamma_ - 1.) / rho;
}