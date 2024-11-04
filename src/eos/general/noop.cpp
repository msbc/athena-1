//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file noop.cpp
//! \brief Implements no-op versions of the general eos functions

// C headers
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../eos.hpp"

void print_backtrace() {
    void *buffer[10];
    int nptrs = backtrace(buffer, 10);  // Get the backtrace
    char **symbols = backtrace_symbols(buffer, nptrs);  // Get symbols for the stack frames

    fprintf(stderr, "Backtrace:\n");
    for (int i = 0; i < nptrs; i++) {
        fprintf(stderr, "%s\n", symbols[i]);
    }

    free(symbols);  // Free the memory allocated by backtrace_symbols
}

Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::PresFromRhoEg" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::EgasFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::AsqFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}

Real EquationOfState::TgasFromRhoEg(Real rho, Real egas) {
  print_backtrace();
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::TgasFromRhoEg" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}

Real EquationOfState::TgasFromRhoP(Real rho, Real pres) {
  std::stringstream msg;
  print_backtrace();
  msg << "### FATAL ERROR in EquationOfState::TgasFromRhoP" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}

Real EquationOfState::EgasFromRhoT(Real rho, Real temp) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::EgasFromRhoT" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}
Real EquationOfState::dTdeFromRhoTgas(Real rho, Real temp) {
  std::stringstream msg;
  msg << "### FATAL ERROR in EquationOfState::dTdeFromRhoTgas" << std::endl
      << "Function should not be called with current configuration." << std::endl;
  ATHENA_ERROR(msg);
  return -1.0;
}