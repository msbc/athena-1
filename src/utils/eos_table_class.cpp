//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_table_class.cpp
//! \brief Implements class EosTable for an EOS lookup table
//========================================================================================

// C headers

// C++ headers
#include <cmath>   // sqrt()
#include <fstream>
#include <iostream> // ifstream
#include <sstream>
#include <stdexcept> // std::invalid_argument
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../inputs/ascii_table_reader.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../parameter_input.hpp"
#include "interp_table.hpp"

// Order of datafields for HDF5 EOS tables
const char *var_names[] = {"p/e(e/rho,rho)", "e/p(p/rho,rho)", "asq*rho/p(p/rho,rho)"};

//----------------------------------------------------------------------------------------
//! \fn void ReadBinaryTable(std::string fn, EosTable *peos_table)
//! \brief Read data from binary EOS table and initialize interpolated table.

void ReadBinaryTable(std::string fn, EosTable *peos_table) {
  std::ifstream eos_file(fn.c_str(), std::ios::binary);
  if (eos_file.is_open()) {
    eos_file.seekg(0, std::ios::beg);
    eos_file.read(reinterpret_cast<char*>(&peos_table->nVar), sizeof(peos_table->nVar));
    eos_file.read(reinterpret_cast<char*>(&peos_table->nEgas), sizeof(peos_table->nEgas));
    eos_file.read(reinterpret_cast<char*>(&peos_table->nRho), sizeof(peos_table->nRho));
    eos_file.read(reinterpret_cast<char*>(&peos_table->logEgasMin),
                  sizeof(peos_table->logEgasMin));
    eos_file.read(reinterpret_cast<char*>(&peos_table->logEgasMax),
                  sizeof(peos_table->logEgasMax));
    eos_file.read(reinterpret_cast<char*>(&peos_table->logRhoMin),
                  sizeof(peos_table->logRhoMin));
    eos_file.read(reinterpret_cast<char*>(&peos_table->logRhoMax),
                  sizeof(peos_table->logRhoMax));
    peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
    eos_file.read(reinterpret_cast<char*>(peos_table->EosRatios.data()),
                  peos_table->nVar * sizeof(peos_table->logRhoMin));
    peos_table->table.SetSize(peos_table->nVar, peos_table->nEgas, peos_table->nRho);
    peos_table->table.SetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
    peos_table->table.SetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
    eos_file.read(reinterpret_cast<char*>(peos_table->table.data.data()),
                  peos_table->nVar * peos_table->nRho * peos_table->nEgas
                  * sizeof(peos_table->logRhoMin));
    eos_file.close();
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in EosTable::EosTable, ReadBinaryTable" << std::endl
        << "Unable to open eos table: " << fn << std::endl;
    ATHENA_ERROR(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ReadBinaryTable(std::string fn, EosTable *peos_table, ParameterInput *pin)
//! \brief Read data from HDF5 EOS table and initialize interpolated table.

void ReadHDF5Table(std::string fn, EosTable *peos_table, ParameterInput *pin) {
#ifndef HDF5OUTPUT
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in EosTable::EosTable, ReadHDF5Table" << std::endl
        << "HDF5 EOS table specified, but HDF5 flag is not enabled."  << std::endl;
    ATHENA_ERROR(msg);
  }
#endif
  bool read_ratios = pin->GetOrAddBoolean("hydro", "eos_read_ratios", true);
  std::string dens_lim_field =
      pin->GetOrAddString("hydro", "EOS_dens_lim_field", "LogDensLim");
  std::string espec_lim_field =
      pin->GetOrAddString("hydro", "EOS_espec_lim_field", "LogEspecLim");
  HDF5TableLoader(fn.c_str(), &peos_table->table, 3, var_names,
                  espec_lim_field.c_str(), dens_lim_field.c_str());
  peos_table->table.GetSize(peos_table->nVar, peos_table->nEgas, peos_table->nRho);
  peos_table->table.GetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
  peos_table->table.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
  peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
  if (read_ratios) {
    std::string ratio_field=pin->GetOrAddString("hydro", "EOS_ratio_field", "ratios");
    int zero[] = {0};
    int pnVar[] = {peos_table->nVar};
    HDF5ReadRealArray(fn.c_str(), ratio_field.c_str(), 1, zero, pnVar,
                      1, zero, pnVar, peos_table->EosRatios);
    if (peos_table->EosRatios(0) <= 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EosTable::EosTable, ReadHDF5Table" << std::endl
          << "Invalid ratio. " << fn.c_str() << ", " << ratio_field << ", "
          << peos_table->EosRatios(0) << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    for (int i=0; i<peos_table->nVar; ++i) peos_table->EosRatios(i) = 1.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ReadAsciiTable(std::string fn, EosTable *peos_table, ParameterInput *pin)
//! \brief Read data from HDF5 EOS table and initialize interpolated table.

void ReadAsciiTable(std::string fn, EosTable *peos_table, ParameterInput *pin) {
  bool read_ratios = pin->GetOrAddBoolean("hydro", "eos_read_ratios", true);
  AthenaArray<Real> *pratios = nullptr;
  if (read_ratios) pratios = &peos_table->EosRatios;
  // If read_ratios then EosRatios.NewAthenaArray is called in ASCIITableLoader
  ASCIITableLoader(fn.c_str(), peos_table->table, pratios);
  peos_table->table.GetSize(peos_table->nVar, peos_table->nEgas, peos_table->nRho);
  peos_table->table.GetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
  peos_table->table.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
  if (!read_ratios) {
    peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
    for (int i=0; i<peos_table->nVar; ++i) peos_table->EosRatios(i) = 1.0;
  }
}

void ReadHelmTable(std::string fn, EosTable *peos_table, ParameterInput *pin) {
  int jmax, imax;
  Real tlo, thi; // log temperature limits
  Real dlo, dhi; // log density limits
  if (pin->GetOrAddBoolean("hydro", "helm_assume_defaults", false)) {
    jmax = pin->GetOrAddInteger("hydro", "helm_temp_n", 201);
    tlo  = pin->GetOrAddReal("hydro", "helm_temp_log_min",  3.0);
    thi  = pin->GetOrAddReal("hydro", "helm_temp_log_max", 13.0);
    imax = pin->GetOrAddInteger("hydro", "helm_dens_n", 541);
    dlo  = pin->GetOrAddReal("hydro", "helm_dens_log_min", -12.0);
    dhi  = pin->GetOrAddReal("hydro", "helm_dens_log_max",  15.0);
  } else {
    jmax = pin->GetInteger("hydro", "helm_temp_n");
    tlo  = pin->GetReal("hydro", "helm_temp_log_min");
    thi  = pin->GetReal("hydro", "helm_temp_log_max");
    imax = pin->GetInteger("hydro", "helm_dens_n");
    dlo  = pin->GetReal("hydro", "helm_dens_log_min");
    dhi  = pin->GetReal("hydro", "helm_dens_log_max");
  }
  //Real tstp  = (thi - tlo)/float(jmax-1);
  //Real tstpi = 1.0/tstp;
  //Real dstp  = (dhi - dlo)/float(imax-1);
  //Real dstpi = 1.0/dstp;

  //init table
  peos_table->table.SetSize(17, imax, jmax);
  peos_table->table.SetX1lim(dlo, dhi);
  peos_table->table.SetX2lim(tlo, thi);
  peos_table->table.GetSize(peos_table->nVar, peos_table->nRho, peos_table->nEgas);
  peos_table->table.GetX1lim(peos_table->logRhoMin, peos_table->logRhoMax);
  peos_table->table.GetX2lim(peos_table->logEgasMin, peos_table->logEgasMax);
  peos_table->EosRatios.NewAthenaArray(peos_table->nVar);
  for (int i=0; i<peos_table->nVar; ++i) peos_table->EosRatios(i) = 1.0;

  std::ifstream file(fn.c_str(), std::ios::in);
  if (! file.good()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EosTable::EosTable, ReadHelmTable" << std::endl
        << "Table data unreadable." << std::endl;
    ATHENA_ERROR(msg);
  }

  // read the helmholtz free energy and its derivatives
  AthenaArray<Real> f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt;
  f.InitWithShallowSlice(peos_table->table.data, 3, 0, 1);
  fd.InitWithShallowSlice(peos_table->table.data, 3, 1, 1);
  ft.InitWithShallowSlice(peos_table->table.data, 3, 2, 1);
  fdd.InitWithShallowSlice(peos_table->table.data, 3, 3, 1);
  ftt.InitWithShallowSlice(peos_table->table.data, 3, 4, 1);
  fdt.InitWithShallowSlice(peos_table->table.data, 3, 5, 1);
  fddt.InitWithShallowSlice(peos_table->table.data, 3, 6, 1);
  fdtt.InitWithShallowSlice(peos_table->table.data, 3, 7, 1);
  fddtt.InitWithShallowSlice(peos_table->table.data, 3, 8, 1);

  std::string line;
  for (int j=0; j<jmax; ++j) {
    for (int i=0; i<imax; ++i) {
      while (std::getline(file, line) && (line[0] == '#')) continue;
      std::stringstream lstream(line);
      lstream >> f(i,j);
      lstream >> fd(i,j);
      lstream >> ft(i,j);
      lstream >> fdd(i,j);
      lstream >> ftt(i,j);
      lstream >> fdt(i,j);
      lstream >> fddt(i,j);
      lstream >> fdtt(i,j);
      lstream >> fddtt(i,j);
    }
  }

  // read the pressure derivative with density table
  AthenaArray<Real> dpdf, dpdfd, dpdft, dpdfdt;
  dpdf.InitWithShallowSlice(peos_table->table.data, 3, 9, 1);
  dpdfd.InitWithShallowSlice(peos_table->table.data, 3, 10, 1);
  dpdft.InitWithShallowSlice(peos_table->table.data, 3, 11, 1);
  dpdfdt.InitWithShallowSlice(peos_table->table.data, 3, 12, 1);
  for (int j=0; j<jmax; ++j) {
    for (int i=0; i<imax; ++i) {
      while (std::getline(file, line) && (line[0] == '#')) continue;
      std::stringstream lstream(line);
      lstream >> dpdf(i,j);
      lstream >> dpdfd(i,j);
      lstream >> dpdft(i,j);
      lstream >> dpdfdt(i,j);
    }
  }

  // read the electron chemical potential table
  for (int j=0; j<jmax; ++j) {
    for (int i=0; i<imax; ++i) {
      while (std::getline(file, line) && (line[0] == '#')) continue;
      //std::stringstream lstream(line);
      //lstream >> ef(i,j);
      //lstream >> efd(i,j);
      //lstream >> eft(i,j);
      //lstream >> efdt(i,j);
    }
  }

  // read the number density table
  AthenaArray<Real> xf, xfd, xft, xfdt;
  xf.InitWithShallowSlice(peos_table->table.data, 3, 13, 1);
  xfd.InitWithShallowSlice(peos_table->table.data, 3, 14, 1);
  xft.InitWithShallowSlice(peos_table->table.data, 3, 15, 1);
  xfdt.InitWithShallowSlice(peos_table->table.data, 3, 16, 1);
  for (int j=0; j<jmax; ++j) {
    for (int i=0; i<imax; ++i) {
      while (std::getline(file, line) && (line[0] == '#')) continue;
      std::stringstream lstream(line);
      lstream >> xf(i,j);
      lstream >> xfd(i,j);
      lstream >> xft(i,j);
      lstream >> xfdt(i,j);
    }
  }

  // close the file
  file.close();
}

// ctor
EosTable::EosTable(ParameterInput *pin) :
    table(), logRhoMin(), logRhoMax(),
    rhoUnit(pin->GetOrAddReal("hydro", "eos_rho_unit", 1.0)),
    eUnit(pin->GetOrAddReal("hydro", "eos_egas_unit", 1.0)),
    hUnit(eUnit/rhoUnit) {
  std::string eos_fn, eos_file_type;
  eos_fn = pin->GetString("hydro", "eos_file_name");
  eos_file_type = pin->GetOrAddString("hydro", "eos_file_type", "auto");

  if (eos_file_type.compare("auto") == 0) {
    std::string ext = eos_fn.substr(eos_fn.find_last_of(".") + 1);
    if (ext.compare("data")*ext.compare("bin") == 0) {
      eos_file_type.assign("binary");
    } else if (ext.compare("hdf5") == 0) {
      eos_file_type.assign("hdf5");
    } else if (ext.compare("tab")*ext.compare("txt")*ext.compare("ascii") == 0) {
      eos_file_type.assign("ascii");
    }
  }
  if (eos_file_type.compare("binary") == 0) { //Raw binary
    ReadBinaryTable(eos_fn, this);
  } else if (eos_file_type.compare("hdf5") == 0) { // HDF5 table
    ReadHDF5Table(eos_fn, this, pin);
  } else if (eos_file_type.compare("ascii") == 0) { // ASCII/text table
    ReadAsciiTable(eos_fn, this, pin);
  } else if (eos_file_type.compare("helm") == 0) { // helmholtz table
    ReadHelmTable(eos_fn, this, pin);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in EosTable::EosTable" << std::endl
        << "EOS table of type '" << eos_file_type << "' not recognized."  << std::endl
        << "Options are 'ascii', 'binary', 'hdf5', and 'helm'." << std::endl;
    ATHENA_ERROR(msg);
  }
}
