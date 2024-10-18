//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file strat.cpp
//! \brief Problem generator for stratified 3D shearing sheet.
//!
//! PURPOSE:  Problem generator for stratified 3D shearing sheet.  Based on the
//!   initial conditions described in "Three-dimensional Magnetohydrodynamic
//!   Simulations of Vertically Stratified Accretion Disks" by Stone, Hawley,
//!   Gammie & Balbus.
//!
//! Several different field configurations and perturbations are possible:
//! - ifield = 1 - Bz=B0 sin(x1) field with zero-net-flux [default]
//! - ifield = 2 - uniform Bz
//! - ifield = 3 - uniform Bz plus sinusoidal perturbation Bz(1+0.5*sin(kx*x1))
//! - ifield = 4 - B=(0,B0cos(kx*x1),B0sin(kx*x1))= zero-net flux w helicity
//! - ifield = 5 - uniform By, but only for |z|<2
//! - ifield = 6 - By with constant beta versus z
//! - ifield = 7 - zero field everywhere
//!
//! - ipert = 1 - random perturbations to P and V [default, used by HGB]
//!
//! Code must be configured using -shear
//!
//! REFERENCE:
//! - Stone, J., Hawley, J., Gammie, C.F. & Balbus, S. A., ApJ 463, 656-673 (1996)
//! - Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992)
//============================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../inputs/hdf5_reader.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifndef HDF5OUTPUT
#error "HDF5OUTPUT must be enabled for this problem generator."
#endif

#define MSBC_DEBUG 1

// TODO(felker): many unused arguments in these functions: time, iout, ...
void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void RadBot(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
            const AthenaArray<Real> &w, FaceField &b,
            AthenaArray<Real> &ir,
            Real time, Real dt,
            int is, int ie, int js, int je, int ks, int ke, int ngh);
void RadTop(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
            const AthenaArray<Real> &w, FaceField &b,
            AthenaArray<Real> &ir,
            Real time, Real dt,
            int is, int ie, int js, int je, int ks, int ke, int ngh);
void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

class OpalOpacityTable {
 public:
  OpalOpacityTable() {}
  OpalOpacityTable(const char *filename) { LoadOpacity(filename); }
  ~OpalOpacityTable() {
    log_rhoT.DeleteAthenaArray();
    log_T.DeleteAthenaArray();
    log_kappa.DeleteAthenaArray();
  }

  void LoadOpacity(const char *filename);
  Real GetOpacity(const Real rho, const Real tgas);
  void SetUnits(const Real rhounit, const Real tunit) {
    OpalOpacityTable::rhounit = rhounit;
    OpalOpacityTable::tunit = tunit;
  }

 private:
  AthenaArray<Real> log_rhoT, log_T, log_kappa;
  std::vector<int> breaks;
  std::vector<Real> break_vals;
  Real rhot_norm, rhot_min, rhot_max;
  Real tunit{1.0};
  Real rhounit{1.0};
};

namespace {
  OpalOpacityTable kR, kP;
  Real rhounit, tunit, lunit;

  Real HistoryBxBy(MeshBlock *pmb, int iout);
  Real HistorydVxVy(MeshBlock *pmb, int iout);

  // Apply a density floor - useful for large |z| regions
  Real dfloor, pfloor, tfloor;
  Real Omega_0, qshear;
  int ic_rows;  // number of rows in the initial condition file
  AthenaArray<Real> empty; // empty array for unused arguments
  InterpTable2D opacity_table;
  Real opacity_norm;
} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (!MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat_rad.cpp InitUserMeshData" << std::endl
        << "This problem generator requires magnetic fields." << std::endl;
    ATHENA_ERROR(msg);
  }

  // get units
  tunit = pin->GetOrAddReal("radiation","T_unit",1.e7);
  rhounit = pin->GetOrAddReal("radiation","density_unit",1.0);
  lunit = pin->GetOrAddReal("radiation","length_unit",1.0);

  // get floors
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));
  tfloor = pin->GetOrAddReal("radiation", "tfloor", 1e-4);

  // shearing sheet parameter
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, HistoryBxBy, "-BxBy");
  EnrollUserHistoryOutput(1, HistorydVxVy, "dVxVy");

  // Enroll user-defined physical source terms
  //   vertical external gravitational potential
  EnrollUserExplicitSourceFunction(VertGrav);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x3, RadBot);
    }
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x3, RadTop);
    }
  }

  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in hb3.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box." << std::endl;
    ATHENA_ERROR(msg);
  }

  // load initial conditions from file
  std::string fn = pin->GetString("problem", "ic_file");
  int comments = pin->GetOrAddInteger("problem", "ic_comments", -1);
  ic_rows = pin->GetOrAddInteger("problem", "ic_rows", -1);
  int cols = pin->GetInteger("problem", "ic_cols");
  int col_z = pin->GetInteger("problem", "ic_col_z");
  int col_rho = pin->GetInteger("problem", "ic_col_rho");
  int col_p = pin->GetInteger("problem", "ic_col_p");

  // if we don't have the number of comments, determine them
  if (comments < 0) {
    if (Globals::my_rank == 0) {
      std::ifstream file(fn);
      std::string line;
      comments= 0;
      while (std::getline(file, line)) {
        if (line[0] == '#') {
          comments++;
        } else {
          break;
        }
      }
      std::cout << "Determined that IC file " << fn << " starts with " << comments
                << " comment lines." << std::endl;
    }
    #ifdef MPI_PARALLEL
    MPI_Bcast(&comments, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #endif
    pin->SetInteger("problem", "ic_comments", comments);
  }

  // if we don't have the number of rows, determine them
  if (ic_rows < 0) {
    if (Globals::my_rank == 0) {
      std::ifstream file(fn);
      std::string line;
      ic_rows = -comments;
      while (std::getline(file, line)) {
        ic_rows++;
      }
      std::cout << "Determined that IC file " << fn << " has " << ic_rows
                << " data lines." << std::endl;
    }
    #ifdef MPI_PARALLEL
    MPI_Bcast(&ic_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #endif
    pin->SetInteger("problem", "ic_rows", ic_rows);
  }

  // Prepare arrays to initial conditions
  AllocateRealUserMeshDataField(5);
  // star profile
  ruser_mesh_data[0].NewAthenaArray(ic_rows);
  ruser_mesh_data[1].NewAthenaArray(ic_rows);
  ruser_mesh_data[2].NewAthenaArray(ic_rows);

  // Alias arrays
  AthenaArray<Real>& z_ic{ruser_mesh_data[0]};
  AthenaArray<Real>& rho_ic{ruser_mesh_data[1]};
  AthenaArray<Real>& p_ic{ruser_mesh_data[2]};

  // read the initial conditions from file
  if (Globals::my_rank == 0) {
    std::string line;
    std::ifstream file(fn);
    for (int n = 0; n < comments; ++n) {
      std::getline(file, line);
    }
    AthenaArray<Real> vals;
    vals.NewAthenaArray(cols);
    for (int n = 0; n < ic_rows; ++n) {
      for (int m = 0; m < cols; ++m) {
        file >> vals(m);
      }
      z_ic(n) = vals(col_z);
      rho_ic(n) = vals(col_rho);
      p_ic(n) = vals(col_p);
      //std::cout << "Loaded IC row " << n << ": z=" << z_ic(n) << ", rho=" << rho_ic(n)
      //          << ", p=" << p_ic(n) << std::endl;
    }
  }

  // Broadcast profile data to other ranks
  #ifdef MPI_PARALLEL
  {
    MPI_Bcast(z_ic.data(), ic_rows, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(rho_ic.data(), ic_rows, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(p_ic.data(), ic_rows, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  #endif

  if (Globals::my_rank == 0) {
    std::cout << "Loaded " << ic_rows << " rows of initial conditions from " << fn << std::endl;
  }

  //opacity_norm = pin->GetOrAddReal("radiation", "opacity_cm_over_simlength", 1);
  //AthenaArray<Real>& opacity{ruser_mesh_data[3]};
  //AthenaArray<Real>& lims{ruser_mesh_data[4]};
  //LoadOpacity(pin, lims, opacity);
//
  //// Initialize the opacity table
  //opacity_table.InitDataWithShallowSlice(opacity);
  //opacity_table.SetX2lim(lims(0), lims(1));
  //opacity_table.SetX1lim(lims(2), lims(3));

  // Initialize the Opal opacity tables
  std::string kR_fn = pin->GetOrAddString("radiation", "kR_file", "kR.h5");
  kR.LoadOpacity(kR_fn.c_str());
  std::string kP_fn = pin->GetOrAddString("radiation", "kP_file", "kP.h5");
  kP.LoadOpacity(kP_fn.c_str());

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  pnrrad->EnrollOpacityFunction(StarOpacity);
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int ifield, ipert;
  Real beta, amp, pres;
  Real iso_cs=1.0;
  Real B0 = 0.0;

  Real SumRd=0.0, SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  // TODO(felker): tons of unused variables in this file: xmin, xmax, rbx, rby, Ly, ky,...
  Real x1, x3;
  //Real xmin, xmax;
  //Real x1f, x2f, x3f;
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  //Real rbx, rby, rbz;
  Real rval;

  // initialize density
  const Real den=1.0;

  // Initialize boxsize
  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  //Real Ly = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  //Real Lz = pmy_mesh->mesh_size.x3max - pmy_mesh->mesh_size.x3min;

  // initialize wavenumbers
  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  //int nwy = pin->GetOrAddInteger("problem","nwy",1);
  //int nwz = pin->GetOrAddInteger("problem","nwz",1);
  Real kx = (2.0*PI/Lx)*(static_cast<Real>(nwx));// nxw=-ve for leading wave
  //Real ky = (2.0*PI/Ly)*(static_cast<Real>(nwy));
  //Real kz = (2.0*PI/Lz)*(static_cast<Real>(nwz));

  // Ensure a different initial random seed for each meshblock.
  std::int64_t iseed = -1 - gid;

  // adiabatic gamma
  Real gam;
  if (GENERAL_EOS) {
    gam = std::numeric_limits<Real>::quiet_NaN();
  } else {
    Real gam = peos->GetGamma();
  }

  if (pmy_mesh->mesh_size.nx3 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat.cpp ProblemGenerator" << std::endl
        << "Stratified shearing sheet only works on a 3D grid" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read problem parameters for initial conditions
  amp = pin->GetReal("problem","amp");
  ipert = pin->GetOrAddInteger("problem","ipert", 1);

  if (MAGNETIC_FIELDS_ENABLED) {
    ifield = pin->GetOrAddInteger("problem","ifield", 1);
    beta = pin->GetReal("problem","beta");
  }

  // Compute pressure based on the EOS.
  if (NON_BAROTROPIC_EOS) {
    pres  = pin->GetOrAddReal("problem","pres",1.0);
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    pres = den*SQR(iso_cs);
  }

  // Compute field strength based on beta.
  if (MAGNETIC_FIELDS_ENABLED) {
    B0 = std::sqrt(static_cast<Real>(2.0*pres/beta));
  }

  Real sigma0 = pin->GetOrAddReal("problem","sigma0",1.0);
  Real prat = pin->GetReal("radiation","prat");

  // Initialize the density, velocity, and pressure from the initial conditions
  // Alias arrays
  AthenaArray<Real>& z_ic = pmy_mesh->ruser_mesh_data[0];
  AthenaArray<Real>& rho_ic = pmy_mesh->ruser_mesh_data[1];
  AthenaArray<Real>& p_ic = pmy_mesh->ruser_mesh_data[2];

  // Initialize primitive values
  int ki = 0;
  Real rho, p, Tgas;
  for (int k = ks; k <= ke; ++k) {
    // Average over face values
    rho = 0.0;
    p = 0.0;
    for (int k0 = 0; k0 < 2; k0++) {
      int index = 0;
      Real z = std::abs(pcoord->x3f(k0 + k));
      while (z > z_ic(index)) {
        index++;
      }
      Real norm = (z - z_ic(index)) / (z_ic(index) - z_ic(index+1));
      if (norm < 0 || norm > 1) {
        std::stringstream msg;
        msg << "### FATAL ERROR in strat.cpp ProblemGenerator" << std::endl
            << "Interpolation error: norm=" << norm << ", z=" << z << ", z_ic=" << z_ic(index-1)
            << ", z_ic=" << z_ic(index) << ", index=" << index << std::endl;
        ATHENA_ERROR(msg);
      }
      rho += rho_ic(index) + (rho_ic(index+1) - rho_ic(index)) * norm;
      p += p_ic(index) + (p_ic(index+1) - p_ic(index)) * norm;
    }
    rho *= 0.5;
    p *= 0.5;
    std::cout << "z=" << pcoord->x3v(k) << ", k=" << k << ", rho=" << rho << ", p=" << p << std::endl;

    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        //x2 = pcoord->x2v(j);
        x3 = pcoord->x3v(k);
        // x1f = pcoord->x1f(i);
        // x2f = pcoord->x2f(j);
        // x3f = pcoord->x3f(k);

        // Initialize perturbations
        // ipert = 1 - random perturbations to P/d and V
        // [default, used by HGB]
        if (ipert == 1) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd = rho*(1.0+2.0*rval);
          if (rd < dfloor) rd = dfloor;
          SumRd += rd;
          if (NON_BAROTROPIC_EOS) {
            rp = p/rho*rd;
            if (rp < pfloor) rp = pfloor;
          }
          rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvx += rd*rvx;

          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvy += rd*rvy;

          rval = amp*(ran2(&iseed) - 0.5);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvz += rd*rvz;
          // no perturbations
        } else {
          rd = rho*std::exp(-x3*x3);
          rvx = 0;
          rvy = 0;
          rvz = 0;
        }

        // Initialize d, M, and P.
        // for_the_future: if FARGO do not initialize the bg shear
        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= rd*qshear*Omega_0*x1;
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          if (GENERAL_EOS) {
            Real egas = peos->EgasFromRhoP(rd, rp);
            Tgas = peos->TgasFromRhoEg(rd, egas);
            phydro->u(IEN,k,j,i) = egas
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
          } else {
            Tgas = rp/rd;
            phydro->u(IEN,k,j,i) = rp/(gam-1.0)
                                  + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                          + SQR(phydro->u(IM2,k,j,i))
                                          + SQR(phydro->u(IM3,k,j,i)))/rd;
          }
        } // Hydro

        // Initialize magnetic field.  For 3D shearing box B1=Bx, B2=By, B3=Bz
        //  ifield = 1 - Bz=B0 std::sin(x1) field with zero-net-flux [default]
        //  ifield = 2 - uniform Bz
        //  ifield = 3 - Bz(1+0.5*sin(kx*x1))
        //  ifield = 4 - B=(0,B0cos(kx*x1),B0sin(kx*x1)) =
        //               zero-net flux w/ helicity
        //  ifield = 5 - uniform By, but only for |z|<2
        //  ifield = 6 - By with constant beta versus z
        //  ifield = 7 - zero field everywhere
        if (MAGNETIC_FIELDS_ENABLED) {
          if (ifield == 1) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 2) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0;
          }
          if (ifield == 3) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = B0*(1.0+0.5*std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(1.0 + 0.5*
                                                     std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 4) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            pfield->b.x3f(k,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0*(std::cos(static_cast<Real>(kx)*x1));
            if (k==ke) pfield->b.x3f(ke+1,j,i) = B0*(std::sin(static_cast<Real>(kx)*x1));
          }
          if (ifield == 5 && std::abs(x3) < 2.0) {
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = B0;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = B0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 6) {
            // net toroidal field with constant \beta with height
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = std::sqrt(den*std::exp(-x3*x3)*SQR(Omega_0)/beta);
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = std::sqrt(den*std::exp(-x3*x3)*
                                                           SQR(Omega_0)/beta);
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
          if (ifield == 7) {
            // zero field everywhere
            pfield->b.x1f(k,j,i) = 0.0;
            pfield->b.x2f(k,j,i) = 0.0;
            pfield->b.x3f(k,j,i) = 0.0;
            if (i==ie) pfield->b.x1f(k,j,ie+1) = 0.0;
            if (j==je) pfield->b.x2f(k,je+1,i) = 0.0;
            if (k==ke) pfield->b.x3f(ke+1,j,i) = 0.0;
          }
        } // MHD


        if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
          for (int ifr=0; ifr<pnrrad->nfreq; ++ifr) {
            for (int n=0; n<pnrrad->nang; ++n) {
               pnrrad->ir(k,j,i,ifr*pnrrad->nang+n) = prat * std::pow(Tgas, 4.0) / PI;
            }
            pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pnrrad->sigma_a(k,j,i,ifr) = sigma0;
            pnrrad->sigma_pe(k,j,i,ifr) = sigma0;
            pnrrad->sigma_p(k,j,i,ifr) = sigma0;
          }
        }
      }
    }
  }

  // For random perturbations as in HGB, ensure net momentum is zero by
  // subtracting off mean of perturbations

  if (ipert == 1) {
    if (lid == pmy_mesh->nblocal - 1) {
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &SumRd,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvy, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &SumRvz, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
      std::int64_t cell_num = pmy_mesh->GetTotalCells();
      SumRvx /= SumRd*cell_num;
      SumRvy /= SumRd*cell_num;
      SumRvz /= SumRd*cell_num;
      for (int b = 0; b < pmy_mesh->nblocal; ++b) {
        Hydro *ph = pmy_mesh->my_blocks(b)->phydro;
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              ph->u(IM1,k,j,i) -= ph->u(IDN,k,j,i)*SumRvx;
              ph->u(IM2,k,j,i) -= ph->u(IDN,k,j,i)*SumRvy;
              ph->u(IM3,k,j,i) -= ph->u(IDN,k,j,i)*SumRvz;
            }
          }
        }
      }
    }
  }
  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if (lid == pmy_mesh->nblocal - 1) {
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  return;
}

//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // free memory
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    kR.~OpalOpacityTable();
    kR.~OpalOpacityTable();
  }

  return;
}


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real fsmooth, xi, sign;
  Real Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real z0 = Lz/2.0;
  Real lambda = 0.1 / z0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);
        // smoothing function
        if (x3 >= 0) {
          sign = -1.0;
        } else {
          sign = 1.0;
        }
        xi = z0/x3;
        fsmooth = SQR( std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign );
        // multiply gravitational potential by smoothing function
        cons(IM3,k,j,i) -= dt*den*SQR(Omega_0)*x3*fsmooth;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= dt*den*SQR(Omega_0)*prim(IVZ,k,j,i)*x3*fsmooth;
        }
      }
    }
  }
  return;
}

//  Here is the lower z outflow boundary.
//  The basic idea is that the pressure and density
//  are exponentially extrapolated in the ghost zones
//  assuming a constant temperature there (i.e., an
//  isothermal atmosphere). The z velocity (NOT the
//  momentum) are set to zero in the ghost zones in the
//  case of the last lower physical zone having an inward
//  flow.  All other variables are extrapolated into the
//  ghost zones with zero slope.

void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Copy field components from last physical zone
  // zero slope boundary for B field
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(kl-k,j,i) = b.x1f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(kl-k,j,i) = b.x2f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(kl-k,j,i) = b.x3f(kl,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        // First calculate the effective gas temperature (Tkl=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tkl = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = prim(IPR,kl,j,i);
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,kl-k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (2.0*Tkl/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IDN,kl-k,j,i)*Tkl;
      }
    }
  }
  return;
}

void RadTop(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
            const AthenaArray<Real> &w, FaceField &b,
            AthenaArray<Real> &ir,
            Real time, Real dt,
            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  //vacuum boundary condition at top
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real miuz = prad->mu(2,ke+k,j,i,n);
            if(miuz > 0.0){
              ir(ke+k,j,i,ifr*prad->nang+n)
                            = ir(ke+k-1,j,i,ifr*prad->nang+n);
            }else{
              ir(ke+k,j,i,ifr*prad->nang+n) = 0.0;
            }


          }
        }
      }
    }
  }
  return;
}

void RadBot(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
            const AthenaArray<Real> &w, FaceField &b,
            AthenaArray<Real> &ir,
            Real time, Real dt,
            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  //vacuum boundary condition at bottom
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        for(int ifr=0; ifr<prad->nfreq; ++ifr){
          for(int n=0; n<prad->nang; ++n){
            Real miuz = prad->mu(2,ks-k,j,i,n);
            if(miuz < 0.0){
              ir(ks-k,j,i,ifr*prad->nang+n)
                            = ir(ks-k+1,j,i,ifr*prad->nang+n);
            }else{
              ir(ks-k,j,i,ifr*prad->nang+n) = 0.0;
            }


          }
        }
      }
    }
  }
  return;
}

// Here is the upper z outflow boundary.
// The basic idea is that the pressure and density
// are exponentially extrapolated in the ghost zones
// assuming a constant temperature there (i.e., an
// isothermal atmosphere). The z velocity (NOT the
// momentum) are set to zero in the ghost zones in the
// case of the last upper physical zone having an inward
// flow.  All other variables are extrapolated into the
// ghost zones with zero slope.
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Copy field components from last physical zone
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu+1; i++) {
          b.x1f(ku+k,j,i) = b.x1f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(ku+k,j,i) = b.x2f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          b.x3f(ku+1+k,j,i) = b.x3f(ku+1,j,i);
        }
      }
    }
  } // MHD

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        // First calculate the effective gas temperature (Tku=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tku = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real pressku = prim(IPR,ku,j,i);
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,ku+k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (2.0*Tku/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IDN,ku+k,j,i)*Tku;
      }
    }
  }
  return;
}


void LoadOpacity(ParameterInput *pin, AthenaArray<Real> &lims, AthenaArray<Real> &opacity) {
  if (!pin->DoesParameterExist("radiation", "opacity_file")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat_rad.cpp ProblemGenerator" << std::endl
        << "Opacity file not specified (radiation/opacity_file)." << std::endl;
    ATHENA_ERROR(msg);
  }
  std::string opacity_file = pin->GetString("radiation", "opacity_file");
  std::string dataset = pin->GetOrAddString("radiation", "opacity_dataset", "kappa");

  if (Globals::my_rank == 0) {
    lims.NewAthenaArray(4);
    HDF5ToRealArray(opacity_file.c_str(), opacity, dataset.c_str());
    if (opacity.GetDim1() > 10000 || opacity.GetDim1() < 0 || opacity.GetDim2() > 10000 || opacity.GetDim2() < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp ProblemGenerator" << std::endl
          << "Opacity data has unexpected shape: " << opacity.GetDim2() << ", " << opacity.GetDim1() << std::endl;
      ATHENA_ERROR(msg);
    }
    AthenaArray<Real> rho;
    HDF5ToRealArray(opacity_file.c_str(), rho, "rho");
    AthenaArray<Real> pres;
    HDF5ToRealArray(opacity_file.c_str(), pres, "pres");
    if (rho.GetDim1() != opacity.GetDim2() || pres.GetDim1() != opacity.GetDim1()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp ProblemGenerator" << std::endl
          << "Opacity shape inconsistent with rho/pres shape." << std::endl
          << "Rho: " << rho.GetDim1() << ", Pres: " << pres.GetDim1() << std::endl
          << "Opacity: " << opacity.GetDim2() << ", " << opacity.GetDim1() << std::endl;
      ATHENA_ERROR(msg);
    }

    lims(0) = std::log10(rho(0));
    lims(1) = std::log10(rho(rho.GetDim1()-1));
    lims(2) = std::log10(pres(0));
    lims(3) = std::log10(pres(pres.GetDim1()-1));
  }
  BroadcastRealArray(opacity);
  BroadcastRealArray(lims);
  return;
}

void OpalOpacityTable::LoadOpacity(const char *opacity_file) {
  if (Globals::my_rank == 0) {
    HDF5ToRealArray(opacity_file, log_kappa, "log_kappa");
    if (log_kappa.GetDim1() > 10000 || log_kappa.GetDim1() < 0 ||
        log_kappa.GetDim2() > 10000 || log_kappa.GetDim2() < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp ProblemGenerator" << std::endl
          << "Opacity data has unexpected shape: "
          << log_kappa.GetDim2() << ", " << log_kappa.GetDim1() << std::endl;
      ATHENA_ERROR(msg);
    }
    HDF5ToRealArray(opacity_file, log_T, "log_T");
    HDF5ToRealArray(opacity_file, log_rhoT, "log_rhoT");
    if (log_T.GetDim1() != log_kappa.GetDim2() ||
        log_rhoT.GetDim1() != log_kappa.GetDim1()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp ProblemGenerator" << std::endl
          << "Opacity shape inconsistent with rho/pres shape." << std::endl
          << "Rho: " << log_T.GetDim1() << ", Pres: " << log_rhoT.GetDim1() << std::endl
          << "Opacity: " << log_kappa.GetDim2() << ", " << log_kappa.GetDim1()
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  BroadcastRealArray(log_kappa);
  BroadcastRealArray(log_T);
  BroadcastRealArray(log_rhoT);

  Real last_diff = -1.0;
  for (int i = 0; i < log_T.GetDim1() - 1; ++i) {
    Real diff = log_T(i+1) - log_T(i);
    if (std::abs(last_diff / diff - 1.0) > 1e-6) {
      breaks.push_back(i);
      break_vals.push_back(log_T(i));
    }
  }
  breaks.push_back(log_T.GetDim1() - 1);
  break_vals.push_back(log_T(log_T.GetDim1() - 1));

  rhot_min = log_rhoT(0);
  rhot_max = log_rhoT(log_rhoT.GetDim1() - 1);
  rhot_norm = (log_rhoT.GetDim1() - 1) / (rhot_max - rhot_min);
}

Real OpalOpacityTable::GetOpacity(const Real rho, const Real tgas) {
  Real logt = log10(std::max(tgas, tfloor) * tunit);
  Real logrhot = log10(std::max(rho, dfloor) * rhounit) - 3.0* logt + 18.0;

  int nt1 = 0;
  int nt2 = 0;
  int i;
  for (i = 0; static_cast<unsigned long>(i) < breaks.size(); ++i) {
    if (logt < break_vals[i]) {
      nt2 = breaks[i];
      break;
    }
    nt1 = breaks[i];
  }

  if (static_cast<unsigned long>(i) == breaks.size() + 1 || i == 0) {
    nt2 = nt1;
  } else {
    Real t1 = break_vals[i-1];
    Real t2 = break_vals[i];
    Real frac = (logt - t1) / (t2 - t1);
    nt2 = nt1 + std::ceil(frac * (breaks[i] - nt1));
    nt1 = nt2 - 1;
  }

  if (MSBC_DEBUG) {
    if (nt1 < 0 || nt1 >= log_T.GetDim1() || nt2 < 0 || nt2 >= log_T.GetDim1()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp OpalOpacityTable::GetOpacity" << std::endl
          << "Temperature index out of bounds: " << nt1 << ", " << nt2 << std::endl;
      ATHENA_ERROR(msg);
    }
    if ((logt < log_T(nt1) || logt > log_T(nt2)) && (nt1 != nt2)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp OpalOpacityTable::GetOpacity" << std::endl
          << "Temperature not bracketed: " << logt << ", " << log_T(nt1) << ", " << log_T(nt2) << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  int nrhot2 = std::ceil(rhot_norm * (logrhot - rhot_min));
  int nrhot1 = nrhot2 - 1;
  if (nrhot1 < 0) nrhot1 = 0;
  if (nrhot2 < 0) nrhot2 = 0;
  if (nrhot1 >= log_rhoT.GetDim1()) nrhot1 = log_rhoT.GetDim1() - 1;
  if (nrhot2 >= log_rhoT.GetDim1()) nrhot2 = log_rhoT.GetDim1() - 1;

  if (MSBC_DEBUG) {
    if (nrhot1 < 0 || nrhot1 >= log_rhoT.GetDim1() || nrhot2 < 0 || nrhot2 >= log_rhoT.GetDim1()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp OpalOpacityTable::GetOpacity" << std::endl
          << "RhoT index out of bounds: " << nrhot1 << ", " << nrhot2 << std::endl;
      ATHENA_ERROR(msg);
    }
    if ((logrhot < log_rhoT(nrhot1) || logrhot > log_rhoT(nrhot2)) && (nrhot1 != nrhot2)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in strat_rad.cpp OpalOpacityTable::GetOpacity" << std::endl
          << "RhoT not bracketed: " << logrhot << ", " << log_rhoT(nrhot1) << ", " << log_rhoT(nrhot2) << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  Real kappa_t1_rho1 = log_kappa(nt1,nrhot1);
  Real kappa_t1_rho2 = log_kappa(nt1,nrhot2);
  Real kappa_t2_rho1 = log_kappa(nt2,nrhot1);
  Real kappa_t2_rho2 = log_kappa(nt2,nrhot2);

  Real rho_1 = log_rhoT(nrhot1);
  Real rho_2 = log_rhoT(nrhot2);
  Real t_1 = log_T(nt1);
  Real t_2 = log_T(nt2);

  Real kappa;

  if (nrhot1 == nrhot2) {
    if(nt1 == nt2) {
      kappa = kappa_t1_rho1;
    } else {
      kappa = kappa_t1_rho1 + (kappa_t2_rho1 - kappa_t1_rho1) *
                              (logt - t_1)/(t_2 - t_1);
    }/* end same T*/
  } else {
    if (nt1 == nt2) {
      kappa = kappa_t1_rho1 + (kappa_t1_rho2 - kappa_t1_rho1) *
                              (logrhot - rho_1)/(rho_2 - rho_1);
    } else {
      kappa = kappa_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                              ((t_2 - t_1) * (rho_2 - rho_1))
            + kappa_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                              ((t_2 - t_1) * (rho_2 - rho_1))
            + kappa_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                              ((t_2 - t_1) * (rho_2 - rho_1))
            + kappa_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                              ((t_2 - t_1) * (rho_2 - rho_1));
    }
  }/* end same rhoT */

  return std::pow(10.0, kappa);
}

void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim) {
  // electron scattering opacity
  Real kappas = 0.2 * (1.0 + 0.7);
  Real kappaa = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        for (int freq=0; freq<pmb->pnrrad->nfreq; freq++) {
          Real rho = prim(IDN,k,j,i);
          Real gast = pmb->peos->TgasFromRhoP(rho, prim(IPR,k,j,i));
          Real kappa = kR.GetOpacity(rho, gast);
          Real kappa_planck = kP.GetOpacity(rho, gast);
          if(kappa < kappas) {
            if(gast < 0.14){
              kappaa = kappa;
              kappa = 0.0;
            } else {
              kappaa = 0.0;
            }
          } else {
            kappaa = kappa - kappas;
            kappa = kappas;
          }
          pmb->pnrrad->sigma_s(k,j,i,freq) = kappa * rho * rhounit * lunit;
          pmb->pnrrad->sigma_a(k,j,i,freq) = kappaa * rho * rhounit * lunit;
          pmb->pnrrad->sigma_pe(k,j,i,freq) = kappa_planck * rho * rhounit * lunit;
          pmb->pnrrad->sigma_p(k,j,i,freq) = pmb->pnrrad->sigma_pe(k,j,i,freq);
        }
      }
    }
  }
}
namespace {

Real HistoryBxBy(MeshBlock *pmb, int iout) {
  Real bxby = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &b = pmb->pfield->bcc;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        bxby -= volume(i)*b(IB1,k,j,i)*b(IB2,k,j,i);
      }
    }
  }
  return bxby;
}


Real HistorydVxVy(MeshBlock *pmb, int iout) {
  Real dvxvy = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        if(!pmb->porb->orbital_advection_defined) {
          vshear = -qshear*Omega_0*pmb->pcoord->x1v(i);
        } else {
          vshear = 0.0;
        }
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) + vshear);
      }
    }
  }
  return dvxvy;
}
} // namespace
