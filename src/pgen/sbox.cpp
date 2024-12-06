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
//! REFERENCE:
//! - Stone, J., Hawley, J., Gammie, C.F. & Balbus, S. A., ApJ 463, 656-673 (1996)
//! - Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992)
//============================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

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
namespace {
// Apply a density floor - useful for large |z| regions
Real dfloor, pfloor;
Real Omega_0, qshear;
} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // shearing sheet parameter
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");

  // Enroll user-defined physical source terms
  //   vertical external gravitational potential
  EnrollUserExplicitSourceFunction(VertGrav);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
  }

  return;
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
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  Real rval;

  // initialize density
  Real den = pin->GetOrAddReal("problem","dens",1.0);
    // Compute pressure based on the EOS.
  if (NON_BAROTROPIC_EOS) {
    pres  = pin->GetOrAddReal("problem","pres",1.0);
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    pres = den*SQR(iso_cs);
  }

  // Initialize boxsize
  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;

  // initialize wavenumbers
  int nwx = pin->GetOrAddInteger("problem","nwx",1);
  Real kx = (2.0*PI/Lx)*(static_cast<Real>(nwx));// nxw=-ve for leading wave

  // Ensure a different initial random seed for each meshblock.
  std::int64_t iseed = -1 - gid;

  // adiabatic gamma
  Real gam = peos->GetGamma();

  if (pmy_mesh->mesh_size.nx3 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat.cpp ProblemGenerator" << std::endl
        << "Stratified shearing sheet only works on a 3D grid" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read problem parameters for initial conditions
  amp = pin->GetReal("problem","amp");
  ipert = pin->GetOrAddInteger("problem","ipert", 1);

  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));

  if (MAGNETIC_FIELDS_ENABLED) {
    ifield = pin->GetOrAddInteger("problem","ifield", 1);
    beta = pin->GetReal("problem","beta");
  }

  // Compute field strength based on beta.
  if (MAGNETIC_FIELDS_ENABLED && Globals::my_rank == 0 && lid == 0) {
    B0 = std::sqrt(static_cast<Real>(2.0*pres/beta));
    std::cout << "B0=" << B0 << std::endl;
  }
  Real asqr0;
  if (NON_BAROTROPIC_EOS) {
    if (GENERAL_EOS) {
      asqr0 = peos->AsqFromRhoP(den, pres);
    } else {
      asqr0 = gam * pres / den;
    }
  } else {
    asqr0 = SQR(iso_cs);
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x3 = pcoord->x3v(k);

        // Initialize perturbations
        // ipert = 1 - random perturbations to P/d and V
        // [default, used by HGB]
        if (NON_BAROTROPIC_EOS && !GENERAL_EOS) {
          Real factor = std::max(1 - 0.5 * (gam - 1) * SQR(x3 * Omega_0) / asqr0, 0.0);
          rd = den * std::pow(factor, 1.0/(gam - 1));
          rp = pres * std::pow(factor, gam/(gam - 1));
        } else {
          rd = den*std::exp(-x3*x3);
        }
        if (ipert == 1) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd *= (1.0+2.0*rval);
          if (rd < dfloor) rd = dfloor;
          SumRd += rd;
          if (NON_BAROTROPIC_EOS) {
            rp *= (1.0+2.0*rval);
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
          rvx = 0;
          rvy = 0;
          rvz = 0;
        }

        // Initialize d, M, and P.
        // for_the_future: if FARGO do not initialize the bg shear
        rd = std::max(rd, dfloor);
        if (NON_BAROTROPIC_EOS) {
          rp = std::max(rp, pfloor);
        }
        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= rd*qshear*Omega_0*x1;
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/(gam-1.0)
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
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
  return;
}

void MeshBlock::UserWorkInLoop() {
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}


void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);
        // multiply gravitational potential by smoothing function
        Real src = dt*den*SQR(pmb->porb->Omega0)*x3;
        cons(IM3,k,j,i) -= src;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= src*prim(IVZ,k,j,i);
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
        prim(IDN,kl-k,j,i) = prim(IDN,kl,j,i);
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        prim(IVZ,kl-k,j,i) = (prim(IVZ,kl,j,i) >= 0.0) ? 0.0 : prim(IVZ,kl,j,i);
        if (NON_BAROTROPIC_EOS) {
          prim(IPR,kl-k,j,i) = prim(IPR,kl,j,i);
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
        Real den = prim(IDN,kl,j,i);
        prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        prim(IVZ,ku+k,j,i) = (prim(IVZ,ku,j,i) <= 0.0) ? 0.0 : prim(IVZ,ku,j,i);
        if (NON_BAROTROPIC_EOS) {
          prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
        }
      }
    }
  }
  return;
}
