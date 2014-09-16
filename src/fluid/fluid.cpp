//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// Primary header
#include "fluid.hpp"

// C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena headers
#include "../athena.hpp"                // array access, macros, Real
#include "../athena_arrays.hpp"         // AthenaArray
#include "bvals/bvals.hpp"              // FluidBoundaryConditions
#include "eos/eos.hpp"                  // FluidEqnOfState
#include "srcterms/srcterms.hpp"        // FluidSourceTerms
#include "integrators/integrators.hpp"  // FluidIntegrator
#include "../mesh.hpp"                  // MeshBlock, Mesh

//======================================================================================
//! \file fluid.cpp
//  \brief implementation of functions in class Fluid
//======================================================================================

// constructor, initializes data structures and parameters

Fluid::Fluid(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_block = pmb;

// Allocate memory for primitive/conserved variables

  int ncells1 = pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (pmy_block->block_size.nx2 > 1) ncells2 = pmy_block->block_size.nx2 + 2*(NGHOST);
  if (pmy_block->block_size.nx3 > 1) ncells3 = pmy_block->block_size.nx3 + 2*(NGHOST);

  u.NewAthenaArray(NVAR,ncells3,ncells2,ncells1);
  w.NewAthenaArray(NVAR,ncells3,ncells2,ncells1);

// Allocate memory for primitive/conserved variables at intermediate-time step

  u1.NewAthenaArray(NVAR,ncells3,ncells2,ncells1);
  w1.NewAthenaArray(NVAR,ncells3,ncells2,ncells1);

  // Allocate memory for metric
  // TODO: this should only be done if we are in GR
  g.NewAthenaArray(NMETRIC, ncells1);
  g_inv.NewAthenaArray(NMETRIC, ncells1);

// Allocate memory for scratch arrays

  dt1_.NewAthenaArray(ncells1);
  dt2_.NewAthenaArray(ncells1);
  dt3_.NewAthenaArray(ncells1);

// Construct ptrs to objects of various classes needed to integrate fluid eqns 

  pf_integrator = new FluidIntegrator(this);
  pf_bcs = new FluidBoundaryConditions(this,pin);
  pf_eos = new FluidEqnOfState(this,pin);
  pf_srcterms = new FluidSourceTerms(this,pin);
}

// destructor

Fluid::~Fluid()
{
  pmy_block = NULL; // MeshBlock destructor will free this memory
  u.DeleteAthenaArray();
  w.DeleteAthenaArray();
  u1.DeleteAthenaArray();
  w1.DeleteAthenaArray();
  g.DeleteAthenaArray();
  g_inv.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  delete pf_integrator;
  delete pf_bcs;
  delete pf_eos;
}

//--------------------------------------------------------------------------------------
// \!fn 
// \brief

void Fluid::NewTimeStep(MeshBlock *pmb)
{
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real min_dt;
  Real wi[NVAR];

  AthenaArray<Real> w = pmb->pfluid->w.ShallowCopy();
  AthenaArray<Real> dt1 = dt1_.ShallowCopy();
  AthenaArray<Real> dt2 = dt2_.ShallowCopy();
  AthenaArray<Real> dt3 = dt3_.ShallowCopy();

  min_dt = (FLT_MAX);
  for (int k=ks; k<=ke; ++k){
  for (int j=js; j<=je; ++j){
    Real& dx2 = pmb->dx2f(j);
    Real& dx3 = pmb->dx3f(k);
#pragma simd
    for (int i=is; i<=ie; ++i){
      wi[IDN]=w(IDN,i);
      wi[IVX]=w(IVX,i);
      wi[IVY]=w(IVY,i);
      wi[IVZ]=w(IVZ,i);
      if (NON_BAROTROPIC_EOS) wi[IEN]=w(IEN,i);
      Real& dx1  = pmb->dx1f(i);
      Real& d_t1 = dt1(i);
      Real& d_t2 = dt2(i);
      Real& d_t3 = dt3(i);

      if (RELATIVISTIC_DYNAMICS)
      {
        d_t1 = dx1;
        d_t2 = dx2;
        d_t3 = dx3;
      }
      else
      {
        Real cs = pf_eos->SoundSpeed(wi);
        d_t1 = dx1/(fabs(wi[IVX]) + cs);
        d_t2 = dx2/(fabs(wi[IVY]) + cs);
        d_t3 = dx3/(fabs(wi[IVZ]) + cs);
      }
    }

// compute minimum of (v1 +/- C)

    for (int i=is; i<=ie; ++i){
      min_dt = std::min(min_dt,dt1(i));
    }
    
// if grid is 2D/3D, compute minimum of (v2 +/- C)

    if (pmb->block_size.nx2 > 1) {
      for (int i=is; i<=ie; ++i){
        min_dt = std::min(min_dt,dt2(i));
      }
    }

// if grid is 3D, compute minimum of (v3 +/- C)

    if (pmb->block_size.nx3 > 1) {
      for (int i=is; i<=ie; ++i){
        min_dt = std::min(min_dt,dt3(i));
      }
    }

  }}

  Mesh *pm = pmb->pmy_domain->pmy_mesh;
  Real old_dt = pm->dt;
  pm->dt = std::min( ((pm->cfl_number)*min_dt) , (2.0*old_dt) );

  return;
}