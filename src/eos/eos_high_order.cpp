//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_high_order.cpp
//! \brief functions for variable conversion at greater than second-order spatial accuracy

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../scalars/scalars.hpp"
#include "eos.hpp"

static AthenaArray<Real> empty;

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitiveCellAverage(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts cell-averaged conserved variables to cell-averaged primitive variables
//! at fourth order accuracy. Wrapper function for specific pointwise conversion routine

void EquationOfState::ConservedToPrimitiveCellAverage(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    AthenaArray<Real> &s, const AthenaArray<Real> &r_old, AthenaArray<Real> &r,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  MeshBlock *pmb = pmy_block_;
  Hydro *ph = pmb->phydro;
  PassiveScalars *ps = pmb->pscalars;

  int nl = 0;
  int nu = NHYDRO - 1;
  int nus = NSCALARS - 1;
  // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this should factor out
  // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
  Real h = pco->dx1f(il);  // pco->dx1f(i); inside loop
  Real C = (h*h)/24.0;

  // Fourth-order accurate approx to cell-centered conserved and primitive variables
  AthenaArray<Real> &u_cc = ph->u_cc, &w_cc = ph->w_cc;
  // Laplacians of cell-averaged conserved and 2nd order accurate primitive variables
  AthenaArray<Real> &laplacian_cc = ph->scr1_nkji_;

    // Passive scalrs
  AthenaArray<Real> &s_cc = (NSCALARS) ? ps->s_cc : empty;
  AthenaArray<Real> &r_cc = (NSCALARS) ? ps->r_cc : empty;
  // Laplacians of cell-averaged conserved and 2nd order accurate primitive variables
  AthenaArray<Real> &s_laplacian_cc = (NSCALARS) ? ps->scr1_nkji_ : empty;

  // Compute and store Laplacian of cell-averaged conserved variables
  pco->Laplacian(cons, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);
  if (NSCALARS) pco->Laplacian(s, s_laplacian_cc, il, iu, jl, ju, kl, ku, nl, nus);

  // Compute fourth-order approximation to cell-centered conserved variables
  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // We do not actually need to store all cell-centered conserved variables,
          // but the ConservedToPrimitive() implementation operates on 4D arrays
          u_cc(n,k,j,i) = cons(n,k,j,i) - C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }
  // Scalars
  for (int n=nl; n<=nus; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // We do not actually need to store all cell-centered conserved variables,
          // but the ConservedToPrimitive() implementation operates on 4D arrays
          s_cc(n,k,j,i) = s(n,k,j,i) - C*s_laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Compute Laplacian of 2nd-order approximation to cell-averaged primitive variables
  pco->Laplacian(prim, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);
  if (NSCALARS) pco->Laplacian(r, s_laplacian_cc, il, iu, jl, ju, kl, ku, nl, nus);

  // Convert cell-centered conserved values to cell-centered primitive values
  ConservedToPrimitive(u_cc, prim_old, b, w_cc, bcc, s_cc, r_old, r, pco, il, iu,
                       jl, ju, kl, ku);

  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // Compute fourth-order approximation to cell-averaged primitive variables
          prim(n,k,j,i) = w_cc(n,k,j,i) + C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }
  for (int n=nl; n<=nus; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // Compute fourth-order approximation to cell-averaged primitive variables
          r(n,k,j,i) = r_cc(n,k,j,i) + C*s_laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Reapply primitive variable floors
  // Cannot fuse w/ above loop since floors are applied to all NHYDRO variables at once
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        ApplyPrimitiveConservedFloors(prim, cons, bcc, r, s, k, j, i);
        if (RELATIVISTIC_DYNAMICS && NSCALARS) // in this case scalars are not floored yet
          ApplyPassiveScalarPrimitiveConservedFloors(s, prim, r, k, j, i);
      }
    }
  }

  return;
}
