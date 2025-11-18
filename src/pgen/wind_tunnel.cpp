//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file default_pgen.cpp
//! \brief Provides default (empty) versions of all functions in problem generator files
//! This means user does not have to implement these functions if they are not needed.
//!
//! The attribute "weak" is used to ensure the loader selects the user-defined version of
//! functions rather than the default version given here.
//!
//! The attribute "alias" may be used with the "weak" functions (in non-defining
//! declarations) in order to have them refer to common no-operation function definition
//! in the same translation unit. Target function must be specified by mangled name
//! unless C linkage is specified.
//!
//! This functionality is not in either the C nor the C++ standard. These GNU extensions
//! are largely supported by LLVM, Intel, IBM, but may affect portability for some
//! architecutres and compilers. In such cases, simply define all 6 of the below class
//! functions in every pgen/*.cpp file (without any function attributes).

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if !NON_BAROTROPIC_EOS || GENERAL_EOS
#error "This problem generator requires the adiabatic equation of state."
#endif

#ifndef HDF5OUTPUT
#error "This problem generator requires HDF5 output."
#endif

//#define USE_UOV

void UserSrc(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void WindflowInnerX1(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void WindflowOuterX1(MeshBlock *pmb, Coordinates *pco,
                     AthenaArray<Real> &a,
                     FaceField &b, Real time, Real dt,
                     int il, int iu, int jl, int ju, int kl, int ku, int ngh);
namespace vars {
  Real rho0, p0, e0, gamma, v0;
}

// 3x members of Mesh class:

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  EnrollUserExplicitSourceFunction(UserSrc);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, WindflowInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, WindflowOuterX1);
  }

  //int nx1 = mesh_size.nx1;
  //int nx2 = mesh_size.nx2;
  //int nx3 = mesh_size.nx3;
  //AllocateRealUserMeshDataField(1);
  //AthenaArray<Real> &data = ruser_mesh_data[0];
  //data.NewAthenaArray(nx3, nx2, nx1);
  //
  //if (Globals::my_rank == 0) {
  //  std::string filename = pin->GetOrAddString("problem", "input_filename", "input.h5");
  //  std::string dataset = pin->GetOrAddString("problem", "dataset", "data");
  //  int start_file[3] = {0, 0, 0};
  //  int count_file[3] = {nx3, nx2, nx1};
  //  int start_mem[3] = {0, 0, 0};
  //  int count_mem[3] = {nx3, nx2, nx1};
  //  HDF5ReadRealArray(filename.c_str(), dataset.c_str(), 3, start_file, count_file, 3,
  //                    start_mem, count_mem, data, true);
  //}

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  // do nothing
  return;
}

// 4x members of MeshBlock class:

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  const int nx1 = block_size.nx1;
  const int nx2 = block_size.nx2;
  const int nx3 = block_size.nx3;

  vars::gamma = peos->GetGamma();
  vars::rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  vars::p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  vars::e0 = vars::p0 / (vars::gamma - 1.0);
  vars::v0 = pin->GetOrAddReal("problem", "v0", -1.0);
  if (vars::v0 < 0.0) {
    Real mach = pin->GetOrAddReal("problem", "mach", 1.0);
    vars::v0 = mach * std::sqrt(vars::gamma * vars::p0 / vars::rho0);
  }

  AllocateRealUserMeshBlockDataField(1);
  AthenaArray<Real> &data = ruser_meshblock_data[0];
  data.NewAthenaArray(1, ncells2, ncells1);

  std::string filename = pin->GetOrAddString("problem", "input_filename", "input.h5");
  std::string dataset = pin->GetOrAddString("problem", "dataset", "data");
  int lx1 = static_cast<int>(loc.lx1);
  int lx2 = static_cast<int>(loc.lx2);
  int lx3 = static_cast<int>(loc.lx3);
  int start_file[3] = {0, lx2 * nx2, lx1 * nx1};
  int count_file[3] = {1, nx2, nx1};
  int start_mem[3] = {0, js, is};
  int count_mem[3] = {1, nx2, nx1};
  HDF5ReadRealArray(filename.c_str(), dataset.c_str(), 3, start_file, count_file, 3,
                    start_mem, count_mem, data, true);
#ifdef USE_UOV
  AllocateUserOutputVariables(1);
  SetUserOutputVariableName(0, "data");
#endif // USE_UOV
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> &data = ruser_meshblock_data[0];
  const Real rho0 = vars::rho0;
  const Real e0 = vars::e0;
  Real rho;

  const bool left = (pmy_mesh->mesh_size.x1min == block_size.x1min);
  for (int k=ks; k<=ke; k++) {
    const bool write_plane = (pmy_mesh->mesh_size.x3max > block_size.x3max) || (k < ke);
    for (int j=js; j<=je; j++) {
#pragma omp simd
      for (int i=is; i<=ie; i++) {
        rho = write_plane ? data(0, j, i) : 0.0;
        phydro->u(IDN,k,j,i) = (rho > 0) ? rho : rho0;
        phydro->u(IM1,k,j,i) = (left && i==is) ? rho * vars::v0 : 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = e0;
        }
        if (NSCALARS > 0) {
          if (rho > 0) {
            for (int n=0; n<NSCALARS; ++n) {
              pscalars->s(n,k,j,i) = rho;
            }
          } else {
            for (int n=0; n<NSCALARS; ++n) {
              pscalars->s(n,k,j,i) = 0.0;
            }
          }
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================

void UserSrc(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  AthenaArray<Real> &data = pmb->ruser_meshblock_data[0];
  Real rho;
  const Real e0 = vars::e0;

  const int ke = pmb->ke > 1 ? pmb->ke - 1 : 0;
  for (int k=pmb->ks; k<=ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        rho = data(0, j, i);
        if (rho > 0) {
          cons(IDN, k, j, i) = rho;
          cons(IM1, k, j, i) = 0.0;
          cons(IM2, k, j, i) = 0.0;
          cons(IM3, k, j, i) = 0.0;
          if (NON_BAROTROPIC_EOS) {
            cons(IEN, k, j, i) = e0;
          }
          if (NSCALARS > 0) {
            for (int n=0; n<NSCALARS; ++n) {
              cons_scalar(n, k, j, i) = rho;
            }
          }
        }
      }
    }
  }
}

void WindflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &a, FaceField &b,
                     Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                     int ngh) {
  const Real rho0 = vars::rho0;
  const Real p0 = vars::p0;
  const Real v0 = vars::v0;
  AthenaArray<Real> &prim = pmb->phydro->w;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,il-i) = rho0;
        prim(IVX,k,j,il-i) = v0;
        prim(IVY,k,j,il-i) = 0.0;
        prim(IVZ,k,j,il-i) = 0.0;
        prim(IPR,k,j,il-i) = p0;
      }
    }
  }
}

void WindflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &a, FaceField &b,
                     Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                     int ngh) {
  const Real rho0 = vars::rho0;
  const Real p0 = vars::p0;
  const Real v0 = vars::v0;
  AthenaArray<Real> &prim = pmb->phydro->w;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IVX,k,j,iu+i) = prim(IVX,k,j,iu) > 0.0 ? prim(IVX,k,j,iu) : 0.0;
        prim(IVY,k,j,iu+i) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
      }
    }
  }
}

#ifdef USE_UOV
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // Prepare scratch arrays
  AthenaArray<Real> &data = ruser_meshblock_data[0];

  // Go through all cells
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0,k,j,i) = data(0, j, i);
      }
    }
  }
  return;
}
#endif // USE_UOV
