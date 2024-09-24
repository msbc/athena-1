//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffer_utils.cpp
//! \brief namespace containing buffer utilities.

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"

// External library headers
#ifdef MPI_PARALLEL
#include <mpi.h>  // MPI_COMM_WORLD, MPI_INFO_NULL
#endif

void BroadcastRealArray(AthenaArray<Real> &array, int root=0) {
  #ifndef MPI_PARALLEL
  return;
  #else

  int dims[MAX_RANK_ARRAY];
  #if MAX_RANK_ARRAY != 6
  #error "BroadcastRealArray has to be updated for MAX_RANK_ARRAY != 6"
  #endif
  dims[0] = array.GetDim6();
  dims[1] = array.GetDim5();
  dims[2] = array.GetDim4();
  dims[3] = array.GetDim3();
  dims[4] = array.GetDim2();
  dims[5] = array.GetDim1();

  MPI_Bcast(dims, MAX_RANK_ARRAY, MPI_INT, root, MPI_COMM_WORLD);
  int n = 1;
  for (int i=0; i<MAX_RANK_ARRAY; ++i) n *= dims[i];
  if (Globals::my_rank != root && !array.IsAllocated()) {
    array.NewAthenaArray(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
  }
  MPI_Bcast(array.data(), n, MPI_ATHENA_REAL, root, MPI_COMM_WORLD);
  #endif // MPI_PARALLEL
}