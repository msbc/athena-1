//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file interp_table.cpp
//! \brief implements functions in class InterpTable2D an intpolated lookup table

// C headers

// C++ headers
//#include <cmath>
#include <sstream>   // stringstream
#include <stdexcept> // std::invalid_argument

// Athena++ headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray
#include "../coordinates/coordinates.hpp" // Coordinates
#include "./interp_table.hpp"

//////////////////////////////////////////
// 1D interpolation class InterpTable1D //
//////////////////////////////////////////

//! A contructor that setts the size of the table with number of variables nvar
//! and length of nx (interpolated dimension)
InterpTable1D::InterpTable1D(const int nvar, const int nx) {
  SetSize(nvar, nx);
}

//! Set size of table
void InterpTable1D::SetSize(const int nvar, const int nx) {
  nvar_ = nvar; // number of variables/tables
  nx_ = nx; // faster indexing dimension
  data.NewAthenaArray(nvar, nx);
}

//! Set the coordinate limits for x
void InterpTable1D::SetXlim(Real xmin, Real xmax) {
  xmin_ = xmin;
  xmax_ = xmax;
  xnorm_ = (nx_ - 1) / (xmax - xmin);
}

void InterpTable1D::GetXlim(Real &xmin, Real &xmax) {
  xmin = xmin_;
  xmax = xmax_;
}

void InterpTable1D::GetSize(int &nvar, int &nx) {
  nvar = nvar_;
  nx = nx_;
}

//! Linear interpolation
Real InterpTable1D::interpolate(int var, Real x) {
  x = (x - xmin_) * xnorm_;
  int xil = static_cast<int>(x); // lower x index
  // if off table, do linear extrapolation
  if (xil < 0) { // below xmin
    xil = 0;
  } else if (xil >= nx_ - 1) { // above xmax
    xil = nx_ - 2;
  }
  Real xrl = 1 + xil - x;  // x residual
  // Sample from the 2 nearest data points and weight appropriately
  return xrl * data(var, xil) + (1-xrl) * data(var, xil+1);
}

//////////////////////////////////////////
// 2D interpolation class InterpTable2D //
//////////////////////////////////////////

//! A contructor that setts the size of the table with number of variables nvar
//! and dimensions nx2 x nx1 (interpolated dimensions)
InterpTable2D::InterpTable2D(const int nvar, const int nx2, const int nx1) {
  SetSize(nvar, nx2, nx1);
}

//! Set size of table
void InterpTable2D::SetSize(const int nvar, const int nx2, const int nx1) {
  nvar_ = nvar; // number of variables/tables
  nx2_ = nx2; // slower indexing dimension
  nx1_ = nx1; // faster indexing dimension
  data.NewAthenaArray(nvar, nx2, nx1);
}

//! Set the corrdinate limits for x1
void InterpTable2D::SetX1lim(Real x1min, Real x1max) {
  x1min_ = x1min;
  x1max_ = x1max;
  x1norm_ = (nx1_ - 1) / (x1max - x1min);
}

//! Set the corrdinate limits for x2
void InterpTable2D::SetX2lim(Real x2min, Real x2max) {
  x2min_ = x2min;
  x2max_ = x2max;
  x2norm_ = (nx2_ - 1) / (x2max - x2min);
}

void InterpTable2D::GetX1lim(Real &x1min, Real &x1max) {
  x1min = x1min_;
  x1max = x1max_;
}


void InterpTable2D::GetX2lim(Real &x2min, Real &x2max) {
  x2min = x2min_;
  x2max = x2max_;
}

void InterpTable2D::GetSize(int &nvar, int &nx2, int &nx1) {
  nvar = nvar_;
  nx2 = nx2_;
  nx1 = nx1_;
}

//! Bilinear interpolation
Real InterpTable2D::interpolate(int var, Real x2, Real x1) {
  Real x, y, xrl, yrl, out;
  x = (x2 - x2min_) * x2norm_;
  y = (x1 - x1min_) * x1norm_;
  int xil = static_cast<int>(x); // lower x index
  int yil = static_cast<int>(y); // lower y index
  int nx = nx2_;
  int ny = nx1_;
  // if off table, do linear extrapolation
  if (xil < 0) { // below xmin
    xil = 0;
  } else if (xil >= nx - 1) { // above xmax
    xil = nx - 2;
  }
  xrl = 1 + xil - x;  // x residual

  if (yil < 0) { // below ymin
    yil = 0;
  } else if (yil >= ny - 1) { // above ymax
    yil = ny - 2;
  }
  yrl = 1 + yil - y;  // y residual

  // Sample from the 4 nearest data points and weight appropriately
  // data is an attribute of the eos class
  out =   xrl  *  yrl  *data(var, xil , yil )
          +   xrl  *(1-yrl)*data(var, xil ,yil+1)
          + (1-xrl)*  yrl  *data(var,xil+1, yil )
          + (1-xrl)*(1-yrl)*data(var,xil+1,yil+1);
  return out;
}

//////////////////////////////////////////
// ND interpolation class InterpTableND //
//////////////////////////////////////////

InterpTableND::InterpTableND(int nvar, int n, int *shape) {
  SetSize(nvar, n, shape);
}

InterpTableND::InterpTableND(AthenaArray<int> shape) {
  SetSize(shape);
}

InterpTableND::InterpTableND(int nvar, AthenaArray<int> shape) {
  SetSize(nvar, shape);
}

InterpTableND::InterpTableND(AthenaArray<Real> src){
  const int shape[] = {src.GetDim6(), src.GetDim5(), src.GetDim4(), src.GetDim3(),
                       src.GetDim2(), src.GetDim1()};
  int n = 0;
  while (shape[n] == 1) n++;
  nvar_ = shape[n];
  ndim_ = 5 - n;
  shape_.NewAthenaArray(ndim_);
  for (int i = 0; i < ndim_; i++)
    shape_(i) = shape[i+n+1];
  InitAux_();
  data.InitWithShallowSlice(src, ndim_ + 1, 0, nvar_);
}

//! Set size of table
void InterpTableND::SetSize(int nvar, int n, int *shape) {
  nvar_ = nvar; // number of variables/tables
  ndim_ = n; // number of dimensions
  shape_.NewAthenaArray(n);
  for (int i=0; i<n; ++i)
    shape_(i) = shape[i];
  InitAux_();
}

void InterpTableND::SetSize(int nvar, AthenaArray<int> shape) {
  SetSize(nvar, shape.GetDim1(), shape.data());
}

void InterpTableND::SetSize(AthenaArray<int> shape) {
  SetSize(shape(0), shape.GetDim1() - 1, shape.data() + 1);
}

void InterpTableND::AllocateData() {
  if (ndim_ == 1) {
    data.NewAthenaArray(nvar_, shape_(0));
  } else if (ndim_ == 2) {
    data.NewAthenaArray(nvar_, shape_(0), shape_(1));
  } else if (ndim_ == 3) {
    data.NewAthenaArray(nvar_, shape_(0), shape_(1), shape_(2));
  } else if (ndim_ == 4) {
    data.NewAthenaArray(nvar_, shape_(0), shape_(1), shape_(2), shape_(3));
  } else if (ndim_ == 5) {
    data.NewAthenaArray(nvar_, shape_(0), shape_(1), shape_(2), shape_(3), shape_(4));
  } else if ((ndim_ == 6) && (nvar_ == 1)) {
    data.NewAthenaArray(shape_(0), shape_(1), shape_(2), shape_(3), shape_(4), shape_(5));
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in InterpTableND::InitData_" << std::endl
        << "Table requires more dimensions than is supported by AthenaArray."
        << std::endl;
    ATHENA_ERROR(msg);
  }
}

void InterpTableND::InitAux_() {
  strides_.NewAthenaArray(ndim_);
  minimums_.NewAthenaArray(ndim_);
  maximums_.NewAthenaArray(ndim_);
  norms_.NewAthenaArray(ndim_);
  location_.NewAthenaArray(ndim_);
  int stride = 1;
  for (int i = 0; i < ndim_; ++i) {
    minimums_(i) = 0.;
    maximums_(i) = shape_(i) - 1.0;
    norms_(i) = 1.0;
    strides_(ndim_ - i - 1) = stride;
    stride *= shape_(i);
  }
}

//! Set the coordinate limits
void InterpTableND::SetLimits(AthenaArray<Real> low, AthenaArray<Real> high) {
  SetLimits(low.data(), high.data());
}

//! Set the coordinate limits
void InterpTableND::SetLimits(Real *low, Real *high) {
  for (int n = 0; n < ndim_; ++n) {
    minimums_(n) = low[n];
    maximums_(n) = high[n];
    norms_(n) = (shape_(n) - 1.0) / (high[n] - low[n]);
  }
}

int InterpTableND::getShape(int n) {
  return shape_(n);
}

Real InterpTableND::getMin(int n) {
  return minimums_(n);
}

Real InterpTableND::getMax(int n) {
  return maximums_(n);
}

int InterpTableND::getNdim() {
  return ndim_;
}

int InterpTableND::getNvar() {
  return nvar_;
}

Real InterpTableND::interpolate(int nvar, AthenaArray<Real> location) {
  return InterpTableND::interpolate(nvar, location.data());
}

Real InterpTableND::interpolate(int nvar, Real *location) {
  for (int n = 0; n < ndim_; ++n) location_(n) = (location[n] - minimums_(n)) * norms_(n);
  return interp_nd(ndim_, shape_.data(), location_.data(), data.data(), strides_.data());
}
// interp_nd(int nd, int *shape, Real *indices, Real *data, int *strides=nullptr)
// Auxiliary functions

//! nD-linear interpolation
Real inline interp_1d(Real x, Real *data, int n, int stride=1) {
  int xil = static_cast<int>(x); // lower x index
  // if off table, do linear extrapolation
  if (xil < 0) { // below xmin
    xil = 0;
  } else if (xil >= n - 1) { // above xmax
    xil = n - 2;
  }
  Real xrl = 1 + xil - x;  // x residual

  // Sample from the 2 nearest data points and weight appropriately
  data += xil * stride;
  return xrl * data[0] + (1-xrl) * data[stride];
}

//! nD-linear interpolation
Real inline interp_nd(int nd, int *shape, Real *indices, Real *data,
                      int *strides=nullptr) {
  int n = shape[0];
  Real x = indices[0];
  if (nd == 1) {
    return interp_1d(x, data, n);
  }
  int xil = static_cast<int>(x); // lower x index
  // if off table, do linear extrapolation
  if (xil < 0) { // below xmin
    xil = 0;
  } else if (xil >= n - 1) { // above xmax
    xil = n - 2;
  }
  Real xrl = 1 + xil - x;  // x residual
  if (!strides) {
    int stride = 1;
    strides = new int[nd];
    for (int n = nd; n >= 0; --n) {
      strides[n] = stride;
      stride *= shape[n];
    }
  }
  return     xrl   * interp_nd(nd-1, shape+1, indices+1, data, strides+1)
         + (1-xrl) * interp_nd(nd-1, shape+1, indices+1, data+strides[0], strides+1);
}
