//#define MYDEBUG
//#define MYDEBUG1
//#define MAINTEST
//#define TEST1
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file helmholtz.cpp
//  \brief implements the Helmholtz EOS in general EOS framework.
//  See http://cococubed.asu.edu/code_pages/eos.shtml and references therein
//======================================================================================

// C headers

// C++ headers
#include <iostream> // ifstream
#include <sstream>  // stringstream

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/interp_table.hpp"
#include "../eos.hpp"

namespace HelmholtzConstants {
  const int nOut = 6;
  const Real forth=4.0/3.0, third=1.0/3.0;
  //const Real avo=6.0221417930e23, kerg=1.380650424e-16, clight=2.99792458e10,
  //                  ssol=5.6704e-5;//, amu=1.66053878283e-24, h=6.6260689633e-27;
  const Real ssol=5.6704e-5;
  const Real qe=4.8032042712e-10, avo=6.0221417930e23, clight=2.99792458e10,
                    kerg=1.380650424e-16;
  const Real asol=4.0*ssol/clight, light2=clight*clight;
  const Real asoli3=asol/3.0;
  //const Real kergavo=kerg*avo, sioncon = (2.0 * PI * amu * kerg)/(h*h)

  // constants for the uniform background coulomb correction
  const Real a1=-0.898004, b1=0.96786, c1=0.220703, d1=-0.86097, e1=2.5269,
                    a2=0.29561, b2=1.9885, c2=0.288675;
  //const Real qe=4.8032042712e-10;
  const Real esqu=qe*qe;
} // namespace HelmholtzConstants

class HelmTable {
 public:
  int jmax, imax;
  Real tlo, thi, tstp, tstpi, dlo, dhi, dstp, dstpi;
  HelmTable(ParameterInput *pin, EosTable *ptable) {
    jmax = ptable->nEgas;
    tlo   = ptable->logEgasMin;
    thi   = ptable->logEgasMax;
    tstp  = (thi - tlo)/static_cast<Real>(jmax-1);
    tstpi = 1.0/tstp;

    imax = ptable->nRho;
    dlo   = ptable->logRhoMin;
    dhi   = ptable->logRhoMax;
    dstp  = (dhi - dlo)/static_cast<Real>(imax-1);
    dstpi = 1.0/dstp;
#ifdef MYDEBUG
    std::cout << "Helm T: " << jmax << ", " << tlo << ", " << thi << '\n';
    std::cout << "Helm d: " << imax << ", " << dlo << ", " << dhi << '\n';
    std::cout << ptable->nVar << '\n';
#endif

    // construct the temperature and density deltas and their inverses
    t.NewAthenaArray(jmax);
    dt_sav.NewAthenaArray(jmax);
    dt2_sav.NewAthenaArray(jmax);
    dti_sav.NewAthenaArray(jmax);
    dt2i_sav.NewAthenaArray(jmax);
    for (int j=0; j<jmax; ++j) {
      Real tsav = tlo + j*tstp;
      t(j) = std::pow(10.0, tsav);
    }
    for (int j=0; j<jmax-1; ++j) {
      Real dth         = t(j+1) - t(j);
      Real dt2         = dth * dth;
      Real dti         = 1.0/dth;
      Real dt2i        = 1.0/dt2;
      //Real dt3i        = dt2i*dti;
      dt_sav(j)   = dth;
      dt2_sav(j)  = dt2;
      dti_sav(j)  = dti;
      dt2i_sav(j) = dt2i;
      //dt3i_sav(j) = dt3i;
    }
    d.NewAthenaArray(imax);
    dd_sav.NewAthenaArray(imax);
    dd2_sav.NewAthenaArray(imax);
    ddi_sav.NewAthenaArray(imax);
    dd2i_sav.NewAthenaArray(imax);
    for (int i=0; i<imax; ++i) {
      Real dsav = dlo + i*dstp;
      d(i) = std::pow(10.0, dsav);
    }
    for (int i=0; i<imax-1; ++i) {
      Real dd          = d(i+1) - d(i);
      Real dd2         = dd * dd;
      Real ddi         = 1.0/dd;
      Real dd2i        = 1.0/dd2;
      //Real dd3i        = dd2i*ddi;
      dd_sav(i)   = dd;
      dd2_sav(i)  = dd2;
      ddi_sav(i)  = ddi;
      dd2i_sav(i) = dd2i;
      //dd3i_sav(i) = dd3i;
    }
#ifdef MYDEBUG
    printf("t(0), d(0): %.16e, %.16e\n", t(0), d(0));
    printf("t(1), d(1): %.16e, %.16e\n", t(1), d(1));
#endif
    // precission for inversion
    prec = pin->GetOrAddReal("hydro", "helm_prec", 1e-8);
    nmax = pin->GetOrAddInteger("hydro", "helm_nmax", 5000);
    Tfloor = pin->GetOrAddBoolean("hydro", "helm_Tfloor", false);

    fi.NewAthenaArray(36);
    // helmholtz free energy and its derivatives
    f.InitWithShallowSlice(ptable->table.data, 3, 0, 1);
    fd.InitWithShallowSlice(ptable->table.data, 3, 1, 1);
    ft.InitWithShallowSlice(ptable->table.data, 3, 2, 1);
    fdd.InitWithShallowSlice(ptable->table.data, 3, 3, 1);
    ftt.InitWithShallowSlice(ptable->table.data, 3, 4, 1);
    fdt.InitWithShallowSlice(ptable->table.data, 3, 5, 1);
    fddt.InitWithShallowSlice(ptable->table.data, 3, 6, 1);
    fdtt.InitWithShallowSlice(ptable->table.data, 3, 7, 1);
    fddtt.InitWithShallowSlice(ptable->table.data, 3, 8, 1);
    // pressure derivative with density table
    dpdf.InitWithShallowSlice(ptable->table.data, 3, 9, 1);
    dpdfd.InitWithShallowSlice(ptable->table.data, 3, 10, 1);
    dpdft.InitWithShallowSlice(ptable->table.data, 3, 11, 1);
    dpdfdt.InitWithShallowSlice(ptable->table.data, 3, 12, 1);
    // number density table
    xf.InitWithShallowSlice(ptable->table.data, 3, 13, 1);
    xfd.InitWithShallowSlice(ptable->table.data, 3, 14, 1);
    xft.InitWithShallowSlice(ptable->table.data, 3, 15, 1);
    xfdt.InitWithShallowSlice(ptable->table.data, 3, 16, 1);
  }

// OutData has length 6 (HelmholtzConstants::nOut)
  void HelmLookupRhoT(Real den, Real temp, Real ye, Real abar,
                      AthenaArray<Real> &OutData) {
    using namespace HelmholtzConstants;  // NOLINT (build/namespace)
    Real din = ye * den;
    Real ytot1 = 1.0/abar;
    Real zbar = ye * abar;
    //hash locate this temperature and density
    int jat = static_cast<int>((std::log10(temp) - tlo)*tstpi);
    jat = std::max(0,std::min(jat,jmax-1));
    int iat = static_cast<int>((std::log10(din) - dlo)*dstpi);
    iat = std::max(0,std::min(iat,imax-1));
#ifdef MYDEBUG
    std::cout << "i,j: " << iat << ", " << jat << '\n';
#endif

    // initialize
    Real deni    = 1.0/den;
    Real tempi   = 1.0/temp;
    Real kt      = kerg * temp;
    Real ktinv   = 1.0/kt;

    ////////////////////////
    // radiation section: //
    ////////////////////////

    //printf("a, t, t4: %.16e, %.16e, %.16e\n", asoli3, temp, temp * temp * temp * temp);
    Real prad    = asoli3 * temp * temp * temp * temp;
    Real dpraddd = 0.0;
    Real dpraddt = 4.0 * prad*tempi;
    //Real dpradda = 0.0;
    //Real dpraddz = 0.0;

    Real erad    = 3.0 * prad*deni;
    //Real deraddd = -erad*deni;
    Real deraddt = 3.0 * dpraddt*deni;
    //Real deradda = 0.0;
    //Real deraddz = 0.0;

    //Real srad    = (prad*deni + erad)*tempi;
    //Real dsraddd = (dpraddd*deni - prad*deni*deni + deraddd)*tempi;
    //Real dsraddt = (dpraddt*deni + deraddt - srad)*tempi;
    //Real dsradda = 0.0;
    //Real dsraddz = 0.0;

    //////////////////
    // ion section: //
    //////////////////

    Real xni     = avo * ytot1 * den;
    Real dxnidd  = avo * ytot1;
    Real dxnida  = -xni * ytot1;

    Real pion    = xni * kt;
    Real dpiondd = dxnidd * kt;
    Real dpiondt = xni * kerg;
    Real dpionda = dxnida * kt;
    Real dpiondz = 0.0;

    Real eion    = 1.5 * pion*deni;
    //Real deiondd = (1.5 * dpiondd - eion)*deni;
    Real deiondt = 1.5 * dpiondt*deni;
    //Real deionda = 1.5 * dpionda*deni;
    //Real deiondz = 0.0;

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
    Real s, x, y, z;

    //x       = abar*abar*std::sqrt(abar) * deni/avo;
    //s       = sioncon * temp;
    //z       = x * s * std::sqrt(s);
    //y       = std::log(z);
    //Real sion    = (pion*deni + eion)*tempi + kergavo * ytot1 * y;
    //Real dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi
    //              - kergavo * deni * ytot1;
    //Real dsiondt = (dpiondt*deni + deiondt)*tempi - (pion*deni + eion) * tempi*tempi
    //              + 1.5 * kergavo * tempi*ytot1;
    //x = avo*kerg/abar;
    //Real dsionda = (dpionda*deni + deionda)*tempi + kergavo*ytot1*ytot1* (2.5 - y);
    //Real dsiondz = 0.0;

    ////////////////////////////////
    // electron-positron section: //
    ////////////////////////////////

    // assume complete ionization
    // Real xnem = xni * zbar;

    // access the table locations only once
    fi(0)  = f(iat,jat);
    fi(1)  = f(iat+1,jat);
    fi(2)  = f(iat,jat+1);
    fi(3)  = f(iat+1,jat+1);
    fi(4)  = ft(iat,jat);
    fi(5)  = ft(iat+1,jat);
    fi(6)  = ft(iat,jat+1);
    fi(7)  = ft(iat+1,jat+1);
    fi(8)  = ftt(iat,jat);
    fi(9)  = ftt(iat+1,jat);
    fi(10) = ftt(iat,jat+1);
    fi(11) = ftt(iat+1,jat+1);
    fi(12) = fd(iat,jat);
    fi(13) = fd(iat+1,jat);
    fi(14) = fd(iat,jat+1);
    fi(15) = fd(iat+1,jat+1);
    fi(16) = fdd(iat,jat);
    fi(17) = fdd(iat+1,jat);
    fi(18) = fdd(iat,jat+1);
    fi(19) = fdd(iat+1,jat+1);
    fi(20) = fdt(iat,jat);
    fi(21) = fdt(iat+1,jat);
    fi(22) = fdt(iat,jat+1);
    fi(23) = fdt(iat+1,jat+1);
    fi(24) = fddt(iat,jat);
    fi(25) = fddt(iat+1,jat);
    fi(26) = fddt(iat,jat+1);
    fi(27) = fddt(iat+1,jat+1);
    fi(28) = fdtt(iat,jat);
    fi(29) = fdtt(iat+1,jat);
    fi(30) = fdtt(iat,jat+1);
    fi(31) = fdtt(iat+1,jat+1);
    fi(32) = fddtt(iat,jat);
    fi(33) = fddtt(iat+1,jat);
    fi(34) = fddtt(iat,jat+1);
    fi(35) = fddtt(iat+1,jat+1);

    // various differences
    Real xt  = std::max((temp - t(jat))*dti_sav(jat), 0.0);
    Real xd  = std::max((din  - d(iat))*ddi_sav(iat), 0.0);
    Real mxt = 1.0 - xt;
    Real mxd = 1.0 - xd;
#ifdef MYDEBUG
    printf("xd, xt: %.16e, %.16e\n", xd, xt);
    printf("din, d(i): %.16e, %.16e, %.16e\n", din, d(iat), ddi_sav(iat));
    printf("tin, t(j): %.16e, %.16e, %.16e\n", temp, t(jat), dti_sav(jat));
#endif

    // the six density and six temperature basis functions
    Real si0t =   psi0(xt);
    Real si1t =   psi1(xt)*dt_sav(jat);
    Real si2t =   psi2(xt)*dt2_sav(jat);

    Real si0mt =  psi0(mxt);
    Real si1mt = -psi1(mxt)*dt_sav(jat);
    Real si2mt =  psi2(mxt)*dt2_sav(jat);

    Real si0d =   psi0(xd);
    Real si1d =   psi1(xd)*dd_sav(iat);
    Real si2d =   psi2(xd)*dd2_sav(iat);

    Real si0md =  psi0(mxd);
    Real si1md = -psi1(mxd)*dd_sav(iat);
    Real si2md =  psi2(mxd)*dd2_sav(iat);

    // derivatives of the weight functions
    Real dsi0t =   dpsi0(xt)*dti_sav(jat);
    Real dsi1t =   dpsi1(xt);
    Real dsi2t =   dpsi2(xt)*dt_sav(jat);

    Real dsi0mt = -dpsi0(mxt)*dti_sav(jat);
    Real dsi1mt =  dpsi1(mxt);
    Real dsi2mt = -dpsi2(mxt)*dt_sav(jat);

    Real dsi0d =   dpsi0(xd)*ddi_sav(iat);
    Real dsi1d =   dpsi1(xd);
    Real dsi2d =   dpsi2(xd)*dd_sav(iat);

    Real dsi0md = -dpsi0(mxd)*ddi_sav(iat);
    Real dsi1md =  dpsi1(mxd);
    Real dsi2md = -dpsi2(mxd)*dd_sav(iat);

    // second derivatives of the weight functions
    Real ddsi0t =   ddpsi0(xt)*dt2i_sav(jat);
    Real ddsi1t =   ddpsi1(xt)*dti_sav(jat);
    Real ddsi2t =   ddpsi2(xt);

    Real ddsi0mt =  ddpsi0(mxt)*dt2i_sav(jat);
    Real ddsi1mt = -ddpsi1(mxt)*dti_sav(jat);
    Real ddsi2mt =  ddpsi2(mxt);

    //Real ddsi0d =   ddpsi0(xd)*dd2i_sav(iat);
    //Real ddsi1d =   ddpsi1(xd)*ddi_sav(iat);
    //Real ddsi2d =   ddpsi2(xd);

    //Real ddsi0md =  ddpsi0(mxd)*dd2i_sav(iat);
    //Real ddsi1md = -ddpsi1(mxd)*ddi_sav(iat);
    //Real ddsi2md =  ddpsi2(mxd);

    // the free energy
    Real free  = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md,
                    si1md, si2md);
#ifdef MYDEBUG
    printf("free: %.16e\n", fi(0));
#endif

    // derivative with respect to density
    Real df_d  = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d,
                    dsi0md, dsi1md, dsi2md);

    // derivative with respect to temperature
    Real df_t = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d,
                   si0md, si1md, si2md);
#ifdef MYDEBUG
    printf("df_t: %.16e, %.16e\n", df_t, fi(4));
#endif


    // derivative with respect to density**2
    //Real df_dd = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, ddsi0d, ddsi1d, ddsi2d,
    //                ddsi0md, ddsi1md, ddsi2md);

    // derivative with respect to temperature**2
    Real df_tt = h5(fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d,
                    si2d, si0md, si1md, si2md);

    // derivative with respect to temperature and density
    Real df_dt = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d,
                    dsi0md, dsi1md, dsi2md);

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    si0t   =  xpsi0(xt);
    si1t   =  xpsi1(xt)*dt_sav(jat);

    si0mt  =  xpsi0(mxt);
    si1mt  =  -xpsi1(mxt)*dt_sav(jat);

    si0d   =  xpsi0(xd);
    si1d   =  xpsi1(xd)*dd_sav(iat);

    si0md  =  xpsi0(mxd);
    si1md  =  -xpsi1(mxd)*dd_sav(iat);

    // derivatives of weight functions
    dsi0t  = xdpsi0(xt)*dti_sav(jat);
    dsi1t  = xdpsi1(xt);

    dsi0mt = -xdpsi0(mxt)*dti_sav(jat);
    dsi1mt = xdpsi1(mxt);

    dsi0d  = xdpsi0(xd)*ddi_sav(iat);
    dsi1d  = xdpsi1(xd);

    dsi0md = -xdpsi0(mxd)*ddi_sav(iat);
    dsi1md = xdpsi1(mxd);

    // look in the pressure derivative only once
    fi(0)  = dpdf(iat,jat);
    fi(1)  = dpdf(iat+1,jat);
    fi(2)  = dpdf(iat,jat+1);
    fi(3)  = dpdf(iat+1,jat+1);
    fi(4)  = dpdft(iat,jat);
    fi(5)  = dpdft(iat+1,jat);
    fi(6)  = dpdft(iat,jat+1);
    fi(7)  = dpdft(iat+1,jat+1);
    fi(8)  = dpdfd(iat,jat);
    fi(9)  = dpdfd(iat+1,jat);
    fi(10) = dpdfd(iat,jat+1);
    fi(11) = dpdfd(iat+1,jat+1);
    fi(12) = dpdfdt(iat,jat);
    fi(13) = dpdfdt(iat+1,jat);
    fi(14) = dpdfdt(iat,jat+1);
    fi(15) = dpdfdt(iat+1,jat+1);

    // pressure derivative with density
    Real dpepdd = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);
    dpepdd = std::max(ye * dpepdd, 1.0e-30);

    // skipping vvv
    // look in the electron chemical potential table only once
    //
    // look in the number density table only once
    // skipping ^^^

    // the desired electron-positron thermodynamic quantities

    // dpepdd at high temperatures and low densities is below the
    // floating point limit of the subtraction of two large terms.
    // since dpresdd doesn't enter the maxwell relations at all, use the
    // bicubic interpolation done above instead of the formally correct expression
    x = din * din;
    Real pele   = x * df_d;
    Real dpepdt = x * df_dt;
    //        dpepdd  = ye * (x * df_dd + 2.0 * din * df_d)
    //s       = dpepdd/ye - 2.0 * din * df_d;
    //Real dpepda  = -ytot1 * (2.0 * pele + s * din);
    //Real dpepdz  = den*ytot1*(2.0 * din * df_d  +  s);


    //x       = ye * ye;
    Real sele    = -df_t * ye;
    Real dsepdt  = -df_tt * ye;
    //Real dsepdd  = -df_dt * x;
    //Real dsepda  = ytot1 * (ye * df_dt * din - sele);
    //Real dsepdz  = -ytot1 * (ye * df_dt * den  + df_t);


    Real eele    = ye*free + temp * sele;
    Real deepdt  = temp * dsepdt;
    //Real deepdd  = x * df_d + temp * dsepdd;
    //Real deepda  = -ye * ytot1 * (free +  df_d * din) + temp * dsepda;
    //Real deepdz  = ytot1* (free + ye * df_d * den) + temp * dsepdz;

    // coulomb section:

    // uniform background corrections only
    // from yakovlev & shalybkov 1989
    // lami is the average ion seperation
    // plasg is the plasma coupling parameter

    z        = forth * PI;
    s        = z * xni;
    Real dsdd     = z * dxnidd;
    Real dsda     = z * dxnida;

    Real lami     = std::pow(s, -third);
    Real inv_lami = 1.0/lami;
    z = -third * lami;
    Real lamidd   = z * dsdd/s;
    Real lamida   = z * dsda/s;

    Real plasg    = zbar*zbar*esqu*ktinv*inv_lami;
#ifdef MYDEBUG
    printf("plasg xni sele: %.16e, %.16e, %.16e\n", plasg, xni, sele);
#endif
    z = -plasg * inv_lami;
    Real plasgdd  = z * lamidd;
    Real plasgda  = z * lamida;
    Real plasgdt  = -plasg*ktinv * kerg;
    Real plasgdz  = 2.0 * plasg/zbar;

    Real ecoul, pcoul, scoul, decouldd, decouldt, decoulda, decouldz, dpcouldd, dpcouldt,
    dpcoulda, dpcouldz, dscouldd, dscouldt, dscoulda, dscouldz;
    // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.0) {
      x        = std::pow(plasg, 0.25);
      y        = avo * ytot1 * kerg;
      ecoul    = y * temp * (a1*plasg + b1*x + c1/x + d1);
      pcoul    = third * den * ecoul;
      Real idkbro = 3.0e0*b1*x - 5.0e0*c1/x+d1*(std::log(plasg)-1.0e0) - e1;
      scoul    = -y * idkbro;
#ifdef MYDEBUG
      //printf("a1, b1, c1, d1, ec: %.16e, %.16e, %.16e, %.16e, %.16e, %.16e\n",
      //       a1, b1, c1, d1, e1, ecoul);
      //printf("scoul, y, x: %.16e, %.16e, %.16e, %.16e\n", scoul, y, x,
      //       3.0*b1*x - 5.0*c1/x + d1 * (std::log(plasg) - 1.0) - e1);
      //printf("log(plasg): %.16e\n", std::log(plasg));
      printf("idk bro: %.16e\n", idkbro);
      printf("%.16e, %.16e, %.16e, %.16e\n", 3.0e0*b1*x, 5.0e0*c1/x,
             d1*(std::log(plasg)-1.0e0), e1);
#endif

      y        = avo*ytot1*kt*(a1 + 0.25/plasg*(b1*x - c1/x));
      decouldd = y * plasgdd;
      decouldt = y * plasgdt + ecoul/temp;
      decoulda = y * plasgda - ecoul/abar;
      decouldz = y * plasgdz;

      y        = third * den;
      dpcouldd = third * ecoul + y*decouldd;
      dpcouldt = y * decouldt;
      dpcoulda = y * decoulda;
      dpcouldz = y * decouldz;

      y        = -avo*kerg/(abar*plasg)*(0.75*b1*x+1.25*c1/x+d1);
      dscouldd = y * plasgdd;
      dscouldt = y * plasgdt;
      dscoulda = y * plasgda - scoul/abar;
      dscouldz = y * plasgdz;

    // yakovlev & shalybkov 1989 equations 102, 103, 104
    } else { // (plasg < 1.0)
      x        = plasg*std::sqrt(plasg);
      y        = std::pow(plasg, b2);
      z        = c2 * x - third * a2 * y;
      pcoul    = -pion * z;
      ecoul    = 3.0 * pcoul/den;
      scoul    = -avo/abar*kerg*(c2*x -a2*(b2-1.0)/b2*y);

      s        = 1.5*c2*x/plasg - third*a2*b2*y/plasg;
      dpcouldd = -dpiondd*z - pion*s*plasgdd;
      dpcouldt = -dpiondt*z - pion*s*plasgdt;
      dpcoulda = -dpionda*z - pion*s*plasgda;
      dpcouldz = -dpiondz*z - pion*s*plasgdz;

      s        = 3.0/den;
      decouldd = s * dpcouldd - ecoul/den;
      decouldt = s * dpcouldt;
      decoulda = s * dpcoulda;
      decouldz = s * dpcouldz;

      s        = -avo*kerg/(abar*plasg)*(1.5*c2*x-a2*(b2-1.0)*y);
      dscouldd = s * plasgdd;
      dscouldt = s * plasgdt;
      dscoulda = s * plasgda - scoul/abar;
      dscouldz = s * plasgdz;
    }

    x   = prad + pion + pele + pcoul;
    y   = erad + eion + eele + ecoul;
    //z   = srad + sion + sele + scoul;

    if ((x < 0.0) || (y < 0.0)) {
      pcoul    = 0.0;
      dpcouldd = 0.0;
      dpcouldt = 0.0;
      dpcoulda = 0.0;
      dpcouldz = 0.0;
      ecoul    = 0.0;
      decouldd = 0.0;
      decouldt = 0.0;
      decoulda = 0.0;
      decouldz = 0.0;
      scoul    = 0.0;
      dscouldd = 0.0;
      dscouldt = 0.0;
      dscoulda = 0.0;
      dscouldz = 0.0;
    }

    // sum all the gas components
    Real pgas    = pion + pele + pcoul;
    Real egas    = eion + eele + ecoul;
    //Real sgas    = sion + sele + scoul;

    Real dpgasdd = dpiondd + dpepdd + dpcouldd;
    Real dpgasdt = dpiondt + dpepdt + dpcouldt;
    //Real dpgasda = dpionda + dpepda + dpcoulda;
    //Real dpgasdz = dpiondz + dpepdz + dpcouldz;

    //Real degasdd = deiondd + deepdd + decouldd;
    Real degasdt = deiondt + deepdt + decouldt;
    //Real degasda = deionda + deepda + decoulda;
    //Real degasdz = deiondz + deepdz + decouldz;

    //Real dsgasdd = dsiondd + dsepdd + dscouldd;
    //Real dsgasdt = dsiondt + dsepdt + dscouldt;
    //Real dsgasda = dsionda + dsepda + dscoulda;
    //Real dsgasdz = dsiondz + dsepdz + dscouldz;

    // add in radiation to get the total
    Real pres    = prad + pgas;
    Real ener    = erad + egas;
    //Real entr    = srad + sgas;

    Real dpresdd = dpraddd + dpgasdd;
    Real dpresdt = dpraddt + dpgasdt;
    //Real dpresda = dpradda + dpgasda;
    //Real dpresdz = dpraddz + dpgasdz;

    //Real denerdd = deraddd + degasdd;
    Real denerdt = deraddt + degasdt;
    //Real denerda = deradda + degasda;
    //Real denerdz = deraddz + degasdz;

    //Real dentrdd = dsraddd + dsgasdd;
    //Real dentrdt = dsraddt + dsgasdt;
    //Real dentrda = dsradda + dsgasda;
    //Real dentrdz = dsraddz + dsgasdz;

    // for the gas
    // the temperature and density exponents (c&g 9.81 9.82)
    // the specific heat at constant volume (c&g 9.92)
    // the third adiabatic exponent (c&g 9.93)
    // the first adiabatic exponent (c&g 9.97)
    // the second adiabatic exponent (c&g 9.105)
    // the specific heat at constant pressure (c&g 9.98)
    // and relativistic formula for the sound speed (c&g 14.29)

    //Real zz        = pgas*deni;
    //Real zzi       = den/pgas;
    //Real chit_gas  = temp/pgas * dpgasdt;
    //Real chid_gas  = dpgasdd*zzi;
    //Real cv_gas    = degasdt;
    //x         = zz * chit_gas/(temp * cv_gas);
    //Real gam3_gas  = x + 1.0;
    //Real gam1_gas  = chit_gas*x + chid_gas;
    //Real nabad_gas = x/gam1_gas;
    //Real gam2_gas  = 1.0/(1.0 - nabad_gas);
    //Real cp_gas    = cv_gas * gam1_gas/chid_gas;
    //z         = 1.0 + (egas + light2)*zzi;
    //Real sound_gas = clight * std::sqrt(gam1_gas/z);

#ifdef MYDEBUG
    printf("pcou, pele, pion, prad: %.16e, %.16e, %.16e, %.16e\n", pcoul, pele, pion,
           prad);
    printf("ecou, eele, eion, erad: %.16e, %.16e, %.16e, %.16e\n", ecoul, eele, eion,
           erad);
#endif

    // for the totals
    Real zz    = pres*deni;
    Real zzi   = den/pres;
    Real chit  = temp/pres * dpresdt;
    Real chid  = dpresdd*zzi;
    Real cv    = denerdt;
    x     = zz * chit/(temp * cv);
    //Real gam3  = x + 1.0;
    Real gam1  = chit*x + chid;
    //Real nabad = x/gam1;
    //Real gam2  = 1.0/(1.0 - nabad);
    //Real cp    = cv * gam1/chid;
    z     = 1.0 + (ener + light2)*zzi;
    //Real sound = clight * std::sqrt(gam1/z);
    Real asq = light2 * gam1/z;

    // maxwell relations; each is zero if the consistency is perfect
    //x   = den * den;
    //Real dse = temp*dentrdt/denerdt - 1.0;
    //Real dpe = (denerdd*x + temp*dpresdt)/pres - 1.0;
    //Real dsp = -dentrdd*x/dpresdt - 1.0;

    OutData(0) = ener * den;    // convert specific energy to energy density
    OutData(1) = denerdt * den; // convert specific energy to energy density
    OutData(2) = pres;
    OutData(3) = dpresdt;
    OutData(4) = asq;
    OutData(5) = temp;
    OutData(6) = dpresdd;
  }

  // index = 0 for internal energy; index = 2 for pressure; var = int energy or pressure
  void HelmInvert(Real rho, Real GuessTemp, Real ye, Real abar, Real var, int index,
                  AthenaArray<Real> &OutData) {
    Real BrakT[] = {t(0), t(jmax-1)};
    Real BrakVal[] = {0, 0};
    Real InvVar = 1.0 / var;
    Real error = 9e9;
    int nlim = nmax;
    Real LastTemp = BrakT[0];
    HelmLookupRhoT(rho, LastTemp, ye, abar, OutData);
    BrakVal[0] = OutData(index) * InvVar - 1.0;
    Real LastErr = BrakVal[0];
    Real delta;
#ifdef MYDEBUG1
    printf("%d: %.16e, %.16e\n", index, var, rho);
    int mode = 0;
#endif
    while (std::abs(error) > prec) {
      if (BrakVal[0] > 0) {//}* BrakVal[1] > 0) {
        HelmLookupRhoT(rho, BrakT[0], ye, abar, OutData);
        Real low = OutData(index);
        // If we've specified Tfloor and we are below Tmin just use Tmin and return
        if (Tfloor && var < low) {
          return;
        }
        HelmLookupRhoT(rho, BrakT[1], ye, abar, OutData);
        Real high =  OutData(index);
        std::stringstream msg;
        const char *varnames[] = {"e_int", "de_int/dT", "P_gas", "dP/dT", "a^2", "T"};
        printf("ERR (%s): %.4e !<= %.4e !<= %.4e\n", varnames[index], low, var,
               high);
        printf("at rho = %.4e, T_bounds = %.4e, %.4e,\n", rho, BrakT[0], BrakT[1]);
        msg << "### FATAL ERROR in EquationOfState inversion (HelmInvert)"
            << std::endl << "Root not bracketed" << std::endl;
        ATHENA_ERROR(msg);
      }
      // if we are outside brackets use bisection method for a step
      if ((GuessTemp <= BrakT[0]) || (GuessTemp >= BrakT[1])) {
        //GuessTemp = 0.5 * (BrakT[0] + BrakT[1]);
        GuessTemp = std::sqrt(BrakT[0] * BrakT[1]);
#ifdef MYDEBUG1
        mode = 1;
#endif
      }
      HelmLookupRhoT(rho, GuessTemp, ye, abar, OutData);
      error = OutData(index) * InvVar - 1.0;
#ifdef MYDEBUG1
      printf("%04d [%.4g, %.4g, %.4g]; %.4g| %d\n", 1000 - nlim, BrakT[0], GuessTemp,
             BrakT[1], error, mode);
#endif
      // update bracketing values
      if (error < 0) {
        BrakT[0] = GuessTemp;
        BrakVal[0] = error;
      } else {
        BrakT[1] = GuessTemp;
        BrakVal[1] = error;
      }
      if (BrakT[1] <= BrakT[0]) {
        HelmLookupRhoT(rho, BrakT[0], ye, abar, OutData);
        Real low = OutData(index);
        // If we've specified Tfloor and we are below Tmin just use Tmin and return
        if (Tfloor && var < low) {
          return;
        }
        HelmLookupRhoT(rho, BrakT[1], ye, abar, OutData);
        Real high = OutData(index);
        std::stringstream msg;
        const char *varnames[] = {"e_int", "de_int/dT", "P_gas", "dP/dT", "a^2", "T"};
        printf("ERR (%s): %.4e !<= %.4e !<= %.4e\n", varnames[index], low, var,
               high);
        printf("at rho = %.4e, T_bounds = %.4e, %.4e,\n", rho, BrakT[0], BrakT[1]);
        msg << "### FATAL ERROR in EquationOfState inversion (HelmInvert)"
            << std::endl << "Root not bracketed" << std::endl;
        ATHENA_ERROR(msg);
      }
      if (std::abs(error) > 1e-2) {
        // secant method step
        delta = error * (GuessTemp - LastTemp) / (error - LastErr);
        LastTemp = GuessTemp;
        GuessTemp -= delta;
        LastErr = error;
#ifdef MYDEBUG1
        mode = 2;
#endif
      } else {
        // Newton–Raphson step
        delta = var * error / OutData(index + 1);
        GuessTemp -= delta;
#ifdef MYDEBUG1
        mode = 3;
#endif
      }
      if (nlim-- < 0) {
        HelmLookupRhoT(rho, BrakT[0], ye, abar, OutData);
        Real low = OutData(index);
        HelmLookupRhoT(rho, BrakT[1], ye, abar, OutData);
        Real high = OutData(index);
        const char *varnames[] = {"e_int", "de_int/dT", "P_gas", "dP/dT", "a^2", "T"};
        printf("ERR (%s): |%.4e|, |%.4e| > %.4e; %d iterations\n", varnames[index],
               low * InvVar - 1.0, high * InvVar - 1.0, prec, nmax);
        printf("%.4e <= %.4e <= %.4e\n", low, var, high);
        printf("at rho = %.4e, T_bounds = %.4e, %.4e,\n", rho, BrakT[0], BrakT[1]);
        std::stringstream msg;
        msg << "### FATAL ERROR in EquationOfState inversion (HelmInvert)"
            << std::endl << "Cannot converge" << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    if (OutData(5) < t(0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState inversion (HelmInvert)"
          << std::endl << "Cannot converge. Recovered T off table." << std::endl;
      ATHENA_ERROR(msg);
    }
    return;
  }

 private:
  Real prec;
  int nmax;
  bool Tfloor;
  AthenaArray<Real> f, ft, ftt, fd, fdd, fdt, fddt, fdtt, fddtt, fi;
  AthenaArray<Real> dpdf, dpdft, dpdfd, dpdfdt;
  AthenaArray<Real> xf, xft, xfd, xfdt;
  AthenaArray<Real> t, d, dt_sav, dt2i_sav, dti_sav, ddi_sav, dd_sav, dd2_sav, dt2_sav,
                    dd2i_sav;

  //quintic hermite polynomial statement functions
  //psi0 and its derivatives
  inline Real psi0(Real z) {
    return std::pow(z, 3)*(z*(-6.0*z + 15.0) - 10.0) + 1.0;
  }
  inline Real dpsi0(Real z) {
    return z * z * ( z * (-30.0 * z + 60.0) - 30.0);
  }
  inline Real ddpsi0(Real z) {
    return z * ( z * (-120.0 * z + 180.0) - 60.0);
  }

  //psi1 and its derivatives
  inline Real psi1(Real z) {
    return z * ( z * z * ( z * (-3.0 * z + 8.0) - 6.0) + 1.0);
  }
  inline Real dpsi1(Real z) {
    return z * z * ( z * (-15.0 * z + 32.0) - 18.0) + 1.0;
  }
  inline Real ddpsi1(Real z) {
    return z * (z * (-60.0 * z + 96.0) - 36.0);
  }

  //psi2  and its derivatives
  inline Real psi2(Real z) {
    return 0.5 * z * z * ( z* ( z * (-z + 3.0) - 3.0) + 1.0);
  }
  inline Real dpsi2(Real z) {
    return 0.5 * z * ( z * (z * (-5.0 * z + 12.0) - 9.0) + 2.0);
  }
  inline Real ddpsi2(Real z) {
    return 0.5 * (z * ( z * (-20.0 * z + 36.0) - 18.0) + 2.0);
  }

  // cubic hermite polynomial statement functions
  // psi0 & derivatives
  inline Real xpsi0(Real z) {
    return z * z * (2.0*z - 3.0) + 1.0;
  }
  inline Real xdpsi0(Real z) {
    return z * (6.0*z - 6.0);
  }

  // psi1 & derivatives
  inline Real xpsi1(Real z) {
    return z * ( z * (z - 2.0) + 1.0);
  }
  inline Real xdpsi1(Real z) {
    return z * (3.0*z - 4.0) + 1.0;
  }

  //biquintic hermite polynomial statement function
  Real h5(AthenaArray<Real> data, Real w0t, Real w1t, Real w2t, Real w0mt, Real w1mt,
          Real w2mt, Real w0d, Real w1d, Real w2d, Real w0md, Real w1md, Real w2md) {
  return data(0)  * w0d * w0t  + data(1)  * w0md * w0t
       + data(2)  * w0d * w0mt + data(3)  * w0md * w0mt
       + data(4)  * w0d * w1t  + data(5)  * w0md * w1t
       + data(6)  * w0d * w1mt + data(7)  * w0md * w1mt
       + data(8)  * w0d * w2t  + data(9)  * w0md * w2t
       + data(10) * w0d * w2mt + data(11) * w0md * w2mt
       + data(12) * w1d * w0t  + data(13) * w1md * w0t
       + data(14) * w1d * w0mt + data(15) * w1md * w0mt
       + data(16) * w2d * w0t  + data(17) * w2md * w0t
       + data(18) * w2d * w0mt + data(19) * w2md * w0mt
       + data(20) * w1d * w1t  + data(21) * w1md * w1t
       + data(22) * w1d * w1mt + data(23) * w1md * w1mt
       + data(24) * w2d * w1t  + data(25) * w2md * w1t
       + data(26) * w2d * w1mt + data(27) * w2md * w1mt
       + data(28) * w1d * w2t  + data(29) * w1md * w2t
       + data(30) * w1d * w2mt + data(31) * w1md * w2mt
       + data(32) * w2d * w2t  + data(33) * w2md * w2t
       + data(34) * w2d * w2mt + data(35) * w2md * w2mt;
  }

  //bicubic hermite polynomial statement function
  Real h3(AthenaArray<Real> data, Real w0t, Real w1t, Real w0mt, Real w1mt, Real w0d,
          Real w1d, Real w0md, Real w1md) {
    return data(0)  * w0d * w0t  + data(1)  * w0md * w0t
         + data(2)  * w0d * w0mt + data(3)  * w0md * w0mt
         + data(4)  * w0d * w1t  + data(5)  * w0md * w1t
         + data(6)  * w0d * w1mt + data(7)  * w0md * w1mt
         + data(8)  * w1d * w0t  + data(9)  * w1md * w0t
         + data(10) * w1d * w0mt + data(11) * w1md * w0mt
         + data(12) * w1d * w1t  + data(13) * w1md * w1t
         + data(14) * w1d * w1mt + data(15) * w1md * w1mt;
  }
};

namespace {
  HelmTable* phelm = nullptr;
  AthenaArray<Real> EosData;
  Real LastTemp;
  Real fixed_ye = -1.0;
  Real fixed_abar = -1.0;
  int i_ye = -1;
  int i_abar = -1;
  int i_temp = -1;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//  \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas, Real* s) {
  Real ye = fixed_ye;
  Real abar = fixed_abar;
  Real temp = LastTemp;
  if (NSCALARS > 0 && i_ye >= 0) {
    ye = s[i_ye] / rho;
  }
  if (NSCALARS > 0 && i_abar >= 0) {
    abar = s[i_abar] / rho;
  }
  if (NSCALARS > 0 && i_temp >= 0) {
    temp = s[i_temp] / rho;
  }
#ifdef TEST1
  phelm->HelmLookupRhoT(rho * rho_unit_, 1e3, EosData);
  if (egas < EosData(0) * inv_egas_unit_) {
    return EosData(0) * inv_egas_unit_;
  }
#endif
  phelm->HelmInvert(rho * rho_unit_, temp, ye, abar, egas * egas_unit_, 0, EosData);
  LastTemp = EosData(5);
  if (NSCALARS > 0 && i_temp >= 0) {
    s[i_temp] = LastTemp * rho;
  }
  return EosData(2) * inv_egas_unit_;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//  \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres, Real* r) {
  //phelm->HelmLookupRhoT(rho, pres, EosData);
  //std::cout << "EgasFromRhoP" << '\n';
  Real ye = fixed_ye;
  Real abar = fixed_abar;
  Real temp = LastTemp;
  if (NSCALARS > 0 && i_ye >= 0) {
    ye = r[i_ye];
  }
  if (NSCALARS > 0 && i_abar >= 0) {
    abar = r[i_abar];
  }
  if (NSCALARS > 0 && i_temp >= 0) {
    temp = r[i_temp];
  }
#ifdef TEST1
  phelm->HelmLookupRhoT(rho * rho_unit_, 1e3, EosData);
  if (pres < EosData(2) * inv_egas_unit_) {
    return EosData(0) * inv_egas_unit_;
  }
#endif
  phelm->HelmInvert(rho * rho_unit_, LastTemp, ye, abar, pres * egas_unit_, 2, EosData);
  LastTemp = EosData(5);
  if (NSCALARS > 0 && i_temp >= 0) {
    r[i_temp] = LastTemp;
  }
  return EosData(0) * inv_egas_unit_;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//  \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres, const Real* r) {
  //phelm->HelmLookupRhoT(rho, pres, EosData);
  //std::cout << "AsqFromRhoP" << '\n';
  Real ye = fixed_ye;
  Real abar = fixed_abar;
  Real temp = LastTemp;
  if (NSCALARS > 0 && i_ye >= 0) {
    ye = r[i_ye];
  }
  if (NSCALARS > 0 && i_abar >= 0) {
    abar = r[i_abar];
  }
  if (NSCALARS > 0 && i_temp >= 0) {
    temp = r[i_temp];
  }
#ifdef TEST1
  phelm->HelmLookupRhoT(rho * rho_unit_, 1e3, EosData);
  if (pres < EosData(2) * inv_egas_unit_) {
    return EosData(4) * inv_vsqr_unit_;
  }
#endif
  phelm->HelmInvert(rho * rho_unit_, LastTemp, ye, abar, pres * egas_unit_, 2, EosData);
  LastTemp = EosData(5);
  return EosData(4) * inv_vsqr_unit_;
}

#if 0
// MSBC: I can't remember why we chose 7 here
void EquationOfState::SevenFromRhoT(Real rho, Real T, AthenaArray<Real> &out) {
  phelm->HelmLookupRhoT(rho * rho_unit_, T, out);
  LastTemp = T;
  out(0) *= inv_egas_unit_;
  out(1) *= inv_egas_unit_;
  out(2) *= inv_egas_unit_;
  out(3) *= inv_egas_unit_;
  out(4) *= inv_vsqr_unit_;
  out(6) *= inv_egas_unit_ * rho_unit_;
}

Real EquationOfState::TFromRhoP(Real rho, Real pres) {
  phelm->HelmInvert(rho * rho_unit_, LastTemp, pres * egas_unit_, 2, EosData);
  LastTemp = EosData(5);
  return LastTemp;
}

Real EquationOfState::PresFromRhoT(Real rho, Real T) {
  phelm->HelmLookupRhoT(rho * rho_unit_, T, EosData);
  LastTemp = T;
  return EosData(2) * inv_egas_unit_;
}

Real EquationOfState::TFromRhoEgas(Real rho, Real egas) {
  phelm->HelmInvert(rho * rho_unit_, LastTemp, egas * egas_unit_, 0, EosData);
  LastTemp = EosData(5);
  return LastTemp;
}
#endif

//----------------------------------------------------------------------------------------
//! void EquationOfState::InitEosConstants(ParameterInput* pin)
//  \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  if (!phelm) phelm = new HelmTable(pin, ptable);
  EosData.NewAthenaArray(HelmholtzConstants::nOut);
  LastTemp = std::pow(10.0, 0.5 * (phelm->tlo + phelm->thi));
  if (pin->DoesParameterExist("hydro", "helm_abar")) {
    fixed_abar = pin->GetReal("hydro", "helm_abar");
  }
  if (pin->DoesParameterExist("hydro", "helm_ye")) {
    fixed_ye = pin->GetReal("hydro", "helm_ye");
  } else if (pin->DoesParameterExist("hydro", "helm_zbar")) {
    if (fixed_abar < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "hydro/helm_abar must be specified if hydro/helm_zbar is." << std::endl;
      ATHENA_ERROR(msg);
    }
    fixed_ye = std::max(1.0e-16, pin->GetReal("hydro", "helm_zbar") / fixed_abar);
  }
  if (NSCALARS==0 && std::min(fixed_abar, fixed_ye) < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
        << "helm_abar and helm_ye (or helm_zbar) must be specified if NSCALARS=0."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  if (pin->DoesParameterExist("hydro", "helm_ye_index")) {
    i_ye = pin->GetInteger("hydro", "helm_ye_index");
    if (i_ye < 0 || i_ye >= NSCALARS) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "hydro/helm_ye_index must be between 0 and NSCALARS (" << NSCALARS << ")."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  if (pin->DoesParameterExist("hydro", "helm_abar_index")) {
    i_abar = pin->GetInteger("hydro", "helm_abar_index");
    if (i_abar < 0 || i_abar >= NSCALARS) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "hydro/helm_abar_index must be between 0 and NSCALARS (" << NSCALARS << ")."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (i_abar == i_ye) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "hydro/helm_abar_index must be different from hydro/helm_ye_index."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (fixed_ye < 0 && i_ye < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "either hydro/helm_ye or hydro/helm_ye_index must be specified."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (fixed_abar < 0 && i_abar < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
          << "either hydro/helm_abar or hydro/helm_abar_index must be specified."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    if (pin->DoesParameterExist("hydro", "helm_temp_index")) {
      i_temp = pin->GetInteger("hydro", "helm_temp_index");
      if (i_temp < 0 || i_temp >= NSCALARS) {
        std::stringstream msg;
        msg << "### FATAL ERROR in EquationOfState::InitEosConstants" << std::endl
            << "hydro/helm_temp_index must be between 0 and NSCALARS ("
            << NSCALARS << ")."
            << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
  ///////////////////
  // test
#ifdef MAINTEST
  Real rho_list[] = {
                      -1.2000000000000000e+01,
                      -1.1000000000000000e+01,
                      -1.0000000000000000e+01,
                      -9.0000000000000000e+00,
                      -8.0000000000000000e+00,
                      -7.0000000000000000e+00,
                      -6.0000000000000000e+00,
                      -5.0000000000000000e+00,
                      -4.0000000000000000e+00,
                      -3.0000000000000000e+00,
                      -2.0000000000000000e+00,
                      -1.0000000000000000e+00,
                      0.0000000000000000e+00,
                      1.0000000000000000e+00,
                      2.0000000000000000e+00,
                      3.0000000000000000e+00,
                      4.0000000000000000e+00,
                      5.0000000000000000e+00,
                      6.0000000000000000e+00,
                      7.0000000000000000e+00,
                      8.0000000000000000e+00,
                      9.0000000000000000e+00,
                      1.0000000000000000e+01,
                      1.1000000000000000e+01,
                      1.2000000000000000e+01,
                      1.3000000000000000e+01,
                      1.4000000000000000e+01,
                      1.5000000000000000e+01,
                      4.2058452966196676e+00,
                      1.0129862822644725e+01,
                      2.5766584519647218e+00,
                      2.3614118173045640e-01,
                      -8.8019100913495780e+00,
                      1.3354073875986661e+01,
                      -6.7616685886840369e+00,
                      -1.0746381669808670e+01,
                      -1.3453761956427019e+00,
                      1.2169963846606585e+01,
                      7.5166571366261294e+00,
                      1.7010857930790877e+00,
                      -6.3777124411303419e+00,
                      -3.7366062655473655e+00,
                      1.4165993948365038e+01,
                      1.3986041228825936e+01,
                      3.3786472603813245e+00,
                      1.1591598832352989e+01,
                      -2.3463727098012228e+00,
                      9.5160876639466032e+00
                    };
  Real temp_list[] = {
                      3.0000000000000000e+00,
                      4.0000000000000000e+00,
                      5.0000000000000000e+00,
                      6.0000000000000000e+00,
                      7.0000000000000000e+00,
                      8.0000000000000000e+00,
                      9.0000000000000000e+00,
                      1.0000000000000000e+01,
                      1.1000000000000000e+01,
                      1.2000000000000000e+01,
                      1.3000000000000000e+01,
                      1.1381931348144851e+01,
                      9.7570094890509491e+00,
                      3.1155589943242417e+00,
                      1.0986874704351557e+01,
                      5.8628915200003870e+00,
                      9.6261258046854170e+00,
                      1.2678093829137069e+01,
                      1.1216961509383435e+01,
                      9.0686831658770330e+00,
                      1.0386295554181906e+01,
                      8.7018246400376746e+00,
                      6.9459561738213313e+00,
                      6.8859742835374531e+00,
                      7.1528300316393914e+00,
                      5.8681615211276359e+00,
                      1.2028328578165263e+01,
                      1.2656772988232268e+01,
                      6.8459476240110000e+00,
                      8.1548048893242360e+00,
                      5.9593483536909213e+00
                    };
  for (int i=0; i<48; ++i) {
    Real rho = std::pow(10.0, rho_list[i]);
    for (int j=0; j<31; ++j) {
      Real temp = std::pow(10.0, temp_list[j]);
      phelm->HelmLookupRhoT(rho, temp, EosData);
      for (int k=0; k<6; ++k) {
        printf("%.16e, ", EosData(k));
      }
      std::cout << '\n';
    }
  }
#endif
#ifdef MYDEBUG
  Real rho, temp;
  std::cout << "Input fluid parameters and retrieve EOS parameters." << '\n'
            << "Non-positive inputs will exit loop." << '\n';
  std::cout << "Input density (mass/volume): ";
  std::cin >> rho;
  std::cout << "Input temperature (K): ";
  std::cin >> temp;
  while (rho > 0 && std::isfinite(rho) && temp >0 && std::isfinite(temp)) {
    printf("d, t: %.16e, %.16e\n", rho, temp);
    phelm->HelmLookupRhoT(rho, temp, EosData);
    std::cout << "e(d, e)          , p(d, e)         , ASq(d, P)\n";
    printf("%.16e, %.16e, %.16e\n", EosData(0), EosData(2), EosData(4));
    std::cout << "Input density (mass/volume): ";
    std::cin >> rho;
    std::cout << "Input temperature (K): ";
    std::cin >> temp;
  }
  std::cout << std::endl;
#endif
  // Test
  //////////////////
  return;
}
