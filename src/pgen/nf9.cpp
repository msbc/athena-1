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
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <fstream>    // ofstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

// TODO(felker): many unused arguments in these functions: time, iout, ...

static AthenaArray<Real> opacitytable;
static AthenaArray<Real> planckopacity;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logrhottable;
static AthenaArray<Real> logttable_planck;
static AthenaArray<Real> logrhottable_planck;

static AthenaArray<Real> ini_profile;
static AthenaArray<Real> blow_data;
static AthenaArray<Real> bup_data;

static int in_line=9995;
static int x1length;
static const Real rhounit = 1e-7;
static const Real tunit = 2e5;
static const Real lunit = 1.1126891737412279e12;

void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck);
void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

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

void RadOutflow_Inner_X3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);


void RadOutflow_Outer_X3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

namespace {
Real HistoryBxBy(MeshBlock *pmb, int iout);
Real HistorydVxVy(MeshBlock *pmb, int iout);
Real HistoryVshear(MeshBlock *pmb, int iout);
Real HistoryFluxes(MeshBlock *pmb, int iout);

// Apply a density floor - useful for large |z| regions
Real dfloor, tfloor, vmax;
Real Omega_0, qshear;
Real temp_goal, tau, growbz,beta;
} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  AllocateUserHistoryOutput(7);
  EnrollUserHistoryOutput(0, HistoryBxBy, "-BxBy");
  EnrollUserHistoryOutput(1, HistorydVxVy, "dVxVy");
  EnrollUserHistoryOutput(2, HistoryVshear, "vshear");
  EnrollUserHistoryOutput(3, HistoryFluxes, "UpperIn");
  EnrollUserHistoryOutput(4, HistoryFluxes, "UpperOut");
  EnrollUserHistoryOutput(5, HistoryFluxes, "LowerIn");
  EnrollUserHistoryOutput(6, HistoryFluxes, "LowerOut");

  // Enroll user-defined physical source terms
  //   vertical external gravitational potential
  EnrollUserExplicitSourceFunction(VertGrav);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
    if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    EnrollUserRadBoundaryFunction(BoundaryFace::inner_x3, RadOutflow_Inner_X3);
    }}
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
    if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x3, RadOutflow_Outer_X3);
  }}

  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in hb3.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box." << std::endl;
    ATHENA_ERROR(msg);
  }

  tfloor = pin->GetOrAddReal("radiation", "tfloor", 0.001);
  dfloor = pin->GetOrAddReal("hydro", "dfloor", 1.e-8);
  Real crat = pin->GetOrAddReal("radiation", "Crat",9166.13);
  vmax = pin->GetOrAddReal("problem", "voc", 0.01);
  growbz = pin->GetOrAddInteger("problem","growbz", 0);
  beta = pin->GetReal("problem","beta");
  vmax*=crat;

  ini_profile.NewAthenaArray(in_line,3);
  FILE *fini;
  if ( (fini=fopen("./fullprof1.txt","r"))==NULL )
  {
     printf("Open input file error MESA profile");
     return;
  }

  for(int j=0; j<in_line; j++){
    for(int i=0; i<3; i++){
      fscanf(fini,"%lf",&(ini_profile(j,i)));
    }
  }
  fclose(fini);
  blow_data.NewAthenaArray(2,NGHOST);
  bup_data.NewAthenaArray(2,NGHOST);
  std::cout << "Initialized blow_data, bup_data" << std::endl;

    if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){

    // the opacity table
    opacitytable.NewAthenaArray(212,46);
    planckopacity.NewAthenaArray(138,37);
    logttable.NewAthenaArray(212);
    logrhottable.NewAthenaArray(46);
    logttable_planck.NewAthenaArray(138);
    logrhottable_planck.NewAthenaArray(37);

    // read in the opacity table
    FILE *fkappa, *flogt, *flogrhot, *fplanck, *flogt_planck, *flogrhot_planck;

    if ( (fkappa=fopen("../optables/aveopacity_combined.txt","r"))==NULL )
    {
      printf("Open input file error aveopacity_combined");
      return;
    }

    if ( (fplanck=fopen("../optables/PlanckOpacity.txt","r"))==NULL )
    {
      printf("Open input file error PlanckOpacity");
      return;
    }

    if ( (flogt=fopen("../optables/logT.txt","r"))==NULL )
    {
      printf("Open input file error logT");
      return;
    }

    if ( (flogrhot=fopen("../optables/logRhoT.txt","r"))==NULL )
    {
      printf("Open input file error logRhoT");
      return;
    }

    if ( (flogt_planck=fopen("../optables/logT_planck.txt","r"))==NULL )
    {
      printf("Open input file error logT_planck");
      return;
    }

    if ( (flogrhot_planck=fopen("../optables/logRhoT_planck.txt","r"))==NULL )
    {
      printf("Open input file error logRhoT_planck");
      return;
    }

    for(int j=0; j<212; j++){
      for(int i=0; i<46; i++){
          fscanf(fkappa,"%lf",&(opacitytable(j,i)));
      }
    }

    for(int j=0; j<138; j++){
      for(int i=0; i<37; i++){
          fscanf(fplanck,"%lf",&(planckopacity(j,i)));
      }
     }


    for(int i=0; i<46; i++){
      fscanf(flogrhot,"%lf",&(logrhottable(i)));
    }

    for(int i=0; i<212; i++){
      fscanf(flogt,"%lf",&(logttable(i)));
    }

    for(int i=0; i<37; i++){
      fscanf(flogrhot_planck,"%lf",&(logrhottable_planck(i)));
    }

    for(int i=0; i<138; i++){
      fscanf(flogt_planck,"%lf",&(logttable_planck(i)));
    }

    fclose(fkappa);
    fclose(flogt);
    fclose(flogrhot);
    fclose(fplanck);
    fclose(flogt_planck);
    fclose(flogrhot_planck);

  }

  return;
}

//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{

  ini_profile.DeleteAthenaArray();
  blow_data.DeleteAthenaArray();
  bup_data.DeleteAthenaArray();

  // free memory
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){

    opacitytable.DeleteAthenaArray();
    logttable.DeleteAthenaArray();
    logrhottable.DeleteAthenaArray();
    planckopacity.DeleteAthenaArray();
    logttable_planck.DeleteAthenaArray();
    logrhottable_planck.DeleteAthenaArray();

  }


  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  Real zbottom = pcoord->x3v(ks-1);
  if(zbottom < pmy_mesh->mesh_size.x3min){
    for(int k=1; k<=NGHOST; ++k){
      Real z = std::abs(pcoord->x3v(ks-k));
      int lleft=0;

      int lright=1;
      while((z > ini_profile(lright,0)) && (lright < in_line-1)){
        lright = lright+1;
      }
      if(lright - lleft > 1) lleft = lright -1;

      Real rho = ini_profile(lleft,2) + (z - ini_profile(lleft,0)) *
                                (ini_profile(lright,2) - ini_profile(lleft,2))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));
      Real tem = ini_profile(lleft,1) + (z - ini_profile(lleft,0)) *
                                (ini_profile(lright,1) - ini_profile(lleft,1))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));

      blow_data(0,k-1) = rho;
      blow_data(1,k-1) = tem;

    }

  }

  Real ztop = pcoord->x3v(ke+1);
  if(ztop > pmy_mesh->mesh_size.x3max){
    for(int k=1; k<=NGHOST; ++k){
      Real z = std::abs(pcoord->x3v(ke+k));
      int lleft=0;

      int lright=1;
      while((z > ini_profile(lright,0)) && (lright < in_line-1)){
        lright = lright+1;
      }
      if(lright - lleft > 1) lleft = lright -1;

      Real rho = ini_profile(lleft,2) + (z - ini_profile(lleft,0)) *
                                (ini_profile(lright,2) - ini_profile(lleft,2))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));
      Real tem = ini_profile(lleft,1) + (z - ini_profile(lleft,0)) *
                                (ini_profile(lright,1) - ini_profile(lleft,1))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));

      bup_data(0,k-1) = rho;
      bup_data(1,k-1) = tem;

    }

  }



  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){

      pnrrad->EnrollOpacityFunction(StarOpacity);

  }else{

  }

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int ifield, ipert;
  Real  amp, pres;
  Real ib0 = 0.0;

  // shearing sheet parameter
  Omega_0 = pin->GetOrAddReal("orbital_advection", "Omega0", 1.0);
  qshear = pin->GetOrAddReal("orbital_advection", "qshear", 1.5);

  Real SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  // TODO(felker): tons of unused variables in this file: xmin, xmax, rbx, rby, Ly, ky,...
  Real x1, x2, x3;
  //Real xmin, xmax;
  Real rvx, rvy, rvz;
  //Real rbx, rby, rbz;
  Real rval;

  Real rd, rho, tem,rp;
  AthenaArray<Real> ir_cm;
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);

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
  if(MAGNETIC_FIELDS_ENABLED){
    ifield = pin->GetOrAddInteger("problem","ifield", 1);
  }

  Real zmax = ini_profile(in_line-1,0);
  Real rho_zmax = ini_profile(in_line-1,2);
  Real tem_zmax = ini_profile(in_line-1,1);

  // Initialize d, M, and P.
  for (int k=ks; k<=ke; k++) {
    x3 = std::abs(pcoord->x3v(k));
    if(x3>zmax){
      tem=tem_zmax;
      rho = dfloor;//rho_zmax*exp(-(SQR(x3)-SQR(zmax))); //not sure if this is correct -AS 1/24
      //rho = std::max(dfloor,rho);
    }else{
      int lleft=0;

      int lright=1;
      while((x3 > ini_profile(lright,0)) && (lright < in_line-1)){
         lright = lright+1;
      }
      if(lright - lleft > 1) lleft = lright -1;

      rho = ini_profile(lleft,2) + (x3 - ini_profile(lleft,0)) *
                                  (ini_profile(lright,2) - ini_profile(lleft,2))
                                 /(ini_profile(lright,0) - ini_profile(lleft,0));
      tem = ini_profile(lleft,1) + (x3 - ini_profile(lleft,0)) *
                                  (ini_profile(lright,1) - ini_profile(lleft,1))
                                 /(ini_profile(lright,0) - ini_profile(lleft,0));

    }

    //printf("%lg %lg %lg\n",x3,rho,tem);
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);

        // Initialize perturbations
        // ipert = 1 - random perturbations to P/d and V
        // [default, used by HGB]
        if (ipert == 1) {
          rval = amp*(ran2(&iseed) - 0.5);
          rd = rho*(1.0+2.0*rval);
	  if (rd < dfloor) rd = dfloor;
          if (NON_BAROTROPIC_EOS) {
            rp = tem*rd;
	    Real pfloor=rd*tfloor;
            if (rp < pfloor) rp = pfloor;
          }
          /*rval = amp*(ran2(&iseed) - 0.5);
          rvx = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvx += rvx;

          rval = amp*(ran2(&iseed) - 0.5);
          rvy = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvy += rvy;

          rval = amp*(ran2(&iseed) - 0.5);
          rvz = 0.4*rval*std::sqrt(pres/den);
          rvz = (0.4/std::sqrt(3.0)) *rval*1e-3;
          SumRvz += rvz;*/
          // no perturbations
        } else {
          rd = rho;
          rp = rho*tem;
	} //yan fei said only random density
	  rvx = 0;
          rvy = 0;
          rvz = 0;


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

	if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
          Real er = tem * tem * tem * tem;
          Real vx = phydro->u(IM1,k,j,i)/rd;
          Real vy = phydro->u(IM2,k,j,i)/rd;
          Real vz = phydro->u(IM3,k,j,i)/rd;
          for(int n=0; n<pnrrad->n_fre_ang; ++n)
            ir_cm(n) = er;

          Real *ir_lab = &(pnrrad->ir(k,j,i,0));
          Real *mux = &(pnrrad->mu(0,k,j,i,0));
          Real *muy = &(pnrrad->mu(1,k,j,i,0));
          Real *muz = &(pnrrad->mu(2,k,j,i,0));

          pnrrad->pradintegrator->ComToLab(vx,vy,vz,mux,muy,muz,ir_cm,ir_lab);
	}// End Rad
	// Opacity will be set during initialization

  }}}


  // Compute pressure based on the EOS.
  if (NON_BAROTROPIC_EOS) {
    pres=ini_profile(0,1)*ini_profile(0,2);
  } else {
    Real iso_cs = peos->GetIsoSoundSpeed();
    pres = den*SQR(iso_cs);
  }


  if (MAGNETIC_FIELDS_ENABLED) {
  // vector potential
  ib0 = std::sqrt(static_cast<Real>(2.0*pres/beta));

  AthenaArray<Real> rax, ray, raz;
  // nxN != ncellsN, in general. Allocate to extend through 2*ghost, regardless # dim
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  rax.NewAthenaArray(nx3, nx2, nx1);
  ray.NewAthenaArray(nx3, nx2, nx1);
  raz.NewAthenaArray(nx3, nx2, nx1);

  AthenaArray<Real> area, len, len_jp1, len_kp1;
  area.NewAthenaArray(nx1);
  len.NewAthenaArray(nx1);
  len_jp1.NewAthenaArray(nx1);
  len_kp1.NewAthenaArray(nx1);

  // set rax
  for (int k=ks; k<=ke+1; k++) {
  for (int j=js; j<=je+1; j++) {
  for (int i=is; i<=ie; i++) {
    Real x1 = pcoord->x1v(i);
    Real x3 = pcoord->x3v(k);
    Real rb = std::sqrt(x1*x1+SQR(std::abs(x3)-0.25));
    if (x3>=0){
       rax(k,j,i) = -ib0*(1+std::cos(PI*rb))/(32*PI);
    }
    if (x3<0){
       rax(k,j,i) = ib0*(1+std::cos(PI*rb))/(32*PI);
    }
  }
  }
  }

    // set ray
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x1 = pcoord->x1v(i);
          Real x3 = pcoord->x3v(k);
	  Real rb = std::sqrt(x1*x1+SQR(std::abs(x3)-0.25));
	  if (x3>=0){
            ray(k,j,i) = -ib0*(1+std::cos(PI*rb))/(32*PI);
          }
	  if (x3<0){
	    ray(k,j,i) = ib0*(1+std::cos(PI*rb))/(32*PI);
	  }
	}
      }
    }

    // set raz
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
            Real x1 = pcoord->x1v(i);
            Real x3 = pcoord->x3v(k);
            Real rb = std::sqrt(x1*x1+SQR(std::abs(x3)-0.25));
            if (x3>=0){
              raz(k,j,i) = -ib0*(1+std::cos(PI*rb))/(32*PI);
            }
            if (x3<0){
              raz(k,j,i) = ib0*(1+std::cos(PI*rb))/(32*PI);
           }
	}
      }
    }

    // calculate b from vector potential
    for (int k=ks; k<=ke  ; k++) {
      for (int j=js; j<=je  ; j++) {
	pcoord->Face1Area(k,j,is,ie+1,area);
        pcoord->Edge3Length(k,j  ,is,ie+1,len);
	pcoord->Edge3Length(k,j+1,is,ie+1,len_jp1);
	pcoord->Edge3Length(k+1,j,is,ie+1,len_kp1);
        for (int i=is; i<=ie+1; i++) {
	  Real x1 = pcoord->x1f(i);
          Real x3 = pcoord->x3f(k);
          Real rbc = std::sqrt(x1*x1+SQR(std::abs(x3)-0.25));
	    if (ifield==0){
              pfield->b.x1f(k,j,i) =0.0;
	    }else if (ifield==1){
	      pfield->b.x1f(k,j,i) = -(len_kp1(i)*ray(k+1,j,i)-len(i)*ray(k,j,i))/area(i);
	    }else  if (ifield==2){
	      pfield->b.x1f(k,j,i) = (len_jp1(i)*raz(k,j+1,i)-len(i)*raz(k,j,i))/area(i)
                                    -(len_kp1(i)*ray(k+1,j,i)-len(i)*ray(k,j,i))/area(i);
	    }
	}
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je  ; j++) {
	pcoord->Face1Area(k,j,is,ie+1,area);
        pcoord->Edge3Length(k,j  ,is,ie+1,len);
        pcoord->Edge3Length(k,j+1,is,ie+1,len_jp1);
        for (int i=is; i<=ie  ; i++) {
	  Real x1 = pcoord->x1f(i);
          Real x3 = pcoord->x3f(k);
            if(ifield==0){
              pfield->b.x3f(k,j,i) =ib0;
	    }else if(ifield==1){
              pfield->b.x3f(k,j,i) = (len(i+1)*ray(k,j,i+1)-len(i)*ray(k,j,i))/area(i);
            }else   if(ifield==2){
              pfield->b.x3f(k,j,i) = (len(i+1)*ray(k,j,i+1)-len(i)*ray(k,j,i))/area(i)
                                 -(len_jp1(i)*rax(k,j+1,i)-len(i)*rax(k,j,i))/area(i);
	    }
	}
      }
    }
    for (int k=ks; k<=ke  ; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie  ; i++) {
        Real x3 = pcoord->x3f(k);
        if((std::abs(x3)<0.8)&&(ifield==2)){
   	  pfield->b.x2f(k,j,i) = ib0/2;
        } else{
            pfield->b.x2f(k,j,i) = 0.0;
          }
	}
      }
    }

    rax.DeleteAthenaArray();
    ray.DeleteAthenaArray();
    raz.DeleteAthenaArray();
    area.DeleteAthenaArray();
    len.DeleteAthenaArray();
    len_jp1.DeleteAthenaArray();
    len_kp1.DeleteAthenaArray();

  pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,   js,je,ks,ke);
  // add magnetic energy
  if (NON_BAROTROPIC_EOS) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(pfield->bcc(IB1,k,j,i))+SQR(pfield->bcc(IB2,k,j,i))
                      +SQR(pfield->bcc(IB3,k,j,i)));
        }
      }
    }
  }
        } // MHD

  // For random perturbations as in HGB, ensure net momentum is zero by
  // subtracting off mean of perturbations
/*
  if (ipert == 1) {
    int cell_num = block_size.nx1*block_size.nx2*block_size.nx3;
    SumRvx /= cell_num;
    SumRvy /= cell_num;
    SumRvz /= cell_num;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IM1,k,j,i) -= rd*SumRvx;
          phydro->u(IM2,k,j,i) -= rd*SumRvy;
          phydro->u(IM3,k,j,i) -= rd*SumRvz;
        }
      }
    }
  }
*/

  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED)
    ir_cm.DeleteAthenaArray();


  return;
}

void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  NRRadiation *prad = pmb->pnrrad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }
  // electron scattering opacity
  Real kappas = 0.2 * (1.0 + 0.7);
  Real kappaa = 0.0;

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);
    Real kappa, kappa_planck;
    rossopacity(rho, gast, kappa, kappa_planck);



    if(kappa < kappas){
      if(gast < 0.14){
        kappaa = kappa;
        kappa = 0.0;
      }else{
        kappaa = 0.0;
      }
    }else{
      kappaa = kappa - kappas;
      kappa = kappas;
    }

    prad->sigma_s(k,j,i,ifr) = kappa * rho * rhounit * lunit;
    prad->sigma_a(k,j,i,ifr) = kappaa * rho * rhounit * lunit;
    prad->sigma_p(k,j,i,ifr) = kappa_planck*rho*rhounit*lunit;
    prad->sigma_pe(k,j,i,ifr) = prad->sigma_p(k,j,i,ifr);
  }


 }}}
 return;
}

void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck)
{


    Real logt = log10(tgas * tunit);
    Real logrhot = log10(rho* rhounit) - 3.0* logt + 18.0;
    int nrhot1_planck = 0;
    int nrhot2_planck = 0;

    int nrhot1 = 0;
    int nrhot2 = 0;

    while((logrhot > logrhottable_planck(nrhot2_planck)) && (nrhot2_planck < 36)){
      nrhot1_planck = nrhot2_planck;
      nrhot2_planck++;
    }
    if(nrhot2_planck==36 && (logrhot > logrhottable_planck(nrhot2_planck)))
      nrhot1_planck=nrhot2_planck;

    while((logrhot > logrhottable(nrhot2)) && (nrhot2 < 45)){
      nrhot1 = nrhot2;
      nrhot2++;
    }
    if(nrhot2==45 && (logrhot > logrhottable(nrhot2)))
      nrhot1=nrhot2;

  /* The data point should between NrhoT1 and NrhoT2 */
    int nt1_planck = 0;
    int nt2_planck = 0;
    int nt1 = 0;
    int nt2 = 0;
    while((logt > logttable_planck(nt2_planck)) && (nt2_planck < 137)){
      nt1_planck = nt2_planck;
      nt2_planck++;
    }
    if(nt2_planck==137 && (logt > logttable_planck(nt2_planck)))
      nt1_planck=nt2_planck;

    while((logt > logttable(nt2)) && (nt2 < 211)){
      nt1 = nt2;
      nt2++;
    }
    if(nt2==211 && (logt > logttable(nt2)))
      nt1=nt2;

    Real kappa_t1_rho1=opacitytable(nt1,nrhot1);
    Real kappa_t1_rho2=opacitytable(nt1,nrhot2);
    Real kappa_t2_rho1=opacitytable(nt2,nrhot1);
    Real kappa_t2_rho2=opacitytable(nt2,nrhot2);

    Real planck_t1_rho1=planckopacity(nt1_planck,nrhot1_planck);
    Real planck_t1_rho2=planckopacity(nt1_planck,nrhot2_planck);
    Real planck_t2_rho1=planckopacity(nt2_planck,nrhot1_planck);
    Real planck_t2_rho2=planckopacity(nt2_planck,nrhot2_planck);


    // in the case the temperature is out of range
    // the planck opacity should be smaller by the
    // ratio T^-3.5
    if(nt2_planck == 137 && (logt > logttable_planck(nt2_planck))){
       Real scaling = pow(10.0, -3.5*(logt - logttable_planck(137)));
       planck_t1_rho1 *= scaling;
       planck_t1_rho2 *= scaling;
       planck_t2_rho1 *= scaling;
       planck_t2_rho2 *= scaling;
    }


    Real rho_1 = logrhottable(nrhot1);
    Real rho_2 = logrhottable(nrhot2);
    Real t_1 = logttable(nt1);
    Real t_2 = logttable(nt2);


    if(nrhot1 == nrhot2){
      if(nt1 == nt2){
        kappa = kappa_t1_rho1;
      }else{
        kappa = kappa_t1_rho1 + (kappa_t2_rho1 - kappa_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1 == nt2){
        kappa = kappa_t1_rho1 + (kappa_t1_rho2 - kappa_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);
      }else{
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
    rho_1 = logrhottable_planck(nrhot1_planck);
    rho_2 = logrhottable_planck(nrhot2_planck);
    t_1 = logttable_planck(nt1_planck);
    t_2 = logttable_planck(nt2_planck);

  /* Now do the same thing for Planck mean opacity */
    if(nrhot1_planck == nrhot2_planck){
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1;
      }else{
        kappa_planck = planck_t1_rho1 + (planck_t2_rho1 - planck_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1 + (planck_t1_rho2 - planck_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);

      }else{
        kappa_planck = planck_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1));
      }
    }/* end same rhoT */

    return;

}

//Adjusts floors, commenting out for now? -AS 1/24
void MeshBlock::UserWorkInLoop() {
  int il=is, iu=ie, jl=js, ju=je, kl=ks, ku=ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju>jl){
     jl -= NGHOST;
     ju += NGHOST;
  }
  if(ku>kl){
    kl -= NGHOST;
    ku += NGHOST;
  }
  Real gam = peos->GetGamma();
  int flag = 0;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        flag = 0;
        Real& u_d  = phydro->u(IDN,k,j,i);
        u_d = (u_d > dfloor) ?  u_d : dfloor;
	Real vmag=std::sqrt(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))/u_d;
	Real valf = std::sqrt((SQR(pfield->bcc(IB1,k,j,i))+ SQR(pfield->bcc(IB2,k,j,i)) +SQR(pfield->bcc(IB3,k,j,i)))/u_d);
	if(vmag>vmax||valf>vmax){
          flag = 1;
          Real rat=std::max((vmag/vmax),SQR(valf/vmax));
          u_d*=rat;
	  phydro->w(IDN,k,j,i)=u_d;
          phydro->w(IVX,k,j,i)=phydro->u(IM1,k,j,i)/u_d;
          phydro->w(IVY,k,j,i)=phydro->u(IM2,k,j,i)/u_d;
          phydro->w(IVZ,k,j,i)=phydro->u(IM3,k,j,i)/u_d;
	}

        if (NON_BAROTROPIC_EOS && flag > 0) {
          Real& w_p  = phydro->w(IPR,k,j,i);
          Real& u_e  = phydro->u(IEN,k,j,i);
          Real& u_m1 = phydro->u(IM1,k,j,i);
          Real& u_m2 = phydro->u(IM2,k,j,i);
          Real& u_m3 = phydro->u(IM3,k,j,i);
	  Real me=0.5*(SQR(pfield->bcc(IB1,k,j,i))+SQR(pfield->bcc(IB2,k,j,i))
                     +SQR(pfield->bcc(IB3,k,j,i)));
	  Real pfloor = u_d*tfloor;
	  w_p = (w_p > pfloor) ?  w_p : pfloor;
          Real di = 1.0/u_d;
          Real kine = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
          u_e = w_p/(gam-1.0)+kine+me;
	}

      }
    }
  }

  if((MAGNETIC_FIELDS_ENABLED)&& (growbz!=0)){
      Real pres=ini_profile(0,1)*ini_profile(0,2);
      Real ib0 = std::sqrt(static_cast<Real>(2.0*pres/beta));
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je  ; j++) {
          for (int i=is; i<=ie  ; i++) {
            pfield->b.x3f(k,j,i) +=ib0/growbz;
            pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);
	    Real me=0.5*(SQR(pfield->bcc(IB1,k,j,i))+SQR(pfield->bcc(IB2,k,j,i))
                     +SQR(pfield->bcc(IB3,k,j,i)));
	    Real kine = 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
			    +SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
	    phydro->u(IEN,k,j,i)=phydro->w(IPR,k,j,i)/(gam-1.0)+kine+me;
          }
        }
      }
   growbz=0;
   printf("Here\n");
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
 //       fsmooth = SQR( std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign );
        // multiply gravitational potential by smoothing function
        cons(IM3,k,j,i) -= dt*den*SQR(pmb->porb->Omega0)*x3;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= dt*den*SQR(pmb->porb->Omega0)*prim(IVZ,k,j,i)*x3;
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
          b.x1f(kl-k,j,i) = 0.0;//b.x1f(kl,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(kl-k,j,i) = 0.0;//b.x2f(kl,j,i);
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
        Real Tkl = blow_data(1,k-1);
        Real den = blow_data(0,k-1);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = Tkl*den;
	  Real pfloor = den*tfloor;
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
	prim(IDN,kl-k,j,i) = prim(IDN,kl,j,i);
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
          prim(IPR,kl-k,j,i) =prim(IPR,kl,j,i);
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
          b.x1f(ku+k,j,i) = 0.0;//b.x1f(ku,j,i);
        }
      }
    }
    for (int k=1; k<=ngh; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu; i++) {
          b.x2f(ku+k,j,i) = 0.0;//b.x2f(ku,j,i);
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
        Real den = bup_data(0,k-1);
        Real Tku = bup_data(1,k-1);
        /*if (NON_BAROTROPIC_EOS) {
          Real pressku = den*Tku;
          Real pfloor=den*tfloor;
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }*/
	prim(IDN,ku+k,j,i) = prim(IDN,ku,j,i);
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
          prim(IPR,ku+k,j,i) = prim(IPR,ku,j,i);
      }
    }
  }
  return;
}

void RadOutflow_Inner_X3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
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

void RadOutflow_Outer_X3(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{

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
        vshear = -pmb->porb->qshear*pmb->porb->Omega0*pmb->pcoord->x1v(i);
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) - vshear);
      }
    }
  }
  return dvxvy;
}

Real HistoryVshear(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  Real vshearsum=0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        vshear = -pmb->porb->qshear*pmb->porb->Omega0*pmb->pcoord->x1v(i);
        vshearsum+=volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*vshear;
      }
    }
  }
  return vshearsum;
}

Real HistoryFluxes(MeshBlock *pmb, int iout) {
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  // print the XR flux in and out of top and bottom
  // of box to a hist file
  Real *weight = &(pmb->pnrrad->wmu(0));
  Real upper_in_fluxUV=0.0;
  Real upper_out_fluxUV=0.0;
  Real lower_in_fluxUV=0.0;
  Real lower_out_fluxUV=0.0;

  if ((pmb->pcoord->x3v(ke+NGHOST))>48.0){
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      Real *cosz_top = &(pmb->pnrrad->mu(2,ke,j,i,0));
      for(int n=0; n<pmb->pnrrad->nang; ++n){
        Real miuz = pmb->pnrrad->mu(2,ke,j,i,n);
        if(miuz < 0.0){
          upper_in_fluxUV+=pmb->pnrrad->ir(ke,j,i,n)*weight[n] * cosz_top[n];
        }
        if(miuz > 0.0){
          upper_out_fluxUV+=pmb->pnrrad->ir(ke,j,i,n)*weight[n] * cosz_top[n];
        }
      }
    }}
  }
  if ((pmb->pcoord->x3v(ks-NGHOST))<-48.0){
    for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      Real *cosz_bot = &(pmb->pnrrad->mu(2,ks,j,i,0));
      for(int n=0; n<pmb->pnrrad->nang; ++n){
        Real miuz = pmb->pnrrad->mu(2,ks,j,i,n);
        if(miuz < 0.0){
          lower_out_fluxUV+=pmb->pnrrad->ir(ks,j,i,n)*weight[n] * cosz_bot[n];
        }
        if(miuz > 0.0){
          lower_in_fluxUV+=pmb->pnrrad->ir(ks,j,i,n)*weight[n] * cosz_bot[n];
        }
      }
    }}
  }

  if(iout==3){
    return upper_in_fluxUV;
  } if(iout==4){
    return upper_out_fluxUV;
  } if(iout==5){
    return lower_in_fluxUV;
  } if(iout==6){
    return lower_out_fluxUV;
  }
}
} // namespace
