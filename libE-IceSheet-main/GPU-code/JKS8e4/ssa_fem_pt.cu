/* Title ....

Copyright (C) 2022  Anjali Sandip, Ludovic Raess and Mathieu Morlighem
 
XXX is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
XXX is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with XXX. If not, see <http://www.gnu.org/licenses/>. */

// -------------------------------------------------------------------------
// Compile as: nvcc ssa_fem_pt.cu -arch=sm_XX 
// arch=sm_XX: TITAN Black=sm_35, TITAN X=sm_52, TITAN Xp=sm_61, Tesla V100=sm_70
// Run as: ./a.out
// -------------------------------------------------------------------------

#include <iomanip>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <cmath>
#include <fstream>
using namespace std;

/*define GPU specific variables*/

#define GPU_ID    0 

#define BLOCK_Xe  256   //optimal block size for JKS2e4 and PIG3e4
#define BLOCK_Xv  256 

// Device norm subroutine
#define blockId       (blockIdx.x)
#define threadId      (threadIdx.x)
#define isBlockMaster (threadIdx.x==0)

#include "helpers.h"

/*CUDA Code*/
__global__ void PT1(double* vx, double* vy, double* alpha, double* beta, int* index,  double* kvx, double* kvy, double* etan,  double* Helem, double* areas, bool* isice, double* Eta_nbe, double* rheology_B, double n_glen, double eta_0, double rele,int nbe){
    // int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix<nbe; ix += blockDim.x * gridDim.x){
        double dvxdx = vx[index[ix*3+0]-1]*alpha[ix*3+0] + vx[index[ix*3+1]-1]*alpha[ix*3+1] + vx[index[ix*3+2]-1]*alpha[ix*3+2];
        double dvxdy = vx[index[ix*3+0]-1]* beta[ix*3+0] + vx[index[ix*3+1]-1]* beta[ix*3+1] + vx[index[ix*3+2]-1]* beta[ix*3+2];
        double dvydx = vy[index[ix*3+0]-1]*alpha[ix*3+0] + vy[index[ix*3+1]-1]*alpha[ix*3+1] + vy[index[ix*3+2]-1]*alpha[ix*3+2];
        double dvydy = vy[index[ix*3+0]-1]* beta[ix*3+0] + vy[index[ix*3+1]-1]* beta[ix*3+1] + vy[index[ix*3+2]-1]* beta[ix*3+2];

        double  eps_xx = dvxdx;
        double  eps_yy = dvydy;
        double  eps_xy = .5*(dvxdy+dvydx);
        double  EII2   = eps_xx*eps_xx + eps_yy*eps_yy + eps_xy*eps_xy + eps_xx*eps_yy;
        double  eta_it = 1.e+14/2.0;

        if (EII2>0.) eta_it = rheology_B[ix]/(2*pow(EII2,(n_glen-1.)/(2*n_glen)));

        /*Skip if no ice*/
        if (isice[ix]){
            etan[ix] = min(exp(rele*log(eta_it) + (1-rele)*log(etan[ix])),eta_0*1e5);
            /*Viscous Deformation*/
            for (int i = 0; i < 3; i++){
                kvx[ix*3+i] = 2 * Helem[ix] * etan[ix] * (2 * eps_xx + eps_yy) * alpha[ix*3+i] * areas[ix] + 2 * Helem[ix] * etan[ix] * eps_xy *  beta[ix*3+i] * areas[ix];
                kvy[ix*3+i] = 2 * Helem[ix] * etan[ix] * (2 * eps_yy + eps_xx) *  beta[ix*3+i] * areas[ix] + 2 * Helem[ix] * etan[ix] * eps_xy * alpha[ix*3+i] * areas[ix];
            }
        }//isice loop

        Eta_nbe[ix] = etan[ix]*areas[ix];
    }
}

//Moving to the next kernel, as kvx cannot be defined and updated in the same kernel
__global__ void PT2_x(double* kvx, double* groundedratio, double* areas, int* index, double* alpha2, double* vx, bool* isice,  int nbe){

    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < nbe; ix += blockDim.x * gridDim.x){
        /*Add basal friction*/
        if (groundedratio[ix] > 0.){
            int n3 = ix * 3;
            double gr_a = groundedratio[ix] * areas[ix];
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    int i_index = index[n3 + i] - 1;
                    double gr_a_alpha2 = gr_a * alpha2[i_index];
                    for (int j = 0; j < 3; j++){
                        int j_index = index[n3 + j] - 1;
                        double gr_a_alpha2_vx = gr_a_alpha2 * vx[j_index];
                        // printf("%d, %f, %f, %d, %f \n", ix, gr_a, gr_a_alpha2, j_index, gr_a_alpha2_vx);
                        if (i == j && j == k){
                            kvx[n3 + k] = isice[ix] * kvx[n3 + k] + gr_a_alpha2_vx / 10.;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
                            kvx[n3 + k] = isice[ix] * kvx[n3 + k] + gr_a_alpha2_vx / 60.;
                        } else{
                            kvx[n3 + k] = isice[ix] * kvx[n3 + k] + gr_a_alpha2_vx / 30.;
                        }
                    }
                }
            }
        }//groundedratio loop
    }
}

__global__ void PT2_y(double* kvy, double* groundedratio, double* areas, int* index, double* alpha2, double* vy, bool* isice,  int nbe){

   for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < nbe; ix += blockDim.x * gridDim.x){
        /*Add basal friction*/
        if (groundedratio[ix] > 0.){
            int n3 = ix * 3;
            double gr_a = groundedratio[ix] * areas[ix];
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    int i_index = index[n3 + i] - 1;
                    double gr_a_alpha2 = gr_a * alpha2[i_index];
                    for (int j = 0; j < 3; j++){
                        int j_index = index[n3 + j] - 1;
                        double gr_a_alpha2_vy = gr_a_alpha2 * vy[j_index];
                        // printf("%d, %f, %f, %d, %f \n", ix, gr_a, gr_a_alpha2, j_index, gr_a_alpha2_vx);
                        if (i == j && j == k){
                            kvy[n3 + k] = isice[ix] * kvy[n3 + k] + gr_a_alpha2_vy / 10.;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
                            kvy[n3 + k] = isice[ix] * kvy[n3 + k] + gr_a_alpha2_vy / 60.;
                        } else{
                            kvy[n3 + k] = isice[ix] * kvy[n3 + k] + gr_a_alpha2_vy / 30.;
                        }
                    }
                }
            }
        }//groundedratio loop
    }
}

//Moving to the next kernel: cannot update kvx and perform indirect access, lines 474 and 475, in the same kernel//
__global__ void PT3(double* kvx, double* kvy, double* Eta_nbe, double* areas, double* eta_nbv, int* index, int* connectivity, int* columns, double* weights, double* ML, double* KVx, double* KVy, double* Fvx, double* Fvy, double* dVxdt, double* dVydt, double* resolx, double* resoly, double* H, double* vx, double* vy, double* spcvx, double* spcvy, double rho, double damp, double relaxation, double eta_b, int nbv){

    double ResVx;
    double ResVy;
    double dtVx;
    double dtVy;

    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix<nbv; ix += blockDim.x * gridDim.x){

        KVx[ix] = 0.;
        KVy[ix] = 0.;

        for(int j=0;j<8;j++){
            if (connectivity[(ix * 8 + j)] != 0){
                KVx[ix] = KVx[ix] + kvx[((connectivity[(ix * 8 + j)])-1) *3 + ((columns[(ix * 8 + j)]))];
                KVy[ix] = KVy[ix] + kvy[((connectivity[(ix * 8 + j)])-1) *3 + ((columns[(ix * 8 + j)]))];
            }
        }

        for (int j = 0; j < 8; j++){
            if (connectivity[(ix * 8 + j)] != 0){
                eta_nbv[ix] = eta_nbv[ix] + Eta_nbe[connectivity[(ix * 8 + j)]-1];
            }
        }

        eta_nbv[ix] =eta_nbv[ix]/weights[ix];

        /*1. Get time derivative based on residual (dV/dt)*/
        ResVx =  1./(rho*max(60.0,H[ix])*ML[ix])*(-KVx[ix] + Fvx[ix]);
ResVy =  1./(rho*max(60.0,H[ix])*ML[ix])*(-KVy[ix] + Fvy[ix]);

        dVxdt[ix] = dVxdt[ix]*damp + ResVx;
        dVydt[ix] = dVydt[ix]*damp + ResVy;

        /*2. Explicit CFL time step for viscous flow, x and y directions*/
        dtVx = rho*resolx[ix]*resolx[ix]/(4*eta_nbv[ix]*(1.+eta_b)*4.1);
        dtVy = rho*resoly[ix]*resolx[ix]/(4*eta_nbv[ix]*(1.+eta_b)*4.1);

        /*3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy*/
        vx[ix] = vx[ix] + relaxation*dVxdt[ix]*dtVx;
        vy[ix] = vy[ix] + relaxation*dVydt[ix]*dtVy;

        /*Apply Dirichlet boundary condition*/
        if (!isnan(spcvx[ix])){
            vx[ix]    = spcvx[ix];
            dVxdt[ix] = 0.;
        }
        if (!isnan(spcvy[ix])){
            vy[ix]    = spcvy[ix];
            dVydt[ix] = 0.;
        }
    }
}

/*Main*/
//int main(){
int main(int argc, char *argv[]){
if (argc < 4) { cerr<<"Wired!!!!!"<<endl; return -1; }

    /*Open input binary file*/
    //const char* inputfile  = "/home/kuanghsu.wang/wais/test/WAIS7e4.bin";
    const char* inputfile  = "/home/kuanghsu.wang/PIG2e6_rand1/test/PIG2e6.bin";
    const char* outputfile = "/home/kuanghsu.wang/PIG2e6_rand1/test/output.outbin";
    FILE* fid = fopen(inputfile,"rb");
    if(fid==NULL) cerr<<"could not open file " << inputfile << " for binary reading or writing";


    /*Get All we need from binary file*/
    int    nbe,nbv,M,N;
    double g,rho,rho_w,yts;
    int    *index           = NULL;
    double *spcvx           = NULL;
    double *spcvy           = NULL;
    double *x               = NULL;
    double *y               = NULL;
    double *H               = NULL;
    double *surface         = NULL;
    double *base            = NULL;
    double *ice_levelset    = NULL;
    double *ocean_levelset  = NULL;
    double *rheology_B_temp = NULL;
    double *vx              = NULL;
    double *vy              = NULL;
    double *friction        = NULL;
    FetchData(fid,&nbe,"md.mesh.numberofelements");
    FetchData(fid,&nbv,"md.mesh.numberofvertices");
    FetchData(fid,&g,"md.constants.g");
    FetchData(fid,&rho,"md.materials.rho_ice");
    FetchData(fid,&rho_w,"md.materials.rho_water");
    FetchData(fid,&yts,"md.constants.yts");
    FetchData(fid,&index,&M,&N,"md.mesh.elements");
    FetchData(fid,&spcvx,&M,&N,"md.stressbalance.spcvx");
    FetchData(fid,&spcvy,&M,&N,"md.stressbalance.spcvy");
    FetchData(fid,&x,&M,&N,"md.mesh.x");
    FetchData(fid,&y,&M,&N,"md.mesh.y");
    FetchData(fid,&H,&M,&N,"md.geometry.thickness");
    FetchData(fid,&surface,&M,&N,"md.geometry.surface");
    FetchData(fid,&base,&M,&N,"md.geometry.base");
    FetchData(fid,&ice_levelset,&M,&N,"md.mask.ice_levelset");
    FetchData(fid,&ocean_levelset,&M,&N,"md.mask.ocean_levelset");
    FetchData(fid,&rheology_B_temp,&M,&N,"md.materials.rheology_B");
    FetchData(fid,&vx,&M,&N,"md.initialization.vx");
    FetchData(fid,&vy,&M,&N,"md.initialization.vy");
    FetchData(fid,&friction,&M,&N,"md.friction.coefficient");

    /*Close input file*/
    if(fclose(fid)!=0) cerr<<"could not close file " << inputfile;

    /*Constants*/
    double n_glen     = 3.;
    double damp       = atof(argv[1]);             // 0.96 for JKS2e4, 0.981 for PIG3e4
    double rele       = atof(argv[2]);             // 1e-1 for JKS2e4, 0.07 for PIG3e4
    double eta_b      = 0.5;
    double eta_0      = 1.e+14/2.;
    int    niter      = 25000;
    int    nout_iter  = 1;                        //change it to 100 for JKS2e4
    double epsi       = 3.171e-7;
    double relaxation = atof(argv[3]);           // 0.7 for JKS2e4, 0.967 for PIG3e4 

  //  printf("damp = %f  , relaxation = %f \n", damp , relaxation);
  //  printf("damp = %f\n rele = %f\n relaxation = %f\n\n", damp , rele, relaxation);


    // Ceiling division to get the close to optimal GRID size
    unsigned int GRID_Xe = 1 + ((nbe - 1) / BLOCK_Xe);
    unsigned int GRID_Xv = 1 + ((nbv - 1) / BLOCK_Xv);

   // GRID_Xe = GRID_Xe - GRID_Xe%80;
    //GRID_Xv = GRID_Xv - GRID_Xv%80;

    cout<<"GRID_Xe="<<GRID_Xe<<endl;
    cout<<"GRID_Xv="<<GRID_Xv<<endl;

    // Set up GPU
    int gpu_id=-1;
    dim3 gridv, blockv;
    dim3 gride, blocke;
    blockv.x = BLOCK_Xv; gridv.x = GRID_Xv;
    blocke.x = BLOCK_Xe; gride.x = GRID_Xe;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d.\n", gpu_id);
    //cudaSetDevice  selects the device, set the gpu id you selected

    /*Initial guesses (except vx and vy that we already loaded)*/
    double* etan = new double[nbe];
    for(int i=0;i<nbe;i++) etan[i] = 1.e+14;
    double* dVxdt = new double[nbv];
    for(int i=0;i<nbv;i++) dVxdt[i] = 0.;
    double* dVydt = new double[nbv];
    for(int i=0;i<nbv;i++) dVydt[i] = 0.;

    /*Manage derivatives once for all*/
    double* alpha   = NULL;
    double* beta    = NULL;
    double* areas   = NULL;
    double* weights = NULL;
    NodalCoeffs(&areas,&alpha,&beta,index,x,y,nbe);
    Weights(&weights,index,areas,nbe,nbv);

    /*MeshSize*/
    double* resolx = new double[nbv];
    double* resoly = new double[nbv];
    MeshSize(resolx,resoly,index,x,y,areas,weights,nbe,nbv);

    /*Physical properties once for all*/
    double* dsdx = new double[nbe];
    double* dsdy = new double[nbe];
    derive_xy_elem(dsdx,dsdy,surface,index,alpha,beta,nbe);
    double* Helem      = new double[nbe];
    double* rheology_B = new double[nbe];
    for(int i=0;i<nbe;i++){
        Helem[i]      = 1./3. * (H[index[i*3+0]-1] + H[index[i*3+1]-1] + H[index[i*3+2]-1]);
        rheology_B[i] = 1./3. * (rheology_B_temp[index[i*3+0]-1] + rheology_B_temp[index[i*3+1]-1] + rheology_B_temp[index[i*3+2]-1]);
    }

    //Initial viscosity//
    double* dvxdx   = new double[nbe];
    double* dvxdy   = new double[nbe];
    double* dvydx   = new double[nbe];
    double* dvydy   = new double[nbe];

    derive_xy_elem(dvxdx,dvxdy,vx,index,alpha,beta,nbe);
    derive_xy_elem(dvydx,dvydy,vy,index,alpha,beta,nbe);

    for(int i=0;i<nbe;i++){
        double eps_xx = dvxdx[i];
        double eps_yy = dvydy[i];
        double eps_xy = .5*(dvxdy[i]+dvydx[i]);
        double EII2 = pow(eps_xx,2) + pow(eps_yy,2) + pow(eps_xy,2) + eps_xx*eps_yy;
        double eta_it = 1.e+14/2.;
        if (EII2>0.) eta_it = rheology_B[i]/(2*pow(EII2,(n_glen-1.)/(2*n_glen)));

        etan[i] = min(eta_it,eta_0*1e5);
        if (isnan(etan[i])){ cerr<<"Found NaN in etan[i]"; return 1;}
    }

    /*Linear integration points order 3*/
    double wgt3[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556 };
    double xg3[]  = {-0.774596669241483, 0.000000000000000, 0.774596669241483 };

    /*Compute RHS amd ML once for all*/
    double* ML            = new double[nbv];
    double* Fvx           = new double[nbv];
    double* Fvy           = new double[nbv];
    double* groundedratio = new double[nbe];
    bool*   isice         = new bool[nbe];
    double level[3];

    for(int i=0;i<nbv;i++){
        ML[i]  = 0.;
        Fvx[i] = 0.;
        Fvy[i] = 0.;
    }
    for(int n=0;n<nbe;n++){
        /*Lumped mass matrix*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                // \int_E phi_i * phi_i dE = A/6 and % \int_E phi_i * phi_j dE = A/12
                if (i==j)
                 ML[index[n*3+j]-1] += areas[n]/6.;
                else
                 ML[index[n*3+j]-1] += areas[n]/12.;
            }
        }
        /*Is there ice at all in the current element?*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        if (level[0]<0 || level[1]<0 || level[2]<0){
            isice[n] = true;
        }
        else{
            isice[n] = false;
            for(int i=0;i<3;i++){
                vx[index[n*3+i]-1] = 0.;
                vy[index[n*3+i]-1] = 0.;
}
            continue;
        }
        /*RHS, 'F ' in equation 22 (Driving Stress)*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (i==j){
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/6.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/6.;
                }
                else{
                    Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/12.;
                    Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/12.;
                }
            }
        }
    }

    /*RHS (Water pressure at the ice front)*/
    //  double level[3];
    for(int n=0;n<nbe;n++){
        /*Determine if there is an ice front there*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        int count = 0;
        for(int i=0;i<3;i++) if (level[i]<0.) count++;
        if (count==1){
            /*Ok this element has an ice front, get indices of the 2 vertices*/
            int seg1[2] = {index[n*3+0]-1,index[n*3+1]-1};
            int seg2[2] = {index[n*3+1]-1,index[n*3+2]-1};
            int seg3[2] = {index[n*3+2]-1,index[n*3+0]-1};
            int pairids[2];
            if (ice_levelset[seg1[0]]>=0 && ice_levelset[seg1[1]]>=0){
                pairids[0] = seg1[0]; pairids[1] = seg1[1];
            }
            else if (ice_levelset[seg2[0]]>=0 && ice_levelset[seg2[1]]>=0){
                pairids[0] = seg2[0]; pairids[1] = seg2[1];
            }
            else if (ice_levelset[seg3[0]]>=0 && ice_levelset[seg3[1]]>=0){
                pairids[0] = seg3[0]; pairids[1] = seg3[1];
            }
              else{
                cerr<<"case not supported";
            }
            /*Get normal*/
            double len = sqrt(pow(x[pairids[1]]-x[pairids[0]],2) + pow(y[pairids[1]]-y[pairids[0]],2) );
            double nx  = +(y[pairids[1]]-y[pairids[0]])/len;
            double ny  = -(x[pairids[1]]-x[pairids[0]])/len;
            /*RHS*/
            for(int gg=0;gg<2;gg++){
                double phi1 = (1.0 -xg3[gg])/2.;
                double phi2 = (1.0 +xg3[gg])/2.;
                double bg = base[pairids[0]]*phi1 + base[pairids[1]]*phi2;
                double Hg = H[pairids[0]]*phi1 + H[pairids[1]]*phi2;
                bg = min(bg,0.0);
                Fvx[pairids[0]] = Fvx[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g* pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi1;
                Fvx[pairids[1]] = Fvx[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi2;
                Fvy[pairids[0]] = Fvy[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi1;
                Fvy[pairids[1]] = Fvy[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi2;
            }
        }
        /*One more thing in this element loop: prepare groundedarea needed later for the calculation of basal friction*/
        level[0] = ocean_levelset[index[n*3+0]-1];
        level[1] = ocean_levelset[index[n*3+1]-1];
        level[2] = ocean_levelset[index[n*3+2]-1];
        if (level[0]>=0. && level[1]>=0. && level[2]>=0.){
            /*Completely grounded*/
            groundedratio[n]=1.;
        }
        else if (level[0]<=0. && level[1]<=0. && level[2]<=0.){
            /*Completely floating*/
            groundedratio[n]=0.;
        }
        else{
            /*Partially floating,*/
            double s1,s2;
            if (level[0]*level[1]>0){/*Nodes 0 and 1 are similar, so points must be found on segment 0-2 and 1-2*/
                s1=level[2]/(level[2]-level[1]);
                s2=level[2]/(level[2]-level[0]);
            }
            else if (level[1]*level[2]>0){ /*Nodes 1 and 2 are similar, so points must be found on segment 0-1 and 0-2*/
                s1=level[0]/(level[0]-level[1]);
                s2=level[0]/(level[0]-level[2]);
            }
            else if (level[0]*level[2]>0){/*Nodes 0 and 2 are similar, so points must be found on segment 1-0 and 1-2*/
                s1=level[1]/(level[1]-level[0]);
                s2=level[1]/(level[1]-level[2]);
            }
            else{
                cerr<<"should not be here...";
}

            if (level[0]*level[1]*level[2]>0.){
                /*two nodes floating, inner triangle is grounded*/
                groundedratio[n]= s1*s2;
            }
            else{
                /*one node floating, inner triangle is floating*/
                groundedratio[n]= (1.-s1*s2);
            }
        }
    }

    /*Finally add calculation of friction coefficient*/
    double* alpha2 = new double[nbv];
    for(int i=0;i<nbv;i++){
        /*Compute effective pressure*/
        double p_ice   = g*rho*H[i];
        double p_water = -rho_w*g*base[i];
        double Neff    = p_ice - p_water;
        if (Neff<0.) Neff=0.;
        /*Compute alpha2*/
        alpha2[i] = pow(friction[i],2)*Neff;
    }

    //prepare head and next vectors for chain algorithm, at this point we have not seen any of the elements, so just set the head to -1 (=stop)
    int* head = new int[nbv];
    int* next  = new int[3*nbe];
    for(int i=0;i<nbv;i++) head[i] = -1;

    //Now construct the chain
    for(int k=0;k<nbe;k++){
        for(int j=0;j<3;j++){
            int i;
            int p = 3*k+j;       //unique linear index of current vertex in index
            i = index[p];
            next[p] = head[i - 1];
            head[i -1] = p + 1;
        }
    }

    //Note: Index array starts at 0, but the node# starts at 1
    //Now we can construct the connectivity matrix
    int MAXCONNECT = 8;
    int* connectivity = new int[nbv*MAXCONNECT];
    int* columns = new int[nbv*MAXCONNECT];

    for(int i=0;i<nbv;i++){
/*Go over all of the elements connected to node I*/
        int count = 0;
        int p=head[i];

        //for (int p = head[i]; p != -1; p = next[p]){
        while (p!= -1){

            int k = p / 3 + 1;     //”row" in index
            int j = (p % 3) - 1;   //"column" in index

            if (j==-1){
                j=2;
                k= k -1;}

            //sanity check
            if (index[p-1] !=i+1){
                cout << "Error occurred"  << endl;;
            }

            //enter element in connectivity matrix
            connectivity[i * MAXCONNECT + count] = k;
            columns[i * MAXCONNECT + count] = j;
            count++;
            p = next[p-1];
        }
    }

    double* device_maxvalx = new double[GRID_Xv];
    double* device_maxvaly = new double[GRID_Xv];
    for(int i=0;i<GRID_Xv;i++) device_maxvalx[i] = 0.;
    for(int i=0;i<GRID_Xv;i++) device_maxvaly[i] = 0.;

    /*------------ now copy all relevant vectors from host to device ---------------*/
    int *d_index = NULL;
    cudaMalloc(&d_index, nbe*3*sizeof(int));
    cudaMemcpy(d_index, index, nbe*3*sizeof(int), cudaMemcpyHostToDevice);

    double *d_vx;
    cudaMalloc(&d_vx, nbv*sizeof(double));
    cudaMemcpy(d_vx, vx, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_vy;
    cudaMalloc(&d_vy, nbv*sizeof(double));
    cudaMemcpy(d_vy, vy, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_alpha;
    cudaMalloc(&d_alpha, nbe*3*sizeof(double));
cudaMemcpy(d_alpha, alpha, nbe*3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_beta;
    cudaMalloc(&d_beta, nbe*3*sizeof(double));
    cudaMemcpy(d_beta, beta, nbe*3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_etan;
    cudaMalloc(&d_etan, nbe*sizeof(double));
    cudaMemcpy(d_etan, etan, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_rheology_B;
    cudaMalloc(&d_rheology_B, nbe*sizeof(double));
    cudaMemcpy(d_rheology_B, rheology_B, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_Helem;
    cudaMalloc(&d_Helem, nbe*sizeof(double));
    cudaMemcpy(d_Helem, Helem, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_areas;
    cudaMalloc(&d_areas, nbe*sizeof(double));
    cudaMemcpy(d_areas, areas, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_weights;
    cudaMalloc(&d_weights, nbv*sizeof(double));
    cudaMemcpy(d_weights, weights, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_ML;
    cudaMalloc(&d_ML, nbv*sizeof(double));
    cudaMemcpy(d_ML, ML, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_Fvx;
    cudaMalloc(&d_Fvx, nbv*sizeof(double));
    cudaMemcpy(d_Fvx, Fvx, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_Fvy;
    cudaMalloc(&d_Fvy, nbv*sizeof(double));
    cudaMemcpy(d_Fvy, Fvy, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_dVxdt;
    cudaMalloc(&d_dVxdt, nbv*sizeof(double));
    cudaMemcpy(d_dVxdt, dVxdt, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_dVydt;
    cudaMalloc(&d_dVydt, nbv*sizeof(double));
    cudaMemcpy(d_dVydt, dVydt, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_resolx;
    cudaMalloc(&d_resolx, nbv*sizeof(double));
cudaMemcpy(d_resolx, resolx, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_resoly;
    cudaMalloc(&d_resoly, nbv*sizeof(double));
    cudaMemcpy(d_resoly, resoly, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_H;
    cudaMalloc(&d_H, nbv*sizeof(double));
    cudaMemcpy(d_H, H, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_spcvx;
    cudaMalloc(&d_spcvx, nbv*sizeof(double));
    cudaMemcpy(d_spcvx, spcvx, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_spcvy;
    cudaMalloc(&d_spcvy, nbv*sizeof(double));
    cudaMemcpy(d_spcvy, spcvy, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_alpha2;
    cudaMalloc(&d_alpha2, nbv*sizeof(double));
    cudaMemcpy(d_alpha2, alpha2, nbv*sizeof(double), cudaMemcpyHostToDevice);

    double *d_groundedratio;
    cudaMalloc(&d_groundedratio, nbe*sizeof(double));
    cudaMemcpy(d_groundedratio, groundedratio, nbe*sizeof(double), cudaMemcpyHostToDevice);

    bool *d_isice;
    cudaMalloc(&d_isice, nbe*sizeof(bool));
    cudaMemcpy(d_isice, isice, nbe*sizeof(bool), cudaMemcpyHostToDevice);

    int *d_connectivity = NULL;
    cudaMalloc(&d_connectivity, nbv*8*sizeof(int));
    cudaMemcpy(d_connectivity, connectivity, nbv*8*sizeof(int), cudaMemcpyHostToDevice);

    int *d_columns = NULL;
    cudaMalloc(&d_columns, nbv*8*sizeof(int));
    cudaMemcpy(d_columns, columns, nbv*8*sizeof(int), cudaMemcpyHostToDevice);

    double* d_device_maxvalx = NULL;
    cudaMalloc(&d_device_maxvalx, GRID_Xv*sizeof(double));
    cudaMemcpy(d_device_maxvalx, device_maxvalx, GRID_Xv*sizeof(double), cudaMemcpyHostToDevice);

    double* d_device_maxvaly = NULL;
    cudaMalloc(&d_device_maxvaly, GRID_Xv*sizeof(double));
    cudaMemcpy(d_device_maxvaly, device_maxvaly, GRID_Xv*sizeof(double), cudaMemcpyHostToDevice);

    /*------------ allocate relevant vectors on host (GPU)---------------*/
    //double *dvxdx = NULL;
    cudaMalloc(&dvxdx,nbe*sizeof(double));

    //double *dvxdy = NULL;
    cudaMalloc(&dvxdy, nbe*sizeof(double));

    //double *dvydx = NULL;
    cudaMalloc(&dvydx, nbe*sizeof(double));

    //double *dvydy = NULL;
    cudaMalloc(&dvydy, nbe*sizeof(double));

    double *KVx = NULL;
    cudaMalloc(&KVx, nbv*sizeof(double));

    double *KVy = NULL;
    cudaMalloc(&KVy, nbv*sizeof(double));

    double *eta_nbv = NULL;
    cudaMalloc(&eta_nbv, nbv*sizeof(double));

    double *Eta_nbe = NULL;
    cudaMalloc(&Eta_nbe, nbe*3*sizeof(double));

    double *kvx = NULL;
    cudaMalloc(&kvx, nbe*3*sizeof(double));

    double *kvy = NULL;
    cudaMalloc(&kvy, nbe*3*sizeof(double));

    //Creating CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Perf
    double time_s = 0.0;
    double mem = (double)1e-9*(double)nbv*sizeof(double);
    int nIO = 8;

    /*Main loop*/
    cout<<"Starting PT loop, nbe="<<nbe<<", nbv="<<nbv<<endl;
    int iter;
    double iterror= -1.0;
    ofstream outfile;
    ofstream errorfile;
    ofstream iterfile;
    outfile.open("icesheet.stat");
    errorfile.open("error.csv");
    iterfile.open("iters.csv");
    outfile<<setprecision(3)<<"damping="<<damp<<setprecision(3)<<", velocity relaxation="<<rele<<setprecision(3)<<", viscosity relaxation="<<relaxation<<endl;
  //  outfile<<"damping="<<damp<<", velocity relaxation="<<rele<<", viscosity relaxation="<<relaxation<<endl;    
  //  cout<<"damping="<<damp<<", viscosity relaxation="<<rele<<", velocity relaxation="<<relaxation<<endl;    
    for(iter=1;iter<=niter;iter++){
    	if (iter==11) tic();

        PT1<<<gride, blocke>>>(d_vx, d_vy, d_alpha, d_beta, d_index, kvx,  kvy, d_etan, d_Helem, d_areas, d_isice, Eta_nbe, d_rheology_B, n_glen, eta_0, rele, nbe);
        cudaDeviceSynchronize();

        PT2_x<<<gride, blocke, 0, stream1>>>(kvx, d_groundedratio, d_areas, d_index, d_alpha2, d_vx, d_isice, nbe);
        cudaStreamSynchronize(stream1);
        PT2_y<<<gride, blocke, 0, stream2>>>(kvy, d_groundedratio, d_areas, d_index, d_alpha2, d_vy, d_isice, nbe);
        cudaStreamSynchronize(stream2);
        // DEBUG: Some stream sync may be missing here

        PT3<<<gridv, blockv>>>(kvx, kvy, Eta_nbe, d_areas, eta_nbv, d_index, d_connectivity, d_columns, d_weights, d_ML, KVx, KVy, d_Fvx, d_Fvy, d_dVxdt, d_dVydt, d_resolx, d_resoly, d_H, d_vx, d_vy, d_spcvx, d_spcvy, rho, damp, relaxation, eta_b, nbv);
        cudaDeviceSynchronize();
	//cout<<"It is running fine."<<endl;
	//cerr<<"It is running bad."<<endl;
        if ((iter % nout_iter) == 0){
           //Get final error estimate/
            __device_max_x(dVxdt);
            __device_max_y(dVydt);
            iterror = max(device_MAXx, device_MAXy);
	    if(iterror ==0) {iterror = NAN;}
            //if(!(iterror>0 || iterror==0 || iterror<0)){printf("\n !! ERROR: err_MAX=NaN \n\n");break;} 
            if((isnan(iterror))){printf("\n !! ERROR: err_MAX=Nan \n\n");break;}
            // Original line: if(!(iterror>0 || iterror==0 || iterror<0)){printf("\n !! ERROR: err_MAX=Nan \n\n");break;}
                cout<<"iter="<<iter<<", err="<<iterror<<endl;
                outfile<<"iter="<<iter<<", err="<<iterror<<endl;
                errorfile<<iterror<<endl;
                iterfile<<iter<<endl;
            if ((iterror < epsi) && (iter > 100)) break;
        }
    }
      outfile<<"iter="<<iter<<", err="<<iterror<<endl;
//    cout<<"iter="<<iter<<", err="<<iterror<<endl;

    time_s = toc(); double gbs = mem/time_s;

    cout<<"Perf: "<<time_s<<" sec. (@ "<<gbs*(iter-10)*nIO<<" GB/s)"<<endl;

    /*Copy results from Device to host*/
    cudaMemcpy(vx, d_vx, nbv*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(vy, d_vy, nbv*sizeof(double), cudaMemcpyDeviceToHost );

    cudaMemcpy(dVxdt, d_dVxdt, nbv*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(dVydt, d_dVydt, nbv*sizeof(double), cudaMemcpyDeviceToHost );
    for(int i=1;i<=nbv;i++){
        outfile<<" dVxdt =  "<<dVxdt[i]<<" m/s2; "<<" dVydt = "<<dVydt[i]<<" m/s2; "<<std::endl;  
    } 
    outfile.close();
    errorfile.close();
    iterfile.close();

    //for(int i=1;i<=nbv;i++){
   // outfile<<" Vx =  "<<vx[i]<<" m/s; "<<" Vy = "<<vy[i]<<" m/s; "<<std::endl;  
  //  } 
  //  outfile.close();

    /*Write output*/
    fid = fopen(outputfile,"wb");
    if (fid==NULL) cerr<<"could not open file " << outputfile << " for binary reading or writing";
    WriteData(fid, "PTsolution", "SolutionType");
    WriteData(fid, vx, nbv, 1, "Vx");
    WriteData(fid, vy, nbv, 1, "Vy");
    if (fclose(fid)!=0) cerr<<"could not close file " << outputfile;

    /*Cleanup and return*/
    delete [] index;
    delete [] x;
    delete [] y;
    delete [] H;
    delete [] surface;
    delete [] base;
    delete [] spcvx;
    delete [] spcvy;
    delete [] ice_levelset;
    delete [] ocean_levelset;
    delete [] rheology_B;
    delete [] rheology_B_temp;
    delete [] vx;
    delete [] vy;
    delete [] friction;
    delete [] alpha2;
    delete [] etan;
    delete [] dVxdt;
    delete [] dVydt;
    delete [] alpha;
    delete [] beta;
    delete [] areas;
    delete [] weights;
    delete [] resolx;
    delete [] resoly;
    delete [] dsdx;
    delete [] dsdy;
    delete [] Helem;
    delete [] ML;
    delete [] Fvx;
    delete [] Fvy;

    cudaFree(d_index);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_etan);
    cudaFree(d_rheology_B);
    cudaFree(d_Helem);
    cudaFree(d_areas);
    cudaFree(d_weights);
    cudaFree(d_ML);
    cudaFree(d_Fvx);
    cudaFree(d_Fvy);
    cudaFree(d_dVxdt);
    cudaFree(d_dVydt);
    cudaFree(d_resolx);
    cudaFree(d_resoly);
    cudaFree(d_H);
    cudaFree(d_spcvx);
    cudaFree(d_spcvy);
    cudaFree(d_alpha2);
    cudaFree(d_groundedratio);
    cudaFree(d_isice);
    cudaFree(d_connectivity);
    cudaFree(d_columns);
    cudaFree(dvxdx);
    cudaFree(dvxdy);
    cudaFree(dvydx);
    cudaFree(dvydy);
    cudaFree(KVx);
    cudaFree(KVy);
    cudaFree(eta_nbv);
    cudaFree(Eta_nbe);
    cudaFree(kvx);
    cudaFree(kvy);
    cudaFree(d_device_maxvalx);
    cudaFree(d_device_maxvaly);

    //Destroying CUDA streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    clean_cuda();
    return 0;
}
