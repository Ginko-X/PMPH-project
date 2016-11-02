#include <cuda_runtime.h>
#include "../include/CudaUtilProj.cu.h"

//{{{KERNELS
__global__ void
d_initTimeline( REAL* d_timeline, const unsigned numT, const REAL t){
    unsigned gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < numT) {
        d_timeline[gid] =  t*gid / (numT-1);
    }
}


__global__ void
d_initNUM( REAL* d_num, unsigned int num_size, const REAL d, unsigned myIndex, const REAL s){
    const unsigned long gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < num_size) {
        d_num[gid] = gid*d - myIndex*d + s;
    }
}


__global__ void
d_initOperator( REAL* d_x, unsigned int x_size, REAL* d_dxx){
    const unsigned long gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < x_size) {
        REAL dxl, dxu;
        if(gid == 0){
            //  lower boundary
            dxl      =  0.0;
            dxu      =  d_x[1] - d_x[0];

            d_dxx[0] =  0.0;
            d_dxx[1] =  0.0;
            d_dxx[2] =  0.0;
            d_dxx[3] =  0.0;
        }else if(gid == x_size-1){
            //  upper boundary
            dxl        =  d_x[x_size-1] - d_x[x_size-2];
            dxu        =  0.0;

            d_dxx[(x_size-1)*4+0] = 0.0;
            d_dxx[(x_size-1)*4+1] = 0.0;
            d_dxx[(x_size-1)*4+2] = 0.0;
            d_dxx[(x_size-1)*4+3] = 0.0;
        }else{
            dxl      = d_x[gid]     - d_x[gid-1];
            dxu      = d_x[gid+1]   - d_x[gid];

            d_dxx[gid*4+0] =  2.0/dxl/(dxl+dxu);
            d_dxx[gid*4+1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
            d_dxx[gid*4+2] =  2.0/dxu/(dxl+dxu);
            d_dxx[gid*4+3] =  0.0;
        }
    }
}

__global__ void
d_setPayoff(REAL* d_result, REAL* d_x, unsigned int x_size, unsigned int y_size, unsigned int z_size){
    unsigned int x = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z*blockIdx.z + threadIdx.z;

    if(x < x_size && y < y_size && z < z_size){
        d_result[z*y_size*x_size + y*x_size + x] = max(d_x[y]-(0.001*z), (REAL)0.0);
    }
}


__global__ void
d_updateParams(REAL* d_varX, REAL* d_varY, REAL* d_x, REAL* d_y, REAL* d_timeline, 
    unsigned int g, REAL alpha, REAL beta, REAL nu, 
    unsigned int numX, unsigned int numY){

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;

    if(i >= numX || j >= numY)
        return;

    d_varX[i*numY+j] = exp(2.0*( beta*log(d_x[i]) +  d_y[j] - 0.5*nu*nu*d_timeline[g]));
    d_varY[i*numY+j] = exp(2.0*( alpha*log(d_x[i]) + d_y[j] - 0.5*nu*nu*d_timeline[g]));

}


#define YX(k,j,i) ((k)*(numY)*(numX)+(j)*(numX)+(i))
#define XY(k,j,i) ((k)*(numY)*(numX)+(j)*(numY)+(i))
#define ZZ(k,j,i) (k*(numZ)*(numZ)+(j)*(numZ)+(i))
#define D4ID(j,i) ((j)*4+(i))

__global__ void
d_explicit_xy_implicit_x(REAL* u, REAL* v, REAL* a, REAL* b, REAL* c,  
    REAL* varX, REAL* varY, REAL* timeline, REAL* dxx, REAL* dyy, REAL* result, 
    unsigned int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
   
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int j = blockDim.y * blockIdx.y + threadIdx.y; //numY
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //numX


    if(k >= outer || j >= numY || i >= numX)
    return;
    
   //  //  explicit x 
   //  u[YX(k,j,i)] =  (1.0/(timeline[g+1]-timeline[g])) *result[XY(k,j,i)];

   //  if(i > 0) {
   //    u[YX(k,j,i)] += 0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,0)] ) 
         //    * result[XY(k,i-1,j)];
   //  }
   //  u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,1)] )
         //    * result[XY(k,i,j)];
   //  if(i < numX-1) {
   //    u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,2)] )
         //    * result[XY(k,i+1,j)];
   //  }

   //  //  explicit y ; RAW v, write u
   //  v[XY(0,0,j)] = 0.0;

   //  if(j > 0) {
   //    v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,0)] )
         // *  result[XY(k,i,j-1)];
   //  }
   //    v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,1)] )
         // *  result[XY(k,i,j)];
   //  if(j < numY-1) {
   //    v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,2)] )
         // *  result[XY(k,i,j+1)];
   //  }
   //  u[YX(k,i,j)] += v[XY(k,i,j)];


    //  implicit x  // write a,b,c
    a[ZZ(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,0)]);
    b[ZZ(k,j,i)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,1)]);
    c[ZZ(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,2)]);
    
}