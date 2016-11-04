
#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.h"

#include "../include/CudaUtilProj.cu.h"

//#include "ProjHost.cu"

#define EPSILON 0.001
#define T 32

#define YX(k,j,i) ((k)*(numY)*(numX)+(j)*(numX)+(i))  //[-][numY][numX]
#define XY(k,j,i) ((k)*(numY)*(numX)+(j)*(numY)+(i)) //[-][numX][numY]
#define ZZ(k,j,i) (k*(numZ)*(numZ)+(j)*(numZ)+(i))    //[-][numZ][numZ]
#define D4ID(j,i) ((j)*4+(i))
#define X4(j,i) ((j)*numX+(i))
#define Y4(j,i) ((j)*numY+(i))


//{{{KERNELS  ------ 
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
d_updateParams(REAL* d_varX, REAL* d_varY, REAL* d_x, REAL* d_y,  REAL* d_timeline,
    int g, REAL alpha, REAL beta, REAL nu, 
    unsigned int numX, unsigned int numY){

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;

    if(i >= numX || j >= numY)
        return;

    d_varX[i*numY+j] = exp(2.0*( beta*log(d_x[i]) +  d_y[j] - 0.5*nu*nu*d_timeline[g]));
    d_varY[i*numY+j] = exp(2.0*( alpha*log(d_x[i]) + d_y[j] - 0.5*nu*nu*d_timeline[g]));

}

__global__ void
d_updateParams_sh(REAL* d_varX, REAL* d_varY, REAL* d_x, REAL* d_y, REAL* d_timeline, 
    unsigned int g, REAL alpha, REAL beta, REAL nu, 
    unsigned int numX, unsigned int numY){

    __shared__ REAL sh_varX[T][T+1], sh_varY[T][T+1]; //

    __shared__ REAL sh_x[T], sh_y[T]; //

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;  //numY
    unsigned int j = blockDim.y*blockIdx.y + threadIdx.y;  //numX
    int tidy = threadIdx.y;
    int tidx = threadIdx.x;

    if(j >= numX || i >= numY)
        return;

    // shared memory store operation
    sh_varX[tidy][tidx] = d_varX[j*numY+i];
    sh_varY[tidy][tidx] = d_varY[j*numY+i]; 
    sh_x[tidy] = d_x[j];
    sh_y[tidx] = d_y[i];
    __syncthreads();

    sh_varX[tidy][tidx] = exp(2.0*( beta*log(sh_x[tidy]) +  sh_y[tidx] - 0.5*nu*nu*d_timeline[g]));
    sh_varY[tidy][tidx] = exp(2.0*( alpha*log(sh_x[tidy]) + sh_y[tidx] - 0.5*nu*nu*d_timeline[g]));

    d_varX[j*numY+i] = sh_varX[tidy][tidx]; 
    d_varY[j*numY+i] = sh_varY[tidy][tidx]; 

}


__global__ void
d_explicit_xy_implicit_x(REAL* u, REAL* v, REAL* a, REAL* b, REAL* c,  
    REAL* varX, REAL* varY, REAL* timeline, REAL* dxx, REAL* dyy, REAL* result, 
    unsigned int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
   
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int j = blockDim.y * blockIdx.y + threadIdx.y; //numY
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //numX


    if(k >= outer || j >= numY || i >= numX)
    return;
    
    //  explicit x 
    u[YX(k,j,i)] =  (1.0/(timeline[g+1]-timeline[g])) *result[XY(k,i,j)];

    if(i > 0) {
      u[YX(k,j,i)] += 0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,0)] ) 
            * result[XY(k,i-1,j)];
    }
    u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,1)] )
            * result[XY(k,i,j)];
    if(i < numX-1) {
      u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[D4ID(i,2)] )
            * result[XY(k,i+1,j)];
    }

    //  explicit y ; RAW v, write u
    v[XY(k,i,j)] = 0.0;

    if(j > 0) {
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,0)] )
         *  result[XY(k,i,j-1)];
    }
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,1)] )
         *  result[XY(k,i,j)];
    if(j < numY-1) {
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[D4ID(j,2)] )
         *  result[XY(k,i,j+1)];
    }
    u[YX(k,j,i)] += v[XY(k,i,j)];


    //  implicit x  // write a,b,c
    a[ZZ(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,0)]);
    b[ZZ(k,j,i)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,1)]);
    c[ZZ(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[D4ID(i,2)]);
    
}



#define UI(k,j,i) ((k)*(middle)*(n)+(j)*(n)+(i))  

__global__ void
d_tridag_implicit_y(
    REAL* a, REAL* b, REAL* c, REAL* r, int n, REAL* u, REAL* uu, // tridag 
    unsigned numX, unsigned numY, unsigned outer, unsigned numZ, unsigned middle){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x; //numX
    unsigned int k = blockDim.y*blockIdx.y + threadIdx.y; //outer

    if(k >= outer || j >= middle)
        return;
 
    REAL   beta;

    u[UI(k,j,0)]  = r[ZZ(k,j,0)]; //u[k*numX*numY + j*numY + i]
    uu[ZZ(k,j,0)] = b[ZZ(k,j,0)]; 

    for(int i=1; i< n; i++) {
        beta  = a[ZZ(k,j,i)] / uu[ZZ(k,j,i-1)];

        uu[ZZ(k,j,i)] = b[ZZ(k,j,i)] - beta*c[ZZ(k,j,i-1)];
        u[UI(k,j,i)]  = r[ZZ(k,j,i)] - beta*u[UI(k,j,i-1)];
    }

    u[UI(k,j,n-1)] = u[UI(k,j,n-1)] / uu[ZZ(k,j,n-1)];
    for(int i=n-2; i>=0; i--) {
        u[UI(k,j,i)]  = (u[UI(k,j,i)]  - c[ZZ(k,j,i)]*u[UI(k,j,i+1)] ) / uu[ZZ(k,j,i)];
    }
}

/*
__global__ void
sh_tridag_implicit_y(  // u = myresult
    REAL* a, REAL* b, REAL* c, REAL* r, int n, REAL* u, REAL* uu, // tridag 
    unsigned numX, unsigned numY, unsigned outer, unsigned numZ, unsigned middle){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x; //numX
    unsigned int k = blockDim.y*blockIdx.y + threadIdx.y; //outer

    if(k >= outer || j >= middle)
        return;

    __shared__ REAL 
        // sh_a[T][T+1],   otherwise Too much shared data
        sh_b[T][T+1],
        sh_c[T][T+1], sh_r[T][T+1],
        sh_u[T][T+1], sh_uu[T][T+1]; //

    int tidy = threadIdx.y;
    int tidx = threadIdx.x;

    REAL   beta;

    for(int ii=0; ii< n; ii+=T) {

        // sh_a[tidy][tidx] = a[k*numZ*numZ +j*tidy + tidx];
        sh_b[tidy][tidx] = b[k*numZ*numZ +j*numZ + tidx];
        sh_c[tidy][tidx] = c[k*numZ*numZ +j*numZ + tidx];
        sh_r[tidy][tidx] = r[k*numZ*numZ +j*numZ + tidx];
        sh_uu[tidy][tidx] = uu[k*numZ*numZ +j*numZ + tidx];
        sh_u[tidy][tidx] = u[k* numX *numY +j*numZ + tidx]; // u and result are different!!
        
        __syncthreads();

        sh_u[tidy][0] = sh_r[tidy][0];
        sh_uu[tidy][0] = sh_b[tidy][0];
        __syncthreads();

        for(int i= 1; i< T; i++){
            // beta  = a[ZZ(k,j,i)] / uu[ZZ(k,j,i-1)];
            beta = a[ZZ(k,j,i)]  / sh_uu[tidy][i-1];
            __syncthreads();

            // uu[ZZ(k,j,i)] = b[ZZ(k,j,i)] - beta*c[ZZ(k,j,i-1)];
            sh_uu[tidy][i] = sh_b[tidy][i] - beta*sh_c[tidy][i-1];

            // u[UI(k,j,i)]  = r[ZZ(k,j,i)] - beta*u[UI(k,j,i-1)];
            sh_u[tidy][i] = sh_r[tidy][i] - beta*sh_u[tidy][i-1];
            __syncthreads();

        }

        // u[UI(k,j,n-1)] = u[UI(k,j,n-1)] / uu[ZZ(k,j,n-1)];
        sh_u[tidy][T-1] = sh_u[tidy][T-1]/ sh_uu[tidy][T-1];
        __syncthreads();

        // read c uu, write u
        for(int i=T-2; i>=0; i--) {  
            // u[UI(k,j,i)]  = (u[UI(k,j,i)]  - c[ZZ(k,j,i)]*u[UI(k,j,i+1)] ) / uu[ZZ(k,j,i)];
            sh_u[tidy][i] = (sh_u[tidy][i] - sh_c[tidy][i]*sh_u[tidy][i+1])/ sh_uu[tidy][tidx];
        }
        __syncthreads();

        // a[k*numZ*numZ +j*tidy + tidx]=sh_a[tidy][tidx] ;
        // b[k*numZ*numZ +j*tidy + tidx]=sh_b[tidy][tidx] ;
        // c[k*numZ*numZ +j*tidy + tidx]=sh_c[tidy][tidx] ;
        u[k*numZ*numZ +j*tidy + tidx]=sh_u[tidy][tidx] ;
        // uu[k*numZ*numZ+j*tidy +tidx] =sh_uu[tidy][tidx];

    }
}
*/

__global__ void
d_tridag_implicit_x(
    REAL* a, REAL* b, REAL* c, REAL* r, int n, REAL* u, REAL* uu, // tridag 
    unsigned numX, unsigned numY, unsigned outer, unsigned numZ, unsigned middle){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x; //numY
    unsigned int k = blockDim.y*blockIdx.y + threadIdx.y; //outer

    if(k >= outer || j >= middle)
        return;
 
    REAL   beta;

    u[UI(k,j,0)]  = r[UI(k,j,0)];
    uu[ZZ(k,j,0)] = b[ZZ(k,j,0)]; //uu size?? [numZ][numZ]

    for(int i=1; i< n; i++) {
        beta  = a[ZZ(k,j,i)] / uu[ZZ(k,j,i-1)];

        uu[ZZ(k,j,i)] = b[ZZ(k,j,i)] - beta*c[ZZ(k,j,i-1)];
        u[UI(k,j,i)]  = r[UI(k,j,i)] - beta*u[UI(k,j,i-1)];
    }

    u[UI(k,j,n-1)] = u[UI(k,j,n-1)] / uu[ZZ(k,j,n-1)];
    for(int i=n-2; i>=0; i--) {
        u[UI(k,j,i)]  = (u[UI(k,j,i)]  - c[ZZ(k,j,i)]*u[UI(k,j,i+1)] ) / uu[ZZ(k,j,i)];
    }
}

__global__ void
d_implicit_y(REAL* u, REAL* v, REAL* a, REAL* b, REAL* c,  REAL* y,  
    REAL* varY, REAL* timeline, REAL* dyy, 
    unsigned int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
   
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y; //numX
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x; //numY


    if(k >= outer || j >= numY || i >= numX)
        return;

    a[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy[D4ID(j,0)]);
    b[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varY[XY(0,i,j)]*dyy[D4ID(j,1)]);
    c[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy[D4ID(j,2)]);
    y[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) * u[YX(k,j,i)] - 0.5*v[XY(k,i,j)];
}


__global__ void
d_implicit_y_trans(REAL* u_tr, REAL* v, REAL* a, REAL* b, REAL* c,  REAL* y,  
    REAL* varY, REAL* timeline, REAL* dyy_tr, 
    unsigned int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
   
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y; //numX
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x; //numY


    if(k >= outer || j >= numY || i >= numX)
        return;

    a[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(0,j)]);
    b[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(1,j)]);
    c[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(2,j)]);
    y[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) * u_tr[XY(k,i,j)] - 0.5*v[XY(k,i,j)];
}



__global__ void
sh_implicit_y(REAL* u_tr, REAL* v, REAL* a, REAL* b, REAL* c,  REAL* y,  
    REAL* varY, REAL* timeline, REAL* dyy_tr, 
    int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
   
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y; //numX
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x; //numY

    if(k >= outer || j >= numY || i >= numX)
        return;

    __shared__ REAL 
        sh_varY[T][T+1],  
        // sh_dyy[T][T+1],
        // sh_u[T][T+1],    
        // sh_v[T][T+1],
        sh_a[T][T+1], 
        sh_b[T][T+1], 
        sh_c[T][T+1], 
        sh_y[T][T+1]; //

    int tidy = threadIdx.y;
    int tidx = threadIdx.x;


    // copy data from global memory to shared memory
    // sh_u[tidy][tidx] = u_tr[XY(k,i,j)];
    sh_a[tidy][tidx] = a[ZZ(k,i,j)];
    sh_b[tidy][tidx] = b[ZZ(k,i,j)];
    sh_c[tidy][tidx] = c[ZZ(k,i,j)];
    // sh_v[tidy][tidx] = v[XY(k,i,j)] ;
    sh_varY[tidy][tidx] = varY[i*numY +j];
    sh_y[tidy][tidx] = y[ZZ(k,i,j)];
    // sh_dyy[tidy][tidx] = (i<4) ? dyy_tr[Y4(i,j)] : 0.0; // need transpose

    __syncthreads();

    // a[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(0,j)]);
    // b[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(1,j)]);
    // c[ZZ(k,i,j)] =       - 0.5*(0.5*varY[XY(0,i,j)]*dyy_tr[Y4(2,j)]);
    // y[ZZ(k,i,j)] = ( 1.0/(timeline[g+1]-timeline[g])) * u[YX(k,j,i)] - 0.5*v[XY(k,i,j)];
    sh_a[tidy][tidx] = - 0.5*(0.5* sh_varY[tidy][tidx] * dyy_tr[Y4(0,j)]);
    sh_b[tidy][tidx] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*sh_varY[tidy][tidx]*dyy_tr[Y4(1,j)]);
    sh_c[tidy][tidx]  =       - 0.5*(0.5*sh_varY[tidy][tidx]*dyy_tr[Y4(2,j)]);
    sh_y[tidy][tidx] = ( 1.0/(timeline[g+1]-timeline[g])) * u_tr[XY(k,i,j)] - 0.5*v[XY(k,i,j)];

    a[ZZ(k,i,j)] = sh_a[tidy][tidx];
    a[ZZ(k,i,j)] = sh_b[tidy][tidx];
    a[ZZ(k,i,j)] = sh_c[tidy][tidx];
    y[ZZ(k,i,j)] = sh_y[tidy][tidx];
}




__global__ void sgmMatTranspose( REAL* A, REAL* trA, int rowsA, int colsA ){
    __shared__ REAL tile[T][T+1];
 
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
  
    unsigned int k = blockDim.z * blockIdx.z + threadIdx.z; //Outer
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y; //numX
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x; //numY
    
    A += k*rowsA*colsA; 
    trA += k*rowsA*colsA;
    
    if( j < colsA && i < rowsA )
        tile[tidy][tidx] = A[i* colsA + j];
    __syncthreads();
    
    i=blockIdx.y*blockDim.y+tidx; 
    j=blockIdx.x*blockDim.x+tidy;
    
    if( j < colsA && i < rowsA )
        trA[j*rowsA+i] = tile[tidx][tidy];
        // trA[XY(k,j,i)] = tile[tidx][tidy];
}



// 2D matrix transpose
__global__ void matTranspose2D(REAL* A, REAL* trA, int rowsA, int colsA){
    __shared__ REAL tile[T][T+1];

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int j = blockIdx.x*T + tidx;
    int i = blockIdx.y*T + tidy;
    
    if( j < colsA && i < rowsA )
        tile[tidy][tidx] = A[i*colsA+j];
    __syncthreads();

    i = blockIdx.y*T + threadIdx.x;
    j = blockIdx.x*T + threadIdx.y;
    
    if( j < colsA && i < rowsA )
        trA[j*rowsA+i] = tile[tidx][tidy];
}



//{{{ wrapper 
void initGrid_GPU(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                    const unsigned numX, const unsigned numY, const unsigned numT, 
                    REAL* d_myX, REAL* d_myY, REAL* d_myTimeline, unsigned myXindex, 
                    unsigned myYindex) {

    const unsigned int BLOCK_SIZE = 256;
    unsigned int NUM_BLOCKS = ceil(numT / (float)BLOCK_SIZE);

    d_initTimeline<<<NUM_BLOCKS,BLOCK_SIZE>>>(d_myTimeline, numT, t);

    NUM_BLOCKS = ceil(numX / (float)BLOCK_SIZE);
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    d_initNUM<<<NUM_BLOCKS,BLOCK_SIZE>>>(d_myX, numX, dx, myXindex, s0);

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    NUM_BLOCKS = ceil(numY / (float)BLOCK_SIZE);
    d_initNUM<<<NUM_BLOCKS,BLOCK_SIZE>>>(d_myY, numY, dy, myYindex, logAlpha);
}

void initOperator_GPU(REAL* d_x, unsigned int x_size, REAL* d_dxx){
    const unsigned int BLOCK_SIZE = 256;
    unsigned int NUM_BLOCKS = ceil(x_size / (float)BLOCK_SIZE);

    d_initOperator<<<NUM_BLOCKS,BLOCK_SIZE>>>(d_x, x_size, d_dxx);
}



void   run_OrigCPU(  
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t, 
                const REAL&           alpha, 
                const REAL&           nu, 
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {


//// ---------- GPU version -------------------- ////
    REAL *h_result; // final result

    // GPU variables
    REAL *d_x, *d_y, *d_timeline, *d_dxx, *d_dyy;
    REAL *d_result, *d_varX, *d_varY;
    REAL *d_a, *d_b, *d_c, *d_yy, *d_yyy, *d_u, *d_v;

    // myXindex myYindex are scalars
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;
    unsigned myYindex = static_cast<unsigned>(numY/2.0);

    unsigned numZ = max(numX,numY);

    int memsize_X = numX * sizeof(REAL);
    int memsize_Y = numY * sizeof(REAL);
    int memsize_T = numT * sizeof(REAL);
    int memsize_XY = numX * numY * sizeof(REAL);
    int memsize_OXY = outer * numX * numY * sizeof (REAL);
    int memsize_OZZ = outer * numZ * numZ * sizeof (REAL);

    // CPU variables
    h_result    = (REAL*) malloc (memsize_OXY);
   
    // GPU variables
    cudaMalloc((void**)&d_result, memsize_OXY);  //[outer][numX][numY]
    cudaMalloc((void**)&d_varX, memsize_XY); //[numX][numY]
    cudaMalloc((void**)&d_varY, memsize_XY); //[numX][numY]
    cudaMalloc((void**)&d_x, memsize_X); //[numX]
    cudaMalloc((void**)&d_y, memsize_Y); //[numY]
    cudaMalloc((void**)&d_timeline, memsize_T); //[numT]
    cudaMalloc((void**)&d_dxx, 4 * memsize_X); //[numX][4]
    cudaMalloc((void**)&d_dyy, 4 * memsize_Y); //[numY][4]

    //a b c yy yyy: [outer][numZ][numZ]
    cudaMalloc((void**)&d_a , memsize_OZZ);
    cudaMalloc((void**)&d_b , memsize_OZZ);
    cudaMalloc((void**)&d_c , memsize_OZZ);
    cudaMalloc((void**)&d_yy , memsize_OZZ); //y in seq code
    cudaMalloc((void**)&d_yyy, memsize_OZZ); //yy in seq code
    cudaMalloc((void**)&d_u , memsize_OXY); //d_u : [outer][numY][numX]
    cudaMalloc((void**)&d_v , memsize_OXY); //d_v : [outer][numX][numY]

// for transpose 
    REAL * d_u_tr;
    REAL * d_dyy_tr;
    cudaMalloc((void**)&d_u_tr , memsize_OXY); //d_u : [outer][numY][numX]
    cudaMalloc((void**)&d_dyy_tr, memsize_Y *4);


//GPU init 
    initGrid_GPU(s0, alpha, nu,t, numX,numY, numT, d_x, d_y, d_timeline, myXindex, myYindex);
    initOperator_GPU( d_x, numX, d_dxx);
    initOperator_GPU( d_y, numY, d_dyy);


 // GPU setPayoff
    dim3 block_3D(32, 32, 1);
    dim3 grid_3D_OXY(ceil(numY/32.0), ceil(numX/32.0), ceil(outer/1.0));
    d_setPayoff<<<grid_3D_OXY, block_3D>>>(d_result, d_x, numY, numX, outer);
   
    dim3 block_2D(T,T);
    dim3 grid_2D_OX(ceil(numX/T), ceil((float)outer/T));
    dim3 grid_2D_OY(ceil(numY/T), ceil((float)outer/T));
    dim3 grid_2D_YX(ceil( numX / T ), ceil( numY / T ));   
    dim3 grid_3D_OYX(ceil(numX/32.0), ceil(numY/32.0),ceil(outer/1.0) );


// timeline loop
for(int g = numT-2;g>=0;--g) { // second outer loop, g


    //GPU updateParams  
    // d_updateParams<<< grid_2D_YX, block_2D >>>(d_varX, d_varY, d_x, d_y, d_timeline,g, 
    //      alpha, beta, nu, numX, numY);
    dim3 block_2D(T,T), grid_2D_XY(ceil( numY / T ),ceil( numX / T )); // sh
    d_updateParams_sh<<< grid_2D_XY, block_2D >>>(d_varX, d_varY, d_x, d_y, d_timeline,g, 
         alpha, beta, nu, numX, numY);
    
    
     // GPU rollback Part_1 
    d_explicit_xy_implicit_x<<<grid_3D_OYX, block_3D>>>(d_u,d_v,d_a,d_b,d_c,
        d_varX,d_varY,d_timeline,d_dxx,d_dyy,d_result, g, numX, numY, outer, numZ);


   // GPU rollback part-2  
    d_tridag_implicit_x <<< grid_2D_OY, block_2D >>> (d_a,d_b,d_c, d_u, numX,d_u,d_yyy,numX,numY,outer,numZ,numY);


   // GPU rollback part 3
    dim3 grid_2D_Y4(1, ceil((float)numY/T));
    matTranspose2D<<< grid_2D_Y4, block_2D >>>(d_dyy, d_dyy_tr, numY, 4);
    sgmMatTranspose <<< grid_3D_OYX, block_3D>>>( d_u, d_u_tr, numY, numX );

    d_implicit_y_trans<<< grid_3D_OXY, block_3D >>>(d_u_tr,d_v,d_a,d_b,d_c, d_yy,
        d_varY,d_timeline, d_dyy_tr, g, numX, numY, outer, numZ);


//----------/GPU rollback 4 
    // dim3 block_2D(T,T,1), grid_2D_OX(ceil(numX/T), ceil((float)outer/T), 1); // 3D kernel is also vaild
    d_tridag_implicit_y <<< grid_2D_OX, block_2D >>> (d_a,d_b,d_c,d_yy,numY,d_result,d_yyy,numX,numY,outer,numZ,numX);
    

} // Timeline loop end


    cudaMemcpy( h_result         , d_result       , memsize_OXY        , cudaMemcpyDeviceToHost);

    // read the final result
    #pragma omp parallel for default(shared) schedule(static) 
    for( unsigned  k = 0; k < outer; ++ k )  //outermost loop k
        res[k] = h_result[XY(k,myXindex,myYindex)];  //  tested OK

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dxx);cudaFree(d_dyy); cudaFree(d_timeline); 
    cudaFree(d_result); cudaFree(d_varX); cudaFree(d_varY);
    cudaFree(d_a); cudaFree(d_b);cudaFree(d_c); cudaFree(d_yy);cudaFree(d_yyy); 
    cudaFree(d_u); cudaFree(d_v);
    
    free(h_result);
 //   #endif
}

