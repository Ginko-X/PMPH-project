// kernels  全部测试正确，保留验证部分代码 比较乱

#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.h"

#include "../include/CudaUtilProj.cu.h"

//#include "ProjHost.cu"

#define EPSILON 0.001
#define T 32

//#define GPU_INIT_TEST // tested OK
//#define GPU_SETPAYOFF_TEST // tested OK
//#define GPU_UPDATE_PARAMS_TEST // tested OK
//#define GPU_ROLLBACK_PART_1_TEST // tested ok
// #define GPU_ROLLBACK_PART_2_TEST

#define YX(k,j,i) ((k)*(numY)*(numX)+(j)*(numX)+(i))  //[-][numY][numX]
#define XY(k,j,i) ((k)*(numY)*(numX)+(j)*(numY)+(i)) //[-][numX][numY]
#define ZZ(k,j,i) (k*(numZ)*(numZ)+(j)*(numZ)+(i))    //[-][numZ][numZ]
#define D4ID(j,i) ((j)*4+(i))


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


    // d_implicit_x_tridag<<< grid_2D_OY, block_2D_OY >>>
//(d_a, d_b, d_c, d_uu, numX, d_u, d_yyy, numX, numY, outer, numZ); 

__global__ void
d_implicit_x_tridag(
    REAL* a, REAL* b, REAL* c, REAL* r, int n, REAL* u, REAL* uu, // tridag 
    unsigned numX, unsigned numY, unsigned outer, unsigned numZ){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x; //numY
    unsigned int k = blockDim.y*blockIdx.y + threadIdx.y; //outer

    if(k >= outer || j >= numY)
        return;
 
    // tridagPar(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]);  
    REAL   beta;

    u[YX(k,j,0)]  = r[YX(k,j,0)];
    uu[ZZ(k,j,0)] = b[ZZ(k,j,0)]; //uu size?? [numZ][numZ]

    for(int i=1; i< n; i++) {
        beta  = a[ZZ(k,j,i)] / uu[ZZ(k,j,i-1)];

        uu[ZZ(k,j,i)] = b[ZZ(k,j,i)] - beta*c[ZZ(k,j,i-1)];
        u[YX(k,j,i)]  = r[YX(k,j,i)] - beta*u[YX(k,j,i-1)];
    }

    u[YX(k,j,n-1)] = u[YX(k,j,n-1)] / uu[ZZ(k,j,n-1)];
    for(int i=n-2; i>=0; i--) {
        u[YX(k,j,i)]  = (u[YX(k,j,i)]  - c[ZZ(k,j,i)]*u[YX(k,j,i+1)] ) / uu[ZZ(k,j,i)];
    }
}

//    d_tridag_y <<< grid_2D_OY, block_2D_OY >>> (d_a,d_b,d_c, d_r, numX,d_u,d_yyy,numX,numY,outer,numZ,numY);

#define UI(k,j,i) ((k)*(middle)*(n)+(j)*(n)+(i))  //[-][numY][numX]

__global__ void
d_tridag_y(
    REAL* a, REAL* b, REAL* c, REAL* r, int n, REAL* u, REAL* uu, // tridag 
    unsigned numX, unsigned numY, unsigned outer, unsigned numZ, unsigned middle){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x; //numY
    unsigned int k = blockDim.y*blockIdx.y + threadIdx.y; //outer

    if(k >= outer || j >= middle)
        return;
 
    // tridagPar(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]);  
    // tridagPar(a[k][i],b[k][i],c[k][i],y[k][i],numY,myResult[k][i],yy[k][i]);

    REAL   beta;

    u[UI(k,j,0)]  = r[ZZ(k,j,0)];
    uu[ZZ(k,j,0)] = b[ZZ(k,j,0)]; //uu size?? [numZ][numZ]

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


// Tested ok
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




// read a b c r, write u
inline void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
) {
    int    i; 
    // int offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}

// check a, b,c,y,yy 
bool check_OZZ(vector<vector<vector<REAL> > >& validator, REAL * arr, 
    unsigned outer, unsigned seond, unsigned inner){
 
    // bool valid = true;
    int numZ = max(seond, inner);
     for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par           
            for(unsigned j=0;j<numZ;j++) { 
                for(unsigned i=0;i<numZ;i++) { 
                    if(j < seond && i < inner)                
                        if (abs(arr[k*numZ*numZ+j*numZ+i] - validator[k][j][i]) > EPSILON ){
                            return false;
                       
                    }
                }
            }
        }
    return true;
}
    // for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
    //         for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop
    //             for(unsigned i=0;i<numX;i++) {

    //                 if (abs(h_u[YX(k,j,i)] - u[k][j][i]) > EPSILON || abs(h_v[XY(k,i,j)] - v[k][i][j]) > EPSILON ) {
    //                     valid = false;
    //                     printf("\n** [h_u] did not validate ! k %d,  j %d, i %d, :  %f != %f **\n",
    //                         k,j,i, h_u[YX(k,j,i)], u[k][j][i]);
    //                     printf("\n** [h_v] did not validate ! k %d,  j %d, i %d, :  %f != %f **\n",
    //                             k,j,i, h_v[XY(k,i,j)], v[k][i][j]);
                       
    //                 }
    //                 break;
    //         }
    //     }
    // }
    // if(!valid){
    //     printf("\n** GPU_ROLLBACK_PART_1_TEST did not validate**\n");
    // }



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

    PrivGlobs  globs(numX, numY, numT);
 
    initGrid    (s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);


    // array expansion on myResult (originally globs.myResult) from [numX][numY] to [outer][numX][numY]
    vector<vector<vector<REAL> > > myResult;
    myResult.resize(outer); 
#pragma omp parallel for default(shared) schedule(static)    
    for(int i=0; i<outer; i++) {
        myResult[i].resize(numX);
        for(int j=0; j<numX; j++){
            myResult[i][j].resize(numY);
       }
    }

    //myVarX myVarY: [numX][numY]
    vector<vector<REAL> > myVarX, myVarY;     
    myVarX.resize(numX);
    myVarY.resize(numX);
    for(int i=0; i<numX; i++){
        myVarX[i].resize(numY);
        myVarY[i].resize(numY);
    }


unsigned numZ = max(numX, numY);


// array expansion on a, b, c, y, yy, [outer][numZ][numZ]
vector<vector<vector<REAL> > > a,b,c,y,yy;
a.resize(outer);
b.resize(outer);
c.resize(outer);
y.resize(outer);
yy.resize(outer);

#pragma omp parallel for default(shared) schedule(static)    
for(int i=0; i<outer; i++) {
    a[i].resize(numZ);
    b[i].resize(numZ);
    c[i].resize(numZ);
    y[i].resize(numZ);
    yy[i].resize(numZ);

    for(int j=0; j<numZ; j++){
       a[i][j].resize(numZ);
       b[i][j].resize(numZ);
       c[i][j].resize(numZ);
       y[i][j].resize(numZ);
       yy[i][j].resize(numZ);
   }
}
 
// array expansion on u,v, u is [outer][numY][numX], v is [outer][numX][]
vector<vector<vector<REAL> > > u,v;
u.resize(outer);
v.resize(outer);

for(int k=0; k<outer; k++){
    u[k].resize(numY);
    for(int i=0; i< numY; i++)
        u[k][i].resize(numX);

    v[k].resize(numX);
    for(int i=0; i< numX; i++)
        v[k][i].resize(numY);
}




//// ---------- GPU version -------------------- ////
// globs vars for gpu
    REAL *h_result; // the final result

    // GPU variables
    REAL *d_x, *d_y, *d_timeline, *d_dxx, *d_dyy;
    REAL *d_result, *d_varX, *d_varY;
    REAL *d_a, *d_b, *d_c, *d_yy, *d_yyy, *d_u, *d_v;
    REAL *d_r;

    // myXindex myYindex are scalars
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;
    unsigned myYindex = static_cast<unsigned>(numY/2.0);


    int memsize_X = numX * sizeof(REAL);
    int memsize_Y = numY * sizeof(REAL);
    int memsize_T = numT * sizeof(REAL);
    int memsize_XY = numX * numY * sizeof(REAL);
    int memsize_OXY = outer * numX * numY * sizeof (REAL);
    int memsize_OZZ = outer * numZ * numZ * sizeof (REAL);

   
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

    cudaMalloc((void**)&d_r , memsize_OZZ); //d_u : [outer][numZ][numZ] // used in Tridag


// For test only
    REAL *h_a, *h_b, *h_c, 
    *h_yy, *h_yyy, 
    *h_u, *h_v;

 // CPU variables
    h_result    = (REAL*) malloc (memsize_OXY);
    h_a         = (REAL *) malloc (memsize_OZZ         );
    h_b         = (REAL *) malloc (memsize_OZZ          );
    h_c         = (REAL *) malloc (memsize_OZZ          );
    h_yy        = (REAL *) malloc (memsize_OZZ      );
    h_yyy       = (REAL *) malloc (memsize_OZZ       );
    h_u        = (REAL *) malloc (memsize_OXY      );
    h_v       = (REAL *) malloc (memsize_OXY       );

  

//GPU init 
    initGrid_GPU(s0, alpha, nu,t, numX,numY, numT, d_x, d_y, d_timeline, myXindex, myYindex);
    initOperator_GPU( d_x, numX, d_dxx);
    initOperator_GPU( d_y, numY, d_dyy);


 // CPU setPayOff         
    #pragma omp parallel for default(shared) schedule(static)  //Kernel-1: 3D
    for( unsigned k = 0; k < outer; ++ k ) {  // outmost loop
        for(unsigned i=0;i<globs.myX.size();++i){
            for(unsigned j=0;j<globs.myY.size();++j) 
                myResult[k][i][j] = max(globs.myX[i]-(0.001*k), (REAL)0.0); 
        }
    }
        

 // GPU setPayoff
    dim3 block_3D_888(8, 8, 8);
    dim3 grid_3D_OXY(ceil(numY/8.0), ceil(numX/8.0), ceil(outer/8.0));
    d_setPayoff<<<grid_3D_OXY, block_3D_888>>>(d_result, d_x, numY, numX, outer);
   


// timeline loop
for(int g = globs.myTimeline.size()-2;g>=0;--g) { // second outer loop, g

    //CPU updateParams(g,alpha,beta,nu,globs);
    #pragma omp parallel for default(shared) schedule(static)  // Kernel-2: 2D
    for(unsigned i=0;i<globs.myX.size();++i){
        for(unsigned j=0;j<globs.myY.size();++j) {
            myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])   
                                  + globs.myY[j]             
                                  - 0.5*nu*nu*globs.myTimeline[g] )
                            );
            myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])   
                                  + globs.myY[j]             
                                  - 0.5*nu*nu*globs.myTimeline[g] )
                           ); // nu*nu
        }
    }
        

    //GPU updateParams  
    int dimy = ceil( numY / T );
    int dimx = ceil( numX / T );
    dim3 block_2D_XY(T,T), grid_2D_XY(dimx,dimy);
    d_updateParams<<< grid_2D_XY, block_2D_XY >>>(d_varX, d_varY, d_x, d_y, d_timeline, 
        g, alpha, beta, nu, numX, numY);
    

    // rollback Part 1, write u,v, a, b, c  
    #pragma omp parallel for default(shared) schedule(static)   // Kernel-3: 3D
    for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
        for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop
            for(unsigned i=0;i<numX;i++) {

                //  explicit x 
                u[k][j][i] =  (1.0/(globs.myTimeline[g+1]-globs.myTimeline[g])) *myResult[k][i][j];

                if(i > 0) { 
                  u[k][j][i] += 0.5*( 0.5*myVarX[i][j]*globs.myDxx[i][0] ) 
                                * myResult[k][i-1][j];
                }
                u[k][j][i]  +=  0.5*( 0.5*myVarX[i][j]*globs.myDxx[i][1] )
                                * myResult[k][i][j];
                if(i < numX-1) {
                  u[k][j][i] += 0.5*( 0.5*myVarX[i][j]*globs.myDxx[i][2] )
                                * myResult[k][i+1][j];
                }

                //  explicit y ; RAW v, write u
                v[k][i][j] = 0.0;

                if(j > 0) {
                  v[k][i][j] +=  ( 0.5*myVarY[i][j]*globs.myDyy[j][0] )
                             *  myResult[k][i][j-1];
                }
                v[k][i][j]  +=   ( 0.5*myVarY[i][j]*globs.myDyy[j][1] )
                             *  myResult[k][i][j];
                if(j < numY-1) {
                  v[k][i][j] +=  ( 0.5*myVarY[i][j]*globs.myDyy[j][2] )
                             *  myResult[k][i][j+1];
                }
                u[k][j][i] += v[k][i][j]; 

                
                //  implicit x  // write a,b,c
                a[k][j][i] =       - 0.5*(0.5*myVarX[i][j]*globs.myDxx[i][0]);
                b[k][j][i] = ( 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g])) - 0.5*(0.5*myVarX[i][j]*globs.myDxx[i][1]);
                c[k][j][i] =       - 0.5*(0.5*myVarX[i][j]*globs.myDxx[i][2]);
            }
        }
    }
    
     // GPU rollback Part_1  
    const dim3 grid_3D_OYX(ceil(numX/8.0), ceil(numY/8.0),ceil(outer/8.0) );
    d_explicit_xy_implicit_x<<<grid_3D_OYX, block_3D_888>>>(d_u,d_v,d_a,d_b,d_c,
        d_varX,d_varY,d_timeline,d_dxx,d_dyy,d_result, g,
        numX, numY, outer, numZ);

    cudaMemcpy( h_a         , d_a       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_b         , d_b       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_c         , d_c       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_u         , d_u       , memsize_OXY        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_v         , d_v       , memsize_OXY        , cudaMemcpyDeviceToHost);

 

        //Part 2 : read a,b,c,u to write u
    #pragma omp parallel for default(shared) schedule(static)  //kernel-4: 2D Kernel or can be merged with the last one to make a 2D kernel
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned j=0;j<numY;j++) {  // Par
                // tridagPar(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]); 
                tridag(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]);  
 
            }
        }
    
    REAL * h_r;
    h_r         = (REAL *) malloc (memsize_OZZ          );
    for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
        for(unsigned j=0;j<numZ;j++) {  // interchanged with the inner loop
            for(unsigned i=0;i<numZ;i++) {
                if(j< numY && i< numX)
                     h_r[ZZ(k,j,i)] = h_u[YX(k,j,i)];
            }
        }
    } 
    cudaMemcpy(d_r,h_r, memsize_OZZ,cudaMemcpyHostToDevice);

  
     dim3 block_2D_OY(8,8), grid_2D_OY(ceil(numY/8.0), ceil(outer/8.0));
// //----------/GPU rollback 2  // Skip Testing this
    d_tridag_y <<< grid_2D_OY, block_2D_OY >>> (d_a,d_b,d_c, d_r, numX,d_u,d_yyy,numX,numY,outer,numZ,numY);
// d_tridag_y <<< grid_2D_OX, block_2D_OX >>> (d_a,d_b,d_c,d_yy,numY,d_result,d_yyy,numX,numY,outer,numZ,numX);

    cudaMemcpy( h_u         , d_u       , memsize_OXY        , cudaMemcpyDeviceToHost);
    // cudaMemcpy( h_yyy         , d_yyy       , memsize_OXY        , cudaMemcpyDeviceToHost);


/*  ~~~~~~ SKIP TEST PART 2~~~~  */
          //prepare h_yyy
    // for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
    //     for(unsigned j=0;j<numZ;j++) {  // interchanged with the inner loop
    //         for(unsigned i=0;i<numZ;i++) {
    //             h_yyy[ZZ(k,j,i)] = yy[k][j][i];
    //         }
    //     }
    // }
    // cudaMemcpy(d_yyy,h_yyy, memsize_OZZ,cudaMemcpyHostToDevice);

    // for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
    //     for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop
    //         for(unsigned i=0;i<numX;i++) {
    //             h_u[YX(k,j,i)] = u[k][j][i];
    //         }
    //     }
    // }
    // cudaMemcpy(d_u,h_u, memsize_OXY,cudaMemcpyHostToDevice);

// Test part2
   bool valid = true;


   for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
        for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop
            for(unsigned i=0;i<numX;i++) {

                if (abs(h_u[YX(k,j,i)] - u[k][j][i]) > EPSILON ) {
                    valid = false;
                   break;
                }
                
            }
        }
    }
    if(!valid){
        for(int i = 0; i <20; i++){
             printf("\n** %f  !=  %f ",
                         h_u[i], u[0][0][i]);
        }
        printf("\n** GPU_ROLLBACK_PART_2_ h_u did not validate**\n");
        break;
    }
    

    //Part 3, write a b c y reading from u,v    // implicit y, 
    #pragma omp parallel for default(shared) schedule(static)  // Kernel-5: 3D
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned i=0;i<numX;i++) { 
                for(unsigned j=0;j<numY;j++) {  
                    a[k][i][j] =       - 0.5*(0.5*myVarY[i][j]*globs.myDyy[j][0]);
                    b[k][i][j] = ( 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g])) - 0.5*(0.5*myVarY[i][j]*globs.myDyy[j][1]);
                    c[k][i][j] =       - 0.5*(0.5*myVarY[i][j]*globs.myDyy[j][2]);
               
                    y[k][i][j] = ( 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g])) *u[k][j][i] - 0.5*v[k][i][j];
                }
            }
        }


    // GPU rollback part 3; tested OK
    const dim3 grid_3D_OXY(ceil(numY/8.0), ceil(numX/8.0), ceil(outer/8.0) );
    d_implicit_y<<< grid_3D_OXY, block_3D_888 >>>(d_u,d_v,d_a,d_b,d_c, d_yy,
        d_varY,d_timeline, d_dyy,
        g, numX, numY, outer, numZ);

    cudaMemcpy( h_a         , d_a       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_b         , d_b       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_c         , d_c       , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_yy        , d_yy      , memsize_OZZ        , cudaMemcpyDeviceToHost);
    cudaMemcpy( h_u         , d_u       , memsize_OXY        , cudaMemcpyDeviceToHost);

    valid = true;
    // Check the result
   // for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
   //          for(unsigned i=0;i<numX;i++) { 
   //              for(unsigned j=0;j<numY;j++) {                 
   //                  if (abs(h_a[ZZ(k,i,j)] - a[k][i][j]) > EPSILON ||
   //                      abs(h_b[ZZ(k,i,j)] - b[k][i][j]) > EPSILON ||
   //                      abs(h_c[ZZ(k,i,j)] - c[k][i][j]) > EPSILON ||
   //                      abs(h_yy[ZZ(k,i,j)] - y[k][i][j]) > EPSILON ) {
                        
   //                          valid = false;
                       
   //                  }
   //                  break;
   //              }
   //          }
   //      }

    for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
            for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop
                for(unsigned i=0;i<numX;i++) {

                    if (abs(h_u[YX(k,j,i)] - u[k][j][i]) > EPSILON || 
                        abs(h_v[XY(k,i,j)] - v[k][i][j]) > EPSILON ) {
                        valid = false;
                        // printf("\n** [h_u] did not validate ! k %d,  j %d, i %d, :  %f != %f **\n",
                        //     k,j,i, h_u[YX(k,j,i)], u[k][j][i]);
                        // printf("\n** [h_v] did not validate ! k %d,  j %d, i %d, :  %f != %f **\n",
                        //         k,j,i, h_v[XY(k,i,j)], v[k][i][j]);
                       
                    }
                    break;
            }
        }
    }
   
    if(!valid){
        printf("\n** GPU_ROLLBACK_PART_3_TEST did not validate**\n");
    }



//Part 4: write myResult reading from a b c y 
    #pragma omp parallel for default(shared) schedule(static)   //kernel-6  
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned i=0;i<numX;i++) { 
                tridag(a[k][i],b[k][i],c[k][i],y[k][i],numY,myResult[k][i],yy[k][i]);
            }
        }
    


//----------/GPU rollback 4 

    dim3 block_2D_OX(8,8), grid_2D_OX(ceil(numX/8.0), ceil(outer/8.0));

    d_tridag_y <<< grid_2D_OX, block_2D_OX >>> (d_a,d_b,d_c,d_yy,numY,d_result,d_yyy,numX,numY,outer,numZ,numX);
    // d_implicit_x_tridag<<< grid_2D_OY, block_2D_OY >>>(d_a, d_b, d_c, d_uu, numX, d_u, d_yyy, numX, numY, outer, numZ); 
    cudaMemcpy( h_result         , d_result       , memsize_OXY        , cudaMemcpyDeviceToHost);

     valid = true;
     for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k, after interchanged //Par        
            for(unsigned i=0;i<numX;i++) {
                for(unsigned j=0;j<numY;j++) {  // interchanged with the inner loop

                    if (abs(h_result[XY(k,i,j)] - myResult[k][i][j]) > EPSILON ){
                        valid = false;
                    }
                    break;
            }
        }
    }
    if(!valid){
        printf("\n** GPU_ROLLBACK_PART_4_TEST did not validate**\n");
    }
} // Timeline loop end

    #pragma omp parallel for default(shared) schedule(static) 
    for( unsigned  k = 0; k < outer; ++ k )  //outermost loop k
        res[k] = h_result[XY(k,myXindex,myYindex)];  //  tested OK


    cudaFree(d_x); cudaFree(d_y); cudaFree(d_dxx);cudaFree(d_dyy); cudaFree(d_timeline); 
    cudaFree(d_result); cudaFree(d_varX); cudaFree(d_varY);
    cudaFree(d_a); cudaFree(d_b);cudaFree(d_c); cudaFree(d_yy);cudaFree(d_yyy); 
    cudaFree(d_u); cudaFree(d_v);
    
    free(h_result);
    free(h_a); free(h_b);free(h_c);
    // free(h_yy);free(h_yyy);
    free(h_u); free(h_v);

 //   #endif
}

