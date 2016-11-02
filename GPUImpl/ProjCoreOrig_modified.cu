#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

#include "../include/CudaUtilProj.cu.h"

#define EPSILON 0.01
#define VALIDATION
#define T 32

//{{{KERNELS
__global__ void
d_initTimeline( REAL* d_timeline, const unsigned int numT, const REAL t){
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
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
            //	lower boundary
            dxl		 =  0.0;
            dxu		 =  d_x[1] - d_x[0];

            d_dxx[0] =  0.0;
            d_dxx[1] =  0.0;
            d_dxx[2] =  0.0;
            d_dxx[3] =  0.0;
        }else if(gid == x_size-1){
            //	upper boundary
            dxl		   =  d_x[x_size-1] - d_x[x_size-2];
            dxu		   =  0.0;

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
d_updateParams(REAL* d_varX, REAL* d_varY, REAL* d_x, REAL* d_y, REAL* d_timeline, unsigned int g,
        REAL alpha, REAL beta, REAL nu, unsigned int numX, unsigned int numY){

    unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;

    if(i >= numX || j >= numY)
        return;


    d_varX[i*numY+j] = exp(2.0*( beta*log(d_x[i])+d_y[j] - 0.5*nu*nu*d_timeline[g]));
    d_varY[i*numY+j] = exp(2.0*( alpha*log(d_x[i]) + d_y[j] - 0.5*nu*nu*d_timeline[g]));

}


#define YX(k,j,i) (k*(numY*numX)+j*numX+i)
#define XY(k,j,i) (k*(numY*numX)+j*numY+i)
#define ZID(k,j,i) (k*(numZ*numZ)+j*numZ+i)
#define DID(j,i) (j*4+i)
__global__ void
d_explicit_xy_implicit_x(REAL* u, REAL* v, REAL* a, REAL* b, REAL* c,  REAL* varX, REAL* varY, REAL* timeline, REAL* dxx, REAL* dyy, REAL* result, unsigned int g, unsigned numX, unsigned numY, unsigned outer, unsigned numZ){
	//for(k, j, i)
   
    unsigned int k = blockDim.z * blockIdx.z + blockIdx.z; //Outer
    unsigned int j = blockDim.y * blockIdx.y + blockIdx.y; //numY
    unsigned int i = blockDim.x * blockIdx.x + blockIdx.x; //numX


    if(k >= outer || j >= numY || i >= numX)
	return;
    
    //  explicit x 
    u[YX(k,j,i)] =  (1.0/(timeline[g+1]-timeline[g])) *result[XY(k,j,i)];

    if(i > 0) {
      u[YX(k,j,i)] += 0.5*( 0.5*varX[XY(0,i,j)]*dxx[DID(i,0)] ) 
		    * result[XY(k,i-1,j)];
    }
    u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[DID(i,1)] )
		    * result[XY(k,i,j)];
    if(i < numX-1) {
      u[YX(k,j,i)]  +=  0.5*( 0.5*varX[XY(0,i,j)]*dxx[DID(i,2)] )
		    * result[XY(k,i+1,j)];
    }

    //  explicit y ; RAW v, write u
    v[XY(0,0,j)] = 0.0;

    if(j > 0) {
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[DID(j,0)] )
		 *  result[XY(k,i,j-1)];
    }
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[DID(j,1)] )
		 *  result[XY(k,i,j)];
    if(j < numY-1) {
      v[XY(k,i,j)] +=  ( 0.5*varY[XY(0,i,j)]*dyy[DID(j,2)] )
		 *  result[XY(k,i,j+1)];
    }
    u[YX(k,i,j)] += v[XY(k,i,j)];


    //  implicit x  // write a,b,c
    a[ZID(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[DID(i,0)]);
    b[ZID(k,j,i)] = ( 1.0/(timeline[g+1]-timeline[g])) - 0.5*(0.5*varX[XY(0,i,j)]*dxx[DID(i,1)]);
    c[ZID(k,j,i)] =       - 0.5*(0.5*varX[XY(0,i,j)]*dxx[DID(i,2)]);
    

}


//}}}


//{{{WRAPPERS


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



//}}}



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
    int    i, offset;
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


      
  // setPayoff(strike, globs);  it's parallel so can be loop-distributed on the outmost loop
  // also need to do array expansion on globs.myResult, i.e.  myResult
#pragma omp parallel for default(shared) schedule(static)  //Kernel-1: 3D
    for( unsigned k = 0; k < outer; ++ k ) {  // outmost loop
        
        // modified setPayoff function below
        for(unsigned i=0;i<globs.myX.size();++i)
        {
            //REAL payoff = max(globs.myX[i]-strike, (REAL)0.0); // move this inside the loop to do privatization
            for(unsigned j=0;j<globs.myY.size();++j) 
                // globs.myResult[i][j] = payoff;   // note that payoff is just a scalar variables,
                myResult[k][i][j] = max(globs.myX[i]-(0.001*k), (REAL)0.0); 
        }
    }
        
  
//--- original code: 
// for(int i = globs.myTimeline.size()-2;i>=0;--i)
//     {
//         updateParams(i,alpha,beta,nu,globs);
//         rollback(i, globs);
//     }
//--- use loop interchange and loop distribution


//modified updateParams(g,alpha,beta,nu,globs);
  // Kernel-2: 3D
    for(int g = globs.myTimeline.size()-2;g>=0;--g) { // second outer loop, g

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
        
        //Part 2 : read a,b,c,u to write u
    #pragma omp parallel for default(shared) schedule(static)  //kernel-4: 2D Kernel or can be merged with the last one to make a 2D kernel
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned j=0;j<numY;j++) {  // Par
                tridagPar(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]);  
            }
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

        //Part 4: write myResult reading from a b c y 
    #pragma omp parallel for default(shared) schedule(static)   //kernel-6  
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned i=0;i<numX;i++) { 
                tridagPar(a[k][i],b[k][i],c[k][i],y[k][i],numY,myResult[k][i],yy[k][i]);
            }
        }


    }



#pragma omp parallel for default(shared) schedule(static) 
for( unsigned  k = 0; k < outer; ++ k )  //outermost loop k
    res[k] = myResult[k][globs.myXindex][globs.myYindex]; // myRes[0][k];

//// ---------- GPU version -------------------- ////
// globs vars for gpu
    REAL *d_x, *d_y, *d_timeline, *d_dxx, *d_dyy;
    REAL *d_result;// *d_varX, *d_varY;
    // REAL *d_a, *d_b, *d_c, *d_yy, *d_yyy, *d_u, *d_v;
    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    unsigned myXindex = static_cast<unsigned>(s0/dx) % numX;
    unsigned myYindex = static_cast<unsigned>(numY/2.0);
    // printf("myXindex : %d, myYindex: %d \n", myXindex, myYindex );


    int memsize_X = numX * sizeof(REAL);
    int memsize_Y = numY * sizeof(REAL);
    int memsize_T = numT * sizeof(REAL);
    int memsize_OXY = outer * numX * numY * sizeof (REAL);

    cudaMalloc((void**)&d_result, memsize_OXY);  //[outer][numX][numY]
    // cudaMalloc((void**)&d_varX, numX*numY*sizeof(REAL)); //[numX][numY]
    // cudaMalloc((void**)&d_varY, numX*numY*sizeof(REAL)); //[numX][numY]
    cudaMalloc((void**)&d_x, memsize_X); //[numX]
    cudaMalloc((void**)&d_y, memsize_Y); //[numY]
    cudaMalloc((void**)&d_timeline, memsize_T); //[numT]
    cudaMalloc((void**)&d_dxx, 4 * memsize_X); //[numX][4]
    cudaMalloc((void**)&d_dyy, 4 * memsize_Y); //[numY][4]

    // a b c yy yyy: [outer][numZ][numZ]
    // cudaMalloc((void**)&d_a , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((void**)&d_b , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((void**)&d_c , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((void**)&d_yy , outer*numZ*numZ*sizeof(REAL)); //y in seq code
    // cudaMalloc((void**)&d_yyy, outer*numZ*numZ*sizeof(REAL)); //yy in seq code

    // cudaMalloc((void**)&d_u , outer*numY*numX*sizeof(REAL)); //d_u : [outer][numY][numX]
    // cudaMalloc((void**)&d_v , outer*numX*numY*sizeof(REAL)); //d_v : [outer][numX][numY]

//GPU init 
    // set myXindex and myYindex, both are scalars
    
    REAL * h_timeline;
    // REAL *h_x, *h_y, *h_timeline, *h_dxx, *h_dyy;
    // h_x         = (REAL *) malloc (memsize_X          );
    // h_y         = (REAL *) malloc (memsize_Y          );
    h_timeline  = (REAL *) malloc (memsize_T          );

    // for(int i = 0; i<numT; i++)
    //     h_timeline[i] = 0;
    //cudaMemcpy(d_timeline, h_timeline_in, memsize_T          , cudaMemcpyHostToDevice);

    
    // initGrid_GPU(s0, alpha, nu,t, numX,numY, numT, d_x, d_y, d_timeline, myXindex, myYindex);
    // initOperator_GPU( d_x, numX, d_dxx);
    // initOperator_GPU( d_y, numY, d_dyy);

    // unsigned int block_size = T*T;
    // unsigned int num_blocks_numT = ceil(numT / (float)block_size);
    for(unsigned i = 0; i< numT; i++)
        h_timeline[i] =  t*gid / (numT-1);
    // t*gid / (numT-1);
    // printf ("num_blocks_numT :%d   block_size: %d", num_blocks_numT, block_size);
    // d_initTimeline<<< num_blocks_numT, block_size >>>(d_timeline, numT, t);

    // unsigned int num_blocks_numX = ceil(numX / (float)block_size);
    // d_initNUM<<<num_blocks_numX,block_size>>>(d_x, numX, dx, myXindex, s0);

    // const REAL stdY = 10.0*nu*sqrt(t);
    // const REAL dy = stdY/numY;
    // const REAL logAlpha = log(alpha);
    // unsigned int num_blocks_numY = ceil(numY / (float)block_size);
    // d_initNUM<<<num_blocks_numY,block_size>>>(d_y, numY, dy, myYindex, logAlpha);


    // h_dxx       = (REAL *) malloc (numX*4*sizeof(REAL)         );
    // h_dyy       = (REAL *) malloc (numY*4*sizeof(REAL)         );

 
    // cudaMemcpy( h_x         , d_x       , numX*sizeof(REAL)           , cudaMemcpyDeviceToHost);
    // cudaMemcpy( h_y         , d_y       , numY*sizeof(REAL)           , cudaMemcpyDeviceToHost);
    // cudaMemcpy( h_timeline,  d_timeline, memsize_T          , cudaMemcpyDeviceToHost);
    // cudaMemcpy( h_dxx       , d_dxx     , numX*4*sizeof(REAL)         , cudaMemcpyDeviceToHost);
    // cudaMemcpy( h_dyy       , d_dyy     , numY*4*sizeof(REAL)         , cudaMemcpyDeviceToHost);


        bool valid = true;
        // for(int i = 0; i < numX; i++){
        //     if(abs(h_x[i]-globs.myX[i]) > EPSILON){
        //         valid = false;
        //         printf("\n** invalid h_x  %f  %f**\n",
        //                   h_x[i], globs.myX[i]);
        //         break;
        //     }
        // }

        // for(int i = 0; i < numY; i++){
        //     if(abs(h_y[i]-globs.myY[i]) > EPSILON){
        //         valid = false;
        //         printf("\n** invalid h_y **\n");
        //         break;
        //     }
        // }

        for(int i = 0; i < numT; i++){
            if(abs(h_timeline[i]-globs.myTimeline[i]) > EPSILON){
                valid = false;
                 printf("\n** invalid h_timeline  %d  %d**\n",
                          h_timeline[i], globs.myTimeline[i]);             
                break;
            }
        }
        // for(int i = 0; i < numX*4; i++){
        //     if(abs(h_dxx[i]-globs.myDxx[i/4][i%4]) > EPSILON){
        //         valid = false;
        //         printf("\n** Invalid h_dxx **\n");                
        //         break;
        //     }
        // }
        // for(int i = 0; i < numY*4; i++){
        //     if(abs(h_dyy[i]-globs.myDyy[i/4][i%4]) > EPSILON){
        //         valid = false;
        //         printf("\n**  Invalid h_dyy **\n");            
        //         break;
        //     }
        // }
        if(!valid){
            printf("\n**Initialization did not validate**\n");
            //return;
        }


    // const dim3 blockSize(8, 8, 8);
    // const dim3 gridSize(ceil(numY/8.0), ceil(numX/8.0), ceil(outer/8.0));
    // d_setPayoff<<<gridSize, blockSize>>>(d_result, d_x, numY, numX, outer);
  
    // REAL *h_result;//, *h_varX, *h_varY,
    // h_result    = (REAL*) malloc (memsize_OXY);
    // h_varX      = (REAL*) malloc (numX*numY*sizeof(REAL)      );
    // h_varY      = (REAL*) malloc (numX*numY*sizeof(REAL)      );
    // cudaMemcpy( h_result    , d_result       , numX*numY*outer*sizeof(REAL), cudaMemcpyDeviceToHost);

    //     for(int k = 0; k < outer; k++)
    //         for(int i = 0; i < globs.myX.size(); i++)
    //             for(int j = 0; j < globs.myY.size(); j++){
    //                 if(abs(h_result[k*numX*numY+i*numY+j]-myResult[k][i][j]) > EPSILON){
    //                     printf("\n**SetPayOff did not validate %f  %f**\n",
    //                             h_result[k*numX*numY+i*numY+j], myResult[k][i][j]);
    //                     break;
    //                 }
    //             }


    // cudaFree(d_timeline); cudaFree(d_result);

} 

//#endif // PROJ_CORE_ORIG


/*Generic Validation function on vectors
template<class T >
bool validate_real_arrs(REAL* arr,  T check){

}
*/
void   run_OrigCPU_(
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
        )
{
    //globs vars for gpu
    // REAL *d_x, *d_y, *d_timeline, *d_dxx, *d_dyy;
    // REAL *d_result, *d_varX, *d_varY;
    // REAL xIndex, yIndex;
    // REAL *d_a, *d_b, *d_c, *d_yy, *d_yyy, *d_u, *d_v;

//    const unsigned int result_size = numX*numY*outer*sizeof(REAL);

    // cudaMalloc((REAL**)&d_result, numX*numY*outer*sizeof(REAL));  //[outer][numX][numY]
    // cudaMalloc((REAL**)&d_varX, numX*numY*sizeof(REAL)); //[numX][numY]
    // cudaMalloc((REAL**)&d_varY, numX*numY*sizeof(REAL)); //[numX][numY]
    // cudaMalloc((REAL**)&d_x, numX*sizeof(REAL)); //[numX]
    // cudaMalloc((REAL**)&d_y, numY*sizeof(REAL)); //[numY]
    // cudaMalloc((REAL**)&d_timeline, numT*sizeof(REAL)); //[numT]
    // cudaMalloc((REAL**)&d_dxx, numX*4*sizeof(REAL)); //[numX][4]
    // cudaMalloc((REAL**)&d_dyy, numY*4*sizeof(REAL)); //[numY][4]

	//Needed in validation as well.
    unsigned numZ = max(numX, numY);

    // a b c yy yyy: [outer][numZ][numZ]
    // cudaMalloc((REAL**)&d_a , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((REAL**)&d_b , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((REAL**)&d_c , outer*numZ*numZ*sizeof(REAL));
    // cudaMalloc((REAL**)&d_yy , outer*numZ*numZ*sizeof(REAL)); //y in seq code
    // cudaMalloc((REAL**)&d_yyy, outer*numZ*numZ*sizeof(REAL)); //yy in seq code

    // cudaMalloc((REAL**)&d_u , outer*numY*numX*sizeof(REAL)); //d_u : [outer][numY][numX]
    // cudaMalloc((REAL**)&d_v , outer*numX*numY*sizeof(REAL)); //d_v : [outer][numX][numY]
   


//#ifdef VALIDATION 
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
    myVarX.resize(numX);    myVarY.resize(numX);
    for(int i=0; i<numX; i++){
        myVarX[i].resize(numY);       myVarY[i].resize(numY);
    }

    // array expansion on a, b, c, y, yy, [outer][numZ][numZ]
    vector<vector<vector<REAL> > > a,b,c,y,yy;
    a.resize(outer); b.resize(outer); c.resize(outer); y.resize(outer); yy.resize(outer);

    #pragma omp parallel for default(shared) schedule(static)    
    for(int i=0; i<outer; i++) {
        a[i].resize(numZ);     b[i].resize(numZ);   c[i].resize(numZ);  y[i].resize(numZ);   yy[i].resize(numZ);
        for(int j=0; j<numZ; j++){
           a[i][j].resize(numZ);       b[i][j].resize(numZ);       c[i][j].resize(numZ);       y[i][j].resize(numZ);       yy[i][j].resize(numZ);
       }
    }
     
    // array expansion on u,v, u is [outer][numY][numX], v is [outer][numX][]
    vector<vector<vector<REAL> > > u,v;
    u.resize(outer); v.resize(outer);

    for(int k=0; k<outer; k++){
        u[k].resize(numY);
        for(int i=0; i< numY; i++)
            u[k][i].resize(numX);

        v[k].resize(numX);
        for(int i=0; i< numX; i++)
            v[k][i].resize(numY);
    }
//#endif


//GPU init 
    // initGrid_GPU(s0, alpha, nu,t, numX,numY, numT, d_x, d_y, d_timeline, xIndex, yIndex);
    // initOperator_GPU( d_x, numX, d_dxx);
    // initOperator_GPU( d_y, numY, d_dyy);


// test initGird_GPU and initOperator_GPU
 //   #ifdef VALIDATION
        PrivGlobs  globs(numX, numY, numT);

        initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
        initOperator(globs.myX,globs.myDxx);
        initOperator(globs.myY,globs.myDyy);


        // REAL *h_x, *h_y, *h_timeline, *h_dxx, *h_dyy;
        // h_x         = (REAL*) malloc (numX*sizeof(REAL)           );
        // h_y         = (REAL*) malloc (numY*sizeof(REAL)           );
        // h_timeline  = (REAL*) malloc (numT*sizeof(REAL)           );
        // h_dxx       = (REAL*) malloc (numX*4*sizeof(REAL)         );
        // h_dyy       = (REAL*) malloc (numY*4*sizeof(REAL)         );

 
        // cudaMemcpy( h_x         , d_x       , numX*sizeof(REAL)           , cudaMemcpyDeviceToHost);
        // cudaMemcpy( h_y         , d_y       , numY*sizeof(REAL)           , cudaMemcpyDeviceToHost);
        // cudaMemcpy( h_timeline  , d_timeline, numT*sizeof(REAL)           , cudaMemcpyDeviceToHost);
        // cudaMemcpy( h_dxx       , d_dxx     , numX*4*sizeof(REAL)         , cudaMemcpyDeviceToHost);
        // cudaMemcpy( h_dyy       , d_dyy     , numY*4*sizeof(REAL)         , cudaMemcpyDeviceToHost);


        // bool valid = true;
        // for(int i = 0; i < numX; i++){
        //     if(abs(h_x[i]-globs.myX[i]) > EPSILON){
        //         valid = false;
        //         break;
        //     }
        // }

        // for(int i = 0; i < numY; i++){
        //     if(abs(h_y[i]-globs.myY[i]) > EPSILON){
        //         valid = false;
        //         break;
        //     }
        // }

        // for(int i = 0; i < numT; i++){
        //     if(abs(h_timeline[i]-globs.myTimeline[i]) > EPSILON){
        //         valid = false;
        //         break;
        //     }
        // }
        // for(int i = 0; i < numX*4; i++){
        //     if(abs(h_dxx[i]-globs.myDxx[i/4][i%4]) > EPSILON){
        //         valid = false;
        //         break;
        //     }
        // }
        // for(int i = 0; i < numY*4; i++){
        //     if(abs(h_dyy[i]-globs.myDyy[i/4][i%4]) > EPSILON){
        //         valid = false;
        //         break;
        //     }
        // }
        // if(!valid){
        //     printf("\n**Initialization did not validate**\n");
        //     //return;
        // }

  //  #endif


  

   //  REAL *h_result, *h_varX, *h_varY,
   //  h_result    = (REAL*) malloc (numX*numY*outer*sizeof(REAL));
   //  h_varX      = (REAL*) malloc (numX*numY*sizeof(REAL)      );
   //  h_varY      = (REAL*) malloc (numX*numY*sizeof(REAL)      );


   // // Test setPayoff
  //  #ifdef VALIDATION

   //  const dim3 blockSize(8, 8, 8);
   //  const dim3 gridSize(ceil(numY/8.0), ceil(numX/8.0), ceil(outer/8.0));
   //  d_setPayoff<<<gridSize, blockSize>>>(d_result, d_x, numY, numX, outer);

    // CPU setPayoff
#pragma omp parallel for default(shared) schedule(static)  //Kernel-1: 3D
    for( unsigned k = 0; k < outer; ++ k ) {  // outmost loop
        for(unsigned i=0;i<globs.myX.size();++i){
            for(unsigned j=0;j<globs.myY.size();++j){
                myResult[k][i][j] = max(globs.myX[i]-(0.001*k), (REAL)0.0);
            }
        }
    }

 //    cudaMemcpy( h_result    , d_result       , numX*numY*outer*sizeof(REAL), cudaMemcpyDeviceToHost);
	// if(globs.myX.size() != numX && globs.myY.size())
	// 	printf("Numx not myX.size()");

 //        for(int k = 0; k < outer; k++)
 //            for(int i = 0; i < globs.myX.size(); i++)
 //                for(int j = 0; j < globs.myY.size(); j++)
 //                    if(abs(h_result[k*numX*numY+i*numY+j]-myResult[k][i][j]) > EPSILON){
 //                        printf("\n**SetPayOff did not validate %f  %f**\n",
 //                                h_result[k*numX*numY+i*numY+j], myResult[k][i][j]);
 //                        return;
 //                    }

 
    for(int g = globs.myTimeline.size()-2;g>=0;--g) { // second outer loop, g

	// { //GPU updateParams
 //        const dim3 blockSize(8, 8, 1);
 //        const dim3 gridSize(ceil(numY/8.0), ceil(numX/8.0), 1);
 //        d_updateParams<<<gridSize, blockSize>>>(d_varX, d_varY, d_x, d_y, d_timeline, g, alpha, beta, nu, numX, numY);
 //    }	

            //CPU updateParams(g,alpha,beta,nu,globs)
            #pragma omp parallel for default(shared) schedule(static)
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
            // cudaMemcpy( h_varX      , d_varX       , numX*numY*sizeof(REAL)      , cudaMemcpyDeviceToHost);
            // cudaMemcpy( h_varY      , d_varY	   , numX*numY*sizeof(REAL)      , cudaMemcpyDeviceToHost);

    //     // check the result of CPU and GPU
	   //  for(int i = 0; i < numX*numY; i++){
    // 		if(abs(h_varX[i] - myVarX[i/numY][i%numY]) > EPSILON || abs(h_varY[i] - myVarY[i/numY][i%numY]) > EPSILON){
    // 			printf("\n**Update Params did not validate %f=%f and %f=%f**\n",
    // 			       h_varX[i], myVarX[i/numY][i%numY], h_varY[i], myVarY[i/numY][i%numY]);
    // 			break;
		  // }
	   //  }

 // 	{ // GPU rollback Part_1
 //        const dim3 blockSize(8, 8, 8);
 //        const dim3 gridSize(ceil(numX/8.0), ceil(numY/8.0), ceil(outer/8.0));
 //        d_explicit_xy_implicit_x<<<gridSize, blockSize>>>(d_u,d_v,d_a,d_b,d_c,d_varX,d_varY,d_timeline,d_dxx,d_dyy,d_result, g, numX, numY, outer, numZ);
	// }
	
        // #ifdef VALIDATION
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
                    c[k][j][i] =       - 05*(0.5*myVarX[i][j]*globs.myDxx[i][2]);
			    }
		    }
		}

        //Part 2 : read a,b,c,u to write u
    #pragma omp parallel for default(shared) schedule(static)  //2D Kernel or can be merged with the last one to make a 2D kernel
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned j=0;j<numY;j++) {  // Par
                tridag(a[k][j],b[k][j],c[k][j],u[k][j],numX,u[k][j],yy[k][j]);
            }
        }

        //Part 3, write a b c y reading from u,v    // implicit y, 
    #pragma omp parallel for default(shared) schedule(static)  // Kernel-4: 3D
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

        //Part 4: write myResult reading from a b c y 
    #pragma omp parallel for default(shared) schedule(static)
        for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop distribution //Par
            for(unsigned i=0;i<numX;i++) {
                tridag(a[k][i],b[k][i],c[k][i],y[k][i],numY,myResult[k][i],yy[k][i]);
            }
        }


	}




#pragma omp parallel for default(shared) schedule(static) 
for( unsigned  k = 0; k < outer; ++ k )  //outermost loop k
    res[k] = myResult[k][globs.myXindex][globs.myYindex]; // myRes[0][k];

    //SHould perhaps be initialized on the gpu instead to save PCI bandwidth. Possibly negl
    /*
     * setPayOff: 
     * INPUT: globs.myX
     * Output: myResult
     *
     * updateParams:
     * input: globs.myTimeline, globs.myX, globs.myY, alpha, beta,
     * output: myVarX, myVarY
     *
     * rollback-1:
     * input: globs.myTimeLine, myResult, 
     * output: 
     *
     * tridagPar:
     *
     * rollback-2:
     * input:
     * output:
     * */

 //   #endif
}

