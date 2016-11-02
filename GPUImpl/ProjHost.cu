#include "ProjKernels.cu.h"


/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
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


