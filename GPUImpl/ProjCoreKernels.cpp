#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

//Par
void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs)
{
//#pragma omp parallel for default(shared) schedule(static)
    for(unsigned i=0;i<globs.myX.size();++i)
        for(unsigned j=0;j<globs.myY.size();++j) {
            globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
        }
}


//Par
void setPayoff(const REAL strike, PrivGlobs& globs )
{
//#pragma omp parallel for default(shared) schedule(static)
        for(unsigned i=0;i<globs.myX.size();++i)
	{
		REAL payoff = max(globs.myX[i]-strike, (REAL)0.0);
		for(unsigned j=0;j<globs.myY.size();++j)
			globs.myResult[i][j] = payoff;
	}
}

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



// modified rollback
void
rollback( const unsigned g, PrivGlobs& globs, const unsigned k, vector<vector<vector<REAL> > >& myResult, 
    vector<vector<vector<REAL> > >& myVarX, vector<vector<vector<REAL> > >& myVarY ) {

    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector<REAL> a(numZ), b(numZ), c(numZ), y(numZ);     // [max(numX,numY)] 
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    //  explicit x
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {
            u[j][i] = dtInv*myResult[k][i][j];

            if(i > 0) { 
              u[j][i] += 0.5*( 0.5*myVarX[k][i][j]*globs.myDxx[i][0] ) 
                            * myResult[k][i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*myVarX[k][i][j]*globs.myDxx[i][1] )
                            * myResult[k][i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*myVarX[k][i][j]*globs.myDxx[i][2] )
                            * myResult[k][i+1][j];
            }
        }
    }

    //  explicit y
    for(j=0;j<numY;j++)
    {
        for(i=0;i<numX;i++) {
            v[i][j] = 0.0;

            if(j > 0) {
              v[i][j] +=  ( 0.5*myVarY[k][i][j]*globs.myDyy[j][0] )
                         *  myResult[k][i][j-1];
            }
            v[i][j]  +=   ( 0.5*myVarY[k][i][j]*globs.myDyy[j][1] )
                         *  myResult[k][i][j];
            if(j < numY-1) {
              v[i][j] +=  ( 0.5*myVarY[k][i][j]*globs.myDyy[j][2] )
                         *  myResult[k][i][j+1];
            }
            u[j][i] += v[i][j]; 
        }
    }

    //  implicit x
    for(j=0;j<numY;j++) {
        for(i=0;i<numX;i++) {  // here a, b,c should have size [numX]
            a[i] =       - 0.5*(0.5*myVarX[k][i][j]*globs.myDxx[i][0]);
            b[i] = dtInv - 0.5*(0.5*myVarX[k][i][j]*globs.myDxx[i][1]);
            c[i] =       - 0.5*(0.5*myVarX[k][i][j]*globs.myDxx[i][2]);
        }
        // here yy should have size [numX]
        tridagPar(a,b,c,u[j],numX,u[j],yy);
    }

    //  implicit y
    for(i=0;i<numX;i++) { 
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
            a[j] =       - 0.5*(0.5*myVarY[k][i][j]*globs.myDyy[j][0]);
            b[j] = dtInv - 0.5*(0.5*myVarY[k][i][j]*globs.myDyy[j][1]);
            c[j] =       - 0.5*(0.5*myVarY[k][i][j]*globs.myDyy[j][2]);
        }

        for(j=0;j<numY;j++)
            y[j] = dtInv*u[j][i] - 0.5*v[i][j];

        // here yy should have size [numY]
        tridagPar(a,b,c,y,numY,myResult[k][i],yy);
    }
}


/*
REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {	
    
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);

    setPayoff(strike, globs);


    for(int i = globs.myTimeline.size()-2;i>=0;--i)
    {
       updateParams(i,alpha,beta,nu,globs);
       
       rollback(i, globs);              
    }
    
    return globs.myResult[globs.myXindex][globs.myYindex];
}

*/

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

   PrivGlobs    globs(numX, numY, numT);
 
    initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
    initOperator(globs.myX,globs.myDxx);
    initOperator(globs.myY,globs.myDyy);


    // array expansion on myResult, myVar, myVarY
    // they are originally [numX][numY], make them [outer][numX][numY]
    vector<vector<vector<REAL> > > myResult, myVarX, myVarY; 
    myResult.resize(outer); myVarX.resize(outer); myVarY.resize(outer);
#pragma omp parallel for default(shared) schedule(static)    
    for(int i=0; i<outer; i++) {
        myResult[i].resize(numX);
        myVarX[i].resize(numX);
        myVarY[i].resize(numX);
        for(int j=0; j<numX; j++){
            myResult[i][j].resize(numY);
            myVarX[i][j].resize(numY);
            myVarY[i][j].resize(numY);
	}
    }

      
  // setPayoff(strike, globs);  it's parallel so can be loop-distributed on the outmost loop
  // also need to do array expansion on globs.myResult, i.e.  myResult
#pragma omp parallel for default(shared) schedule(static)
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
#pragma omp parallel for default(shared) schedule(static)
    for( unsigned k = 0; k < outer; ++ k ) {  //outermost loop k
       
        for(int g = globs.myTimeline.size()-2;g>=0;--g) { // second outer loop, g

           //modified updateParams(g,alpha,beta,nu,globs);
            for(unsigned i=0;i<globs.myX.size();++i)
                for(unsigned j=0;j<globs.myY.size();++j) {
                    myVarX[k][i][j] = exp(2.0*(  beta*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
                    myVarY[k][i][j] = exp(2.0*(  alpha*log(globs.myX[i])   
                                          + globs.myY[j]             
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
                }

           // modified rollback                
	       rollback(g, globs, k, myResult, myVarX, myVarY);   // rollback(i, globs);           
    	   res[k] =  myResult[k][globs.myXindex][globs.myYindex];
        }

    }
 } 

//#endif // PROJ_CORE_ORIG
