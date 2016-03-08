#include "PlaneStressTri6.h"

PlaneStressTri6::PlaneStressTri6( PlaneStressStiffness * _stiff, 
				  PlaneStressElementType type,
				  int _componentNum ){

}

PlaneStressTri6::~PlaneStressTri6(){

}

/*
  Evaluates the shape function at pt = [xi, eta], and its derivatives
*/
PlaneStressTri6::getShapeFunctions( const double pt[], double N[], 
				    double Na[], double Nb[]){
  
  // shape function values
  N[0] = 2.0(1.0-(pt[0]+pt[1]))*(0.5-(pt[0]+pt[1]));
  N[1] = 2.0*pt[0]*(pt[0]-0.5);
  N[2] = 2.0*pt[1]*(pt[1]-0.5);
  N[3] = 2.0*pt[0]*(1.0-(pt[0]+pt[1]));
  N[4] = 4.0*pt[0]*pt[1];
  N[5] = 2.0*pt[1]*(1.0-(pt[0]+pt[1]));

  // derivative of shape function with respect to xi
  Na[0] = 4.0*(pt[0]+pt[1]-0.75);
  Na[1] = 4.0*pt[1]-1.0;
  Na[2] = 0.0;
  Na[3] = 2.0-4.0*pt[0]-2.0*pt[1];
  Na[4] = 4.0*pt[1];
  Na[5] = -2.0*pt[1];

  // derivative of shape function with respect to eta
  Nb[0] = 4.0*(pt[0]+pt[1]-0.75); 
  Nb[1] = 0.0;
  Nb[2] = 4.0*pt[1]-1.0;
  Nb[3] = -2.0*pt[0];
  Nb[4] = 4.0*pt[0];
  Nb[5] = 2.0*(1.0-pt[0]-2.0*pt[1]);
  
}

/*
  Evaluates the shape function at pt = [xi, eta]
*/
PlaneStressTri6::getShapeFunctions( const double pt[], double N[]){
  
  // shape function values
  N[0] = 2.0(1.0-(pt[0]+pt[1]))*(0.5-(pt[0]+pt[1]));
  N[1] = 2.0*pt[0]*(pt[0]-0.5);
  N[2] = 2.0*pt[1]*(pt[1]-0.5);
  N[3] = 2.0*pt[0]*(1.0-(pt[0]+pt[1]));
  N[4] = 4.0*pt[0]*pt[1];
  N[5] = 2.0*pt[1]*(1.0-(pt[0]+pt[1]));
  
}

PlaneStressTri6::getNumGassPts(){
    return 4;
}

PlaneStressTri6::getGaussWtsPts( const int num, double pt[] ){
    
}