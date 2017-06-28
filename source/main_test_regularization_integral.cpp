// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "test_regularization_integral.h"


// ------------------------------------------------------
//				MAIN - 3D Model
// ------------------------------------------------------

int main()
{
	//Prepared for 3 zones: objects, background and invalid measurements
	const unsigned int refine_levels = 1;
	Mod3DfromRGBD mod3D;

	mod3D.with_reg = true;
	mod3D.max_iter = 20; 
	mod3D.Kn = 0.f;
	mod3D.K_m = 1.f;
	mod3D.K_tp = 5.f;
	mod3D.adap_mult = 1.f;	//0.001 for gradDecent
	mod3D.robust_kernel = 1;

	//Background
	mod3D.tau = 5.f;
	mod3D.peak_margin = 0.1f;
	mod3D.alpha = 5e-4f*sqrtf(mod3D.tau); 

	
	//Load images
	mod3D.createImageFromSphere();

	//Compute the normals
	mod3D.computeDataNormals();

	//Create the 3D scene
	mod3D.initializeScene();

	//Create initial mesh
	mod3D.loadInitialMesh();
	mod3D.refineMeshOneLevel();
	mod3D.refineMeshOneLevel();

	//mod3D.createTopologyRefiner();
	//mod3D.understandLimitSurfaceEvaluation();
	//mod3D.showLocalPoints();


	//					Solve
	//----------------------------------------------
	for (unsigned int k = 0; k < refine_levels; k++)
	{
		mod3D.createTopologyRefiner();
		mod3D.computeInitialCorrespondences();
		//mod3D.solveGradientDescent();
		mod3D.solveLMSparseJ();
	}

	//Render the final solution
	for (unsigned int r=0; r<7-refine_levels; r++)
		mod3D.refineMeshToShow();
	mod3D.showRenderedModel();

	system::os::getch();
	return 0;
}

