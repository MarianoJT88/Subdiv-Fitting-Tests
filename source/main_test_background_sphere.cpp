// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "test_background_sphere.h"


// ------------------------------------------------------
//				MAIN - 3D Model (1 banana)
// ------------------------------------------------------

int main()
{
	//Prepared for 3 zones: objects, background and invalid measurements
	
	const bool solve_DT = false;
	const unsigned int refine_levels = 1;
	Mod3DfromRGBD mod3D;
	mod3D.solve_DT = solve_DT;
	mod3D.max_iter = 100; 
	mod3D.Kn = 0.f;
	mod3D.adap_mult = 0.001f;
	mod3D.robust_kernel = 1;

	if (solve_DT)
	{
		mod3D.nsamples_approx = 100;
		//mod3D.alpha = 0.4f/float(mod3D.nsamples_approx);	//For DT
		mod3D.alpha = 0.05f/float(mod3D.nsamples_approx);	//For DT^2
	}
	else
	{
		mod3D.tau = 2.5f;
		mod3D.eps = 0.1f;
		mod3D.alpha = 5e-4f*sqrtf(mod3D.tau); 
	}
	
	//Load images
	mod3D.createImageFromSphere();

	//Compute DT (if using DT)
	if (solve_DT)	mod3D.computeDistanceTransform();

	//Compute the normals
	mod3D.computeDataNormals();

	//Create the 3D scene
	mod3D.initializeScene();

	//Create initial mesh
	mod3D.loadInitialMesh();


	//					Solve
	//----------------------------------------------
	for (unsigned int k = 0; k < refine_levels; k++)
	{
		mod3D.createTopologyRefiner();
		mod3D.computeInitialCorrespondences();
		//if (solve_DT)	mod3D.solveWithDT();
		if (solve_DT)	mod3D.solveWithDT2();
		else			mod3D.solveGradientDescent();
	}


	//mod3D.saveResults();

	for (unsigned int r=0; r<7-refine_levels; r++)
		mod3D.refineMeshToShow();
	mod3D.showRenderedModel();

	system::os::getch();
	return 0;
}

