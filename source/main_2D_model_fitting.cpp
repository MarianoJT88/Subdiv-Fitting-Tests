// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "2D_model_fitting.h"


// ------------------------------------------------------
//					MAIN - 2D Model 
// ------------------------------------------------------

int main()
{	
	const bool solve_DT = true;
	const unsigned int refine_levels = 4;
	const unsigned int num_images = 5;
	const unsigned int downsample = 1;
	Mod2DfromRGBD mod2D(num_images, downsample);


	mod2D.im_dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/";
	mod2D.image_set = 1;
	mod2D.solve_DT = solve_DT;

	mod2D.nsamples_approx = 200; //Only used for DT
	mod2D.max_iter = 40;
	mod2D.robust_kernel = 1;
	mod2D.Kn = 0.005f;
	mod2D.adap_mult = 1.f;

	if (solve_DT)
	{
		mod2D.nsamples_approx = 200;		
		mod2D.alpha = 0.2f*mod2D.num_images/float(mod2D.nsamples_approx);
	}
	else
	{
		mod2D.tau = 3.f;
		mod2D.eps = 0.1f;
		mod2D.alpha = 0.02f*mod2D.num_images/mod2D.tau;
	}


	//Load input data
	mod2D.loadDepthFromImages();

	//Segment the "images"
	mod2D.segmentFromDepth();

	//Compute the DT function (if using DT)
	if (solve_DT)	mod2D.computeDistanceTransform();

	//Compute normals
	mod2D.computeDataNormals();

	//Create the 3D scene
	mod2D.initializeScene();

	//Find initial poses for the cameras
	mod2D.computeInitialCameraPoses();

	//Create initial mesh
	mod2D.loadInitialMesh();
	//mod2D.refineMeshOneLevel();
	//mod2D.refineMeshOneLevel();


	//					Solve
	//----------------------------------------------
	for (unsigned int k = 0; k < refine_levels; k++)
	{
		mod2D.computeInitialUDataterm();
		mod2D.computeInitialUBackground();
		//mod2D.solveViaFiniteDifferences();
		if (solve_DT)	mod2D.solveDT_GradientDescent();
		else			mod2D.solveSK_GradientDescent();
		//else			mod2D.solveSK_LM();
		mod2D.showSubSurface();

		if (k < refine_levels-1)
			mod2D.refineMeshOneLevel();

		mod2D.max_iter += 10;
		mod2D.Kn *= 0.5f;
		mod2D.adap_mult = 1.f;
		printf("\n New level \n");
	}

	//system::os::getch();
	//mod2D.refineMeshOneLevel();
	//mod2D.computeInitialCorrespondences();
	//mod2D.computeTransCoordAndResiduals();
	mod2D.showSubSurface();

	system::os::getch();
	return 0;
}

