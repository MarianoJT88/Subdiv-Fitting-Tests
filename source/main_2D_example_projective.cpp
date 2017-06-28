// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "2D_example_projective.h"


// ------------------------------------------------------
//					MAIN - 2D Model 
// ------------------------------------------------------

int main()
{	
	const bool solve_DT = false;
	const unsigned int refine_levels = 5;
	Mod2DfromRGBD mod2D;
	mod2D.solve_DT = solve_DT;
	mod2D.max_iter = 150;
	//mod2D.createDepthScan();
	mod2D.loadDepthFromImages();

	//Segment the objects
	mod2D.segmentFromDepth();
	if (solve_DT)	mod2D.computeDistanceTransform();

	//Create the 3D scene
	mod2D.initializeScene();

	//Find initial poses for the cameras
	mod2D.computeInitialCameraPoses();

	//Create initial mesh
	mod2D.loadInitialMesh();
	//mod2D.refineMeshOneLevel();


	//					Solve
	//----------------------------------------------
	for (unsigned int k = 0; k < refine_levels; k++)
	{
		mod2D.computeInitialCorrespondences();
		//mod2D.solveViaFiniteDifferences();
		if (solve_DT)	mod2D.solveWithDT();
		else			mod2D.solveGradientDescent();

		if (k < refine_levels-1)
			mod2D.refineMeshOneLevel();
	}

	//mod2D.computeTransCoordAndResiduals();
	//mod2D.showSubSurface();

	system::os::getch();
	return 0;
}

