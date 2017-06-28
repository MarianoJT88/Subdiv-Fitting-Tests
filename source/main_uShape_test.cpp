// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "uShape_test.h"


// ------------------------------------------------------
//				MAIN - 3D Model
// ------------------------------------------------------

int main()
{
	//Prepared for 3 zones: object, background and invalid measurements
	const bool solve_DT = true;
	const unsigned int refine_levels = 1; //3
	Mod3DfromRGBD mod3D;

	//Reg parameters
	mod3D.with_reg_normals = true;				mod3D.Kr_total = 0.5f/float(mod3D.downsample); //0.5  Must be a function of the resolution and the number of images!
	mod3D.regularize_unitary_normals = true;
	mod3D.with_reg_edges = true;				mod3D.Ke_total = 0.01f/float(mod3D.downsample); //0.01
	mod3D.with_reg_membrane = false;			mod3D.K_m = 0.05f/float(mod3D.downsample);
	mod3D.with_reg_thin_plate = false;			mod3D.K_tp = 0.1f/float(mod3D.downsample);

	//Dataterm parameters
	mod3D.Kn = 0.001f;	//0.005f
	mod3D.truncated_res = 1.1f; //0.1f 
	mod3D.truncated_resn = 2.f; //1.f

	mod3D.solve_DT = solve_DT;
	mod3D.max_iter = 100; //2
	mod3D.adap_mult = 1.f;
	mod3D.f_folder = "C:/Users/jaimez/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/For the paper/video/Exp1/";

	//DT
	mod3D.nsamples_approx = 5000;
	mod3D.alpha_DT = 0.f; //37.f/float(mod3D.nsamples_approx*mod3D.downsample);

	//Raycast
	mod3D.tau = 4.f;
	mod3D.peak_margin = 0.1f;
	mod3D.alpha_raycast = 0.5f/(sqrtf(mod3D.tau)*float(mod3D.downsample)); 

	
	//Load images
	mod3D.loadUShape();

	//Segment the objects
	mod3D.computeDepthSegmentationUShape();
	mod3D.computeDistanceTransform();

	//Compute the normals
	mod3D.computeDataNormals();

	//Create the 3D scene
	mod3D.initializeScene();

	//Find initial poses for the cameras
	mod3D.computeInitialCameraPoses();

	//Create initial mesh
	//mod3D.loadInitialMesh();
	mod3D.loadInitialMeshUShape();
	mod3D.refineMeshOneLevel();
	mod3D.refineMeshOneLevel();



	//					Solve
	//----------------------------------------------
	for (unsigned int k = 0; k < refine_levels; k++)
	{
		mod3D.createTopologyRefiner();
		mod3D.computeInitialCorrespondences();
		if (solve_DT)	mod3D.solveDTwithLMNoCameraOptimization();
		//else			mod3D.solveGradientDescent();
		//else			mod3D.solveLMSparseJ();
		else			mod3D.solveLMWithoutCameraOptimization();

		if (k < refine_levels - 1)
			mod3D.refineMeshOneLevel();

		//mod3D.max_iter += 15;				if (mod3D.num_verts > 500)		mod3D.max_iter = 6;
		//mod3D.alpha *= 1.5f;
		//mod3D.Kr_total *= 0.8f;
		mod3D.adap_mult = 1.f;
		mod3D.tau *= 0.5f;

		printf("\n New level");
	}

	//Render the final solution
	//for (unsigned int r=0; r<7-refine_levels; r++)
	//	mod3D.refineMeshToShow();
	//mod3D.showRenderedModel();

	//mod3D.saveResults();

	mod3D.takePictureLimitSurface(true);
	mrpt::system::sleep(1000);
	mod3D.takePictureLimitSurface(true);
	mrpt::system::sleep(1000);

	const float cam_azi = mod3D.window.getCameraAzimuthDeg();
	const float cam_zoom = mod3D.window.getCameraZoom();
	const float cam_elev = mod3D.window.getCameraElevationDeg();
	float camx, camy, camz; mod3D.window.getCameraPointingToPoint(camx, camy, camz);
	printf("\n Cam azimuth = %f, cam zoom = %f, cam elev = %f, camx = %f, camy = %f, camz = %f", cam_azi, cam_zoom, cam_elev, camx, camy, camz);


	system::os::getch();
	return 0;
}

