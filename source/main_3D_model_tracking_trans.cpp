// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "3D_model_fitting.h"

// ------------------------------------------------------
//					MAIN - 3D Tracking
// ------------------------------------------------------

int main()
{
	const bool solve_DT = true; //See what happens when the correspondences project out of the image domain.
	const unsigned int num_images = 1;
	const unsigned int downsample = 8; //4 - It even works with 16
	const unsigned int im_set = 4; //Not used now
	Mod3DfromRGBD mod3D(num_images, downsample, im_set);

	//mod3D.im_dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting-Private/data/";
	//mod3D.im_dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting-Private/data/";

	mod3D.im_set_dir = "C:/Users/Mariano/Dropbox/My RGBD sequences/Me tracking 1/";

	//mod3D.f_folder = "C:/Users/jaimez/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/cube/";
					//"C:/Users/Mariano/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/For the paper/video/Exp3/DT1/";
	
	mod3D.paper_visualization = false;
	mod3D.solve_DT = solve_DT;
	mod3D.vis_errors = true;

	mod3D.max_iter = 15;
	mod3D.adap_mult = 1.f;	//Check the lines that connect to the same correspondence, that should not happen	

	//Aux to test the cube
	mod3D.seg_from_background = false;
	mod3D.small_initialization = false;

	//Reg parameters
	mod3D.with_reg_trans = false;			mod3D.K_trans_total = 1.f; //10.f; //Temporary
	mod3D.with_reg_edges_tracking = false;	mod3D.K_ini_total = 500.f; //This term is terrible
	mod3D.with_reg_consRot = false;			mod3D.K_rot = 0.095f; 	//mod3D.s_reg = 1;	
	mod3D.with_rot_penalization = false;	mod3D.K_pen_rot = 15.f; 
	mod3D.with_reg_normals_tracking = true;	mod3D.Kr_total = 0.005f; mod3D.s_reg = 3;


	//Dataterm parameters
	mod3D.Kp = 0.5f*float(square(downsample))/float(num_images);
	mod3D.Kn = 0.00025f*float(square(downsample))/float(num_images);
	mod3D.truncated_res = 0.3f; //0.1f
	mod3D.truncated_resn = 1.f; 
	mod3D.K_cam_prior = 0.f;	//Don't use it!

	//Background term parameters
	if (solve_DT)
	{
		mod3D.nsamples_approx = 5000;		
		mod3D.alpha = 1.f/float(mod3D.nsamples_approx);
	}
	else
	{
		mod3D.adaptive_tau = true;
		mod3D.tau_max = 30.f/float(downsample); //3.5f
		mod3D.eps_rel = 0.1; //0.1f
		mod3D.alpha = 0.005f*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images);  //0.0005f
	}

	//Load first image of the sequence
	mod3D.loadImageFromSequence(1, true);
	//mod3D.loadInputs();

	//Create initial mesh
	mod3D.loadMeshFromFile();
	
	//Find initial poses for the cameras
	//mod3D.computeInitialCameraPoses();

	//Compute the distance transform (necessary in any case now)
	mod3D.computeDistanceTransformOpenCV();

	//Compute the normals 
	mod3D.computeDataNormals();


	//Create the 3D scene and show initial status
	mod3D.initializeScene();
	mod3D.showCamPoses();
	mod3D.showMesh();


	//Fit
	mod3D.createTopologyRefiner();
	mod3D.computeInitialUDataterm();
	mod3D.computeInitialUBackground();
	if (mod3D.with_reg_consRot) mod3D.computeIniNormalsAtVerts();


	if (solve_DT)	mod3D.solveDT2_Tracking();
	else			mod3D.solveSK_Tracking();
	//else			mod3D.solveNB_Tracking();




	//Tracking
	//--------------------------------------------
	for (unsigned int k=3; k<30; k+=2)
	{	
		//Load new image of the sequence
		mod3D.loadImageFromSequence(k, false);

		//Compute the distance transform (necessary in any case now)
		mod3D.computeDistanceTransformOpenCV();

		//Compute the normals 
		mod3D.computeDataNormals();

		//Create the 3D scene and show initial status
		mod3D.showNewData();
		mod3D.showDTAndDepth();
		mod3D.showMesh();

		//Fit
		mod3D.computeInitialUDataterm();
		mod3D.computeInitialUBackground();

		mod3D.adap_mult = 1.f;
		if (solve_DT)	mod3D.solveDT2_Tracking();
		else			mod3D.solveSK_Tracking();
		//else			mod3D.solveNB_Tracking();
	}

	//Show final model
	for (unsigned int r=0; r<3; r++) //4
		mod3D.refineMeshToShow();
	mod3D.showRenderedModel();

	////Save silhouette
	//mod3D.saveSilhouetteToFile();


	system::os::getch();
	return 0;
}


