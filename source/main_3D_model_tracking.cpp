// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "3D_model_fitting.h"

// ------------------------------------------------------
//					MAIN - 3D Tracking
// ------------------------------------------------------

//Fix two erroneous results

int main()
{

	const unsigned int num_images = 1; //5
	const unsigned int downsample = 4; //4
	const unsigned int seq_ID = 1; //1 - Manu tracking 1, 2 - Me tracking 1
	Mod3DfromRGBD mod3D(num_images, downsample, 2); //Last argument must just be different from 1 and 5

	//mod3D.im_dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting-Private/data/";
	//mod3D.im_dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting-Private/data/";

	if (seq_ID == 1)		mod3D.im_set_dir = "D:/My RGBD sequences/Manu tracking 1/"; //"C:/Users/Mariano/Dropbox/My RGBD sequences/Manu tracking 1/"; 
	else if (seq_ID == 2)	mod3D.im_set_dir = "D:/My RGBD sequences/Mariano tracking 1/";
						
	mod3D.f_folder = "C:/Users/jaimez/Desktop/results_background_term/track_exp hiding arms/DTall/";
					//"C:/Users/Mariano/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/For the paper/video/Exp3/DT1/";

	//Flags
	const bool solve_SK = false;
	const bool solve_DT = true;
	const bool safe_DT = solve_DT ? false : 1; //Change it only when evaluating DT (used the safe one to compute adaptive tau)
	mod3D.solve_DT = solve_DT;
	mod3D.solve_SK = solve_SK;

	mod3D.paper_visualization = false;
	mod3D.paper_vis_no_mesh = true;
	mod3D.vis_errors = true;
	
	const bool continuous_tracking = true;
	mod3D.optimize_cameras = false;
	mod3D.save_energy = false;

	mod3D.with_color = false;
	mod3D.unk_per_vertex = mod3D.with_color ? 4 : 3;


	//Solver
	mod3D.ctf_level = 5; //Current one
	mod3D.max_iter = 150; //30
	mod3D.convergence_ratio = 0.995f;


	//Reg parameters
	mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
	mod3D.with_reg_normals_good = false;		mod3D.Kr_total = 0.0005f; //0.003f
	mod3D.with_reg_normals_4dir = false;
	mod3D.with_reg_atraction = false;			mod3D.K_atrac_total = 2.f; //0.5
	mod3D.with_reg_edges_iniShape = false;		mod3D.K_ini_total = 0.0001f; //0.002f
	mod3D.with_reg_arap = true;					mod3D.K_arap = 4.f; //3.5
	mod3D.with_reg_rot_arap = true;				mod3D.K_rot_arap = 0.1f; //0.03
												mod3D.K_color_reg = 0.001f;

	//Dataterm parameters
	mod3D.fit_normals_old = false;
	mod3D.Kp = 0.5f*float(square(downsample))/float(num_images); //1.f
	mod3D.Kn = 0.001f*float(square(downsample))/float(num_images); //0.005f
	mod3D.Kc = 0.001f*float(square(downsample))/float(num_images);
	mod3D.truncated_res = 0.5f; //0.1f
	mod3D.truncated_resn = 1.f; 

	//Background term parameters
	if (solve_DT)
	{
		mod3D.nsamples_approx = 5000;		
		mod3D.trunc_threshold_DT = 5.f;
		mod3D.alpha = 1000.f/(float(mod3D.nsamples_approx)*mod3D.trunc_threshold_DT); //0.2 for DT, 1000 for BG
		
	}
	if (solve_SK)
	{
		mod3D.adaptive_tau = true;
		mod3D.tau_max = 32.f/float(downsample); //3.5f
		mod3D.eps_rel = 0.1; //0.1f
		mod3D.alpha = 0.0025*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images);  //0.001f
	}
	
	//Create initial mesh
	mod3D.loadMeshFromFile();

	//Create file to save energies
	if (mod3D.save_energy)
		mod3D.createFileToSaveEnergy();

	//Find initial poses for the cameras
	//mod3D.computeInitialCameraPoses();



	//Tracking
	//--------------------------------------------
	//For the experiment about basin of convergence: Me tracking 1 from 83 till 112, incr = 4
	//For the experiment on tracking: Manu 1 from 151 till 210, incr = 3 (another option? Manu 2 from 37 to 120);
	const unsigned int first_image = 153;
	const unsigned int last_image = 206;
	const unsigned int incr = 2; 
	for (unsigned int k=first_image; k<last_image; k+=incr)
	{	

		if ((k>first_image)&&(!continuous_tracking))
			mod3D.loadMeshFromFile();
		
		//Load new image of the sequence
		if (k == first_image) 	mod3D.loadImageFromSequence(k, true, seq_ID);
		else					mod3D.loadImageFromSequence(k, false, seq_ID);

		//Compute the distance transform (necessary in any case now)
		//mod3D.computeDistanceTransformOpenCV(safe_DT);
		mod3D.computeTruncatedDTOpenCV();

		//Compute the normals 
		mod3D.computeDataNormals();

		//Create and/or update the 3D scene
		if (k == first_image)
		{
			if (mod3D.paper_visualization)	mod3D.initializeSceneDataArch();
			else							mod3D.initializeScene();
		}
			
		if (!mod3D.paper_visualization)
		{
			mod3D.showNewData(true); //antes showCamPoses en vez de newdata
			mod3D.showDTAndDepth();
			mod3D.showMesh(); 
			//system::os::getch();
		}
		//else
		//	mod3D.takePictureLimitSurface(mod3D.paper_vis_no_mesh);
		//mod3D.takePictureDataArch();

		//Fit
		if ((k == first_image)|| !continuous_tracking)
			mod3D.createTopologyRefiner();

		mod3D.computeInitialUDataterm();
		if (solve_SK)
			mod3D.computeInitialUBackground();

		mod3D.adap_mult = 0.01f; //*** Check it ***
		if (solve_SK)			mod3D.solveSK_Arap();
		//else if (solve_DT)	mod3D.solveDT2_Arap();
		else if (solve_DT)		mod3D.solveBG_Arap();		
		else					mod3D.solveNB_Arap();

		if (mod3D.save_energy)
			mod3D.saveCurrentEnergyInFile(true);

		if (mod3D.paper_visualization)
			mod3D.takePictureLimitSurface(mod3D.paper_vis_no_mesh);
	}

	//if (mod3D.paper_visualization)
	//	mod3D.takePictureLimitSurface(mod3D.paper_vis_no_mesh);

	////Show final model
	//for (unsigned int r=0; r<3; r++) //4
	//	mod3D.refineMeshToShow();
	//mod3D.showRenderedModel();

	if (mod3D.save_energy)
		mod3D.f_energy.close();

	system::os::getch();
	return 0;
}


//Weights for the experiments in the CVPR paper
//-------------------------------------------------------------------------------
////Reg parameters
//mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
//mod3D.with_reg_normals_good = true;			mod3D.Kr_total = 0.0005f; //0.003f
//mod3D.with_reg_normals_4dir = false;
//mod3D.with_reg_atraction = false;			mod3D.K_atrac_total = 2.f; //0.5
//mod3D.with_reg_edges_iniShape = false;		mod3D.K_ini_total = 0.0001f; //0.002f
//mod3D.with_reg_arap = true;					mod3D.K_arap = 1.5f; //0.3 without backg
//mod3D.with_reg_rot_arap = true;				mod3D.K_rot_arap = 0.01f;

////Dataterm parameters
//mod3D.Kp = 0.5f*float(square(downsample))/float(num_images); //1.f
//mod3D.Kn = 0.001f*float(square(downsample))/float(num_images); //0.005f
//mod3D.truncated_res = 0.2f; //0.1f
//mod3D.truncated_resn = 1.f; 
//
////Background term parameters
//if (solve_DT)
//{
//	mod3D.nsamples_approx = 5000;		
//	mod3D.alpha = 0.25f/float(mod3D.nsamples_approx);
//}
//else
//{
//	mod3D.adaptive_tau = true;
//	mod3D.tau_max = 16.f/float(downsample); //3.5f
//	mod3D.eps_rel = 0.1; //0.1f
//	mod3D.alpha = 0.002f*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images);
//}

//Weights for the experiment with different initializations (Manu tracking)
//-------------------------------------------------------------------------------
////Reg parameters
//mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f;
//mod3D.with_reg_normals_good = true;			mod3D.Kr_total = 0.0005f;
//mod3D.with_reg_normals_4dir = false;
//mod3D.with_reg_atraction = false;			mod3D.K_atrac_total = 2.f;
//mod3D.with_reg_edges_iniShape = false;		mod3D.K_ini_total = 0.0001f; 
//mod3D.with_reg_arap = true;					mod3D.K_arap = 3.5f; 
//mod3D.with_reg_rot_arap = true;				mod3D.K_rot_arap = 0.03f; 
//
////Dataterm parameters
//mod3D.Kp = 0.5f*float(square(downsample))/float(num_images); //1.f
//mod3D.Kn = 0.001f*float(square(downsample))/float(num_images); //0.005f
//mod3D.truncated_res = 0.5f; //0.1f
//mod3D.truncated_resn = 1.f; 
//
////Background term parameters
//if (solve_DT)
//{
//	mod3D.nsamples_approx = 5000;		
//	mod3D.alpha = 0.2f/float(mod3D.nsamples_approx);
//}
//else
//{
//	mod3D.adaptive_tau = true;
//	mod3D.tau_max = 32.f/float(downsample);
//	mod3D.eps_rel = 0.1; 
//	mod3D.alpha = 0.001*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images);
//}

