// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

//				Datasets
//-------------------------------------------------
// 1 - cube
// 2 - images Teddy
// 3 - Teddy remove background
// 4 - person front
// 5 - u shape
// 6 - robert dataset (topology known)
// 7 - dataset_teddy2 (topology known)
// 8 - sculpture1
// 9 - sculpture2
// 10 - motorbike1
// 11 - moped2


#include "3D_model_fitting.h"

//Implement joint optimization with background and test it with the arch

// ------------------------------------------------------
//					MAIN - 3D Model
// ------------------------------------------------------

int main(int argc, char* argv[])
{
	float cam_perturbation = 0.1f;
	if (argc > 1)
		cam_perturbation = stof(argv[1]);

	printf("\n Cam_perturbation = %f", cam_perturbation);
	
	const unsigned int num_images = 5; 
	const unsigned int downsample = 4; 
	const unsigned int im_set = 5;
	Mod3DfromRGBD mod3D(num_images, downsample, im_set);

	//Dirs to read and write
	mod3D.im_dir = "C:/Users/jaimez/programs/GitHub/Subdiv-Fitting-Tests/data/";
	//mod3D.f_folder = "C:/Users/jaimez/Desktop/new background term/BG noreg/";


	//Flags
	const bool solve_DT = true;
	const bool safe_DT = solve_DT ? true : 1; //Change it only when evaluating DT (used the safe one to compute adaptive tau)
	mod3D.solve_DT = solve_DT;

	mod3D.paper_visualization = false;
	mod3D.paper_vis_no_mesh = true;
	mod3D.vis_errors = true;

	mod3D.optimize_cameras = true;
	mod3D.fix_first_camera = false;
	const bool perturb_cam_poses = false;

	mod3D.small_initialization = false;
	mod3D.save_energy = false;


	//Parameters: solver
	const unsigned int refine_levels =  1;
	mod3D.ini_size = 0.3f;	//Only used for image_set = 3 when the segmentation is not perfect	
	mod3D.max_iter = 50; //100
	mod3D.adap_mult = 0.01f;
	mod3D.convergence_ratio = 0.99995f;

	//Parameters: regularization
	mod3D.with_reg_normals_good = true;			mod3D.Kr_total = 0.01; //0.01 //0.002f - true
	mod3D.with_reg_normals_4dir = false;
	mod3D.with_reg_ctf = false;					mod3D.K_ctf_total = 50.f; //500.f;
	mod3D.with_reg_atraction = true;			mod3D.K_atrac_total = 0.15f; //0.3f //0.15 - true

	//Parameters: data term
	mod3D.Kp = 0.3f*float(square(downsample))/float(num_images); //1.f
	mod3D.Kn = 0.0001f*float(square(downsample))/float(num_images); //0.0001f
	mod3D.truncated_res = 0.1f; //0.1f
	mod3D.truncated_resn = 1.f; //1.f

	//Parameters: background term
	if (solve_DT)
	{
		mod3D.nsamples_approx = 5000;	
		mod3D.trunc_threshold_DT = 5.f;
		mod3D.alpha = 30.f*float(square(downsample))/(float(mod3D.nsamples_approx*mod3D.num_images)*mod3D.trunc_threshold_DT); //0.05 //50 for BG
		
	}
	else
	{
		mod3D.adaptive_tau = true;
		mod3D.tau_max = 32.f/float(downsample); //32
		mod3D.eps_rel = 0.1; //0.1f
		mod3D.alpha = 0.01f*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images); //0.001
	}

	////mod3D.chooseParameterSet(0)

	
	//Load images and segment them
	mod3D.loadInputs();

	//Find initial poses for the cameras
	mod3D.computeInitialCameraPoses();
	if (perturb_cam_poses)
		mod3D.perturbInitialCameraPoses(cam_perturbation, 0.f); //Teddy - 0, 0.025, 0.05, 0.075, 0.1, 0.15 

	//Compute the distance transform (necessary in any case now)
	//mod3D.computeDistanceTransformOpenCV(safe_DT);
	//mod3D.tau_pixel[0].fill(mod3D.tau_max);
	mod3D.computeTruncatedDTOpenCV();

	//Compute the normals
	mod3D.computeDataNormals();

	//Create initial mesh
	mod3D.loadInitialMesh();

	//Create file to save energies
	if (mod3D.save_energy)
		mod3D.createFileToSaveEnergy();


	//Create the 3D scene and show initial status
	if ((im_set == 5)&&(mod3D.paper_visualization))
		mod3D.initializeSceneDataArch();
	else
		mod3D.initializeScene();
	if (!mod3D.paper_visualization) 
	{
		mod3D.showCamPoses();
		mod3D.showMesh();
	}

	//If I want to refine before starting
	mod3D.refineMeshOneLevel(); mod3D.ctf_level--;
	//mod3D.refineMeshOneLevel();
	//mod3D.refineMeshOneLevel();


	//					Solve
	//----------------------------------------------
	for (unsigned int k = mod3D.ctf_level-1; k < refine_levels; k++)
	{
		mod3D.createTopologyRefiner();
		mod3D.computeInitialUDataterm();
		mod3D.computeInitialUBackground();
		//printf("\n Iteration %d, ctf_level = %d", k, mod3D.ctf_level); system::os::getch();

		if (k >= 0)
		{
			//if (solve_DT)	mod3D.solveDT2_LM_Joint(false);
			if (solve_DT)	mod3D.solveBG_LM_Joint(false);
			//else			mod3D.solveNB_LM_Joint(false);	
			else			mod3D.solveSK_LM_Joint(false);
			//else			mod3D.solveSK_LM_Joint_BuildJtJ();

			//if (solve_DT)	mod3D.solveDT_GradientDescent();
			//else			mod3D.solveSK_GradientDescent();

			//if (k < refine_levels - 1)
				//mod3D.refineMeshOneLevel();
		}

		if (k < refine_levels - 1)
		{
			if (k>0) {mod3D.refineMeshOneLevel(); mod3D.alpha *= 10.f;} //Just for the new term because it changes its weight through levels
			else {mod3D.Kr_total = 0.0025f; mod3D.K_atrac_total = 0.15f; printf("\n Entra aqui *************");} //0.002, 0.15
		
			mod3D.max_iter = 50; //30; //8/(k+1);				
			mod3D.adap_mult = 0.1f;
			mod3D.Kn *= 0.9f;
			mod3D.Kr_total *= 0.5f;
			mod3D.Ke_total *= 0.5f;
			mod3D.alpha *= 0.75;
			mod3D.K_atrac_total *= 0.75f;
		}

		printf("\n New level \n");
	}


	//Render the final solution
	if (mod3D.paper_visualization)
	{
		mod3D.takePictureLimitSurface(mod3D.paper_vis_no_mesh);
		mrpt::system::sleep(1000);
		//mod3D.takePictureLimitSurface(true);
		//mrpt::system::sleep(1000);
	}
	else
	{
		//for (unsigned int r=0; r<8-mod3D.ctf_level; r++) //4
		while (mod3D.num_verts < 30000)
			mod3D.refineMeshToShow();
		//mod3D.showRenderedModel();
	}

	if (mod3D.save_energy)
		mod3D.f_energy.close();

	//Save cam poses
	//mod3D.saveResultsCamExperiment();

	system::os::getch();
	return 0;
}



//Parameters - camera perturbation experiment

////Parameters: solver
//const unsigned int refine_levels =  4;
//mod3D.ini_size = 0.3f;	//Only used for image_set = 3 when the segmentation is not perfect	
//mod3D.max_iter = 50;
//mod3D.adap_mult = 1.f;
//mod3D.convergence_ratio = 0.999f;

////Parameters: regularization
//mod3D.with_reg_normals_good = true;			mod3D.Kr_total = 0.01; //0.002f - true
//mod3D.with_reg_normals_4dir = false;
//mod3D.with_reg_ctf = false;					mod3D.K_ctf_total = 50.f; //500.f;
//mod3D.with_reg_atraction = true;			mod3D.K_atrac_total = 0.3f; //0.15 - true

////Parameters: data term
//mod3D.Kp = 0.3f*float(square(downsample))/float(num_images); //1.f
//mod3D.Kn = 0.0001f*float(square(downsample))/float(num_images); //0.0001f
//mod3D.truncated_res = 0.1f; //0.1f
//mod3D.truncated_resn = 1.f; //1.f

////Parameters: background term
//if (solve_DT)
//{
//	mod3D.nsamples_approx = 5000;		
//	mod3D.alpha = 0.05*float(square(downsample))/float(mod3D.nsamples_approx*mod3D.num_images);
//	mod3D.trunc_threshold_DT = 8.f;
//}
//else
//{
//	mod3D.adaptive_tau = true;
//	mod3D.tau_max = 32.f/float(downsample); //32
//	mod3D.eps_rel = 0.1; //0.1f
//	mod3D.alpha = 0.01f*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images); //0.001 ?? I don't know which one
//}
//...
//mod3D.refineMeshOneLevel(); //mod3D.ctf_level--;


	
