// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "mesh_smoothing.h"

//Implement joint optimization with background and test it with the arch

// ------------------------------------------------------
//					MAIN - 3D Model
// ------------------------------------------------------

int main()
{
	MeshSmoother mesh_ref;

	mesh_ref.mesh_path = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/meshes/mesh_teddy1.obj";
					//"C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/mesh.obj";
	mesh_ref.data_per_face = 3*3;

	//Load the mesh to refine
	mesh_ref.loadTargetMesh();

	//Optimization and visualization parameters
	mesh_ref.vis_errors = false;
	mesh_ref.max_iter = 50;
	mesh_ref.adap_mult = 1.f;

	//Reg parameters
	mesh_ref.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
	mesh_ref.with_reg_normals_good = false;			mesh_ref.Kr_total = 0.0001f; //0.002f; //0.2f*mod3D.num_images
	mesh_ref.with_reg_normals_4dir = false;
	mesh_ref.with_reg_edges = false;	  				mesh_ref.Ke_total = 0.02f/float(mesh_ref.num_faces);
	mesh_ref.with_reg_ctf = false;					mesh_ref.K_ctf_total = 1.f; //500.f;
	mesh_ref.with_reg_atraction = true;				mesh_ref.K_atrac_total = 0.5f; ///float(mesh_ref.num_faces);

	//Dataterm parameters
	mesh_ref.Kp = 30000.f/float(mesh_ref.num_points);
	mesh_ref.Kn = 0.f; //0.001f/float(mesh_ref.num_points); 
	mesh_ref.truncated_res = 1.f; //0.1f
	mesh_ref.truncated_resn = 1.f; 


	//Generate the data to fit from the target mesh
	mesh_ref.generateDataFromMesh();

	//Create the 3D scene and show initial status
	mesh_ref.initializeScene();



	//					Solve
	//----------------------------------------------
	mesh_ref.createTopologyRefiner();
	mesh_ref.computeInitialUDataterm();
	mesh_ref.solveNB_LM_Joint();

	//mesh_ref.refineMeshOneLevel();
	//mesh_ref.createTopologyRefiner();
	//mesh_ref.computeInitialUDataterm();
	//mesh_ref.solveNB_LM_Joint();


	system::os::getch();
	return 0;
}

