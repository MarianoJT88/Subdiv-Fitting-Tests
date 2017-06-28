// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "KinectFusion_datatest.h"

//Implement joint optimization with background and test it with the arch

// ------------------------------------------------------
//					MAIN - 3D Model
// ------------------------------------------------------

int main()
{
	const unsigned int refine_levels = 5;
	const unsigned int num_images = 5; //5
	const unsigned int downsample = 4; //4
	const unsigned int im_set = 1;
	Mod3DfromRGBD mod3D(num_images, downsample, im_set);

	mod3D.im_dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/";
				 //"C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/";
	mod3D.f_folder = "C:/Users/jaimez/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/cube/";
					 //"C:/Users/Mariano/Dropbox/OpenSubdiv-Model-Fitting/videos and pictures/For the paper/video/Exp3/DT1/";

	mod3D.vis_errors = true;

	
	//Load poses, images and the mesh
	mod3D.loadPoses();
	mod3D.loadImages();
	mod3D.loadMesh();

	//Segment the objects
	mod3D.computeSegmentationFromBoundingBox();
	////mod3D.computeDepthSegmentationFromPlane();
	////mod3D.computeDepthSegmentationClosestObject();
	////mod3D.saveSegmentationToFile(3);


	////Compute the normals
	//mod3D.computeDataNormals();

	//Create the 3D scene
	mod3D.initializeScene();

	//Visualize the data
	mod3D.showCamPoses();
	mod3D.showImages();
	mod3D.showMesh();
	mod3D.showRenderedModel();

	////Create initial mesh
	//mod3D.loadInitialMesh();
	////mod3D.refineMeshOneLevel();
	////mod3D.refineMeshOneLevel();


	////Render the final solution
	//mod3D.showRenderedModel();


	system::os::getch();
	return 0;
}

