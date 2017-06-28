// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "jacobian_comparison.h"


// ------------------------------------------------------
//				MAIN - 3D Model (1 banana)
// ------------------------------------------------------

int main()
{
	Mod3DfromRGBD jacob_comp;
	jacob_comp.loadImagesFromDisc();

	//Segment the objects
	jacob_comp.computeDepthSegmentation();

	//Create the 3D scene
	jacob_comp.initializeScene();

	//Find initial poses for the cameras
	jacob_comp.computeInitialCameraPoses();

	//Create initial mesh
	jacob_comp.loadInitialMesh();

	//Create topology refiner
	jacob_comp.createTopologyRefiner();

	//Refine one level
	jacob_comp.refineMeshOneLevel(); //I have to create a routine to initialize the internal points for any arbitrary initial mesh...
	jacob_comp.createTopologyRefiner();
	jacob_comp.refineMeshOneLevel(); 
	jacob_comp.createTopologyRefiner();

	//Compute initial internal points
	jacob_comp.computeInitialIntPointsClosest();

	//Compare Jacobians
	jacob_comp.computeJacobianAnalytical();
	jacob_comp.computeJacobianFiniteDifferences();
	//jacob_comp.compareJacobians();


	//Compare them
	const float sum_dif_block1 = (jacob_comp.J - jacob_comp.J_findif).leftCols(3 * jacob_comp.num_verts).cwiseAbs().sum();

	printf("\n sum differences jacobian block 1 = %f", sum_dif_block1);

	//Compare blocks of the gradients with respect to the control vertices
	//SparseMatrix<float> block_fd = banana.J_findif.block(0, 0, 10, 10);
	//cout << endl << "Block of J_findif: " << endl << block_fd;

	//SparseMatrix<float> block_an = banana.J.block(0, 0, 10, 10);
	//cout << endl << "Block of J_analytical: " << endl << block_an;

	//Compare blocks of the gradients with respect to the control vertices
	//SparseMatrix<float> block_fd = banana.J_findif.block(0, 3 * banana.num_verts, 10, 6*banana.num_images);
	//cout << endl << "Block of J_findif: " << endl << block_fd;

	//SparseMatrix<float> block_an = banana.J.block(0, 3 * banana.num_verts, 10, 6*banana.num_images);
	//cout << endl << "Block of J_analytical: " << endl << block_an;

	//Compare blocks of the gradients with respect to the internal points
	//const unsigned int col_offset = 3 * banana.num_verts + 6 * banana.num_images;
	//SparseMatrix<float> block_fd = banana.J_findif.block(0, col_offset, 10, 10);
	//cout << endl << "Block of J_findif: " << endl << block_fd;

	//SparseMatrix<float> block_an = banana.J.block(0, col_offset, 10, 10);
	//cout << endl << "Block of J_analytical: " << endl << block_an;


	system::os::getch();
	return 0;
}



