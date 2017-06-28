// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "face_transition_tests_3D.h"


// ------------------------------------------------------
//				MAIN - 3D Model (1 banana)
// ------------------------------------------------------

int main()
{
	TestTransitions3D face_tests;

	//Create the 3D scene
	face_tests.initializeScene();

	//Create initial mesh
	face_tests.loadInitialMesh();

	//Create topology refiner
	face_tests.createTopologyRefiner();

	//Test
	//face_tests.testSingleParticle();
	face_tests.testMultipleParticles();
	//face_tests.testGradientDescentGravity();

	system::os::getch();
	return 0;
}

