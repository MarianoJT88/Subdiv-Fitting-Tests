// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


//MRPT
#include <mrpt/system.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl.h>
#include <mrpt/math.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

//System
#include <stdio.h>
#include <string.h>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::opengl;
using namespace mrpt::math;
using namespace Eigen;
using namespace std;



class Mod2DfromRGBD {
public:

	//Camera
	float fovh_d, fovv_d;
	unsigned int rows, cols, downsample;
	vector<poses::CPose3D> cam_poses;
	vector<Matrix4f> cam_trans;
	vector<Matrix4f> cam_trans_inv;
	vector<Matrix4f> cam_ini;
	vector<Matrix<float, 6, 1>> cam_mfold, cam_mfold_old;

	//Flags
	bool model_behind_camera;
	bool solve_DT;
	
	//Images
	int image_set;			//1 -> images head 1, 2 -> images head 2, 3 -> cloth ground
	unsigned int num_images;
	string	im_dir;
	vector<ArrayXf> depth;
	vector<ArrayXf> x_image;
	vector<ArrayXf> nx_image, ny_image;

	//Segmentation and invalid mask
	vector<Array<bool, Dynamic, 1>> valid;
	vector<Array<bool, Dynamic, 1>> is_object;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<int, 2, Dynamic>		face_verts;
	Array<int, 2, Dynamic>		face_adj;
	Array<float, 2, Dynamic>	vert_coords, vert_coords_old;

	//Internal points
	vector<ArrayXf> u1, u1_old, u1_old_outer;
	vector<ArrayXi> uface, uface_old, uface_old_outer;
	vector<ArrayXf> mx, my;
	vector<ArrayXf> mx_t, my_t;
	vector<ArrayXf> nx, ny;
	vector<ArrayXf> nx_t, ny_t;

	//Increments for the unknowns
	Array<float, 2, Dynamic>	vert_incrs;
	vector<Matrix<float, 6, 1>> cam_incrs;
	vector<ArrayXf> u1_incr;

	//Residuals
	vector<ArrayXf> res_x, res_y;
	vector<ArrayXf> res_nx, res_ny;
	vector<ArrayXf> res_d1;

	//Weights/jacobians
	vector<Array<float*, Dynamic, 1>> w_contverts, w_derverts;
	vector<Array<float*, Dynamic, 1>> u1_der, u1_der2;

	//LM variables
	typedef Triplet<float> Tri;
	vector<Tri> j_elem;
	SparseMatrix<float> J;
	VectorXf R, increments;

	//Parameters
	unsigned int robust_kernel; //I leave it here in case we want to test different kernels, but for now it is unused 
	unsigned int max_iter;
	float sz_x, sz_xi, adap_mult;
	float tau, alpha, Kn, eps;
	float K_cam_prior;

	//DT
	unsigned int nsamples_approx, nsamples;
	vector<ArrayXf> DT;
	vector<ArrayXf> DT_grad;
	vector<ArrayXf> pixel_DT;
	Array<float*, Dynamic, 1> w_DT;
	Array<float*, Dynamic, 1> u1_der_DT;
	ArrayXf mx_DT, my_DT;
	ArrayXf u1_DT;
	ArrayXi uface_DT;

	//Visualization
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;


	//----------------------------------------------------------------------
	//								Methods
	//----------------------------------------------------------------------
	Mod2DfromRGBD(unsigned int num_im, unsigned int downsamp);

	//Visualization
	void initializeScene();
	void showCamPoses();
	void showMesh();
	void showSubSurface();
	void showJacobiansBackground();

	//Initialization and segmentation
	void createDepthScan();					//Old way of generating synthetic data.
	void loadDepthFromImages();
	void loadInitialMesh();
	void computeDataNormals();
	void computeInitialCameraPoses();
	void computeInitialUDataterm();
	void computeInitialUBackground();
	void segmentFromDepth();


	//Solvers
	void solveSK_GradientDescent();
	void solveSK_LM();
	void updateInternalPointCrossingEdges(unsigned int i, unsigned int u, bool adaptive);
	void test_FiniteDifferences();
	float computeEnergyOverall();
	void computeTransCoordAndResiduals();
	void computeCameraTransfandPosesFromTwist();

	void optimizeUDataterm_LM();
	void optimizeUBackground_LM();


	//Subdivision surfaces
	void evaluateSubDivSurface();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int u);
	void refineMeshOneLevel();

	//Distance transform
	void sampleSurfaceForDTBackground();
	void computeDistanceTransform();
	void solveDT_GradientDescent();
	float computeEnergyDTOverall();
};

//Comments
//- LM is not going to work properly after refinement because of the lack of regularization (the increments for some cont. vertices will not be constrained).
//- cam_trans and cam_trans_inv! - The names are the opposite to what you would expect from the paper...
//- DT solved with LM is not implemented, only DT (without square) with gradient descent

