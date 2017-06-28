// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


//Associated to MRPT
#include <mrpt/system.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl.h>
#include <mrpt/math.h>

//Associated to Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

//System
//#include <cstdio>
//#include <cstring>
//#include <cassert>
//#include <cfloat>
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
	bool new_cam_pose_model;
	bool model_behind_camera;
	bool solve_DT;
	int image_set;
	float min_depth;
	
	//Images
	unsigned int num_images;
	vector<ArrayXf> depth;
	vector<ArrayXf> x_image;
	vector<ArrayXf> x_t, y_t;

	//Segmentation used to say whether a point belong to the object or not (only for depth)
	vector<Array<bool, Dynamic, 1>> valid;
	vector<Array<bool, Dynamic, 1>> is_object;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<bool, Dynamic, 1>		is_quad;
	Array<int, 2, Dynamic>		face_verts;	//Maximum possible size if we allow for triangles and quads
	Array<int, 2, Dynamic>		face_adj;
	Array<float, 2, Dynamic>	vert_coords, vert_coords_old;

	//Internal points
	vector<ArrayXf> u1, u1_old, u1_old_outer;
	vector<ArrayXi> uface, uface_old, uface_old_outer;
	vector<ArrayXf> mx, my;
	vector<ArrayXf> mx_t, my_t;

	//Increments for the unknowns, residuals and sub-surface weights
	Array<float, 2, Dynamic>	vert_incrs;
	vector<Matrix<float, 6, 1>> cam_incrs;
	vector<ArrayXf> u1_incr;
	vector<ArrayXf> res_depth;
	vector<ArrayXf> res_d1;
	vector<Array<float*, Dynamic, 1>> w_contverts;
	vector<Array<float*, Dynamic, 1>> u1_der;


	//Scene
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;

	//Methods
	Mod2DfromRGBD();
	void initializeScene();
	void showCamPoses();
	void showMesh();
	void showSubSurface();
	void createDepthScan();
	void loadDepthFromImages();
	void loadInitialMesh();
	//void computeDepthSegmentation();
	void computeInitialCameraPoses();
	void computeInitialU();
	void computeInitialV();
	void computeInitialCorrespondences();
	void computeInitialMesh();
	void computeTransCoordAndResiduals();
	void computeEnergyMaximization();
	void computeEnergyMinimization();
	float computeEnergyOverall();
	void segmentFromDepth();

	//Solve the max (inner-background) and min(inner-foreground and outer) problem with gradient descent
	float sz_x, sz_xi, sz_uf, sz_ub;
	float tau, alpha, Kz, Kclose, Kproj;
	float cam_prior;
	void solveGradientDescent();
	void updateInternalPointCrossingEdges(unsigned int i, unsigned int u, bool adaptive);
	void solveViaFiniteDifferences();

	//LM solvers
	void rayCastingLMBackgroundPerPixel();
	void rayCastingLMForegroundPerPixel();
	vector<float> energy_vec;
	unsigned int max_iter;

	//Subdivision surfaces
	void evaluateSubDivSurface();
	void evaluateSubDivSurfaceOnlyBackground();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int u);
	void refineMeshOneLevel();

	//Distance transform
	vector<ArrayXf> DT;
	vector<ArrayXf> DT_grad;
	vector<ArrayXf> pixel_DT;
	Array<float*, Dynamic, 1> w_DT;
	Array<float*, Dynamic, 1> u1_der_DT;
	ArrayXf mx_DT, my_DT;
	ArrayXf u1_DT;
	ArrayXi uface_DT;
	unsigned int nsamples_approx, nsamples;
	void sampleSurfaceForDTBackground();
	void computeDistanceTransform();
	void solveWithDT();
	float computeEnergyDTOverall();

};



// Returns the normalized version of the input vector
inline void normalize(float *n);

// Returns the cross product of v1 and v2                            
inline void cross_prod(float const *v1, float const *v2, float* vOut);

// Returns the dot product of v1 and v2
inline float dot_prod(float const *v1, float const *v2);


// Be careful with some wrong segmentations!!!!

//Problems with Distance transform
//- There is no good solution for invalid measurment
//- We would need an infinite sampling of the model to force the model to be within the real silhouette
//- Some samples of the model might be out of the image plane
