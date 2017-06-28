// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

//Associated to Opensubdiv
#define M_PI       3.14159265358979323846
#define max_num_w  16
//#include <math.h>	// To define M_PI, but it does not work
#include <iso646.h> //To define the words and, not, etc. as operators in Windows
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/far/stencilTable.h>

//Associated to MRPT
#include <mrpt/system.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/utils/CImage.h>
#include <mrpt/opengl.h>
#include <mrpt/math.h>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Associated to Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

//System
//#include <cstdio>
//#include <cstring>
//#include <cassert>
//#include <cfloat>
#include <stdio.h>
#include <string.h>


using namespace OpenSubdiv;
using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::opengl;
using namespace mrpt::math;
using namespace Eigen;
using namespace std;



// Vertex container implementation.
struct Vertex {

	// Minimal required interface ----------------------
	Vertex() { }

	void Clear(void * = 0) {
		point[0] = point[1] = point[2] = 0.0f;
	}

	void AddWithWeight(Vertex const & src, float weight) {
		point[0] += weight * src.point[0];
		point[1] += weight * src.point[1];
		point[2] += weight * src.point[2];
	}

	// Public interface ------------------------------------
	void SetPosition(float x, float y, float z) {
		point[0] = x;
		point[1] = y;
		point[2] = z;
	}

	float point[3];
};


// Limit frame container implementation -- this interface is not strictly
// required but follows a similar pattern to Vertex.
struct LimitFrame {

	void Clear(void * = 0) {
		point[0] = point[1] = point[2] = 0.0f;
		deriv1[0] = deriv1[1] = deriv1[2] = 0.0f;
		deriv2[0] = deriv2[1] = deriv2[2] = 0.0f;
	}

	void AddWithWeight(Vertex const & src,
		float weight, float d1Weight, float d2Weight) {

		point[0] += weight * src.point[0];
		point[1] += weight * src.point[1];
		point[2] += weight * src.point[2];

		deriv1[0] += d1Weight * src.point[0];
		deriv1[1] += d1Weight * src.point[1];
		deriv1[2] += d1Weight * src.point[2];

		deriv2[0] += d2Weight * src.point[0];
		deriv2[1] += d2Weight * src.point[1];
		deriv2[2] += d2Weight * src.point[2];
	}

	float point[3],
		deriv1[3],
		deriv2[3];
};


class Mod3DfromRGBD {
public:

	//Camera
	float fovh_i, fovv_i;
	float fovh_d, fovv_d;
	unsigned int rows, cols, downsample;
	vector<poses::CPose3D> cam_poses;
	vector<Matrix4f> cam_trans;
	vector<Matrix4f> cam_trans_inv;
	vector<Matrix4f> cam_ini;
	vector<Matrix<float, 6, 1>> cam_mfold, cam_mfold_old;

	//Flags
	bool solve_DT;
	bool with_reg_normals;
	bool with_reg_edges;
	bool with_reg_membrane;
	bool with_reg_thin_plate;
	bool regularize_unitary_normals;
	
	//Images
	unsigned int num_images;
	unsigned int image_set;
	unsigned int ctf_level;
	vector<ArrayXXf> intensity;
	vector<ArrayXXf> depth;
	vector<ArrayXXf> x_image;
	vector<ArrayXXf> y_image;
	vector<ArrayXXf> nx_image, ny_image, nz_image;

	//Segmentation used to say whether a point belong to the object or not (only for depth)
	vector<Array<bool, Dynamic, Dynamic>> valid;
	vector<Array<bool, Dynamic, Dynamic>> is_object;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<bool, Dynamic, 1>		is_quad;
	Array<int, 4, Dynamic>		face_verts;	//Maximum possible size if we allow for triangles and quads
	Array<int, 4, Dynamic>		face_adj;
	Array<float, 3, Dynamic>	vert_coords, vert_coords_old;

	//Internal points
	vector<ArrayXXf> u1, u1_old, u1_old_outer;
	vector<ArrayXXf> u2, u2_old, u2_old_outer;
	vector<ArrayXXi> uface, uface_old, uface_old_outer;
	vector<ArrayXXf> mx, my, mz;
	vector<ArrayXXf> mx_t, my_t, mz_t;
	vector<ArrayXXf> nx, ny, nz;
	vector<ArrayXXf> nx_t, ny_t, nz_t;
	vector<ArrayXXf> nx_reg, ny_reg, nz_reg, inv_reg_norm;


	//Increments for the unknowns, residuals and sub-surface weights
	Array<float, 3, Dynamic>	vert_incrs;
	vector<Matrix<float, 6, 1>> cam_incrs;
	vector<ArrayXXf> u1_incr;
	vector<ArrayXXf> u2_incr;
	vector<ArrayXXf> res_x, res_y, res_z;
	vector<ArrayXXf> res_nx, res_ny, res_nz;
	vector<ArrayXXf> res_d1, res_d2;
	vector<ArrayXXi> w_indices;						//Indices for non-zeros entrances (w_contverts). However, w_u1 and w_u2 have more zeros
	vector<ArrayXXf> w_contverts, w_u1, w_u2;		//New format, each cols stores the coefficients of a given pixel (cols grows with v and then jumping to the next u)
	vector<Array<float*, Dynamic, Dynamic>> u1_der, u2_der;

	vector<ArrayXXi> w_size_reg;
	vector<ArrayXXf> w_u1_reg, w_u2_reg;
	vector<Array<float*, Dynamic, Dynamic>> u1_der_reg, u2_der_reg;
	vector<Array<float*,Dynamic,Dynamic>> n_der_u1, n_der_u2;
	Matrix<float, max_num_w, max_num_w> Q_m_sqrt, Q_tp_sqrt;

	//Parameters
	unsigned int max_iter;
	unsigned int s_reg;
	unsigned int robust_kernel; //0 - original truncated quadratic, 1 - 2 parabolas with peak
	float truncated_res, truncated_resn; //Truncated dataterm
	float sz_x, sz_xi, sz_uf, sz_ub, adap_mult;
	float Kn, Kr_total, Kr, Ke, Ke_total;
	float tau, alpha_raycast, alpha_DT, cam_prior, peak_margin;


	//Visualization
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;

	//----------------------------------------------------------------------
	//								Methods
	//----------------------------------------------------------------------
	Mod3DfromRGBD();

	//Visualization
	void initializeScene();
	void showCamPoses();
	void showMesh();
	void showSubSurface();
	void showRenderedModel();
	void takePictureLimitSurface(bool last);

	void loadUShape();
	void loadInitialMesh();
	void loadInitialMeshUShape();
	void computeDataNormals();
	void computeDepthSegmentationUShape();
	void computeCameraTransfandPosesFromTwist();
	void computeInitialCameraPoses();
	void computeInitialU();
	void searchBetterU();
	void searchBetterV();
	void computeInitialV();
	void computeInitialCorrespondences();
	void computeInitialMesh();
	void computeTransCoordAndResiduals();
	void computeNormalDerivativesPixel(unsigned int i,unsigned int v,unsigned int u);
	float computeEnergyOverall();
	float last_energy;

	//Solve the max(inner-background) and min(inner-foreground and outer) problem with gradient descent
	float solveGradientDescentBackgroundCam2D();
	void solveLM();
	void solveLMSparseJ();
	void solveLMWithoutCameraOptimization();
	void solveLMOnlyBackground();
	void solveLMOnlyForeground();
	void solveLMOnlyForegroundAndCameras();
	void updateInternalPointCrossingEdges(unsigned int i, unsigned int v, unsigned int u, bool adaptive);

	void solveGradientDescent();
	void vertIncrRegularizationNormals();
	void vertIncrRegularizationEdges();

	void rayCastingLMBackgroundPerPixel();
	void rayCastingLMForegroundPerPixel();

	void fillJacobianRegNormals(unsigned int &J_row);
	void fillJacobianRegEdges(unsigned int &J_row);
	void fillJacobianRegMembrane(unsigned int &J_row);
	void fillJacobianRegThinPlate(unsigned int &J_row);
	float computeEnergyRegNormals();
	float computeEnergyRegEdges();
	float computeEnergyRegMembrane();
	float computeEnergyRegThinPlate();
	float K_m, K_tp;

	typedef Triplet<float> Tri;
	vector<Tri> j_elem;
	SparseMatrix<float> J;
	VectorXf R, increments;

	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void evaluateSubDivSurfaceRegularization();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u);
	void evaluateSubDivSurfaceOnlyBackground();
	void refineMeshOneLevel();
	void refineMeshToShow();

	//Distance transform
	vector<ArrayXXf> DT;
	vector<ArrayXXf> DT_grad_u, DT_grad_v;
	ArrayXf w_DT;
	ArrayXi w_indices_DT;
	Array<float*, Dynamic, 1> u1_der_DT, u2_der_DT;
	ArrayXf mx_DT, my_DT, mz_DT;
	ArrayXf u1_DT, u2_DT;
	ArrayXi uface_DT;
	vector<ArrayXf> pixel_DT_u, pixel_DT_v;
	unsigned int nsamples_approx, nsamples;
	void sampleSurfaceForDTBackground();
	void computeDistanceTransform();
	void solveWithDT();
	void solveDTwithLMNoCameraOptimization();
	float solveWithDTBackgroundCam2D();
	float computeEnergyDTOverall();
	float computeEnergyDT2Overall();

	//Permutations from solution to analyze convergence
	ArrayXXf energy_disp_raycast, energy_disp_DT;
	void runRaycastFromDisplacedMesh();

	//Results to analyze
	vector<float> energy_data, energy_background, energy_reg;
	std::ofstream f_res;
	string f_folder;
	void saveResults();
};



// Returns the normalized version of the input vector
inline void normalize(float *n);

// Returns the cross product of v1 and v2                            
inline void cross_prod(float const *v1, float const *v2, float* vOut);

// Returns the dot product of v1 and v2
inline float dot_prod(float const *v1, float const *v2);



//The gradient descent works for the old background kernel
