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
#include <mrpt/opengl.h>
#include <mrpt/math.h>

//Associated to Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

//System
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
	float fovh_d, fovv_d;
	unsigned int rows, cols, downsample;

	//Sphere
	float xs, ys, zs, r;
	
	//Images
	unsigned int num_images;
	unsigned int image_set;
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
	vector<ArrayXXf> w_contverts, w_u1, w_u2;		//New format, each cols stores the coefficients of a given pixel (cols grows with v and then jumps to the next u)
	vector<Array<float*, Dynamic, Dynamic>> u1_der, u2_der;

	vector<ArrayXXi> w_size_reg;
	vector<ArrayXXf> w_u1_reg, w_u2_reg;
	vector<Array<float*, Dynamic, Dynamic>> u1_der_reg, u2_der_reg;
	vector<Array<float*,Dynamic,Dynamic>> n_der_u1, n_der_u2;
	Matrix<float, max_num_w, max_num_w> Q_m_sqrt, Q_tp_sqrt;

	//LM
	typedef Triplet<float> Tri;
	vector<Tri> j_elem;
	SparseMatrix<float> J;
	VectorXf R, increments;

	//Parameters
	unsigned int robust_kernel; //0 - original truncated quadratic, 1 - 2 parabolas with peak
	float adap_mult, tau, alpha, cam_prior, Kn, peak_margin, K_m, K_tp;
	unsigned int max_iter;
	unsigned int s_reg;
	bool with_reg;

	//Scene
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;


	//Methods
	Mod3DfromRGBD();

	//Visualization
	void initializeScene();
	void showCamPoses();
	void showMesh();
	void showSubSurface();
	void showRenderedModel();

	//Initialization and segmentation
	void createImageFromSphere();
	void loadInitialMesh();
	void computeDataNormals();
	void computeDepthSegmentation();
	void computeDepthSegmentationFromPlane();
	void computeInitialUDataterm();
	void computeInitialUBackground();
	void computeInitialCorrespondences();
	void computeInitialMesh();


	//Solver
	void solveLMSparseJ();
	void updateInternalPointCrossingEdges(unsigned int i, unsigned int v, unsigned int u, bool adaptive);
	void rayCastingLMBackgroundPerPixel();
	void rayCastingLMForegroundPerPixel();
	void searchBetterUDataterm();
	void searchBetterUBackground();
	void computeTransCoordAndResiduals();
	void computeNormalDerivativesPixel(unsigned int i, unsigned int v, unsigned int u);
	float computeEnergyOverall();

	void testIfMembraneWorks();
	void fillJacobianRegMembrane(unsigned int &J_row);
	void fillJacobianRegThinPlate(unsigned int &J_row);
	float computeEnergyMembrane();
	float computeEnergyThinPlate();


	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u);
	void refineMeshOneLevel();
	void refineMeshToShow();
	void showLocalPoints();
	void understandLimitSurfaceEvaluation();

	//Results to analyze
	vector<float> energy_foreground, energy_background;
	std::ofstream f_res;
	void saveResults();

};



