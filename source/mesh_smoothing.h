// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

//Associated to Opensubdiv
#define M_PI       3.14159265358979323846
#define max_num_w  18		//If using ENDCAP_BSPLINE_BASIS

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

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

//Associated to Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/SparseCholesky>

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



// Vertex container implementation
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


// Limit frame container implementation
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


class MeshSmoother {
public:

	//Data
	unsigned int num_points;
	unsigned int data_per_face;
	ArrayXXf data_coords;
	ArrayXXf data_normals;
	ArrayXf n_weights;

	//Flags
	bool with_reg_normals;
	bool with_reg_normals_good;
	bool with_reg_normals_4dir;
	bool with_reg_edges;
	bool with_reg_ctf;
	bool with_reg_atraction;
	
	//Images
	string mesh_path;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<bool, Dynamic, 1>		is_quad;
	Array<int, 4, Dynamic>		face_verts;	//Maximum possible size if we allow for triangles and quads
	Array<int, 4, Dynamic>		face_adj;
	Array<float, 3, Dynamic>	vert_coords, vert_coords_old, vert_coords_reg, vert_coords_target;

	//Internal points
	ArrayXf u1, u1_old, u1_old_outer;
	ArrayXf u2, u2_old, u2_old_outer;
	ArrayXi uface, uface_old, uface_old_outer;
	ArrayXf mx, my, mz;
	ArrayXf nx, ny, nz;

	vector<ArrayXXf> mx_reg, my_reg, mz_reg;
	vector<ArrayXXf> nx_reg, ny_reg, nz_reg, inv_reg_norm;


	//Increments for the unknowns, residuals and sub-surface weights
	Array<float, 3, Dynamic>	vert_incrs;
	ArrayXf u1_incr;
	ArrayXf u2_incr;
	ArrayXf res_x, res_y, res_z;
	ArrayXf res_nx, res_ny, res_nz;
	ArrayXf res_d1, res_d2;
	ArrayXXi w_indices;						//Indices for non-zeros entrances (w_contverts). However, w_u1 and w_u2 have more zeros
	ArrayXXf w_contverts, w_u1, w_u2;		//New format, each cols stores the coefficients of a given pixel (cols grows with v and then jump to the next u)
	Array<float*, Dynamic, 1> u1_der, u2_der;
	Array<float*, Dynamic, 1> n_der_u1, n_der_u2;

	vector<ArrayXi> w_indices_reg;
	vector<ArrayXXf> w_u1_reg, w_u2_reg, w_contverts_reg;
	vector<Array<float*, Dynamic, Dynamic>> u1_der_reg, u2_der_reg;


	//Parameters
	unsigned int max_iter;
	unsigned int s_reg;
	float adap_mult;			//For both GD and LM
	float sz_x, sz_xi;			//For GD
	float truncated_res, truncated_resn; //Truncated dataterm
	float Kp, Kn, Kr_total, Kr, Ke, Ke_total, K_ctf, K_ctf_total, K_atrac, K_atrac_total;	//Weights

	//LM
	typedef Triplet<float> Tri;
	vector<Tri> j_elem;
	SparseMatrix<float,0> J;	//Col-major
	VectorXf R, increments;
	MatrixXf JtJ_dense;
	VectorXf JtR;

	//Visualization
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;
	bool					vis_errors;

	//Save results/data
	string f_folder;


	//----------------------------------------------------------------------
	//								Methods
	//----------------------------------------------------------------------
	MeshSmoother();

	//Visualization
	void initializeScene();
	void showOriginalSurface();
	void showSubSurface();
	void showMeshAndCorrespondences();

	//Initialization and segmentation
	void loadImagesStandard();
	void loadTargetMesh();
	void generateDataFromMesh();
	void computeInitialUDataterm();

	//Data from KinectFusion
	void loadMeshFromKinectFusion();

	//Solvers
	void solveNB_GradientDescent();
	void vertIncrRegularizationNormals();
	void vertIncrRegularizationEdges();

	void solveNB_LM_Joint();
	void updateInternalPointCrossingEdges(unsigned int i);
	void optimizeUDataterm_LM();

	void fillJacobianEpPixel(unsigned int i, unsigned int &J_row);
	void fillJacobianEpPixelJoint(unsigned int i, unsigned int &J_row);
	void fillJacobianEnPixel(unsigned int i, unsigned int &J_row);
	void fillJacobianEnPixelJoint(unsigned int i, unsigned int &J_row);
	void fillJacobianRegNormals(unsigned int &J_row);
	void fillJacobianRegNormalsGood(unsigned int &J_row);
	void fillJacobianRegNormals4dir(unsigned int &J_row);
	void fillJacobianRegEdges(unsigned int &J_row);
	void fillJacobianRegCTF(unsigned int &J_row);
	void fillJacobianRegAtraction(unsigned int &J_row);

	float computeEnergyRegNormals();
	float computeEnergyRegNormalsGood();
	float computeEnergyRegNormals4dir();
	float computeEnergyRegEdges();
	float computeEnergyRegCTF();
	float computeEnergyRegAtraction();
	float computeEnergyNB();

	void computeTransCoordAndResiduals();
	void computeNormalDerivativesPixel(unsigned int i);

	void searchBetterUDataterm();


	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void evaluateSubDivSurfaceRegularization();
	void evaluateSubDivSurfacePixel(unsigned int i);
	void refineMeshOneLevel();
	void refineMeshToShow();

};



//									Info
//-------------------------------------------------------------------------------------------------


