// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

//Associated to Opensubdiv
#define M_PI       3.14159265358979323846
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
#include <mrpt/utils.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl.h>
#include <mrpt/math/eigen_frwds.h>
#include <mrpt/math.h>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Associated to Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

//System
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cfloat>
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
	vector<Matrix4f> cam_ini;
	vector<Matrix<float, 6, 1>> cam_mfold;
	
	//Images
	unsigned int num_images;
	vector<ArrayXf> intensity;
	vector<ArrayXf> depth;
	vector<ArrayXf> x_image;
	vector<ArrayXf> y_image;
	vector<ArrayXf> x_t, y_t, z_t;

	//Segmentation used to say whether a point belong to the object or not (only for depth)
	vector<Array<bool, Dynamic, Dynamic>>	d_labeling;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<bool, Dynamic, 1>		is_quad;
	Array<int, 4, Dynamic>		face_verts;	//Maximum possible size if we allow for triangles and quads
	Array<int, 4, Dynamic>		face_adj;
	Array<float, 3, Dynamic>	vert_coords;

	//Internal points
	vector<ArrayXXf> u1;
	vector<ArrayXXf> u2;
	vector<ArrayXi> uface;
	vector<ArrayXXf> mx, my, mz;

	//Increments for the unknowns, residuals and sub-surface weights
	vector<ArrayXXf> u1_incr;
	vector<ArrayXXf> u2_incr;
	vector<ArrayXXf> res_x, res_y, res_z;
	vector<Array<float*,Dynamic, Dynamic>> w_contverts;
	vector<Array<float*, Dynamic, Dynamic>> u1_der;
	vector<Array<float*, Dynamic, Dynamic>> u2_der;

	//Jacobians
	typedef Triplet<float> Tri;
	SparseMatrix<float> J, J_findif;
	vector<Tri> j_elem, j_elem_fd;

	//Scene
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;

	//Methods
	Mod3DfromRGBD();
	void initializeScene();
	void showCamPoses();
	void showMesh();
	void showSubSurface();
	void loadImagesFromDisc();
	void loadInitialMesh();
	void computeDepthSegmentation();
	void computeInitialCameraPoses();
	void computeInitialIntPointsPinHole();
	void computeInitialIntPointsOrtographic();
	void computeInitialIntPointsExpanding();
	void computeInitialIntPointsClosest();
	void computeInitialMesh();
	void computeTransCoordAndResiduals();

	//Jacobians computation
	void computeJacobianFiniteDifferences();
	void computeJacobianAnalytical();
	void compareJacobians();
	void fillGradControlVertices();
	void fillGradCameraPoses();
	void fillGradInternalPoints();

	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void refineMeshOneLevel();

};



// Returns the normalized version of the input vector
inline void normalize(float *n);

// Returns the cross product of v1 and v2                            
inline void cross_prod(float const *v1, float const *v2, float* vOut);

// Returns the dot product of v1 and v2
inline float dot_prod(float const *v1, float const *v2);


