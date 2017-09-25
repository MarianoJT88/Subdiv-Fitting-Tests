// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

//Associated to Opensubdiv
#define M_PI       3.14159265358979323846
#define max_num_w  16		//If using ENDCAP_BSPLINE_BASIS

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

//Associated to Eigen
#include <Eigen/Dense>
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


class Mod3DfromRGBD {
public:

	//Camera
	float fovh_d, fovv_d;
	unsigned int rows, cols, downsample;
	float fx, fy;

	//Cam poses
	vector<poses::CPose3D> cam_poses;
	vector<Matrix4f> cam_trans;
	vector<Matrix4f> cam_trans_inv;
	vector<Matrix4f> cam_ini;
	vector<Matrix<float, 6, 1>> cam_mfold, cam_mfold_old;
	vector<MatrixXf> trajectory;
	vector<double> timestamps;
	
	//Images
	unsigned int num_images;
	unsigned int image_set;
	unsigned int ctf_level;
	string im_dir;
	ArrayXXf depth_background;
	vector<ArrayXXf> intensity;	//Not used
	vector<ArrayXXf> depth;
	vector<ArrayXXf> x_image;
	vector<ArrayXXf> y_image;
	vector<ArrayXXf> nx_image, ny_image, nz_image;
	vector<ArrayXf> n_weights;

	//Segmentation used to say whether a point belongs to the object or not (only for depth)
	vector<Array<bool, Dynamic, Dynamic>> valid;
	vector<Array<bool, Dynamic, Dynamic>> is_object;

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
	vector<ArrayXXi> uface;
	vector<ArrayXXf> mx, my, mz;
	vector<ArrayXXf> nx, ny, nz;


	//Visualization
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;
	bool					vis_errors;

	//Save results/data
	string f_folder;


	//----------------------------------------------------------------------
	//								Methods
	//----------------------------------------------------------------------
	Mod3DfromRGBD(unsigned int num_im, unsigned int downsamp, unsigned int im_set);

	//Visualization
	void initializeScene();
	void showCamPoses();
	void showImages();
	void showMesh();
	void showSubSurface();
	void showRenderedModel();

	//Initialization and segmentation
	void computeSegmentationFromBoundingBox();
	void computeDepthSegmentationClosestObject();
	void computeDepthSegmentationFromPlane();
	void saveSegmentationToFile(unsigned int image);
	void loadPoses();
	void loadImages();
	void loadMesh();
	void computeDataNormals();
	void computeInitialCameraPoses();
	void computeCameraTransfandPosesFromTwist();


	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u);
	void refineMeshOneLevel();
	void refineMeshToShow();

};


