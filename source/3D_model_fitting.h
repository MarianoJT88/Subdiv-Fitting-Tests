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
#include <random>


using namespace OpenSubdiv;
using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::opengl;
using namespace mrpt::math;
using namespace Eigen;
using namespace std;


typedef Triplet<float> Tri;
typedef pair<float, unsigned int> pair_J;

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

struct Vertex_color {

	// Minimal required interface ----------------------
	Vertex_color() { }

	void Clear(void * = 0) {
		color = 0.f;
	}

	void AddWithWeight(Vertex_color const & src, float weight) {
		color += weight * src.color;

	}

	// Public interface ------------------------------------
	void SetColor(float c) {
		color = c;
	}

	float color;
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

struct LimitFrameColor {

	void Clear(void * = 0) {
		color = 0.0f;
		deriv1 = 0.0f;
		deriv2 = 0.0f;
	}

	void AddWithWeight(Vertex_color const & src,
		float weight, float d1Weight, float d2Weight) {

		color += weight * src.color;
		deriv1 += d1Weight * src.color;
		deriv2 += d2Weight * src.color;

	}

	float color, deriv1, deriv2;
};

struct LimitFrame2der {

	void Clear(void * = 0) {
		point[0] = point[1] = point[2] = 0.f;
		deriv1[0] = deriv1[1] = deriv1[2] = 0.f;
		deriv2[0] = deriv2[1] = deriv2[2] = 0.f;
		der_ss[0] = der_ss[1] = der_ss[2] = 0.f;
		der_st[0] = der_st[1] = der_st[2] = 0.f;
		der_tt[0] = der_tt[1] = der_tt[2] = 0.f;
	}

	void AddWithWeight(Vertex const & src,
		float weight, float d1Weight, float d2Weight, float dss_w, float dst_w, float dtt_w)
	{
		for (unsigned int k=0; k<3; k++)
		{
			const float point_c = src.point[k];
			point[k] += weight * point_c;
			deriv1[k] += d1Weight * point_c;
			deriv2[k] += d2Weight * point_c;
			der_ss[k] += dss_w * point_c;
			der_st[k] += dst_w * point_c;
			der_tt[k] += dtt_w * point_c;
		}
	}

	float point[3];
	float deriv1[3], deriv2[3];
	float der_ss[3], der_st[3], der_tt[3];
};


class Mod3DfromRGBD {
public:

	//Camera
	float fovh_d, fovv_d;
	unsigned int rows, cols, downsample;
	float fx, fy;

	//Cam poses
	vector<poses::CPose3D> cam_poses;
	vector<Matrix4f> cam_trans, cam_trans_inv, cam_ini;
	vector<Matrix<float, 6, 1>> cam_mfold, cam_mfold_old, cam_pert;
	Matrix4f mat_der_xi[6]; //Generators

	//Transformations for tracking
	vector<Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_arap;
	vector<Vector3f> rot_mfold, rot_mfold_old;


	//Flags
	bool solve_DT;
	bool with_reg_normals;
	bool with_reg_normals_good;
	bool with_reg_normals_4dir;
	bool with_reg_edges;
	bool with_reg_edges_iniShape;
	bool with_reg_ctf;
	bool with_reg_atraction;
	bool with_reg_arap;
	bool with_reg_rot_arap;
	bool behind_cameras;
	bool seg_from_background;
	bool paper_visualization;
	bool paper_vis_no_mesh;
	bool optimize_cameras;
	bool small_initialization;
	bool adaptive_tau;
	bool save_energy;
	bool fix_first_camera;
	bool with_color;
	unsigned int unk_per_vertex;
	
	//Images
	unsigned int num_images;
	unsigned int image_set;
	unsigned int ctf_level;
	string im_dir, im_set_dir;
	ArrayXXf depth_background;
	vector<ArrayXXf> intensity;	//Not used
	vector<ArrayXXf> depth, x_image, y_image;
	vector<Matrix3Xf> xyz_image;
	vector<Matrix3Xf> normals_image;
	vector<VectorXf> n_weights;

	//KinectFusion data
	vector<MatrixXf> trajectory;
	vector<double> timestamps, timestamps_im;
	vector<string> files_depth, files_color;


	//Segmentation used to say whether a point belongs to the object or not (only for depth)
	vector<Array<bool, Dynamic, Dynamic>> valid;
	vector<Array<bool, Dynamic, Dynamic>> is_object;

	//Mesh topology
	unsigned int num_verts;
	unsigned int num_faces;
	Array<bool, Dynamic, 1>		is_quad;
	Array<int, 4, Dynamic>		face_verts;	//Maximum possible size if we allow for triangles and quads
	Array<int, 4, Dynamic>		face_adj;
	Array<int, 16, Dynamic>		opposite_verts;
	Array<float, 3, Dynamic>	vert_coords, vert_coords_old, vert_coords_reg;
	Array<float, 1, Dynamic>	vert_colors, vert_colors_old;
	Array<int, Dynamic, Dynamic> neighbourhood; //Each column stores all the vertices in the neigh. of a vertex (v = column)
	ArrayXi valence;
	unsigned int num_eq_arap;

	//Correspondences
	vector<ArrayXXf> u1, u1_old, u1_old_outer;
	vector<ArrayXXf> u2, u2_old, u2_old_outer;
	vector<ArrayXXi> uface, uface_old, uface_old_outer;
	vector<VectorXf> surf_color;
	vector<Matrix3Xf> surf, surf_t, surf_reg;
	vector<Matrix3Xf> normals, normals_t;
	vector<Matrix3Xf> normals_reg;
	vector<VectorXf> inv_reg_norm;


	//Increments for the unknowns, residuals and sub-surface weights
	vector<ArrayXXf> u1_incr, u2_incr;
	vector<Matrix3Xf> res_pos, res_normals;
	vector<Matrix2Xf> res_pixels;
	vector<VectorXf> res_color;
	vector<ArrayXXi> w_indices;						//Indices for non-zeros entrances (w_contverts). However, w_u1 and w_u2 have more zeros
	vector<ArrayXXf> w_contverts, w_u1, w_u2;		//New format, each cols stores the coefficients of a given pixel (cols grows with v and then jump to the next u)
	vector<Matrix3Xf> u1_der, u2_der;
	vector<VectorXf> u1_der_color, u2_der_color;
	vector<ArrayXXf> tau_pixel;

	vector<ArrayXi> w_indices_reg;
	vector<ArrayXXf> w_u1_reg, w_u2_reg, w_contverts_reg;
	vector<Matrix3Xf> u1_der_reg, u2_der_reg;
	vector<Matrix3Xf> n_der_u1, n_der_u2;


	//Parameters
	unsigned int max_iter;
	unsigned int s_reg;
	float convergence_ratio;
	float adap_mult;
	float tau_max, alpha, eps_rel;
	float truncated_res, truncated_resn; //Truncated dataterm
	float Kp, Kn, Kc;
	float Kr_total, Kr, K_ctf, K_ctf_total, K_atrac, K_atrac_total;	//Weights
	float Ke, Ke_total, K_ini, K_ini_total;
	float K_arap, K_rot_arap, K_color_reg;
	float ini_size;
	float max_depth_segmentation, max_radius_segmentation, plane_res_segmentation;

	//LM
	vector<Tri> j_elem;
	SparseMatrix<float,0> J, JtJ;	//Col-major
	VectorXf R, increments, JtR;

	//DT
	unsigned int nsamples_approx, nsamples, trunc_threshold_DT;
	vector<ArrayXXf> DT, DT_grad_u, DT_grad_v;
	ArrayXXf w_DT, w_indices_DT, w_u1_DT, w_u2_DT;
	Array<float*, Dynamic, 1> u1_der_DT, u2_der_DT;
	ArrayXf u1_dernorm_DT, u2_dernorm_DT;
	ArrayXf mx_DT, my_DT, mz_DT;
	ArrayXf nx_DT, ny_DT, nz_DT, norm_n_DT;
	ArrayXf u1_DT, u2_DT;
	ArrayXi uface_DT;
	vector<ArrayXf> pixel_DT_u, pixel_DT_v;

	//Visualization
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;
	bool					vis_errors;

	//Save results/data
	string			f_folder;
	std::ofstream	f_energy;


	//----------------------------------------------------------------------
	//								Methods
	//----------------------------------------------------------------------
	Mod3DfromRGBD(unsigned int num_im, unsigned int downsamp, unsigned int im_set);

	//Visualization
	void initializeScene();
	void initializeSceneDataArch();
	void showCamPoses();
	void showNewData(bool new_pointcloud = false);
	void showDTAndDepth();
	void showMesh();
	void showSubSurface();
	void showRenderedModel();
	void takePictureLimitSurface(bool last);
	void takePictureDataArch();
	void saveSceneAsImage();
	void drawRayAndCorrespondence(unsigned int i, unsigned int v, unsigned int u);

	//Segmentation
	void computeDepthSegmentationDepthRange(float min_d, float max_d);
	void computeDepthSegmentationGrowRegFromCenter();
	void computeDepthSegmentationClosestObject();
	void computeDepthSegmentationFromPlane();
	void computeDepthSegmentationFromBackground();
	void computeDepthSegmentationFromBoundingBox();
	void saveSegmentationToFile(unsigned int image);

	//Input - Output
	void loadImagesStandard();
	void loadInputs();
	void createImagesFromCube();
	void loadInitialMesh();
	void loadInitialMeshUShape();
	void saveMeshToFile();
	void readAssocFileKinectFusion();
	void loadPosesFromKinectFusion();
	void loadImagesFromKinectFusion();
	void loadMeshFromKinectFusion();
	void loadMeshFromFile();
	void saveSilhouetteToFile();
	void loadImageFromSequence(int im_num, bool first_image, int seq_ID);
	void createFileToSaveEnergy();
	void saveCurrentEnergyInFile(bool insert_newline);
	void saveResultsCamExperiment();

	//Others
	void computeInitialCameraPoses();
	void perturbInitialCameraPoses(float per_trans, float per_rot);
	void computeInitialUDataterm();
	void computeInitialUBackground();
	void computeDataNormals();
	void computeCameraTransfandPosesFromTwist();
	void rotMatricesFromTwist();
	void computeTransCoordAndResiduals();
	void findOppositeVerticesFace();
	void findNeighbourhoodsForArap();
	void initializeRegForFitting();
	void chooseParameterSet(unsigned int exp_ID);

	//Solvers
	void solveNB_LM_Joint(bool verbose);
	void solveSK_LM();
	void solveSK_LM_Joint(bool verbose);
	bool updateInternalPointCrossingEdges(unsigned int i, unsigned int v, unsigned int u);
	void optimizeUDataterm_LM();
	void optimizeUBackground_LM();
	void solveSK_Arap();
	void solveNB_Arap();


	//Fill J
	void fill_J_EpPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_EpPixelJoint(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_EnPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_EnPixelJoint(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_BackSKPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_BackDT2(unsigned int i, unsigned int &J_row);
	void fill_J_BackBS(unsigned int i, unsigned int &J_row);
	void fill_J_BackBG(unsigned int i, unsigned int &J_row);
	void fill_J_RegNormals(unsigned int &J_row);
	void fill_J_RegNormalsCurvature(unsigned int &J_row);
	void fill_J_RegEdges(unsigned int &J_row);
	void fill_J_RegEdgesIniShape(unsigned int &J_row);
	void fill_J_RegCTF(unsigned int &J_row);
	void fill_J_RegAtraction(unsigned int &J_row);
	void fill_J_fixFirstCamera(unsigned int &J_row);
	void fill_J_RegVertColor(unsigned int &J_row);

	void fill_J_EpArap(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_EnArap(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row);
	void fill_J_RegArap(unsigned int &J_row);
	void fill_J_RegRotArap(unsigned int &J_row);

	//Energies
	float computeEnergyRegNormals();
	float computeEnergyRegNormalsGood();
	float computeEnergyRegNormals4dir();
	float computeEnergyRegEdges();
	float computeEnergyRegEdgesIniShape();
	float computeEnergyRegCTF();
	float computeEnergyRegAtraction();
	float computeEnergyRegArap();
	float computeEnergyRegRotArap();
	float computeEnergyRegVertColor();

	float computeEnergySK();
	float computeEnergyNB();

	//Searches
	void searchBetterUDataterm();
	void searchBetterUBackground();

	//Subdivision surfaces
	Far::TopologyRefiner *refiner;
	Far::PatchTable *patchTable;
	std::vector<Vertex> verts;
	std::vector<Vertex_color> verts_c;
	void createTopologyRefiner();
	void evaluateSubDivSurface();
	void evaluateSubDivSurfaceRegularization();
	void evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u);
	//void computeNormalDerivatives_FinDif(unsigned int i, unsigned int v, unsigned int u);
	void computeNormalDerivatives_Analyt(unsigned int i, unsigned int v, unsigned int u);
	void refineMeshOneLevel();
	void refineMeshToShow();

	//Distance transform
	void sampleSurfaceForDTBackground();
	//void computeDistanceTransform();
	void computeDistanceTransformOpenCV(bool safe_DT);
	void solveDT2_LM();
	void solveDT2_LM_Joint(bool verbose);
	void solveDT2_Arap();
	float computeEnergyDT2();

	//Minimum-Background-surface term (BS)
	void sampleSurfaceForBSTerm();
	void evaluateSurfaceForBSSamples();
	void computeTruncatedDTOpenCV();
	void solveBS_LM_Joint(bool verbose);
	void solveBS_Arap();
	float computeEnergyBS();

	//Minimum-Background-gradients term (BG)
	void evaluateSurfaceForBGSamples();
	void solveBG_LM_Joint(bool verbose);
	void solveBG_Arap();
	float computeEnergyBG();

};



//									Info
//-------------------------------------------------------------------------------------------------
// Regularization is imposed twice between some samples on the edges of faces (I mean in "reg_normals") 
// DT: samples do not interpolate between the pixels values (after projection), they are rounded to the closest
// Problem with normals with the first level (cube) -> Instability in the optimization, observed always after the search
// Problem with the sparse solver and the truncated dataterms: some correspondences can be underdetermined (no equations for them) which breaks the solver.
//      Solution: I introduce some gradient and zero residual to constrain the unknown (it should stay still)

// Idea 1: regularization on the correspondences to avoid wrong associations and irreversible deformations of the surface. It is complex!
// Idea 3: performance: Build a sparse JtJ directly (then we would save the costly J.transpose()*J -> I tried it and it was not better...)

