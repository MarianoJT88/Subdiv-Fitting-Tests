// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "KinectFusion_datatest.h"


Mod3DfromRGBD::Mod3DfromRGBD(unsigned int num_im, unsigned int downsamp, unsigned int im_set)
{
	num_images = num_im;
	image_set = im_set;
	downsample = downsamp; //It can be set to 1, 2, 4, etc.

	fovh_d = utils::DEG2RAD(58.6f); fovv_d = utils::DEG2RAD(45.6f);
	rows = 480/downsample; cols = 640/downsample;
		
	fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	fy = float(rows) / (2.f*tan(0.5f*fovv_d));

	ctf_level = 1;


	//Cameras
	cam_poses.resize(num_images);
	cam_trans.resize(num_images);
	cam_trans_inv.resize(num_images);
	cam_mfold.resize(num_images); cam_mfold_old.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	depth_background.resize(rows,cols);
	intensity.resize(num_images);
	depth.resize(num_images); x_image.resize(num_images); y_image.resize(num_images);
	nx_image.resize(num_images); ny_image.resize(num_images); nz_image.resize(num_images); n_weights.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		intensity[i].resize(rows, cols);
		depth[i].resize(rows, cols); x_image[i].resize(rows, cols); y_image[i].resize(rows, cols);
		nx_image[i].resize(rows, cols); ny_image[i].resize(rows, cols); nz_image[i].resize(rows, cols); n_weights[i].resize(rows*cols);
		is_object[i].resize(rows, cols); valid[i].resize(rows, cols);
	}

	//Correspondences
	u1.resize(num_images); u2.resize(num_images); uface.resize(num_images); 
	mx.resize(num_images); my.resize(num_images); mz.resize(num_images);
	nx.resize(num_images); ny.resize(num_images); nz.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1[i].resize(rows, cols); u2[i].resize(rows, cols); uface[i].resize(rows, cols); 
		mx[i].resize(rows, cols); my[i].resize(rows, cols); mz[i].resize(rows, cols);
		nx[i].resize(rows, cols); ny[i].resize(rows, cols); nz[i].resize(rows, cols);
	}	
}

void Mod3DfromRGBD::loadPoses()
{
	string dir, name;
	if (image_set == 1)			dir = im_dir + "dataset_robert/";

	//Open Trajectory file
	std::ifstream	f_trajectory;
	string filename = dir;
	filename.append("robert_traj.txt");

	cout << endl << "Filename: " << filename;
	f_trajectory.open(filename.c_str());

	if (f_trajectory.fail())
		throw std::runtime_error("\nError finding the trajectory file.");

	// first load all groundtruth timestamps and poses
    std::string line;
    while (std::getline(f_trajectory, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestamp;
        float tx, ty, tz;
        float qx, qy, qz, qw;
        if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
            break;

        timestamps.push_back(timestamp);

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        Eigen::Vector3f tVec(tx, ty, tz);
        pose.topRightCorner(3,1) = tVec;
        //Eigen::Quaternionf quat(qw, qx, qy, qz);
		Eigen::Quaternionf quat(qw, qx, qy, qz);
        pose.topLeftCorner(3,3) = quat.toRotationMatrix();

		CPose3D pose_corr = CPose3D(0,0,0,0,-M_PI/2, M_PI/2);
		CMatrixDouble44 mat_aux; pose_corr.getHomogeneousMatrix(mat_aux);
		Matrix4f mat_corr = mat_aux.cast<float>();


		MatrixXf aux_pose = pose*mat_corr;
        trajectory.push_back(aux_pose);
    }

    // align all poses so that initial pose is identity matrix
	bool firstPoseIsIdentity = false;
    if (firstPoseIsIdentity && !trajectory.empty())
    {
        Eigen::Matrix4f initPose = trajectory[0];

		//Correction
        for (int i = 0; i < trajectory.size(); ++i)
            trajectory[i] = initPose.inverse() * trajectory[i];
    }

	f_trajectory.close();
}

void Mod3DfromRGBD::loadImages()
{
	string dir, name;
	if (image_set == 1)			dir = im_dir + "dataset_robert/";
	
	float depth_scale = 1.f / 5000.f;
	float depth_offset = 0.f;

	char aux[30];
	const float norm_factor = 1.f/255.f;
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	//Images
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Depth
		float ind = i*(trajectory.size()-2)/(num_images) + 1;
		sprintf_s(aux, "robert2/depth/%.6f.png", ind);
		name = dir + aux;

		cout << endl << "Filename: " << name;

		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		const float disp_v = 0.5f*(rows - 1);

		cv::Mat im_d = cv::imread(name, -1);
		cv::Mat depth_float;

		im_d.convertTo(depth_float, CV_32FC1, depth_scale);
		for (unsigned int u = 0; u<cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample) + depth_offset;
				depth[i](v, u) = d;
				x_image[i](v, u) = (u - disp_u)*d*inv_fd;
				y_image[i](v, u) = (v - disp_v)*d*inv_fd;
			}

		//Load their poses into the right variables
		cam_ini[i] = trajectory.at(ind).block<4,4>(0,0);
		CMatrixDouble44 mat = cam_ini[i];
		cam_poses[i] = CPose3D(mat);
	}

}

void Mod3DfromRGBD::loadMesh()
{
	//Open file
	string dir;
	if (image_set == 1)			dir = im_dir + "dataset_robert/";

	std::ifstream	f_mesh;
	dir.append("robert_212_5.obj");

	cout << endl << "Filename: " << dir;
	f_mesh.open(dir.c_str());
	
	
	//Detect the number of vertices and faces
	num_verts = 0;
	num_faces = 0;
	string line; 
	while (std::getline(f_mesh, line))
    {
        //const char* token = line + strspn(line, " \t"); // ignore space and tabs
		if (line.at(0) == 'v' && line.at(1) == ' ')
			num_verts++;

		if (line.at(0) == 'f')
			num_faces++;
	}
	printf("\nReading mesh: num_verts = %d, num_faces = %d", num_verts, num_faces);


	//Load the vertices
	vert_coords.resize(3, num_verts);

	f_mesh.clear();
	f_mesh.seekg(0);
	bool is_vert = true;
	unsigned int cont = 0;
	printf("\nReading vertices...");

	while (is_vert)
	{
		std::getline(f_mesh, line);
		if (line.at(0) == 'v' && line.at(1) == ' ') 
		{
			std::istringstream iss(line);
			char aux_v;
			iss >> aux_v >> vert_coords(0,cont) >> vert_coords(1,cont) >> vert_coords(2,cont);	
			cont++;
		}
		else
			is_vert = false;
	}

	//Load the faces
	face_verts.resize(4, num_faces);
	cont = 0;
	printf("\nReading faces...");

	while (line.at(0) == 'v')
		std::getline(f_mesh, line);

	do
	{
		std::istringstream iss(line);
		char aux_c1, aux_c2; int face_id;
		iss >> aux_c1;
		for (unsigned int k=0; k<4; k++)
		{
			iss >> face_verts(k,cont) >> aux_c1 >> aux_c2 >> face_id;
			//printf("\n Read: %d %c %c %d", face_verts(k,cont), aux_c1, aux_c2, face_id);
		}
		
		cont++;

	} while (std::getline(f_mesh, line));

	face_verts -= 1;
         


	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);


	//Find the adjacent faces to every face
	face_adj.resize(4, num_faces);
	face_adj.fill(-1);
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int k = 0; k < 4; k++)
		{
			unsigned int kinc = k + 1; if (k + 1 == 4) kinc = 0;
			float edge[2] = {face_verts(k,f), face_verts(kinc,f)};
			for (unsigned int fa = 0; fa < num_faces; fa++)
			{
				if (f == fa) continue;
				char found = 0;
				for (unsigned int l = 0; l < 4; l++)
					if ((face_verts(l, fa) == edge[0]) || (face_verts(l, fa) == edge[1]))
						found++;

				if (found > 1)
				{
					face_adj(k, f) = fa;
					break;
				}
			}
		}
}

void Mod3DfromRGBD::computeDepthSegmentationClosestObject()
{
	vector<Array2i> buffer;
	Array<bool,Dynamic,Dynamic> checked(rows,cols);
	const float dist_thres = 0.0001f; //It is actually a threshold for the distance square(^2)
	
	//For every image, we find the closest area and expand it. 
	//We also check it afterwards and reject it if it does not fulfill certain conditions
	for (unsigned int i = 0; i < num_images; i++)
	{
		checked.fill(false);
		valid[i].fill(true);
		
		//Find the closest point
		float min_dist = 10.f;
		Array2i min_coords; min_coords.fill(0);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (depth[i](v, u) > 0.f)
				{
					const float dist_origin = square(depth[i](v, u)) + square(x_image[i](v, u)) + square(y_image[i](v, u));
					if (dist_origin < min_dist)
					{
						min_dist = dist_origin;
						min_coords(0) = v; min_coords(1) = u;
					}
				}
				else
					valid[i](v, u) = false;

		//Expand from the closest point
		buffer.push_back(min_coords);
		while (!buffer.empty())
		{
			const int v = buffer.back()(0);
			const int u = buffer.back()(1);
			buffer.pop_back();
			checked(v, u) = true;

			for (int k = -1; k < 2; k++)
				for (int l = -1; l < 2; l++)
				{
					Array2i ind; ind(0) = v + l; ind(1) = u + k;

					if ((ind(0) < 0) || (ind(1) < 0) || (ind(0) >= rows) || (ind(1) >= cols))
						continue;

					else if (checked(ind(0), ind(1)))
						continue;

					const float dist = square(depth[i](v, u) - depth[i](ind(0), ind(1)))
									+ square(x_image[i](v, u) - x_image[i](ind(0), ind(1)))
									+ square(y_image[i](v, u) - y_image[i](ind(0), ind(1)));

					if (dist < dist_thres)
						buffer.push_back(ind);
				}
		}

		//Copy result to the segmentation
		is_object[i].swap(checked);
	}
}

void Mod3DfromRGBD::computeDepthSegmentationFromPlane()
{
	const float max_depth = 1.f;
	const float res_threshold = 0.05f;
	vector<Vector3f> alpha; alpha.resize(num_images);

	//RANSAC
	//E(a) = sum((a1*x + a2*y + a3*z - 1)^2)
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		std::vector<float> depth_sorted;
		
		//First find the number of pixels used
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) < max_depth))
					depth_sorted.push_back(depth[i](v,u));

		std::sort(depth_sorted.begin(), depth_sorted.end());

		const float percent_furthest = 0.1f;
		const float depth_threshold = depth_sorted[ceil((1.f - percent_furthest)*depth_sorted.size())];

		vector<Vector3f> inliers;
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if ((depth[i](v,u) > depth_threshold)&&(depth[i](v,u) < max_depth))
				{
					Vector3f point; point <<  depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					inliers.push_back(point);
				}

		const unsigned int num_iter_ransac = 5;

		for (unsigned int r=0; r<num_iter_ransac; r++)
		{
			//Create matrices to solve the system
			unsigned int num_points = inliers.size();
			MatrixXf A(num_points, 3);
			VectorXf b(num_points); b.fill(1.f);
			VectorXf residuals(num_points); residuals.fill(1.f);

			for (unsigned int k=0; k<num_points; k++)
				A.row(k) = inliers[k].transpose();

			MatrixXf AtA, AtB;
			AtA.multiply_AtA(A);
			AtB.multiply_AtB(A,b);
			alpha[i] = AtA.ldlt().solve(AtB);
			residuals = A*alpha[i]-b;

			//Find new inliers
			inliers.clear();
			for (unsigned int u=0; u<cols; u++)
				for (unsigned int v=0; v<rows; v++)
					if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) < max_depth))
					{
						Vector3f point; point <<  depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
						const float res = abs((alpha[i].transpose()*point).value() - 1.f);
						if (res < res_threshold)
							inliers.push_back(point);
					}
		}


		//Check which points belong to the plane
		is_object[i].fill(true);
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) < max_depth))
				{
					Matrix<float, 1, 3> r; r<< depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					if (abs((r*alpha[i]).value() - 1.f) < res_threshold)
						is_object[i](v,u) = false;
				}
				else if ((depth[i](v,u) == 0.f)||(depth[i](v,u) > max_depth))
					is_object[i](v,u) = false;
			}

		//Filter the mask to reject false positives
		//First, reject those points far from the center of gravity of the object
		Array3f cent_of_grav; cent_of_grav.fill(0.f);
		unsigned int num_points = 0;
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
				{
					Array3f point; point << depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					cent_of_grav += point;
					num_points++;
				}
		cent_of_grav /= num_points;

		const float max_radius = 0.3f;
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
					if (sqrtf(square(depth[i](v,u) - cent_of_grav(0)) + square(x_image[i](v,u) - cent_of_grav(1)) +  square(y_image[i](v,u) - cent_of_grav(2))) > max_radius)
						is_object[i](v,u) = false;

		//Build invalid mask
		valid[i].fill(true);
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (depth[i](v,u) == 0.f)
					valid[i](v,u) = false;
	}

	////I check whether it works or not...
	//global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	//window.resize(1000, 900);
	//window.setPos(900, 0);
	//window.setCameraZoom(3);
	//window.setCameraAzimuthDeg(0);
	//window.setCameraElevationDeg(45);
	//window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	//scene = window.get3DSceneAndLock();

	//// Lights:
	//scene->getViewport()->setNumberOfLights(2);
	//mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	//light0.light_ID = 0;
	////light0.setPosition(2.5f,0,0.f,1.f);
	//light0.setDirection(0.f, 0.f, -1.f);

	//mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	//light1.light_ID = 1;
	//light1.setPosition(0.0f, 0, 0.f, 1.f);
	//light1.setDirection(0.f, 1.f, 0.f);

	////Reference
	//opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	//reference->setScale(0.2f);
	//scene->insert(reference);

	////Points
	//CPose3D pose_aux = CPose3D(0,0,0,0,0,0);
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//	points->setPointSize(3.f);
	//	points->enablePointSmooth(true);
	//	points->setPose(pose_aux);
	//	scene->insert(points);

	//	//Insert points
	//	//float r, g, b;
	//	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//	for (unsigned int v = 0; v < rows; v++)
	//		for (unsigned int u = 0; u < cols; u++)
	//		{
	//			if (is_object[i](v, u))
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 1.f, 0.f, 0.f);
	//			else
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 0.1f, 0.1f, 0.1f);
	//		}

	//	//Sample the plane
	//	opengl::CPointCloudColouredPtr plane = opengl::CPointCloudColoured::Create();
	//	plane->setPointSize(3.f);
	//	plane->enablePointSmooth(true);
	//	plane->setPose(pose_aux);
	//	scene->insert(plane);
	//	for (unsigned int s = 0; s < 20; s++)
	//		for (unsigned int t = 0; t < 20; t++)
	//		{
	//			const float x = 0.f + s*0.1f;
	//			const float y = -0.5f + t*0.05f;
	//			const float z = (1.f - alpha[i](0)*x - alpha[i](1)*y)/alpha[i](2);
	//			plane->push_back(x, y, z, 0.f , 1.f, 0.f);
	//		}

	//	pose_aux.y_incr(1.5f);
	//}

	//window.unlockAccess3DScene();
	//window.repaint();
	//system::os::getch();
}

void Mod3DfromRGBD::computeSegmentationFromBoundingBox()
{
	const float cube_edge = 1.f;
	
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (depth[i](v,u) == 0.f)
				{
					valid[i](v,u) = false;
					is_object[i](v,u) = true;
				}
				else
				{
					valid[i](v,u) = true;

					//Transform the points to the global coordinate system
					const float x_t = trajectory[i](0,0)*depth[i](v,u) + trajectory[i](0,1)*x_image[i](v,u) + trajectory[i](0,2)*y_image[i](v,u) + trajectory[i](0,3);
					const float y_t = trajectory[i](1,0)*depth[i](v,u) + trajectory[i](1,1)*x_image[i](v,u) + trajectory[i](1,2)*y_image[i](v,u) + trajectory[i](1,3);
					const float z_t = trajectory[i](2,0)*depth[i](v,u) + trajectory[i](2,1)*x_image[i](v,u) + trajectory[i](2,2)*y_image[i](v,u) + trajectory[i](2,3);

					if ((abs(x_t) < 0.5f*cube_edge)&&(abs(y_t) < 0.5f*cube_edge)&&(abs(z_t) < 0.5f*cube_edge))
						is_object[i](v,u) = true;
					else
						is_object[i](v,u) = false;			
				}					
			}
}


void Mod3DfromRGBD::saveSegmentationToFile(unsigned int image)
{
	string dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/";
	string name;
	char aux[30];
	
	//Save segmentation
	sprintf_s(aux, "seg_image.png");
	name = dir + aux;
	cv::Mat seg_cv(rows, cols, CV_16U);
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
			if (is_object[image](v,u) == true)
				seg_cv.at<unsigned short>(rows-1-v,u) = 0;
			else if (valid[image](v,u) == true)
				seg_cv.at<unsigned short>(rows-1-v,u) = 10000;
			else
				seg_cv.at<unsigned short>(rows-1-v,u) = 20000;
		}

	cv::imwrite(name.c_str(), seg_cv);
}

void Mod3DfromRGBD::computeDataNormals()
{
	ArrayXXf rx_ninv(rows, cols), ry_ninv(rows, cols);
	ArrayXXf dx_u(rows, cols), dy_u(rows, cols);
	ArrayXXf dx_v(rows, cols), dz_v(rows, cols);
	for (unsigned int i = 0; i<num_images; i++)
	{
		//Compute connectivity
		rx_ninv.fill(1.f); ry_ninv.fill(1.f);
		for (unsigned int u = 0; u < cols-1; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				const float norm_dx = square(x_image[i](v, u+1) - x_image[i](v, u))
									+ square(depth[i](v, u+1) - depth[i](v, u));
				if (norm_dx  > 0.f)
					rx_ninv(v, u) = norm_dx; // sqrt(norm_dx);
			}

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows-1; v++)
			{
				const float norm_dy = square(y_image[i](v+1, u) - y_image[i](v, u))
									+ square(depth[i](v+1, u) - depth[i](v, u));

				if (norm_dy > 0.f)
					ry_ninv(v, u) = norm_dy; // sqrt(norm_dy);
			}

		//Spatial derivatives
		for (unsigned int v = 0; v < rows; v++)
		{
			for (unsigned int u = 1; u < cols-1; u++)
				if (is_object[i](v,u))
				{
					dx_u(v, u) = (rx_ninv(v, u-1)*(depth[i](v, u+1) - depth[i](v, u)) + rx_ninv(v, u)*(depth[i](v, u) - depth[i](v, u-1)))/(rx_ninv(v, u)+rx_ninv(v, u-1));
					dy_u(v, u) = (rx_ninv(v, u-1)*(x_image[i](v, u+1) - x_image[i](v, u)) + rx_ninv(v, u)*(x_image[i](v, u) - x_image[i](v, u-1)))/(rx_ninv(v, u)+rx_ninv(v, u-1));
				}
			dx_u(v, 0) = dx_u(v, 1); dx_u(v, cols-1) = dx_u(v, cols-2);
			dy_u(v, 0) = dy_u(v, 1); dy_u(v, cols-1) = dy_u(v, cols-2);
		}

		for (unsigned int u = 0; u < cols; u++)
		{
			for (unsigned int v = 1; v < rows-1; v++)
				if (is_object[i](v,u))
				{
					dx_v(v, u) = (ry_ninv(v-1, u)*(depth[i](v+1, u) - depth[i](v, u)) + ry_ninv(v, u)*(depth[i](v, u) - depth[i](v-1, u)))/(ry_ninv(v, u)+ry_ninv(v-1, u));
					dz_v(v, u) = (ry_ninv(v-1, u)*(y_image[i](v+1, u) - y_image[i](v, u)) + ry_ninv(v, u)*(y_image[i](v, u) - y_image[i](v-1, u)))/(ry_ninv(v, u)+ry_ninv(v-1, u));
				}

			dx_v(0, u) = dx_v(1, u); dx_v(rows-1, u) = dx_v(rows-2, u);
			dz_v(0, u) = dz_v(1, u); dz_v(rows-1, u) = dz_v(rows-2, u);
		}

		//Find the normals
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v,u))	
				{
					const float v1[3] = {dx_v(v, u), 0.f, dz_v(v, u)};
					const float v2[3] = {dx_u(v, u), dy_u(v, u), 0.f};

					const float nx = v1[1] * v2[2] - v1[2] * v2[1];
					const float ny = v1[2] * v2[0] - v1[0] * v2[2];
					const float nz = v1[0] * v2[1] - v1[1] * v2[0];
					const float norm = sqrtf(nx*nx + ny*ny + nz*nz);

					if (norm > 0.f)
					{
						nx_image[i](v, u) = nx/norm;
						ny_image[i](v, u) = ny/norm;
						nz_image[i](v, u) = nz/norm;
					}
					else
					{
						nx_image[i](v, u) = 0.f;
						ny_image[i](v, u) = 0.f;
						nz_image[i](v, u) = 0.f;
					}
				}

		//Filter
		//-----------------------------------------------------
		//Compute gaussian mask
		float v_mask[5] = {1, 4, 6, 4, 1};
		Matrix<float, 5, 5> mask;
		for (unsigned int k = 0; k<5; k++)
			for (unsigned int j = 0; j<5; j++)
				mask(k,j) = v_mask[k]*v_mask[j]/256.f;

		//Apply the filter N times
		const unsigned int num_times_filter = round(max(1.f, log2f(rows/30)));
		const float max_dist = 0.1f; //This value should be a function of the size of the object!!!

		for (unsigned int k=1; k<=num_times_filter; k++)
			for (int u = 0; u < cols; u++)
				for (int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						float n_x = 0.f, n_y = 0.f, n_z = 0.f, sum = 0.f;
						for (int k = -2; k<3; k++)
						for (int l = -2; l<3; l++)
						{
							const int ind_u = u+k, ind_v = v+l;
							if ((ind_u >= 0)&&(ind_u < cols)&&(ind_v >= 0)&&(ind_v < rows)&&(is_object[i](ind_v,ind_u)))
							{
								const float abs_dist = sqrtf(square(depth[i](ind_v,ind_u) - depth[i](v,u)) + square(x_image[i](ind_v,ind_u) - x_image[i](v,u)) + square(y_image[i](ind_v,ind_u) - y_image[i](v,u)));

								if (abs_dist < max_dist)
								{
									const float aux_w = mask(l+2, k+2)*(max_dist - abs_dist);
									n_x += aux_w*nx_image[i](ind_v, ind_u);
									n_y += aux_w*ny_image[i](ind_v, ind_u);
									n_z += aux_w*nz_image[i](ind_v, ind_u);
									sum += aux_w;
								}
							}
						}

						nx_image[i](v,u) = n_x/sum;
						ny_image[i](v,u) = n_y/sum;
						nz_image[i](v,u) = n_z/sum;

						//Renormalize
						const float inv_norm = 1.f/sqrtf(square(nx_image[i](v,u)) + square(ny_image[i](v,u)) + square(nz_image[i](v,u)));
						nx_image[i](v,u) *= inv_norm;
						ny_image[i](v,u) *= inv_norm;
						nz_image[i](v,u) *= inv_norm;
					}
	}

	//Compute normal weights using distances (depth) of the surrounding pixels
	//----------------------------------------------------------------------------
	//Compute average distance between the points:
	float aver_dist;
	unsigned int cont = 0;
	for (unsigned int i=0; i<num_images; i++)
		for (int u = 0; u < cols-1; u++)
			for (int v = 0; v < rows-1; v++)
				if (is_object[i](v,u) && is_object[i](v+1,u) && is_object[i](v,u+1))
				{
					aver_dist += abs(depth[i](v,u) - depth[i](v+1,u)) + abs(depth[i](v,u) - depth[i](v,u+1));
					cont += 2;
				}

	aver_dist /= cont;
	const float w_constant = 0.000025f/square(aver_dist);	

	//Compute the weights
	for (unsigned int i=0; i<num_images; i++)
		for (int u = 0; u < cols; u++)
			for (int v = 0; v < rows; v++)
				if (is_object[i](v,u))
				{
					float sum_dist = 0.f;
					
					for (int k = -1; k<2; k++)
					for (int l = -1; l<2; l++)
					{
						const int ind_u = u+k, ind_v = v+l;
						if ((ind_u >= 0)&&(ind_u < cols)&&(ind_v >= 0)&&(ind_v < rows))
							sum_dist += sqrtf(square(depth[i](ind_v,ind_u) - depth[i](v,u)));
					}

					n_weights[i](v +rows*u) = exp(-w_constant*sum_dist); 
				}
}

void Mod3DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(max(int(num_images), 6));
	cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

	if (image_set == 1) //bananas
	{	
		cam_poses[1].setFromValues(0.f, -0.19f, 0.03f, utils::DEG2RAD(20.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.35f, 0.35f, -0.11f, utils::DEG2RAD(-65.8f), utils::DEG2RAD(-11.44f), utils::DEG2RAD(-20.1f));
		cam_poses[3].setFromValues(0.61f, 0.33f, -0.19f, utils::DEG2RAD(-97.f), utils::DEG2RAD(-20.f), utils::DEG2RAD(-43.f));
		cam_poses[4].setFromValues(0.71f, -0.33f, -0.28f, utils::DEG2RAD(158.f), utils::DEG2RAD(-49.f), utils::DEG2RAD(11.5f));
	}

	//Get and store the initial transformation matrices
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Transformation
		CMatrixDouble44 aux;
		cam_poses[i].getHomogeneousMatrix(aux);
		cam_trans[i] = aux.cast<float>();

		const Matrix3f rot_mat = cam_trans[i].block<3, 3>(0, 0).transpose();
		const Vector3f tra_vec = cam_trans[i].block<3, 1>(0, 3);

		cam_trans_inv[i].topLeftCorner<3, 3>() = rot_mat;
		cam_trans_inv[i].block<3, 1>(0, 3) = -rot_mat*tra_vec;
		cam_trans_inv[i].row(3) << 0.f, 0.f, 0.f, 1.f;
		cam_ini[i] = cam_trans_inv[i].topLeftCorner<4, 4>();
		cam_mfold[i].assign(0.f);
	}

	showCamPoses();
}

void Mod3DfromRGBD::computeCameraTransfandPosesFromTwist()
{
	for (unsigned int i=0; i<num_images; i++)
	{
		Matrix4f kai_mat;
		kai_mat << 0.f, -cam_mfold[i](5), cam_mfold[i](4), cam_mfold[i](0),
				cam_mfold[i](5), 0.f, -cam_mfold[i](3), cam_mfold[i](1),
				-cam_mfold[i](4), cam_mfold[i](3), 0.f, cam_mfold[i](2),
				0.f, 0.f, 0.f, 0.f;

		const Matrix4f new_trans = kai_mat.exp();
		const MatrixXf prod = new_trans*cam_ini[i];
		cam_trans_inv[i] = prod.topLeftCorner<4, 4>(); //It don't know why but it crashes if I do the assignment directly

		const Matrix3f rot_mat = cam_trans_inv[i].block<3, 3>(0, 0).transpose();
		const Vector3f tra_vec = cam_trans_inv[i].block<3, 1>(0, 3);
		cam_trans[i].topLeftCorner<3, 3>() = rot_mat;
		cam_trans[i].block<3, 1>(0, 3) = -rot_mat*tra_vec;
		cam_trans[i].row(3) << 0.f, 0.f, 0.f, 1.f;

		CMatrixDouble44 mat = cam_trans[i];
		cam_poses[i] = CPose3D(mat);
	}
}



void Mod3DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1200, 1000); //window.resize(1000, 900);
	window.setPos(200, 0); //window.setPos(900, 0);
	window.setCameraZoom(1.4f); //window.setCameraZoom(3); //Exp2 - 1.8, 1.8, 2.0; Exp3 = 1.4;
	window.setCameraAzimuthDeg(225);//window.setCameraAzimuthDeg(0); //Exp2 = 205; Exp3 = 225;
	window.setCameraElevationDeg(25);//window.setCameraElevationDeg(45);	//Exp2 = 8; //Exp3 = 25;
	window.setCameraPointingToPoint(0.6f, 0.f, 0.f);//window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	scene = window.get3DSceneAndLock();

	// Lights:
	scene->getViewport()->setNumberOfLights(2);
	mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	light0.light_ID = 0;
	light0.setPosition(-0.5f, 0.f, 0.5f, 0.f);

	mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	light1.light_ID = 1;
	//light1.setPosition(0.0f, 0, 0.f, 1.f);
	//light1.setDirection(0.f, 1.f, 0.f);


	//Extra viewport for the distance transform
	//COpenGLViewportPtr gl_view_aux = scene->createViewport("DT");
	//gl_view_aux->setViewportPosition(10, 10, 300, 200);
	//utils::CImage DT_image;
	//DT_image.setFromMatrix(DT[0].matrix(), false);
	//DT_image.flipVertical();
	//DT_image.normalize();
	//gl_view_aux->setImageView(DT_image);


	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	control_mesh->setLineWidth(2.f);
	control_mesh->setPointSize(8.f);
	control_mesh->setEdgeColor(0.9f, 0.f, 0.f);
	control_mesh->setVertColor(0.6f, 0.f, 0.f);
	scene->insert(control_mesh);

	////Vertex numbers
	//for (unsigned int v = 0; v < 2000; v++)
	//{
	//	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	//	vert_nums->setString(std::to_string(v));
	//	vert_nums->setScale(0.02f);
	//	vert_nums->setColor(0.5, 0, 0);
	//	scene->insert(vert_nums);
	//}


	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	reference->setScale(0.05f);
	scene->insert(reference);

	//Frustums
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CFrustumPtr frustum = opengl::CFrustum::Create(0.01f, 0.1f, utils::RAD2DEG(fovh_d), utils::RAD2DEG(fovv_d), 1.5f, true, false);
	//	frustum->setColor(0.4f, 0.f, 0.f);
	//	scene->insert(frustum);
	//}

	//Points
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
		points->setPointSize(4.f);
		points->enablePointSmooth(true);
		scene->insert(points);
	}

	//Surface normals
	const float fact = 0.01f;
	for (unsigned int i = 0; i < num_images; i++)
	{	
		opengl::CSetOfLinesPtr normals = opengl::CSetOfLines::Create();
		normals->setColor(0, 0, 0.8f);
		normals->setLineWidth(1.f);
		scene->insert(normals);

		////Insert points (they don't change through the optimization process)
		//float r, g, b;
		//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
		//normals->setColor(r, g, b);
		//for (unsigned int v = 0; v < rows; v++)
		//	for (unsigned int u = 0; u < cols; u++)
		//	if (is_object[i](v, u))
		//		normals->appendLine(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 
		//		depth[i](v, u) + fact*nx_image[i](v, u), x_image[i](v, u) + fact*ny_image[i](v, u), y_image[i](v, u) + fact*nz_image[i](v, u));
	}


	//3D Model
	opengl::CMesh3DPtr model = opengl::CMesh3D::Create();
	model->setPose(CPose3D(0.f, 1.5f, 0.f, 0.f, 0.f, 0.f));
	model->enableShowVertices(false);
	model->enableShowEdges(false);
	model->enableShowFaces(true);
	model->enableFaceNormals(true);
	//model->enableTransparency(true);
	model->setFaceColor(0.7f, 0.7f, 0.8f, 1.f);
	scene->insert(model);


	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showRenderedModel()
{
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;

	const float num_ref = 4;
	
	for (unsigned int r=0; r<num_ref-ctf_level; r++)
	{
		typedef Far::TopologyDescriptor Descriptor;

		Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
		Sdc::Options options;
		options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);

		//Fill the topology of the mesh
		Descriptor desc;
		desc.numVertices = num_verts_now;
		desc.numFaces = num_faces_now;

		int *vertsperface; vertsperface = new int[num_faces_now];
		for (unsigned int i = 0; i < num_faces_now; i++)
			vertsperface[i] = 4;

		desc.numVertsPerFace = vertsperface;
		desc.vertIndicesPerFace = face_verts_now.data();

		//Instantiate a FarTopologyRefiner from the descriptor.
		refiner_now = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
			Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

		// Uniformly refine the topolgy once
		refiner_now->RefineUniform(Far::TopologyRefiner::UniformOptions(1));
		const Far::TopologyLevel mesh_level = refiner_now->GetLevel(1);

		// Allocate and fill a buffer for the old vertex primvar data
		vector<Vertex> old_verts; old_verts.resize(num_verts_now);
		for (unsigned int v = 0; v<num_verts_now; v++)
			old_verts[v].SetPosition(vert_coords_now(0, v), vert_coords_now(1, v), vert_coords_now(2, v));

		// Allocate and fill a buffer for the new vertex primvar data
		verts_now.resize(mesh_level.GetNumVertices());
		Vertex *src = &old_verts[0], *dst = &verts_now[0];
		Far::PrimvarRefiner(*refiner_now).Interpolate(1, src, dst);


		//									Create the new mesh from it
		//---------------------------------------------------------------------------------------------------------
		num_verts_now = mesh_level.GetNumVertices();
		num_faces_now = mesh_level.GetNumFaces();

		//Fill the vertices per face
		face_verts_now.resize(4, num_faces_now);
		for (unsigned int f = 0; f < num_faces_now; f++)
		{
			Far::ConstIndexArray face_v = mesh_level.GetFaceVertices(f);
			for (unsigned int v = 0; v < face_v.size(); v++)
				face_verts_now(v, f) = face_v[v];
		}

		//Fill the 3D coordinates of the vertices
		vert_coords_now.resize(3, num_verts_now); 
		for (unsigned int v = 0; v < verts_now.size(); v++)
		{
			vert_coords_now(0, v) = verts_now[v].point[0];
			vert_coords_now(1, v) = verts_now[v].point[1];
			vert_coords_now(2, v) = verts_now[v].point[2];
		}
	}


	scene = window.get3DSceneAndLock();

	//Coordinates conversion
	const Matrix4f &mytrans_inv = cam_trans_inv[3];

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	//for (unsigned int cv = 0; cv < num_verts_now; cv++)
	//{
	//	const float vx = mytrans_inv(0, 0)*vert_coords_now(0, cv) + mytrans_inv(0, 1)*vert_coords_now(1, cv) + mytrans_inv(0, 2)*vert_coords_now(2, cv) + mytrans_inv(0, 3);
	//	const float vy = mytrans_inv(1, 0)*vert_coords_now(0, cv) + mytrans_inv(1, 1)*vert_coords_now(1, cv) + mytrans_inv(1, 2)*vert_coords_now(2, cv) + mytrans_inv(1, 3);
	//	const float vz = mytrans_inv(2, 0)*vert_coords_now(0, cv) + mytrans_inv(2, 1)*vert_coords_now(1, cv) + mytrans_inv(2, 2)*vert_coords_now(2, cv) + mytrans_inv(2, 3);

	//	vert_coords_now(0,cv) = vx;
	//	vert_coords_now(1,cv) = vy;
	//	vert_coords_now(2,cv) = vz;
	//}
	is_quad.resize(num_faces_now, 1); is_quad.fill(true);
	model->loadMesh(num_verts_now, num_faces_now, is_quad, face_verts_now, vert_coords_now);


	//opengl::CMesh3DPtr mesh = scene->getByClass<CMesh3D>(0);

	//vert_coords_now.resize(3, num_verts);
	//for (unsigned int cv = 0; cv < num_verts; cv++)
	//{
	//	vert_coords_now(0,cv) = mytrans_inv(0, 0)*vert_coords(0, cv) + mytrans_inv(0, 1)*vert_coords(1, cv) + mytrans_inv(0, 2)*vert_coords(2, cv) + mytrans_inv(0, 3);
	//	vert_coords_now(1,cv) = mytrans_inv(1, 0)*vert_coords(0, cv) + mytrans_inv(1, 1)*vert_coords(1, cv) + mytrans_inv(1, 2)*vert_coords(2, cv) + mytrans_inv(1, 3);
	//	vert_coords_now(2,cv) = mytrans_inv(2, 0)*vert_coords(0, cv) + mytrans_inv(2, 1)*vert_coords(1, cv) + mytrans_inv(2, 2)*vert_coords(2, cv) + mytrans_inv(2, 3);
	//}
	//is_quad.resize(num_faces, 1); is_quad.fill(true);

	//mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords_now);


	window.unlockAccess3DScene();
	window.repaint();	

	system::sleep(10);
}

void Mod3DfromRGBD::showCamPoses()
{
	scene = window.get3DSceneAndLock();

	for (unsigned int k=0; k<trajectory.size(); k++)
	{
		//Reference
		opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
		reference->setScale(0.03f);

		CMatrixDouble mat = trajectory.at(k);
		CPose3D pose(mat);
		reference->setPose(pose);
		scene->insert(reference);
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showMesh()
{
	scene = window.get3DSceneAndLock();

	//Control mesh
	opengl::CMesh3DPtr control_mesh = scene->getByClass<CMesh3D>(0);
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);

	////Show vertex numbers
	//for (unsigned int v = 0; v < num_verts; v++)
	//{
	//	opengl::CText3DPtr vert_nums = scene->getByClass<CText3D>(v);
	//	vert_nums->setLocation(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));
	//}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showImages()
{
	scene = window.get3DSceneAndLock();

	//Show correspondences and samples for DT (if solving with DT)
	for (unsigned int i = 0; i < num_images; ++i)
	{
		CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(i);
		points->setPose(cam_poses[i]);

		//Insert points
		float r, g, b;
		if (vis_errors)	{r = 0.f; g = 0.7f; b = 0.f;}
		else			{utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);}
			
		for (unsigned int v = 0; v < rows; v++)
			for (unsigned int u = 0; u < cols; u++)
				if (depth[i](v,u) < 1.f)
				{
					if (is_object[i](v,u))
						points->push_back(depth[i](v,u), x_image[i](v, u), y_image[i](v,u), 0, 1, 0);
					else
						points->push_back(depth[i](v,u), x_image[i](v, u), y_image[i](v,u), 1, 0, 0);
				}
	}

	window.unlockAccess3DScene();
	window.repaint();
}


void Mod3DfromRGBD::createTopologyRefiner()
{
	typedef Far::TopologyDescriptor Descriptor;

	Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
	// Adpative refinement is only supported for CATMARK
	// Scheme LOOP is only supported if the mesh is purely composed of triangles

	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);

	//Fill the topology of the mesh
	Descriptor desc;
	desc.numVertices = num_verts;
	desc.numFaces = num_faces;

	int *vertsperface = new int[num_faces];
	for (unsigned int i = 0; i < num_faces; i++)
	{
		if (is_quad(i))		vertsperface[i] = 4;
		else				vertsperface[i] = 3;
	}

	desc.numVertsPerFace = vertsperface;
	desc.vertIndicesPerFace = face_verts.data();

	//Instantiate a FarTopologyRefiner from the descriptor.
	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
				Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

	const int maxIsolation = 0; //Don't change it!
	refiner->RefineAdaptive( Far::TopologyRefiner::AdaptiveOptions(maxIsolation));


	// Generate a set of Far::PatchTable that we will use to evaluate the surface limit
	Far::PatchTableFactory::Options patchOptions;
	patchOptions.endCapType = Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;

	//Far::PatchTable const * patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
	patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);

	// Compute the total number of points we need to evaluate patchtable.
	// we use local points around extraordinary features.
	int nRefinerVertices = refiner->GetNumVerticesTotal();
	int nLocalPoints = patchTable->GetNumLocalPoints();

	// Create a buffer to hold the position of the refined verts and
	// local points, then copy the coarse positions at the beginning.
	verts.clear();
	verts.resize(nRefinerVertices + nLocalPoints);
	memcpy(&verts[0], vert_coords.data(), num_verts * 3 * sizeof(float));

	// Interpolate vertex primvar data : they are the control vertices
	// of the limit patches (see far_tutorial_0 for details)
	Vertex * src = &verts[0];
	for (int level = 1; level <= maxIsolation; ++level)
	{
		Vertex * dst = src + refiner->GetLevel(level - 1).GetNumVertices();
		Far::PrimvarRefiner(*refiner).Interpolate(level, src, dst);
		src = dst;
	}

	// Evaluate local points from interpolated vertex primvars.
	patchTable->ComputeLocalPointValues(&verts[0], &verts[nRefinerVertices]);
}

void Mod3DfromRGBD::evaluateSubDivSurface()
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils(); ///printf("\n Num of stencils - %d", nstencils);
	Far::Stencil *st = new Far::Stencil[nstencils];

	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);
	
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	//Far::PtexIndices ptexIndices(*refiner);  // Far::PtexIndices helps to find indices of ptex faces.

	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int i = 0; i<num_images; ++i)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (valid[i](v,u))
				{
					// Locate the patch corresponding to the face ptex idx and (s,t)
					Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface[i](v,u), u1[i](v,u), u2[i](v,u)); assert(handle);

					// Evaluate the patch weights, identify the CVs and compute the limit frame:
					patchTable->EvaluateBasis(*handle, u1[i](v,u), u2[i](v,u), pWeights, dsWeights, dtWeights);

					Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

					LimitFrame eval; eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

					//Save the 3D coordinates
					mx[i](v, u) = eval.point[0];
					my[i](v, u) = eval.point[1];
					mz[i](v, u) = eval.point[2];

					//Compute the normals
					nx[i](v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
					ny[i](v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
					nz[i](v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];


				}
}


void Mod3DfromRGBD::refineMeshOneLevel()
{
	typedef Far::TopologyDescriptor Descriptor;

	Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);

	//Fill the topology of the mesh
	Descriptor desc;
	desc.numVertices = num_verts;
	desc.numFaces = num_faces;

	int *vertsperface; vertsperface = new int[num_faces];
	for (unsigned int i = 0; i < num_faces; i++)
	{
		if (is_quad(i)) vertsperface[i] = 4;
		else			vertsperface[i] = 3;
	}

	desc.numVertsPerFace = vertsperface;
	desc.vertIndicesPerFace = face_verts.data();

	//Instantiate a FarTopologyRefiner from the descriptor.
	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
		Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

	// Uniformly refine the topolgy once
	refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(1));
	const Far::TopologyLevel mesh_level = refiner->GetLevel(1);

	// Allocate and fill a buffer for the old vertex primvar data
	vector<Vertex> old_verts; old_verts.resize(num_verts);
	for (unsigned int v = 0; v<num_verts; v++)
		old_verts[v].SetPosition(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));

	// Allocate and fill a buffer for the new vertex primvar data
	verts.resize(mesh_level.GetNumVertices());
	Vertex *src = &old_verts[0], *dst = &verts[0];
	Far::PrimvarRefiner(*refiner).Interpolate(1, src, dst);


	//									Create the new mesh from it
	//---------------------------------------------------------------------------------------------------------
	num_verts = mesh_level.GetNumVertices();
	num_faces = mesh_level.GetNumFaces();

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	for (unsigned int f = 0; f < num_faces; f++)
	{
		if (mesh_level.GetFaceVertices(f).size() == 4)
			is_quad(f) = true;
		else
		{
			is_quad(f) = false;
			printf("\n Warning!!!! Some faces are not Quad and the algorithm is more than likely to crash!!");
		}
	}
	
	//Fill the vertices per face
	face_verts.resize(4, num_faces);
	for (unsigned int f = 0; f < num_faces; f++)
	{
		Far::ConstIndexArray face_v = mesh_level.GetFaceVertices(f);
		for (int v = 0; v < face_v.size(); v++)
			face_verts(v, f) = face_v[v];
	}

	//Find the adjacent faces to every face
	face_adj.resize(4, num_faces);
	face_adj.fill(-1);
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int k = 0; k < 4; k++)
		{
			unsigned int kinc = k + 1; if (k + 1 == 4) kinc = 0;
			float edge[2] = {face_verts(k, f), face_verts(kinc, f)};
			for (unsigned int fa = 0; fa < num_faces; fa++)
			{
				if (f == fa) continue;
				char found = 0;
				for (unsigned int l = 0; l < 4; l++)
					if ((face_verts(l, fa) == edge[0]) || (face_verts(l, fa) == edge[1]))
						found++;

				if (found > 1)
				{
					face_adj(k, f) = fa;
					break;
				}
			}
		}

	//Fill the 3D coordinates of the vertices
	vert_coords.resize(3, num_verts);
	for (unsigned int v = 0; v < verts.size(); v++)
	{
		vert_coords(0, v) = verts[v].point[0];
		vert_coords(1, v) = verts[v].point[1];
		vert_coords(2, v) = verts[v].point[2];
	}

	//Show the mesh on the 3D Scene
	ctf_level++;
}

void Mod3DfromRGBD::refineMeshToShow()
{
	typedef Far::TopologyDescriptor Descriptor;

	Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
	Sdc::Options options;
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);

	//Fill the topology of the mesh
	Descriptor desc;
	desc.numVertices = num_verts;
	desc.numFaces = num_faces;

	int *vertsperface; vertsperface = new int[num_faces];
	for (unsigned int i = 0; i < num_faces; i++)
	{
		if (is_quad(i)) vertsperface[i] = 4;
		else			vertsperface[i] = 3;
	}

	desc.numVertsPerFace = vertsperface;
	desc.vertIndicesPerFace = face_verts.data();

	//Instantiate a FarTopologyRefiner from the descriptor.
	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
		Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

	// Uniformly refine the topolgy once
	refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(1));
	const Far::TopologyLevel mesh_level = refiner->GetLevel(1);

	// Allocate and fill a buffer for the old vertex primvar data
	vector<Vertex> old_verts; old_verts.resize(num_verts);
	for (unsigned int v = 0; v<num_verts; v++)
		old_verts[v].SetPosition(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));

	// Allocate and fill a buffer for the new vertex primvar data
	verts.resize(mesh_level.GetNumVertices());
	Vertex *src = &old_verts[0], *dst = &verts[0];
	Far::PrimvarRefiner(*refiner).Interpolate(1, src, dst);


	//									Create the new mesh from it
	//---------------------------------------------------------------------------------------------------------
	num_verts = mesh_level.GetNumVertices();
	num_faces = mesh_level.GetNumFaces();

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	for (unsigned int f = 0; f < num_faces; f++)
	{
		if (mesh_level.GetFaceVertices(f).size() == 4)
			is_quad(f) = true;
		else
		{
			is_quad(f) = false;
			printf("\n Warning!!!! Some faces are not Quad and the algorithm is more than likely to crash!!");
		}
	}

	//Fill the vertices per face
	face_verts.resize(4, num_faces);
	for (unsigned int f = 0; f < num_faces; f++)
	{
		Far::ConstIndexArray face_v = mesh_level.GetFaceVertices(f);
		for (int v = 0; v < face_v.size(); v++)
			face_verts(v, f) = face_v[v];
	}

	//Fill the 3D coordinates of the vertices
	vert_coords.resize(3, num_verts);
	for (unsigned int v = 0; v < verts.size(); v++)
	{
		vert_coords(0, v) = verts[v].point[0];
		vert_coords(1, v) = verts[v].point[1];
		vert_coords(2, v) = verts[v].point[2];
	}

	showRenderedModel();
}



void Mod3DfromRGBD::evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	// Locate the patch corresponding to the face ptex idx and (s,t)
	Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface[i](v, u), u1[i](v, u), u2[i](v, u)); assert(handle);

	// Evaluate the patch weights, identify the CVs and compute the limit frame:
	patchTable->EvaluateBasis(*handle, u1[i](v, u), u2[i](v, u), pWeights, dsWeights, dtWeights);

	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

	LimitFrame eval; eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

	//Save the 3D coordinates
	mx[i](v, u) = eval.point[0];
	my[i](v, u) = eval.point[1];
	mz[i](v, u) = eval.point[2];


	//Compute the normals
	nx[i](v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	ny[i](v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	nz[i](v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
}







