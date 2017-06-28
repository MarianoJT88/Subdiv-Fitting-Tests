// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "2D_example_projective.h"


Mod2DfromRGBD::Mod2DfromRGBD()
{
	num_images = 5;
	fovh_d = utils::DEG2RAD(60.f);
	downsample = 1; //It can be set to 1, 2, 4, etc.
	rows = 480/downsample; cols = 640/downsample;
	new_cam_pose_model = true;
	model_behind_camera = false;
	solve_DT = false;
	image_set = 4;
	cam_prior = 0.f;
	Kz = 1.f; 
	Kclose = 20.f;
	Kproj = 0.0001f;
	max_iter = 100;
	min_depth = 0.4f;

	cam_poses.resize(num_images);
	cam_incrs.resize(num_images);
	cam_trans.resize(num_images);
	cam_trans_inv.resize(num_images);
	cam_mfold.resize(num_images); cam_mfold_old.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	depth.resize(num_images); x_image.resize(num_images); 
	x_t.resize(num_images); y_t.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images);
	DT.resize(num_images); DT_grad.resize(num_images); 
	for (unsigned int i = 0; i < num_images; i++)
	{
		depth[i].resize(cols); 
		x_image[i].resize(cols);
		x_t[i].resize(cols); y_t[i].resize(cols);
		is_object[i].resize(cols); valid[i].resize(cols);
		DT[i].resize(cols); DT_grad[i].resize(cols); 
	}

	//Internal points
	u1.resize(num_images); u1_old.resize(num_images); u1_old_outer.resize(num_images);
	uface.resize(num_images); uface_old.resize(num_images); uface_old_outer.resize(num_images);
	u1_incr.resize(num_images);
	res_depth.resize(num_images); 
	res_d1.resize(num_images);
	mx.resize(num_images); my.resize(num_images); 
	mx_t.resize(num_images); my_t.resize(num_images);
	u1_der.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1[i].resize(cols); u1_old.resize(cols); u1_old_outer.resize(num_images);
		uface[i].resize(cols); uface_old.resize(cols); uface_old_outer.resize(cols);
		u1_incr[i].resize(cols);
		res_depth[i].resize(cols); 
		res_d1[i].resize(cols);
		mx[i].resize(cols); my[i].resize(cols);
		mx_t[i].resize(cols); my_t[i].resize(cols); 
		u1_der[i].resize(cols);
		for (unsigned int u = 0; u < cols; u++)
			u1_der[i](u) = new float[2];
	}
}

void Mod2DfromRGBD::createDepthScan()
{
	for (unsigned int i = 1; i <= num_images; i++)
	{
		//Depth
		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		float d;

		for (unsigned int u = 0; u < cols; u++)
		{
			if ((u > 0.25f*cols) && (u < 0.75f*cols))
				d = 0.1f*sin(10.f*float(u)/float(cols)) + 0.4f;
			else
				d = 3.f;

			depth[i - 1](u) = d;
			x_image[i - 1](u) = (float(u) - disp_u)*d*inv_fd;
		}
	}
}

void Mod2DfromRGBD::loadDepthFromImages()
{
	string dir;
	if (image_set == 2)
		dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/images head1/";
	else if (image_set == 3)
		dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/images head2/";
	else if (image_set == 4)
		dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/cloth ground/";
	string name;
	char aux[30];
	const float norm_factor = 1.f / 255.f;

	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	for (unsigned int i = 1; i <= num_images; i++)
	{
		////Intensity
		//sprintf_s(aux, "i%d.png", i);
		//name = dir + aux;

		//cv::Mat im_i = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		//for (unsigned int u = 0; u < cols; u++)
		//	intensity[i - 1](u) = norm_factor*im_i.at<unsigned char>(rows*downsample/2, im_cols - 1 - u*downsample);


		//Depth
		sprintf_s(aux, "d%d.png", i);
		name = dir + aux;
		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);

		cv::Mat im_d = cv::imread(name, -1);
		cv::Mat depth_float;

		im_d.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
		for (unsigned int u = 0; u < cols; u++)
		{
			const float d = depth_float.at<float>(im_rows / 2, im_cols - 1 - u*downsample);
			depth[i - 1](u) = d;
			x_image[i - 1](u) = (u - disp_u)*d*inv_fd;
		}
	}
}

void Mod2DfromRGBD::segmentFromDepth()
{
	const float depth_limit = 1.f;

	//Simple threshold for depth
	for (unsigned int i = 0; i < num_images; i++)
	{
		valid[i].fill(true);
		is_object[i].fill(false);
		for (unsigned int u = 0; u < cols; u++)
		{
			if (depth[i](u) == 0.f)
				valid[i](u) = false;
			else if (depth[i](u) <= depth_limit)
				is_object[i](u) = true;
		}
	}
}


void Mod2DfromRGBD::loadInitialMesh()
{
	//Initial mesh - A cube...
	num_verts = 4;
	num_faces = 4;

	//Resize the weights
	w_contverts.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		w_contverts[i].resize(cols);
		for (unsigned int u = 0; u < cols; u++)
			w_contverts[i](u) = new float[num_verts];
	}

	////Fill the type of poligons (triangles or quads)
	//is_quad.resize(num_faces, 1);
	//is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(2, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1;
	face_verts.col(1) << 1, 2;
	face_verts.col(2) << 2, 3;
	face_verts.col(3) << 3, 0;

	//cout << endl << "Face vertices: " << endl << face_verts;

	//Find the adjacent faces to every face
	face_adj.resize(2, num_faces);
	face_adj.fill(-1);
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int fa = 0; fa < num_faces; fa++)
		{
			if (f == fa) continue;
			char found = 0;
			if (face_verts(1, fa) == face_verts(0, f))
			{
				face_adj(0, f) = fa;
				face_adj(1, fa) = f;
			}
			else if (face_verts(0, fa) == face_verts(1, f))
			{
				face_adj(1, f) = fa;
				face_adj(0, fa) = f;
			}
		}

	//cout << endl << "Face adjacency: " << endl << face_adj;


	//Fill the 3D coordinates of the vertices
	//Place the cube in the right place - Get the bounding box of the 3D point cloud
	float min_x = 10.f, min_y = 10.f;
	float max_x = -10.f, max_y = -10.f;

	for (unsigned int i = 0; i < num_images; i++)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
			{
				//Compute the 3D coordinates according to the camera pose
				const float x_t = mytrans(0, 0)*depth[i](u) + mytrans(0, 1)*x_image[i](u) + mytrans(0, 3);
				const float y_t = mytrans(1, 0)*depth[i](u) + mytrans(1, 1)*x_image[i](u) + mytrans(1, 3);

				if (x_t < min_x) 	min_x = x_t;
				if (x_t > max_x)	max_x = x_t;
				if (y_t < min_y) 	min_y = y_t;
				if (y_t > max_y)	max_y = y_t;
			}
	}

	const float x_margin = 0.1f*(max_x - min_x);
	const float y_margin = 0.1f*(max_y - min_y);

	vert_incrs.resize(2, num_verts);
	vert_coords.resize(2, num_verts); vert_coords_old.resize(2, num_verts);
	vert_coords.col(0) << min_x - x_margin, min_y - y_margin;
	vert_coords.col(1) << max_x + x_margin, min_y - y_margin;
	vert_coords.col(3) << min_x - x_margin, max_y + y_margin;
	vert_coords.col(2) << max_x + x_margin, max_y + y_margin;

	//Show the mesh on the 3D Scene
	showMesh();
}

//void Mod2DfromRGBD::computeDepthSegmentation()
//{
//	//Every valid measurement is assumed to be part of the object in this example
//	for (unsigned int i = 0; i < num_images; i++)
//	{
//		is_object[i].fill(false);
//		valid[i].fill(true);
//
//		for (unsigned int u = 0; u < cols; u++)
//		{
//			if (depth[i](u) > 0.f)
//				is_object[i](u) = true;
//
//			else
//				valid[i](u) = false;
//		}
//	}
//}

void Mod2DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(max(int(num_images), 5));

	if (image_set == 2) //images head 1
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.38f, -0.69f, 0.f, utils::DEG2RAD(60.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.50f, 0.75f, 0.f, utils::DEG2RAD(-71.5f), 0.f, 0.f);
		cam_poses[3].setFromValues(0.75f, 0.87f, 0.f, utils::DEG2RAD(-89.f), 0.f, 0.f);
	}
	else if (image_set == 3) //images head 2
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.33f, -0.48f, 0.f, utils::DEG2RAD(63.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.25f, 0.5f, 0.f, utils::DEG2RAD(-60.f), 0.f, 0.f);
		cam_poses[3].setFromValues(0.96f, -0.32f, 0.f, utils::DEG2RAD(148.f), 0.f, 0.f);
	}
	else if (image_set == 4) //cloth ground
	{ 
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.42f, 0.61f, 0.f, utils::DEG2RAD(-66.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.77f, 0.71f, 0.f, utils::DEG2RAD(-92.f), 0.f, 0.f);
		cam_poses[3].setFromValues(1.08f, -0.7f, 0.f, utils::DEG2RAD(119.f), 0.f, 0.f);
		cam_poses[4].setFromValues(1.25f, 0.54f, 0.f, utils::DEG2RAD(-132.f), 0.f, 0.f);
	}


	//Store the transformation matrices and the manifold values
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

		//Manifold
		if (new_cam_pose_model)
			cam_mfold[i].assign(0.f);
		else
		{
			Matrix4d log_trans = aux.log(); //Matrix4f log_trans = cam_trans[i].log(); <- Precision problem
			cam_mfold[i](0) = log_trans(0, 3); cam_mfold[i](1) = log_trans(1, 3); cam_mfold[i](2) = log_trans(2, 3);
			cam_mfold[i](3) = -log_trans(1, 2); cam_mfold[i](4) = log_trans(0, 2); cam_mfold[i](5) = -log_trans(0, 1);
		}

		//cout << endl << "Pose: " << endl << cam_poses[i];
		//cout << endl << "Manifold values" << endl << cam_mfold[i].transpose();
		//cout << endl << "Transformation matrix" << endl << cam_trans[i];
	}

	showCamPoses();


	//char c = 'y';
	//while (c == 'y')
	//{
	//	printf("\n Do you want to change the pose of any image [y/n]? ");
	//	cin >> c;

	//	if (c != 'y')
	//		break;

	//	int num;
	//	printf("\n Which image do you want to move? (from 1 to num_images-1, the first one cannot be moved): ");
	//	cin >> num;

	//	if (num >= num_images)
	//		printf("\n We don't have so many images");
	//	
	//	else
	//	{
	//		printf("\n Use characters q w e for positive increments");
	//		printf("\n Use characters a s d for negative increments");
	//		printf("\n Push p to finish with the pose of this image");

	//		int pushed_key = 0, stop = 0;
	//		const float incr_t = 0.01f, incr_r = 0.05f;
	//		CPose3D &pose = cam_poses[num];
	//		Matrix<float, 6, 1> &mfold = cam_mfold[num];
	//		Matrix4f &trans = cam_trans[num];
	//			
	//		while (!stop)
	//		{
	//			if (window.keyHit())
	//				pushed_key = window.getPushedKey();
	//			else
	//				pushed_key = 0;

	//			switch (pushed_key)
	//			{
	//			//Positive increments
	//			case  'q':
	//				pose.x_incr(incr_t);
	//				break;
	//			case 'w':
	//				pose.y_incr(incr_t);
	//				break;
	//			case 'e':
	//				pose.setYawPitchRoll(pose.yaw() + incr_r, pose.pitch(), pose.roll());
	//				break;

	//			//negative increments
	//			case  'a':
	//				pose.x_incr(-incr_t);
	//				break;
	//			case 's':
	//				pose.y_incr(-incr_t);
	//				break;
	//			case 'd':
	//				pose.setYawPitchRoll(pose.yaw() - incr_r, pose.pitch(), pose.roll());
	//				break;

	//			case 'p':
	//				stop = 1;
	//				break;
	//			}

	//			if (pushed_key)
	//				showCamPoses();
	//		}

	//		cout << endl << "Final pose for the image " << num << ":" << endl << pose;
	//	}
	//}
}

void Mod2DfromRGBD::computeInitialU()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXf> u1_ini;
	u1_ini.resize(num_faces);
	const unsigned int num_samp = max(int(2000/num_faces), 1);
	const float fact = 1.f / float(num_samp-1);
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		u1_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
			u1_ini[f](u) = float(u)*fact;
	}

	//Evaluate the surface
	vector<ArrayXf> x_ini, y_ini;
	x_ini.resize(num_faces); y_ini.resize(num_faces);

	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp);
		y_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
		{
			const float u1 = fact*u;
			Array<float, 2, 6> vert_sums;
			const unsigned int face_l = face_adj(0, f);
			const unsigned int face_r = face_adj(1, f);

			const float coef[6] = {(u1 + 2.f) / 3.f, (u1 + 1.f) / 3.f, u1 / 3.f, (u1 + 1.f)/2.f, u1/2.f, u1};
			vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
			vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, f)) + (1.f - coef[1])*vert_coords.col(face_verts(0, f));
			vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
			vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
			vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			//Save the 3D coordinates
			x_ini[f](u) = vert_sums(0, 5);
			y_ini[f](u) = vert_sums(1, 5);
		}
	}

	//Find the correspondence with minimal energy for each point (data)
	//-----------------------------------------------------------------
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);

	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		Matrix4f &mytrans = cam_trans_inv[i];

		//Compute the transformed points
		vector<ArrayXf> x_ini_t, z_ini_t;
		x_ini_t.resize(num_faces); z_ini_t.resize(num_faces);
		for (unsigned int f = 0; f < num_faces; f++)
		{
			x_ini_t[f].resize(num_samp);
			z_ini_t[f].resize(num_samp);

			for (unsigned int uu = 0; uu < num_samp; uu++)
			{
				//Compute the 3D coordinates of the subdiv point after the relative transformation
				z_ini_t[f](uu) = mytrans(0, 0)*x_ini[f](uu) + mytrans(0, 1)*y_ini[f](uu) + mytrans(0, 3);
				x_ini_t[f](uu) = mytrans(1, 0)*x_ini[f](uu) + mytrans(1, 1)*y_ini[f](uu) + mytrans(1, 3);

				if (z_ini_t[f](uu) <= 0.f)
				{
					printf("\n Problem computing the initial internal points for the background!! Model points too close to the cameras");
					model_behind_camera = true;
				}
			}
		}

		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
			{
				float min_energy = 100000.f, u1_min = 0.f, energy;
				unsigned int f_min = 0;

				for (unsigned int f = 0; f < num_faces; ++f)
					for (unsigned int uu = 0; uu < num_samp; uu++)
					{
						//Project point onto the image plane (I assume that no point has z-coordinate == 0)
						const float u_pixel = fx*(x_ini_t[f](uu) / z_ini_t[f](uu)) + disp_u;
						const float pix_error = square(u_pixel - float(u));
						//const float depth_error = square(z_ini_t[f](uu) - depth[i](u));
						const float depth_dist = square(z_ini_t[f](uu));

						if ((energy = pix_error + Kclose*depth_dist) < min_energy)
						{
							min_energy = energy;
							u1_min = u1_ini[f](uu);
							f_min = f;
						}
					}

				u1[i](u) = u1_min;
				uface[i](u) = f_min;
			}
	}
}

void Mod2DfromRGBD::computeInitialV()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXf> u1_ini;
	u1_ini.resize(num_faces); 
	const unsigned int num_samp = max(int(200 / num_faces), 1);
	const float fact = 1.f / float(num_samp-1);
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		u1_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
			u1_ini[f](u) = float(u)*fact;
	}

	//Evaluate the surface
	vector<ArrayXf> x_ini, y_ini;
	x_ini.resize(num_faces); y_ini.resize(num_faces);

	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp);
		y_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
		{
			const float u1 = fact*u;
			Array<float, 2, 6> vert_sums;
			const unsigned int face_l = face_adj(0, f);
			const unsigned int face_r = face_adj(1, f);

			const float coef[6] = {(u1 + 2.f) / 3.f, (u1 + 1.f) / 3.f, u1 / 3.f, (u1 + 1.f) / 2.f, u1 / 2.f, u1};
			vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
			vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, f)) + (1.f - coef[1])*vert_coords.col(face_verts(0, f));
			vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
			vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
			vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			//Save the 3D coordinates
			x_ini[f](u) = vert_sums(0, 5);
			y_ini[f](u) = vert_sums(1, 5);
		}
	}

	//Find the one that projects closer to the corresponding pixel (using Pin-Hole model)
	//-----------------------------------------------------------------------------------
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		const Matrix4f &mytrans = cam_trans_inv[i];

		//Compute the transformed points
		vector<ArrayXf> x_ini_t, z_ini_t;
		x_ini_t.resize(num_faces); z_ini_t.resize(num_faces);
		for (unsigned int f = 0; f < num_faces; f++)
		{
			x_ini_t[f].resize(num_samp);
			z_ini_t[f].resize(num_samp);

			for (unsigned int uu = 0; uu < num_samp; uu++)
			{
				//Compute the 3D coordinates of the subdiv point after the relative transformation
				z_ini_t[f](uu) = mytrans(0, 0)*x_ini[f](uu) + mytrans(0, 1)*y_ini[f](uu) + mytrans(0, 3);
				x_ini_t[f](uu) = mytrans(1, 0)*x_ini[f](uu) + mytrans(1, 1)*y_ini[f](uu) + mytrans(1, 3);

				if (z_ini_t[f](uu) <= 0.f)
				{
					printf("\n Problem computing the initial internal points for the background!! Model points too close to the cameras");
					model_behind_camera = true;
				}
			}
		}

		for (unsigned int u = 0; u < cols; u++)
			if (!is_object[i](u))
			{
				float min_dist = 100000.f, u1_min = 0.f, u2_min = 0.f;
				unsigned int f_min = 0;

				for (unsigned int f = 0; f < num_faces; f++)
					for (unsigned int uu = 0; uu < num_samp; uu++)
					{
						//Project point onto the image plane (I assume that no point has z-coordinate == 0)
						const float u_pixel = fx*(x_ini_t[f](uu) / z_ini_t[f](uu)) + disp_u;
						const float pix_dist = square(u_pixel - float(u));

						if (pix_dist < min_dist)
						{
							min_dist = pix_dist;
							u1_min = u1_ini[f](uu);
							f_min = f;
						}
					}

				u1[i](u) = u1_min;
				uface[i](u) = f_min;
			}
	}
}


void Mod2DfromRGBD::computeInitialCorrespondences()
{
	//Compute initial internal points for the foreground
	computeInitialU();

	//Compute initial internal points for the background
	computeInitialV();

	//Evaluate Subdiv surface
	evaluateSubDivSurface();

	//Show Subdiv surface
	//showSubSurface();
}

void Mod2DfromRGBD::computeTransCoordAndResiduals()
{
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	model_behind_camera = false;
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix4f &mytrans_inv = cam_trans_inv[i];

		for (unsigned int u = 0; u < cols; u++)
		{
			mx_t[i](u) = mytrans_inv(0, 0)*mx[i](u) + mytrans_inv(0, 1)*my[i](u) + mytrans_inv(0, 3);
			my_t[i](u) = mytrans_inv(1, 0)*mx[i](u) + mytrans_inv(1, 1)*my[i](u) + mytrans_inv(1, 3);

			if (is_object[i](u))
				res_depth[i](u) = mx_t[i](u) - depth[i](u);
				//res_depth[i](u) = mx_t[i](u);


			if (mx_t[i](u) <= 0.f)
			{
				printf("\n Depth coordinate of the internal correspondence is equal or inferior to zero after the transformation!!!");
				model_behind_camera = true;
			}
			const float u_proj = fx*(my_t[i](u) / mx_t[i](u)) + disp_u;
			res_d1[i](u) = float(u) - u_proj;

		}
	}
}


void Mod2DfromRGBD::updateInternalPointCrossingEdges(unsigned int i, unsigned int u, bool adaptive)
{
	//Check if crossing borders
	float u_incr = u1_incr[i](u);
	float u1_old;
	unsigned int face;
	if (adaptive)
	{
		u1_old = this->u1_old[i](u);
		face = uface_old[i](u);
	}
	else
	{
		u1_old = u1[i](u);
		face = uface[i](u);
	}

	float u1_new = u1_old + u_incr;
	bool crossing = true;

	while (crossing)
	{
		//Find the new face	and the coordinates of the crossing point within the old face and the new face
		unsigned int face_new;
		//printf("\n u_incr before update = %f", u_incr);

		if (u1_new < 0.f)
		{
			face_new = face_adj(0, face);
			const float du_prev = -u1_old;
			u1_old = 1.f;
			u_incr -= du_prev;
		}
		else
		{
			face_new = face_adj(1, face);
			const float du_prev = 1.f - u1_old;
			u1_old = 0.f;
			u_incr -= du_prev;
		}

		u1_new = u1_old + u_incr;
		face = face_new;

		crossing = (u1_new < 0.f) || (u1_new > 1.f);
	}

	u1[i](u) = u1_new;
	uface[i](u) = face;
}

void Mod2DfromRGBD::computeEnergyMaximization()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			if (!is_object[i](u))
				energy += square(res_d1[i](u));
	
	//printf("\n Energy max = %f", energy);
	energy_vec.push_back(energy);
}

void Mod2DfromRGBD::computeEnergyMinimization()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
				energy += square(res_d1[i](u)) + Kclose*square(mx_t[i](u));

	energy_vec.push_back(energy);
	//printf("\n Energy min = %f", energy);
}

float Mod2DfromRGBD::computeEnergyOverall()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
		{
			if (is_object[i](u))
				energy += Kproj*(square(res_d1[i](u)) +  Kclose*square(mx_t[i](u))) + Kz*square(res_depth[i](u));

			else if (valid[i](u))
			{
				const float d_squared = square(res_d1[i](u));

				//Truncated quadratic
				if (d_squared < square(tau))	energy -= alpha*d_squared / square(tau);
				else							energy -= alpha;
			}
		}

		//Camera prior - Keep close to the initial pose
		for (unsigned int l = 0; l < 6; l++)
			energy += 0.5f*cam_prior*square(cam_mfold[i](l));
	}

	//printf("\n Energy overall = %f", energy);
	return energy;
}


void Mod2DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1000, 900);
	window.setPos(900, 0);
	window.setCameraZoom(2);
	window.setCameraAzimuthDeg(0);
	window.setCameraElevationDeg(90);
	window.setCameraPointingToPoint(0.f, 0.f, 0.f);
	window.getDefaultViewport()->setCustomBackgroundColor(utils::TColorf(1.f, 1.f, 1.f));

	scene = window.get3DSceneAndLock();

	// Lights:
	//scene->getViewport()->setNumberOfLights(1);
	//mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	//light0.light_ID = 0;
	//light0.setPosition(2.5f,0,0.f,1.f);
	//light0.setDirection(0.f, 0.f, -1.f);

	//Control mesh
	opengl::CPointCloudPtr control_verts = opengl::CPointCloud::Create();
	control_verts->setPointSize(4.f);
	control_verts->setColor(0.5f, 0.5f, 0.5f);
	scene->insert(control_verts);

	opengl::CSetOfLinesPtr mesh_edges = opengl::CSetOfLines::Create();
	mesh_edges->setLineWidth(1.f);
	mesh_edges->setColor(0.5f, 0.5f, 0.5f);
	scene->insert(mesh_edges);

	////Vertex numbers
	//for (unsigned int v = 0; v < 26; v++)
	//{
	//	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	//	vert_nums->setString(std::to_string(v));
	//	vert_nums->setScale(0.03f);
	//	vert_nums->setColor(0.5, 0, 0);
	//	scene->insert(vert_nums);
	//}

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	reference->setScale(0.05f);
	scene->insert(reference);

	//Frustums
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CFrustumPtr frustum = opengl::CFrustum::Create(0.01f, 0.1f, utils::RAD2DEG(fovh_d), 0.f , 1.f, true, false);
		frustum->setColor(0.4f, 0.f, 0.f, 0.99f);
		scene->insert(frustum);
	}

	//Points
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CPointCloudPtr points = opengl::CPointCloud::Create();
		points->setPointSize(3.f);
		points->enablePointSmooth(true);
		scene->insert(points);

		//Insert points (they don't change through the optimization process)
		float r, g, b;
		utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
		points->setColor(r, g, b);
		//points->setColor(0.f, 0.9f, 0.f);
		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
				points->insertPoint(depth[i](u), x_image[i](u), 0.f);
	}

	//Points
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CPointCloudPtr points = opengl::CPointCloud::Create();
		points->setPointSize(3.f);
		points->enablePointSmooth(true);
		scene->insert(points);
	}

	//Correspondences (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(4.f);
	points->setColorA(0.95f);
	points->enablePointSmooth(true);
	scene->insert(points);

	//Correspondences projected onto the image plane
	opengl::CPointCloudColouredPtr proj = opengl::CPointCloudColoured::Create();
	proj->setPointSize(4.f);
	proj->setColorA(0.95f);
	proj->enablePointSmooth(true);
	scene->insert(proj);

	//Whole subdivision surface
	opengl::CSetOfLinesPtr subsurface = opengl::CSetOfLines::Create();
	subsurface->setLineWidth(1.f);
	subsurface->setColor(0.9f, 0.f, 0.f, 0.8f);
	scene->insert(subsurface);

	//Connecting data and correspondences
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CSetOfLinesPtr conect = opengl::CSetOfLines::Create();
		float r, g, b;
		utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
		conect->setColor(r, g, b);
		conect->setLineWidth(1.f);
		//conect->setColor(0.f, 0.6f, 0.6f);
		scene->insert(conect);
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod2DfromRGBD::showCamPoses()
{
	scene = window.get3DSceneAndLock();

	for (unsigned int i = 0; i < num_images; i++)
	{
		//Points (bananas)
		opengl::CPointCloudPtr points = scene->getByClass<CPointCloud>(i+1);
		points->setPose(cam_poses[i]);
		//points->clear();

		//float r, g, b;
		//utils::colormap(mrpt::utils::cmJET, float(i)/float(num_images), r, g, b);
		////points->setColor(r, g, b, 0.98f);
		//points->setColor(0.f, 0.8f, 0.f, 0.95f);

		//for (unsigned int u = 0; u < cols; u++)
		//	if (is_object[i](u))
		//		points->insertPoint(depth[i](u), x_image[i](u), 0.f);

		//Cameras
		opengl::CFrustumPtr frustum = scene->getByClass<CFrustum>(i);
		frustum->setPose(cam_poses[i]);
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod2DfromRGBD::showMesh()
{
	scene = window.get3DSceneAndLock();

	//Control mesh
	opengl::CPointCloudPtr control_verts = scene->getByClass<CPointCloud>(0);
	control_verts->clear();
	for (unsigned int k = 0; k < num_verts; k++)
		control_verts->insertPoint(vert_coords(0, k), vert_coords(1, k), 0.f);


	opengl::CSetOfLinesPtr mesh_edges = scene->getByClass<CSetOfLines>(0);
	mesh_edges->clear();
	for (unsigned int f = 0; f < num_faces; f++)
	{
		const unsigned int &v0 = face_verts(0, f);
		const unsigned int &v1 = face_verts(1, f);
		mesh_edges->appendLine(vert_coords(0, v0), vert_coords(1, v0), 0.f, vert_coords(0, v1), vert_coords(1, v1), 0.f);
	}

	////Show vertex numbers
	//for (unsigned int v = 0; v < num_verts; v++)
	//{
	//	opengl::CText3DPtr vert_nums = scene->getByClass<CText3D>(v);
	//	vert_nums->setLocation(vert_coords(0, v), vert_coords(1, v), 0.f);
	//}

	//// Set per-particle direction using the limit tangent (display as 'Streak')
	//for (int sample = 0; sample<nsamples; ++sample)
	//{
	//	float const * tan1 = samples[sample].deriv1;
	//	printf("%f %f %f\n", tan1[0], tan1[1], tan1[2]);
	//}

	//// Output particle positions for the bi-tangent
	//for (int sample = 0; sample<nsamples; ++sample)
	//{
	//	float const * tan2 = samples[sample].deriv2;
	//	printf("%f %f %f\n", tan2[0], tan2[1], tan2[2]);
	//}

	//    COpenGLViewportPtr vp_labels = odo.m_scene->createViewport("labels");
	//    vp_labels->setViewportPosition(0.7,0.05,240,180);
	//    //vp_labels->s ew("main");
	//    //vp_labels->setTransparent(true);

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod2DfromRGBD::showSubSurface()
{
	scene = window.get3DSceneAndLock();

	//Show the correspondences
	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0); 
	points->clear();

	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
		{
			if (is_object[i](u))
				points->push_back(mx[i](u), my[i](u), 0.f, 0.f, 0.f, 1.f);

			else if (!solve_DT)
			{
				//const float r = min(1.f, 0.5f*square(res_d1[i](u)));
				//const float g = 1 - r;
				points->push_back(mx[i](u), my[i](u), 0.f, 0.f, 0.f, 0.f);
			}
		}

	if (solve_DT)
	{
		for (unsigned int k = 0; k < nsamples; k++)
			points->push_back(mx_DT(k), my_DT(k), 0.f, 0.5f, 0.f, 0.5f);
	}

	//Show the correspondences projected onto the image plane
	//points = scene->getByClass<CPointCloudColoured>(1);
	//points->clear();
	//for (unsigned int i = 0; i < num_images; ++i)
	//	for (unsigned int u = 0; u < cols; u++)
	//	{
	//		if (is_object[i](u))
	//			points->push_back(0.1f, 0.1f*my[i](u) / mx[i](u), 0.f, 0.f, 0.f, 1.f);

	//		else
	//		{
	//			//const float r = min(1.f, 0.5f*square(res_d1[i](u)));
	//			//const float g = 1 - r;
	//			points->push_back(0.1f, 0.1f*my[i](u) / mx[i](u), 0.f, 0.f, 0.f, 0.f);
	//		}
	//	}

	//Show the whole surface
	CSetOfLinesPtr subsurface = scene->getByClass<CSetOfLines>(1);
	subsurface->clear();

	const unsigned int samples = 20;
	const float inv_samp = 1.f / float(samples);
	TPoint2D lastp, inip;

	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int u = 0; u < samples; u++)
		{
			const float u1 = inv_samp*u;
			Array<float, 2, 6> vert_sums;
			const unsigned int face_l = face_adj(0, f);
			const unsigned int face_r = face_adj(1, f);

			const float coef[6] = {(u1 + 2.f) / 3.f, (u1 + 1.f) / 3.f, u1 / 3.f, (u1 + 1.f) / 2.f, u1 / 2.f, u1};
			vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
			vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, f)) + (1.f - coef[1])*vert_coords.col(face_verts(0, f));
			vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
			vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
			vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			//Store the first point for the last line
			if ((u == 0) && (f == 0)) { inip.x = vert_sums(0, 5); inip.y = vert_sums(1, 5); }
			else						subsurface->appendLine(lastp.x, lastp.y, 0.f, vert_sums(0, 5), vert_sums(1, 5), 0.f);

			lastp.x = vert_sums(0, 5); lastp.y = vert_sums(1, 5);
		}
	subsurface->appendLine(lastp.x, lastp.y, 0.f, inip.x, inip.y, 0.f);


	//Connecting lines
	for (unsigned int i = 0; i < num_images; i++)
	{
		CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(2+i);
		conect->clear();
		conect->setPose(cam_poses[i]);
		for (unsigned int u = 0; u < cols; u+=5)
			if (is_object[i](u))
				conect->appendLine(depth[i](u), x_image[i](u), 0.f, mx_t[i](u), my_t[i](u), 0.f);
	}


	window.unlockAccess3DScene();
	window.repaint();
}

void Mod2DfromRGBD::evaluateSubDivSurfaceOnlyBackground()
{
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			if (!is_object[i](u))
				evaluateSubDivSurfacePixel(i, u);
}

void Mod2DfromRGBD::evaluateSubDivSurfacePixel(unsigned int i, unsigned int u)
{
	Array<float, 2, 6> vert_sums;
	const unsigned int face_m = uface[i](u);
	const unsigned int face_l = face_adj(0, face_m);
	const unsigned int face_r = face_adj(1, face_m);

	//Compute the weights and do recursive evaluation according to De Boor's algorithm
	const float coef[6] = {(u1[i](u) + 2.f) / 3.f, (u1[i](u) + 1.f) / 3.f, u1[i](u) / 3.f, (u1[i](u) + 1.f) / 2.f, u1[i](u) / 2.f, u1[i](u)};
	vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
	vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, face_m)) + (1.f - coef[1])*vert_coords.col(face_verts(0, face_m));
	vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
	vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
	vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
	vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

	mx[i](u) = vert_sums(0, 5);
	my[i](u) = vert_sums(1, 5);

	//Compute the derivatives
	Array<float, 2, 3> vert_dif;
	vert_dif.col(0) = (vert_coords.col(face_verts(1, face_l)) - vert_coords.col(face_verts(0, face_l)));
	vert_dif.col(1) = (vert_coords.col(face_verts(1, face_m)) - vert_coords.col(face_verts(0, face_m)));
	vert_dif.col(2) = (vert_coords.col(face_verts(1, face_r)) - vert_coords.col(face_verts(0, face_r)));
	vert_sums.col(3) = coef[3] * vert_dif.col(1) + (1.f - coef[3])*vert_dif.col(0);
	vert_sums.col(4) = coef[4] * vert_dif.col(2) + (1.f - coef[4])*vert_dif.col(1);
	vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

	u1_der[i](u)[0] = vert_sums(0, 5);
	u1_der[i](u)[1] = vert_sums(1, 5);
}

void Mod2DfromRGBD::refineMeshOneLevel()
{
	Array<float, 2, Dynamic> new_coords;
	new_coords.resize(2, 2 * num_verts);

	//Obtain the coordinates after refinement
	for (unsigned int v = 0; v < num_verts-1; v++)
		new_coords.col(1 + 2*v) = 0.5f*(vert_coords.col(v) + vert_coords.col(v + 1));
	new_coords.col(2 * num_verts - 1) = 0.5f*(vert_coords.col(num_verts - 1) + vert_coords.col(0));

	for (unsigned int v = 1; v < num_verts-1; v++)
		new_coords.col(2 * v) = 0.125f*vert_coords.col(v - 1) + 0.125f*vert_coords.col(v + 1) + 0.75f*vert_coords.col(v);
	new_coords.col(0) = 0.125f*vert_coords.col(num_verts - 1) + 0.125f*vert_coords.col(1) + 0.75f*vert_coords.col(0); //Special case to close the loop
	new_coords.col(2*num_verts - 2) = 0.125f*vert_coords.col(num_verts - 2) + 0.125f*vert_coords.col(0) + 0.75f*vert_coords.col(num_verts - 1); //Special case to close the loop

	vert_coords.resize(2, 2 * num_verts);
	vert_incrs.resize(2, 2 * num_verts);
	vert_coords.swap(new_coords);


	//Create the new mesh from with them
	num_verts *= 2;
	num_faces *= 2;

	//Resize the weights
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
				w_contverts[i](u) = new float[num_verts];

	//Fill the vertices per face
	face_verts.resize(2, num_faces);
	for (unsigned int f = 0; f < num_faces; f++)
	{
		face_verts(0, f) = f;
		face_verts(1, f) = f + 1;
	}
	face_verts(1, num_faces - 1) = 0;

	//cout << endl << "Face vertices: " << endl << face_verts;

	//Find the adjacent faces to every face
	face_adj.resize(2, num_faces);
	face_adj.fill(-1);
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int fa = 0; fa < num_faces; fa++)
		{
			if (f == fa) continue;
			char found = 0;
			if (face_verts(1, fa) == face_verts(0, f))
			{
				face_adj(0, f) = fa;
				face_adj(1, fa) = f;
			}
			else if (face_verts(0, fa) == face_verts(1, f))
			{
				face_adj(1, f) = fa;
				face_adj(0, fa) = f;
			}
		}

	//cout << endl << "Face adjacency: " << endl << face_adj;

	//Show the mesh on the 3D Scene
	showMesh();
}


void Mod2DfromRGBD::solveGradientDescent()
{
	float adap_mult = 0.1f;
	sz_x = 0.001f; sz_xi = 0.000000f*num_faces; //sz_xi = 0.0005f;
	tau = 1.f; alpha = 0.02f; cam_prior = 0.f; //cam_prior = 500.f; //alpha = 0.01f
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);

	//Create the matrices templates for the cam-pose derivatives
	Matrix4f mat_der_xi[6];
	for (unsigned int l = 0; l < 6; l++)
		mat_der_xi[l].assign(0.f);

	mat_der_xi[0](0, 3) = 1.f;
	mat_der_xi[1](1, 3) = 1.f;
	mat_der_xi[2](2, 3) = 1.f;
	mat_der_xi[3](1, 2) = -1.f; mat_der_xi[3](2, 1) = 1.f;
	mat_der_xi[4](0, 2) = 1.f; mat_der_xi[4](2, 0) = -1.f;
	mat_der_xi[5](0, 1) = -1.f; mat_der_xi[5](1, 0) = 1.f;


	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		vert_incrs.fill(0.f);
		evaluateSubDivSurface(); //To compute the new weights associated to the control vertices
		computeTransCoordAndResiduals();


		//								Compute the gradients
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			uface_old_outer[i] = uface[i];

			//Fast access to camera matrices and clean increments
			const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);
			cam_incrs[i].fill(0.f);

			for (unsigned int u = 0; u < cols; u++)
			{
				//Warning
				if (mx_t[i](u) <= 0.f)
				{
					model_behind_camera = true;
					printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
				}
				
				//Foreground
				if (is_object[i](u))
				{				
					Matrix<float, 1, 2> J_pi;
					J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);

					const float J_phi = 2.f*Kproj*res_d1[i](u);
					const Matrix<float, 1, 2> J_mult = J_phi*J_pi*T_inv;

					//Control vertices
					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						const float ww = w_contverts[i](u)[cp];
						vert_incrs(0, cp) += J_mult(0)*ww;	//Understand this term!!!!!! *****************************
						vert_incrs(1, cp) += J_mult(1)*ww;

						vert_incrs(0, cp) += 2.f*Kz*res_depth[i](u)*T_inv(0,0)*ww;
						vert_incrs(1, cp) += 2.f*Kz*res_depth[i](u)*T_inv(0,1)*ww;

						//vert_incrs(0, cp) += 2.f*Kclose*mx_t[i](u)*T_inv(0,0)*ww;
						//vert_incrs(1, cp) += 2.f*Kclose*mx_t[i](u)*T_inv(0,1)*ww;
					}

					//Camera pose
					Vector4f m_t; m_t << mx_t[i](u), my_t[i](u), 0.f, 1.f;
					for (unsigned int l = 0; l < 6; l++) 
					{
						Vector2f aux_prod = (mat_der_xi[l] * m_t).block<2, 1>(0, 0);
						cam_incrs[i](l) += (J_phi*J_pi*aux_prod).value();

						cam_incrs[i](l) += 2.f*Kz*res_depth[i](u)*aux_prod(0);
					}
				}

				//Background
				else if (valid[i](u) && (abs(res_d1[i](u)) < tau))
				{
					Matrix<float, 1, 2> J_pi;
					J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);

					const float J_phi = 2.f*res_d1[i](u) / square(tau);
					const Matrix<float, 1, 2> J_mult = J_phi*J_pi*T_inv;

					//Control vertices
					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						const float ww = w_contverts[i](u)[cp];
						vert_incrs(0, cp) += -alpha*J_mult(0)*ww;
						vert_incrs(1, cp) += -alpha*J_mult(1)*ww;
					}

					//Camera pose
					Vector4f m_t; m_t << mx_t[i](u), my_t[i](u), 0.f , 1.f;
					for (unsigned int l = 0; l < 6; l++)
					{
						Vector2f aux_prod = (mat_der_xi[l] * m_t).block<2, 1>(0, 0);
						cam_incrs[i](l) += -alpha*(J_phi*J_pi*aux_prod).value();
					}
				}
			}

			//Camera prior - Keep close to the initial pose
			for (unsigned int l = 0; l < 6; l++)
				cam_incrs[i](l) += cam_prior*cam_mfold_old[i](l);
		}

		energy_increasing = true;
		unsigned int cont = 0.f;


		//Update
		while (energy_increasing)
		{
			//Control vertices
			vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
			{
				cam_mfold[i] = cam_mfold_old[i] - adap_mult*sz_xi*cam_incrs[i];
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
				//cout << endl << "Camera increments: " << sz_xi*cam_incrs[i].transpose();
			}

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				uface[i] = uface_old_outer[i];
			}

			evaluateSubDivSurface();
			rayCastingLMBackgroundPerPixel();
			rayCastingLMForegroundPerPixel();
			new_energy = computeEnergyOverall();

			//if ((new_energy <= last_energy) && (!model_behind_camera))
			//{
			//	energy_increasing = false;
			//	adap_mult *= 1.2f;
			//	//printf("\n Energy decreasing: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			//}
			//else
			//{
			//	adap_mult *= 0.8f;
			//	model_behind_camera = false;
			//	//printf("\n Energy increasing -> repeat: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			//}

			//cont++;
			//if (cont > 50) energy_increasing = false;
			energy_increasing = false;
		}

		const float runtime = 1000.f*clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();
		system::sleep(10);

		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		//if (new_energy > last_energy - 0.0000001f)
		//	break;
	}

	printf("\n Average runtime = %f", aver_runtime/max_iter);
}


void Mod2DfromRGBD::rayCastingLMBackgroundPerPixel()
{
	float aver_lambda = 0.f;
	int cont = 0;
	
	//Iterative solver
	float lambda;
	float energy_ratio;
	float energy_old, energy;
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float limit_uincr = 2.f;

	//Compute the residuals
	computeTransCoordAndResiduals();

	//Initialize the old variables
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1_old[i] = u1[i];
		uface_old[i] = uface[i];
	}

	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);
		const Matrix4f &mytrans_inv = cam_trans_inv[i];

		for (unsigned int u = 0; u < cols; u++)
			if (!is_object[i](u) && valid[i](u))
			{
				energy = square(res_d1[i](u));
				energy_ratio = 2.f;
				lambda = 100000.f;

				while (energy_ratio > 1.00002f)
				{
					//Old equal to new for the next iteration
					u1_old[i](u) = u1[i](u);
					uface_old[i](u) = uface[i](u);
					energy_old = energy;

					//Fill the Jacobian with the gradients with respect to the internal points
					if (mx_t[i](u) <= 0.f)
					{
						printf("\n Warning!! The model is behind the camera. Problems with the projection");
						model_behind_camera = true;
						return;
					}

					Vector2f J_mu; J_mu << u1_der[i](u)[0], u1_der[i](u)[1];
					Vector2f J_pi; J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);
					const float J = -(J_pi.transpose()*T_inv*J_mu).value();

					bool energy_increasing = true;

					while (energy_increasing)
					{
						//Solve with LM
						u1_incr[i](u) = J*res_d1[i](u) / (square(J) + lambda);

						if (abs(u1_incr[i](u)) > limit_uincr)
						{
							lambda *= 2.f;
							continue;
						}

						//Update variable
						const float u1_new = u1_old[i](u) + u1_incr[i](u);
						if ((u1_new < 0.f) || (u1_new > 1.f))
							updateInternalPointCrossingEdges(i, u, true);
						else
						{
							u1[i](u) = u1_new;
							uface[i](u) = uface_old[i](u);
						}

						//Re-evaluate the mesh with the new parametric coordinates
						evaluateSubDivSurfacePixel(i, u);

						//Compute the residuals
						mx_t[i](u) = mytrans_inv(0, 0)*mx[i](u) + mytrans_inv(0, 1)*my[i](u) + mytrans_inv(0, 3);
						my_t[i](u) = mytrans_inv(1, 0)*mx[i](u) + mytrans_inv(1, 1)*my[i](u) + mytrans_inv(1, 3);
						const float u_proj = fx*(my_t[i](u) / mx_t[i](u)) + disp_u;
						res_d1[i](u) = float(u) - u_proj;

						//Compute the energy associated to this pixel
						energy = square(res_d1[i](u));

						if (energy > energy_old)
						{
							lambda *= 2.f;
							//printf("\n Energy is higher than before");
							//cout << endl << "Lambda updated: " << lambda;
						}
						else
						{
							energy_increasing = false;

							aver_lambda += lambda;
							cont++;

							lambda *= 0.5f;
							//printf("\n Energy is lower than before");
							//cout << endl << "Lambda updated: " << lambda;
						}
					}

					energy_ratio = energy_old / energy;
				}

				//if (energy > 5.f)
				//	printf("\n Pixel %d, res_d1 = %f, u1 = %f, face = %d", u, res_d1[i](u), u1[i](u), uface[i](u));
			}
	}

	//printf("\n Aver lambda = %f", aver_lambda / cont);

	//Compute energy of the ray-casting problem
	//computeEnergyMaximization();
	//printf("\n Final energy = %f", energy_vec.back());
}

void Mod2DfromRGBD::rayCastingLMForegroundPerPixel()
{	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float limit_uincr = 0.05f;

	//Compute the residuals
	computeTransCoordAndResiduals();

	//Initialize the old variables
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1_old[i] = u1[i];
		uface_old[i] = uface[i];
	}

	//Solve with LM
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);
		const Matrix4f &mytrans_inv = cam_trans_inv[i];

		for (unsigned int u = 0; u < cols; u++)
		if (is_object[i](u))
		{
			energy = square(res_d1[i](u)) + Kclose*square(mx_t[i](u));
			energy_ratio = 2.f;
			lambda = 0.1f;

			while (energy_ratio > 1.00005f)
			{
				//Old equal to new for the next iteration
				u1_old[i](u) = u1[i](u);
				uface_old[i](u) = uface[i](u);
				energy_old = energy;
				int cont = 0;

				//Fill the Jacobian with the gradients with respect to the internal points
				if (mx_t[i](u) <= 0.f)
				{
					printf("\n Warning!! The model is behind the camera. Problems with the projection");
					model_behind_camera = true;
					return;
				}

				Vector2f J_mu; J_mu << u1_der[i](u)[0], u1_der[i](u)[1];
				Vector2f J_pi; J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);
				const float J_proj = (J_pi.transpose()*T_inv*J_mu).value();
				const float J_depth = Kz*(T_inv.row(0)*J_mu).value();
				const float J_close = Kclose*(T_inv.row(0)*J_mu).value();

				//Vector3f J; J << J_proj, J_depth, J_close;
				//Vector3f R; R << res_d1[i](u), res_depth[i](u), mx_t[i](u) - min_depth;
				Vector2f J; J << J_proj, J_close;
				Vector2f R; R << res_d1[i](u), mx_t[i](u);
				bool energy_increasing = true;

				while (energy_increasing)
				{					
					//Solve with LM
					u1_incr[i](u) = -(J.transpose()*R).value() / ((J.transpose()*J).value() + lambda);
					if (abs(u1_incr[i](u)) > limit_uincr*float(num_faces))
					{
						lambda *= 2.f;
						continue;
					}
					//Update variable
					const float u1_new = u1_old[i](u) + u1_incr[i](u);
					if ((u1_new < 0.f) || (u1_new > 1.f))
						updateInternalPointCrossingEdges(i, u, true);
					else
					{
						u1[i](u) = u1_new;
						uface[i](u) = uface_old[i](u);
					}

					//Re-evaluate the mesh with the new parametric coordinates
					evaluateSubDivSurfacePixel(i, u);

					//Compute the residuals
					mx_t[i](u) = mytrans_inv(0, 0)*mx[i](u) + mytrans_inv(0, 1)*my[i](u) + mytrans_inv(0, 3);
					my_t[i](u) = mytrans_inv(1, 0)*mx[i](u) + mytrans_inv(1, 1)*my[i](u) + mytrans_inv(1, 3);
					const float u_proj = fx*(my_t[i](u) / mx_t[i](u)) + disp_u;
					res_d1[i](u) = float(u) - u_proj;
					res_depth[i](u) = mx_t[i](u) - depth[i](u);

					//Compute the energy associated to this pixel
					energy = square(res_d1[i](u)) + Kclose*square(mx_t[i](u));

					if (energy > energy_old)
					{
						lambda *= 2.f;
						//printf("\n Energy is higher than before");
					}
					else
					{
						energy_increasing = false;
						lambda *= 0.5f;
						//printf("\n Energy is lower than before");
					}

					cont++;
					if (cont > 5) energy_increasing = false;
				}

				energy_ratio = energy_old / energy;
			}

			//if (abs(u1_incr[i](u)) > 0.1f)
			//	printf("\n Im %d, energy = %.3f, u1 = %.3f, res_d1 = %.4f, res_depth = %.4f, l = %.2f", i, energy, u1[i](u), res_d1[i](u), res_depth[i](u), lambda);
		}
	}

	//Compute energy of the ray-casting problem
	//computeEnergyMinimization();
	//printf("\n Final energy = %f", energy_vec.back());

	//Show
	//showSubSurface();
	//system::sleep(5);
	//printf("\n Iteration %d", i);

}

void Mod2DfromRGBD::solveViaFiniteDifferences()
{
	ArrayXXf grad_X; grad_X.resize(2, num_verts);
	const float incr = 0.001f;
	float grad_sz = 0.0001f;
	tau = 1.f; alpha = 4e-4f; //Truncated quadratic
	//tau = 20.f; alpha = 10e-3f; //Geman-McClure
 	const unsigned int iter = 3;
	float last_energy, new_energy, incr_energy;
	bool energy_increasing;

	computeTransCoordAndResiduals();
	new_energy = computeEnergyOverall();

	for (unsigned int i = 0; i < iter; i++)
	{
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;

		//Compute the gradients
		for (unsigned int cv = 0; cv < num_verts; cv++)
			for (unsigned int k = 0; k < 2; k++)
			{
				vert_coords(k, cv) += incr;

				evaluateSubDivSurface();
				rayCastingLMBackgroundPerPixel();
				rayCastingLMForegroundPerPixel();

				incr_energy = computeEnergyOverall();
				grad_X(k, cv) = (incr_energy - last_energy) / incr;

				vert_coords(k, cv) -= incr;
			}

		//cout << endl << "Grad_X finite differences: " << endl << grad_X;

		energy_increasing = true;

		while (energy_increasing)
		{
			//Update
			vert_coords = vert_coords_old - grad_sz*grad_X;

			//Check whether the energy is increasing or decreasing
			computeInitialCorrespondences();
			evaluateSubDivSurface();
			rayCastingLMBackgroundPerPixel();
			rayCastingLMForegroundPerPixel();
			new_energy = computeEnergyOverall();

			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				grad_sz *= 2.f;
				//printf("\n Energy decreasing: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			}
			else
			{
				grad_sz *= 0.5f;
				//printf("\n Energy increasing -> repit: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			}
		}

		showMesh();
		system::sleep(1);

		printf("\n New_energy = %f, last_energy = %f", new_energy, last_energy);
		if (new_energy > last_energy - 0.000001f)
			break;
	}
}

void Mod2DfromRGBD::evaluateSubDivSurface()
{
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
		{	
			Array<float, 2, 6> vert_sums;
			const unsigned int face_m = uface[i](u);
			const unsigned int face_l = face_adj(0, face_m);
			const unsigned int face_r = face_adj(1, face_m);

			//Compute the weights and do recursive evaluation according to De Boor's algorithm
			const float coef[6] = {(u1[i](u) + 2.f) / 3.f, (u1[i](u) + 1.f) / 3.f, u1[i](u) / 3.f, (u1[i](u) + 1.f) / 2.f, u1[i](u) / 2.f, u1[i](u)};
			vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
			vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, face_m)) + (1.f - coef[1])*vert_coords.col(face_verts(0, face_m));
			vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
			vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
			vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			mx[i](u) = vert_sums(0, 5);
			my[i](u) = vert_sums(1, 5);

			//Compute the derivatives
			Array<float, 2, 3> vert_dif;
			vert_dif.col(0) = (vert_coords.col(face_verts(1, face_l)) - vert_coords.col(face_verts(0, face_l)));
			vert_dif.col(1) = (vert_coords.col(face_verts(1, face_m)) - vert_coords.col(face_verts(0, face_m)));
			vert_dif.col(2) = (vert_coords.col(face_verts(1, face_r)) - vert_coords.col(face_verts(0, face_r)));
			vert_sums.col(3) = coef[3] * vert_dif.col(1) + (1.f - coef[3])*vert_dif.col(0);
			vert_sums.col(4) = coef[4] * vert_dif.col(2) + (1.f - coef[4])*vert_dif.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			u1_der[i](u)[0] = vert_sums(0, 5);
			u1_der[i](u)[1] = vert_sums(1, 5);

			//Check derivatives
			//const float incr = 0.001f;
			//const float coef_i[6] = {(u1[i](u) + 2.f + incr) / 3.f, (u1[i](u) + 1.f + incr) / 3.f, (u1[i](u) + incr) / 3.f, (u1[i](u) + 1.f + incr) / 2.f, (u1[i](u) + incr) / 2.f, u1[i](u) + incr};
			//vert_sums.col(0) = coef_i[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef_i[0])*vert_coords.col(face_verts(0, face_l));
			//vert_sums.col(1) = coef_i[1] * vert_coords.col(face_verts(1, face_m)) + (1.f - coef_i[1])*vert_coords.col(face_verts(0, face_m));
			//vert_sums.col(2) = coef_i[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef_i[2])*vert_coords.col(face_verts(0, face_r));
			//vert_sums.col(3) = coef_i[3] * vert_sums.col(1) + (1.f - coef_i[3])*vert_sums.col(0);
			//vert_sums.col(4) = coef_i[4] * vert_sums.col(2) + (1.f - coef_i[4])*vert_sums.col(1);
			//vert_sums.col(5) = coef_i[5] * vert_sums.col(4) + (1.f - coef_i[5])*vert_sums.col(3);
			//const float u1der1 = (vert_sums(0, 5) - mx[i](u)) / incr;
			//const float u1der2 = (vert_sums(1, 5) - my[i](u)) / incr;
			//printf("\n Analytic der = %f, %f, findif der = %f, %f", u1_der[i](u)[0], u1_der[i](u)[1], u1der1, u1der2);

			//Compute the weights associated to the control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				w_contverts[i](u)[k] = 0.f;
			w_contverts[i](u)[face_verts(0, face_l)] = (1.f - coef[0])*(1.f - coef[3])*(1.f - coef[5]);
			w_contverts[i](u)[face_verts(0, face_m)] = coef[0]*(1.f - coef[3])*(1.f - coef[5])
														+ (1.f - coef[1])*(coef[3]*(1-coef[5]) + (1.f - coef[4])*coef[5]);
			w_contverts[i](u)[face_verts(1, face_m)] = (1 - coef[2])*coef[4]*coef[5]
														+ coef[1]*(coef[3] * (1 - coef[5]) + (1.f - coef[4])*coef[5]);
			w_contverts[i](u)[face_verts(1, face_r)] = coef[2]*coef[4]*coef[5];
			//printf("\n Weights: %f, %f, %f, %f", w_contverts[i](u)[face_verts(0, face_l)], w_contverts[i](u)[face_verts(0, face_m)], w_contverts[i](u)[face_verts(1, face_m)], w_contverts[i](u)[face_verts(1, face_r)]);
			//printf("\n Sum weights: %f", w_contverts[i](u)[face_verts(0, face_l)] + w_contverts[i](u)[face_verts(0, face_m)] + w_contverts[i](u)[face_verts(1, face_m)] + w_contverts[i](u)[face_verts(1, face_r)]);

		}
}

void Mod2DfromRGBD::computeDistanceTransform()
{
	for (unsigned int i = 0; i < num_images; i++)
	{
		//"Expand" the segmentation to its surrounding invalid pixels
		Array<bool, Dynamic, 1> big_segment = is_object[i];
		vector<int> buffer;
		for (int u = 0; u < cols-1; u++)
			if (big_segment(u) != big_segment(u + 1))
			{
				if (big_segment(u) == true)	buffer.push_back(u);
				else						buffer.push_back(u);
			}

		while (!buffer.empty())
		{
			const int ind = buffer.back();
			buffer.pop_back();

			if ((ind == 0) || (ind == cols-1))
				continue;
			else
			{
				if ((valid[i](ind - 1) == false) && (big_segment(ind - 1) == false))
				{
					buffer.push_back(ind - 1);
					big_segment(ind - 1) = true;
				}

				if ((valid[i](ind + 1) == false) && (big_segment(ind + 1) == false))
				{
					buffer.push_back(ind + 1);
					big_segment(ind + 1) = true;
				}
			}
		}
			
		
		//Compute the distance tranform
		for (int u = 0; u < cols; u++)
		{
			if (big_segment(u))
				DT[i](u) = 0.f;

			//Find the closest pixel which belongs to the object
			else	
			{
				bool found = false;
				int dist = 1;
				while (!found)
				{
					const int ind_r = min(u + dist, int(cols) - 1);
					const int ind_l = max(u - dist, 0);
					if (big_segment(ind_r) || big_segment(ind_l))
					{
						found = true;
						DT[i](u) = dist;
					}
					dist++;
				}
			}
		}

		//Compute the gradient of the distance transform
		for (int u = 1; u < cols-1; u++)
			DT_grad[i](u) = 0.5f*(DT[i](u + 1) - DT[i](u - 1));
		
		DT_grad[i](0) = DT[i](1) - DT[i](0);
		DT_grad[i](cols-1) = DT[i](cols-1) - DT[i](cols-2);
	}
}

void Mod2DfromRGBD::solveWithDT()
{
	//nsamples_approx = 200; 
	//float last_energy, new_energy, aver_runtime = 0.f;
	//bool energy_increasing;
	//float adap_mult = 1.f;
	//sz_x = 0.001f; sz_xi = 0.0001f*num_verts;
	//alpha = 5e-5f;
	//utils::CTicTac clock;

	//const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	//const float disp_u = 0.5f*float(cols - 1);

	////Create the matrices templates for the cam-pose derivatives
	//Matrix4f mat_der_xi[6];
	//for (unsigned int l = 0; l < 6; l++)
	//	mat_der_xi[l].assign(0.f);

	//mat_der_xi[0](0, 3) = 1.f;
	//mat_der_xi[1](1, 3) = 1.f;
	//mat_der_xi[2](2, 3) = 1.f;
	//mat_der_xi[3](1, 2) = -1.f; mat_der_xi[3](2, 1) = 1.f;
	//mat_der_xi[4](0, 2) = 1.f; mat_der_xi[4](2, 0) = -1.f;
	//mat_der_xi[5](0, 1) = -1.f; mat_der_xi[5](1, 0) = 1.f;


	//computeTransCoordAndResiduals();
	//sampleSurfaceForDTBackground();
	//new_energy = computeEnergyDTOverall();

	//for (unsigned int iter = 0; iter < max_iter; iter++)
	//{	
	//	clock.Tic();
	//	
	//	//Update old variables
	//	last_energy = new_energy;
	//	vert_coords_old = vert_coords;
	//	cam_mfold_old = cam_mfold;
	//	vert_incrs.fill(0.f);
	//	for (unsigned int i = 0; i < num_images; i++)
	//		cam_incrs[i].fill(0.f);
	//	evaluateSubDivSurface(); //To compute the new weights associated to the control vertices
	//	computeTransCoordAndResiduals();
	//	sampleSurfaceForDTBackground();


	//	//Compute the gradients
	//	for (unsigned int i = 0; i < num_images; i++)
	//	{
	//		//Keep the last solution for u
	//		u1_old_outer[i] = u1[i];
	//		uface_old_outer[i] = uface[i];
	//		
	//		const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);

	//		for (unsigned int u = 0; u < cols; u++)
	//		{
	//			if (is_object[i](u))
	//			{
	//				//Control vertices
	//				for (unsigned int cp = 0; cp < num_verts; cp++)
	//				{
	//					const float ww = w_contverts[i](u)[cp];
	//					vert_incrs(0, cp) += -2.f*res_x[i](u)*ww;
	//					vert_incrs(1, cp) += -2.f*res_y[i](u)*ww;
	//				}

	//				//Camera pose	
	//				Vector4f t_point(4, 1); t_point << x_t[i](u), y_t[i](u), 0.f, 1.f;
	//				for (unsigned int l = 0; l < 6; l++)
	//				{
	//					MatrixXf prod = mat_der_xi[l] * t_point;
	//					cam_incrs[i](l) += 2.f*(res_x[i](u)*prod(0) + res_y[i](u)*prod(1));
	//				}
	//			}
	//		}

	//		//Background term with DT
	//		for (unsigned int s = 0; s < nsamples; s++)
	//		{
	//			Vector4f t_point(4, 1); t_point << mx_DT(s), my_DT(s), 0.f, 1.f;
	//			const float mx_t_DT = cam_trans_inv[i].row(0)*t_point;
	//			const float my_t_DT = cam_trans_inv[i].row(1)*t_point;

	//			if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

	//			Matrix<float, 1, 2> J_pi;
	//			J_pi << fx*my_t_DT / square(mx_t_DT), -fx / mx_t_DT;

	//			const Matrix<float, 1, 2> J_mult = DT_grad[i](int(pixel_DT[i](s)))*J_pi*T_inv;

	//			//Control vertices
	//			for (unsigned int cp = 0; cp < num_verts; cp++)
	//			{
	//				const float ww = w_DT(s)[cp];
	//				vert_incrs(0, cp) += -alpha*J_mult(0)*ww;
	//				vert_incrs(1, cp) += -alpha*J_mult(1)*ww;
	//			}

	//			//Camera pose
	//			Vector4f m_t; m_t << mx_t_DT, my_t_DT, 0.f, 1.f;
	//			for (unsigned int l = 0; l < 6; l++)
	//			{
	//				Vector2f aux_prod = (-mat_der_xi[l] * m_t).block<2, 1>(0, 0);
	//				cam_incrs[i](l) += alpha*(DT_grad[i](int(pixel_DT[i](s)))*J_pi*aux_prod).value();
	//			}
	//		}
	//	}

	//	energy_increasing = true;
	//	unsigned int cont = 0.f;

	//	//Update the control vertices
	//	while (energy_increasing)
	//	{
	//		//Update
	//		vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

	//		for (unsigned int i = 0; i < num_images; i++)
	//		{
	//			cam_mfold[i] = cam_mfold_old[i] - adap_mult*sz_xi*cam_incrs[i];
	//			Matrix4f kai_mat;
	//			kai_mat << 0.f, -cam_mfold[i](5), cam_mfold[i](4), cam_mfold[i](0),
	//				cam_mfold[i](5), 0.f, -cam_mfold[i](3), cam_mfold[i](1),
	//				-cam_mfold[i](4), cam_mfold[i](3), 0.f, cam_mfold[i](2),
	//				0.f, 0.f, 0.f, 0.f;

	//			const Matrix4f new_trans = kai_mat.exp();
	//			const MatrixXf prod = new_trans*cam_ini[i];
	//			cam_trans[i] = prod.topLeftCorner<4, 4>(); //It don't know why but it crashes if I do the assignment directly

				//const Matrix3f rot_mat = cam_trans[i].block<3, 3>(0, 0).transpose();
				//const Vector3f tra_vec = cam_trans[i].block<3, 1>(0, 3);
				//cam_trans_inv[i].topLeftCorner<3, 3>() = rot_mat;
				//cam_trans_inv[i].block<3, 1>(0, 3) = -rot_mat*tra_vec;
				//cam_trans_inv[i].row(3) << 0.f, 0.f, 0.f, 1.f;

	//			CMatrixDouble44 mat_ini = cam_ini[i].cast<double>();
	//			CMatrixDouble44 mat_new = new_trans.cast<double>();
	//			cam_poses[i] = CPose3D(mat_new) + CPose3D(mat_ini);
	//			//cout << endl << "Camera increments: " << sz_xi*cam_incrs[i].transpose();
	//		}

	//		//Check whether the energy is increasing or decreasing
	//		for (unsigned int i = 0; i < num_images; i++)
	//		{
	//			u1[i] = u1_old_outer[i];
	//			uface[i] = uface_old_outer[i];
	//		}
	//		//computeInitialIntPoints();
	//		evaluateSubDivSurface();
	//		findClosestPointLMForegroundPerPixel();
	//		sampleSurfaceForDTBackground();
	//		new_energy = computeEnergyDTOverall();

	//		if (new_energy <= last_energy)
	//		{
	//			energy_increasing = false;
	//			adap_mult *= 2.f;
	//			//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
	//		}
	//		else
	//		{
	//			adap_mult *= 0.5f;
	//			//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
	//		}

	//		cont++;
	//		if (cont > 10) energy_increasing = false;
	//	}

	//	const float runtime = 1000.f*clock.Tac();
	//	aver_runtime += runtime;

	//	showCamPoses();
	//	showMesh();
	//	showSubSurface();
	//	system::sleep(20);

	//	printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
	//	//if (new_energy > last_energy - 0.0000001f)
	//	//	break;
	//}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

float Mod2DfromRGBD::computeEnergyDTOverall()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
		{
			if (is_object[i](u))
				energy += square(res_d1[i](u)) + Kz*square(res_depth[i](u));
		}

		for (unsigned int s = 0; s < nsamples; s++)
		{
			energy += alpha*DT[i](pixel_DT[i](s));
		}
	}

	//printf("\n Energy overall DT = %f", energy);
	return energy;
}

void Mod2DfromRGBD::sampleSurfaceForDTBackground()
{
	const unsigned int nsamples_pface = nsamples_approx / num_faces;
	nsamples = nsamples_pface*num_faces;
	w_DT.resize(nsamples);	
	u1_der_DT.resize(nsamples);
	mx_DT.resize(nsamples); my_DT.resize(nsamples);
	u1_DT.resize(nsamples); uface_DT.resize(nsamples);
	pixel_DT.resize(num_images);

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);

	
	const float fact = 1.f / float(nsamples_pface); //**********************
	for (unsigned int f = 0; f < num_faces; ++f)
		for (unsigned int u = 0; u < nsamples_pface; u++)
		{
			const unsigned int ind = f*nsamples_pface + u;
			u1_DT(ind) = float(u)*fact;
			uface_DT(ind) = f;
			w_DT(ind) = new float[num_verts];
			u1_der_DT(ind) = new float[2];
		}

	//Evaluate the surface
	for (unsigned int s = 0; s < nsamples; s++)
	{
		const float u1 = u1_DT(s);
		Array<float, 2, 6> vert_sums;
		const unsigned int face_m = uface_DT(s);
		const unsigned int face_l = face_adj(0, face_m);
		const unsigned int face_r = face_adj(1, face_m);

		const float coef[6] = { (u1 + 2.f) / 3.f, (u1 + 1.f) / 3.f, u1 / 3.f, (u1 + 1.f) / 2.f, u1 / 2.f, u1 };
		vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
		vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, face_m)) + (1.f - coef[1])*vert_coords.col(face_verts(0, face_m));
		vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
		vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
		vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
		vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

		//Save the 3D coordinates
		mx_DT(s) = vert_sums(0, 5);
		my_DT(s) = vert_sums(1, 5);

		//Derivatives
		Array<float, 2, 3> vert_dif;
		vert_dif.col(0) = (vert_coords.col(face_verts(1, face_l)) - vert_coords.col(face_verts(0, face_l)));
		vert_dif.col(1) = (vert_coords.col(face_verts(1, face_m)) - vert_coords.col(face_verts(0, face_m)));
		vert_dif.col(2) = (vert_coords.col(face_verts(1, face_r)) - vert_coords.col(face_verts(0, face_r)));
		vert_sums.col(3) = coef[3] * vert_dif.col(1) + (1.f - coef[3])*vert_dif.col(0);
		vert_sums.col(4) = coef[4] * vert_dif.col(2) + (1.f - coef[4])*vert_dif.col(1);
		vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

		u1_der_DT(s)[0] = vert_sums(0, 5);
		u1_der_DT(s)[1] = vert_sums(1, 5);

		//Weights - control vertices
		for (unsigned int k = 0; k < num_verts; k++)
			w_DT(s)[k] = 0.f;

		w_DT(s)[face_verts(0, face_l)] = (1.f - coef[0])*(1.f - coef[3])*(1.f - coef[5]);
		w_DT(s)[face_verts(0, face_m)] = coef[0] * (1.f - coef[3])*(1.f - coef[5])
			+ (1.f - coef[1])*(coef[3] * (1 - coef[5]) + (1.f - coef[4])*coef[5]);
		w_DT(s)[face_verts(1, face_m)] = (1 - coef[2])*coef[4] * coef[5]
			+ coef[1] * (coef[3] * (1 - coef[5]) + (1.f - coef[4])*coef[5]);
		w_DT(s)[face_verts(1, face_r)] = coef[2] * coef[4] * coef[5];
	}

	//Compute the pixel to which the samples project
	for (unsigned int i = 0; i < num_images; i++)
	{
		pixel_DT[i].resize(nsamples);
		const Matrix4f &T_inv = cam_trans_inv[i];

		for (unsigned int s = 0; s < nsamples; s++)
		{
			//Camera pose
			Vector4f m_DT; m_DT << mx_DT(s), my_DT(s), 0.f, 1.f;
			Vector2f m_t_DT = T_inv.topRows(2)*m_DT;
			const float pixel_proj = fx*m_t_DT(1) / m_t_DT(0) + disp_u;
			pixel_DT[i](s) = roundf(min(float(cols - 1), max(0.f, pixel_proj)));
			//printf("\n Pixel proj = %f", pixel_DT[i](s));
		}
	}
}


inline void normalize(float *n)
{
	float rn = 1.0f / sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
	n[0] *= rn;
	n[1] *= rn;
	n[2] *= rn;
}


inline void cross_prod(float const *v1, float const *v2, float* vOut)
{
	vOut[0] = v1[1] * v2[2] - v1[2] * v2[1];
	vOut[1] = v1[2] * v2[0] - v1[0] * v2[2];
	vOut[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline float dot_prod(float const *v1, float const *v2)
{
	return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}



