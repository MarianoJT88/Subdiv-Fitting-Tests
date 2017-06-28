// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "2D_model_fitting.h"


Mod2DfromRGBD::Mod2DfromRGBD(unsigned int num_im, unsigned int downsamp)
{
	num_images = num_im;
	fovh_d = utils::DEG2RAD(60.f);
	downsample = downsamp; //It can be set to 1, 2, 4, etc.
	rows = 480/downsample; cols = 640/downsample;
	model_behind_camera = false;
	solve_DT = false;
	image_set = 3;
	K_cam_prior = 0.f;
	max_iter = 100;
	Kn = 0.005f;
	alpha = 0.01f;
	tau = 2.f;

	cam_poses.resize(num_images);
	cam_incrs.resize(num_images);
	cam_trans.resize(num_images);
	cam_trans_inv.resize(num_images);
	cam_mfold.resize(num_images); cam_mfold_old.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	depth.resize(num_images); x_image.resize(num_images);
	nx_image.resize(num_images); ny_image.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images);
	DT.resize(num_images); DT_grad.resize(num_images); 
	for (unsigned int i = 0; i < num_images; i++)
	{
		depth[i].resize(cols); 
		x_image[i].resize(cols);
		is_object[i].resize(cols); valid[i].resize(cols);
		DT[i].resize(cols); DT_grad[i].resize(cols); 
		nx_image[i].resize(cols); ny_image[i].resize(cols);
	}

	//Internal points
	u1.resize(num_images); u1_old.resize(num_images); u1_old_outer.resize(num_images);
	uface.resize(num_images); uface_old.resize(num_images); uface_old_outer.resize(num_images);
	u1_incr.resize(num_images);
	res_x.resize(num_images); res_y.resize(num_images);
	res_nx.resize(num_images); res_ny.resize(num_images);
	res_d1.resize(num_images);
	mx.resize(num_images); my.resize(num_images); 
	mx_t.resize(num_images); my_t.resize(num_images);
	u1_der.resize(num_images); u1_der2.resize(num_images);
	nx.resize(num_images); ny.resize(num_images);
	nx_t.resize(num_images); ny_t.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1[i].resize(cols); u1_old.resize(cols); u1_old_outer.resize(num_images);
		uface[i].resize(cols); uface_old.resize(cols); uface_old_outer.resize(cols);
		u1_incr[i].resize(cols);
		res_x[i].resize(cols); res_y[i].resize(cols); 
		res_nx[i].resize(cols); res_ny[i].resize(cols);
		res_d1[i].resize(cols);
		mx[i].resize(cols); my[i].resize(cols);
		mx_t[i].resize(cols); my_t[i].resize(cols); 
		u1_der[i].resize(cols); u1_der2[i].resize(cols);
		nx[i].resize(cols); ny[i].resize(cols);
		nx_t[i].resize(cols); ny_t[i].resize(cols);
		for (unsigned int u = 0; u < cols; u++)
		{
			u1_der[i](u) = new float[2];
			u1_der2[i](u) = new float[2];
		}
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

void Mod2DfromRGBD::computeDataNormals()
{
	ArrayXf r(cols);
	for (unsigned int i = 0; i<num_images; i++)
	{
		//Compute connectivity
		r.fill(1.f);
		for (unsigned int u = 0; u < cols-1; u++)
		{
			const float dist = square(depth[i](u+1) - depth[i](u)) + square(x_image[i](u+1) - x_image[i](u));
			if (dist  > 0.f)
				r(u) = sqrt(dist);
		}

		//Find the normals
		for (unsigned int u = 1; u < cols-1; u++)
		{
			if (is_object[i](u))
			{
				const float ddepth = (r(u-1)*(depth[i](u+1)-depth[i](u)) + r(u)*(depth[i](u) - depth[i](u-1)))/(r(u)+r(u-1));
				const float dx = (r(u-1)*(x_image[i](u+1)-x_image[i](u)) + r(u)*(x_image[i](u) - x_image[i](u-1)))/(r(u)+r(u-1));
				const float norm = sqrtf(square(ddepth) + square(dx));
				if (norm > 0.f)
				{
					nx_image[i](u) = -dx/norm;
					ny_image[i](u) = ddepth/norm;
				}
				else
				{
					nx_image[i](u) = 0.f;
					ny_image[i](u) = 0.f;;
				}
			}
			else
			{
				nx_image[i](u) = 0.f;
				ny_image[i](u) = 0.f;
			}
		}
		nx_image[i](0) = 0.f; ny_image[i](0) = 0.f;
		nx_image[i](cols-1) = 0.f; ny_image[i](cols-1) = 0.f;

		//Filter 
		const float mask[7] = {0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
		for (int u = 0; u < cols; u++)
		{
			if (is_object[i](u))
			{
				float n_x = 0.f, n_y = 0.f, sum = 0.f;
				for (int k = -3; k<4; k++)
				{
					const int index = u+k;
					if ((index > 0)&&(index<cols)&&(is_object[i](index)))
					{
						n_x += mask[k+3]*nx_image[i](index);
						n_y += mask[k+3]*ny_image[i](index);
						sum += mask[k+3];
					}
				}

				nx_image[i](u) = n_x/sum;
				ny_image[i](u) = n_y/sum;
			}
		}
	}
}

void Mod2DfromRGBD::loadDepthFromImages()
{
	string dir, name;
	if (image_set == 1) 		dir = im_dir + "images head1/";
	else if (image_set == 2)	dir = im_dir + "images head2/";
	else if (image_set == 3)	dir = im_dir + "cloth ground/";

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
	w_derverts.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		w_contverts[i].resize(cols);
		w_derverts[i].resize(cols);
		for (unsigned int u = 0; u < cols; u++)
		{
			w_contverts[i](u) = new float[num_verts];
			w_derverts[i](u) = new float[num_verts];
		}
	}

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


	//Fill the 2D coordinates of the vertices
	//Place the cube in the right place - Get the bounding box of the 2D point cloud
	float min_x = 10.f, min_y = 10.f;
	float max_x = -10.f, max_y = -10.f;

	for (unsigned int i = 0; i < num_images; i++)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
			{
				//Compute the 2D coordinates according to the camera pose
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

void Mod2DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(max(int(num_images), 5));

	if (image_set == 1) //images head 1
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.33f, -0.57f, 0.f, utils::DEG2RAD(60.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.29f, 0.62f, 0.f, utils::DEG2RAD(-60.f), 0.f, 0.f);
		cam_poses[3].setFromValues(0.69f, 0.77f, 0.f, utils::DEG2RAD(-92.f), 0.f, 0.f);
		cam_poses[4].setFromValues(1.1f, -0.73f, 0.f, utils::DEG2RAD(135.f), 0.f, 0.f);
	}
	else if (image_set == 2) //images head 2
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.33f, -0.48f, 0.f, utils::DEG2RAD(63.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.25f, 0.5f, 0.f, utils::DEG2RAD(-60.f), 0.f, 0.f);
		cam_poses[3].setFromValues(0.96f, -0.32f, 0.f, utils::DEG2RAD(148.f), 0.f, 0.f);
	}
	else if (image_set == 3) //cloth ground
	{ 
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.42f, 0.61f, 0.f, utils::DEG2RAD(-66.f), 0.f, 0.f);
		cam_poses[2].setFromValues(0.77f, 0.71f, 0.f, utils::DEG2RAD(-92.f), 0.f, 0.f);
		cam_poses[3].setFromValues(1.08f, -0.7f, 0.f, utils::DEG2RAD(119.f), 0.f, 0.f);
		cam_poses[4].setFromValues(1.25f, 0.54f, 0.f, utils::DEG2RAD(-132.f), 0.f, 0.f);

		////Perturb initial poses
		//const float scale_trans = 0.1f;
		//const float scale_rot = 0.5f;
		//for (unsigned int i=0; i<num_images; i++)
		//{
		//	const float incr_x = scale_trans*float(rand() % 1000)/1000.f;
		//	cam_poses[i].x_incr(incr_x);
		//	const float incr_y = scale_trans*float(rand() % 1000)/1000.f;
		//	cam_poses[i].y_incr(incr_y);
		//	const float incr_phi = scale_rot*float(rand() % 1000)/1000.f;
		//	cam_poses[i] += CPose3D(0,0,0,incr_phi,0,0);
		//}
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
		cam_mfold[i].assign(0.f);
	}

	showCamPoses();


	////Uncomment this to be able to move the camera poses manually to find a good initial value for them
	////--------------------------------------------------------------------------------------------------
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

void Mod2DfromRGBD::computeInitialUDataterm()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXf> u1_ini;
	u1_ini.resize(num_faces);
	const unsigned int num_samp = max(int(300/num_faces), 1);
	const float fact = 1.f / float(num_samp-1);
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		u1_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
			u1_ini[f](u) = float(u)*fact;
	}

	//Evaluate the surface
	vector<ArrayXf> x_ini, y_ini, nx_ini, ny_ini;
	x_ini.resize(num_faces); y_ini.resize(num_faces);
	nx_ini.resize(num_faces); ny_ini.resize(num_faces);

	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp); y_ini[f].resize(num_samp);
		nx_ini[f].resize(num_samp); ny_ini[f].resize(num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
		{
			const float u1 = fact*u;
			Array<float, 2, 6> vert_sums;
			const unsigned int face_l = face_adj(0, f);
			const unsigned int face_r = face_adj(1, f);
			const unsigned int face_m = f;

			const float coef[6] = {(u1 + 2.f) / 3.f, (u1 + 1.f) / 3.f, u1 / 3.f, (u1 + 1.f)/2.f, u1/2.f, u1};
			vert_sums.col(0) = coef[0] * vert_coords.col(face_verts(1, face_l)) + (1.f - coef[0])*vert_coords.col(face_verts(0, face_l));
			vert_sums.col(1) = coef[1] * vert_coords.col(face_verts(1, face_m)) + (1.f - coef[1])*vert_coords.col(face_verts(0, face_m));
			vert_sums.col(2) = coef[2] * vert_coords.col(face_verts(1, face_r)) + (1.f - coef[2])*vert_coords.col(face_verts(0, face_r));
			vert_sums.col(3) = coef[3] * vert_sums.col(1) + (1.f - coef[3])*vert_sums.col(0);
			vert_sums.col(4) = coef[4] * vert_sums.col(2) + (1.f - coef[4])*vert_sums.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);

			//Save the 3D coordinates
			x_ini[f](u) = vert_sums(0, 5);
			y_ini[f](u) = vert_sums(1, 5);

			//Compute the normals
			Array<float, 2, 3> vert_dif;
			vert_dif.col(0) = (vert_coords.col(face_verts(1, face_l)) - vert_coords.col(face_verts(0, face_l)));
			vert_dif.col(1) = (vert_coords.col(face_verts(1, face_m)) - vert_coords.col(face_verts(0, face_m)));
			vert_dif.col(2) = (vert_coords.col(face_verts(1, face_r)) - vert_coords.col(face_verts(0, face_r)));
			vert_sums.col(3) = coef[3] * vert_dif.col(1) + (1.f - coef[3])*vert_dif.col(0);
			vert_sums.col(4) = coef[4] * vert_dif.col(2) + (1.f - coef[4])*vert_dif.col(1);
			vert_sums.col(5) = coef[5] * vert_sums.col(4) + (1.f - coef[5])*vert_sums.col(3);
			const float inv_norm = 1.f/sqrtf(square(vert_sums(1, 5)) + square(vert_sums(0, 5)));
			nx_ini[f](u) = inv_norm*vert_sums(1, 5);
			ny_ini[f](u) = -inv_norm*vert_sums(0, 5);
		}
	}

	//For each data point, find the surface sample which produces the minimum residual
	//---------------------------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
			{
				//Compute the 3D coordinates of the observed point after the relative transformation
				const float x = mytrans(0, 0)*depth[i](u) + mytrans(0, 1)*x_image[i](u) + mytrans(0, 3);
				const float y = mytrans(1, 0)*depth[i](u) + mytrans(1, 1)*x_image[i](u) + mytrans(1, 3);
				const float n_x = mytrans(0, 0)*nx_image[i](u) + mytrans(0, 1)*ny_image[i](u);
				const float n_y = mytrans(1, 0)*nx_image[i](u) + mytrans(1, 1)*ny_image[i](u);

				float min_dist = 1000.f, u1_min = 0.f, dist;
				unsigned int f_min = 0;

				for (unsigned int f = 0; f < num_faces; ++f)
					for (unsigned int uu = 0; uu < num_samp; uu++)
						if ((dist = square(x - x_ini[f](uu)) + square(y - y_ini[f](uu)) + Kn*(square(n_x - nx_ini[f](uu)) + square(n_y - ny_ini[f](uu)))) < min_dist)
						{
							min_dist = dist;
							u1_min = u1_ini[f](uu);
							f_min = f;
						}

				u1[i](u) = u1_min;
				uface[i](u) = f_min;
			}
	}
}

void Mod2DfromRGBD::computeInitialUBackground()
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

	//For each background pixel, find the surface sample that projects closer to it
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
			{
				res_x[i](u) = depth[i](u) - mx_t[i](u);
				res_y[i](u) = x_image[i](u) - my_t[i](u);

				nx_t[i](u) = mytrans_inv(0, 0)*nx[i](u) + mytrans_inv(0, 1)*ny[i](u);
				ny_t[i](u) = mytrans_inv(1, 0)*nx[i](u) + mytrans_inv(1, 1)*ny[i](u);
				const float inv_norm = 1.f/sqrtf(square(nx_t[i](u)) + square(ny_t[i](u)));
				res_nx[i](u) = nx_image[i](u) - inv_norm*nx_t[i](u);
				res_ny[i](u) = ny_image[i](u) - inv_norm*ny_t[i](u);
			}
			else
			{
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

float Mod2DfromRGBD::computeEnergyOverall()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{		
		for (unsigned int u = 0; u < cols; u++)
		{		
			if (is_object[i](u))
				energy += square(res_x[i](u)) + square(res_y[i](u)) + Kn*(square(res_nx[i](u)) + square(res_ny[i](u)));

			else if (valid[i](u))
			{
				const float d_squared = square(res_d1[i](u));
				if (d_squared < square(tau) && (d_squared > square(eps)))
					energy += alpha*square(1.f - abs(res_d1[i](u))/tau);	
				else if (d_squared <= square(eps))
					energy += alpha*(1.f - eps/tau)*(1.f - d_squared/(eps*tau));
			}
		}

		//Camera prior - Keep close to the initial pose
		for (unsigned int l = 0; l < 6; l++)
			energy += 0.5f*K_cam_prior*square(cam_mfold[i](l));
	}

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

	//Correspondences (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(4.f);
	points->setColorA(0.95f);
	points->enablePointSmooth(true);
	scene->insert(points);

	////Correspondences projected onto the image plane
	//opengl::CPointCloudColouredPtr proj = opengl::CPointCloudColoured::Create();
	//proj->setPointSize(4.f);
	//proj->setColorA(0.95f);
	//proj->enablePointSmooth(true);
	//scene->insert(proj);

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
		//Points
		opengl::CPointCloudPtr points = scene->getByClass<CPointCloud>(i+1);
		points->setPose(cam_poses[i]);

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
	const float fact_norm = 0.03f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(2+i);
		conect->clear();
		conect->setPose(cam_poses[i]);
		for (unsigned int u = 0; u < cols; u+=1)
			if (is_object[i](u))
			{
				conect->appendLine(depth[i](u), x_image[i](u), 0.f, mx_t[i](u), my_t[i](u), 0.f);
				//const float tangent_norm_inv = 1.f/sqrtf(square(u1_der[i](u)[0]) + square(u1_der[i](u)[1]));
				//conect->appendLine(mx[i](u), my[i](u), 0.f,
				//					mx[i](u) + fact_norm*tangent_norm_inv*normal[i](u)[0], my[i](u) + fact_norm*tangent_norm_inv*normal[i](u)[1], 0.f);
				//conect->appendLine(depth[i](u), x_image[i](u), 0.f, depth[i](u) + fact_norm*nx_image[i](u), x_image[i](u) + fact_norm*ny_image[i](u), 0.f);	
			}
	}


	window.unlockAccess3DScene();
	window.repaint();
}

void Mod2DfromRGBD::showJacobiansBackground()
{
	scene = window.get3DSceneAndLock();

	//Choose the control vertex
	const unsigned int cv = 1;

	CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(2);
	conect->clear();
	conect->setPose(CPose3D(0,0,0,0,0,0));

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float alpha_sqrt = sqrtf(alpha);


	//Jacobian lines lines
	const float fact_norm = 0.01f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);

		for (unsigned int u = 0; u < cols; u+=1)
			if (!is_object[i](u) && (valid[i](u)) && (abs(res_d1[i](u)) < tau))  
			{
					Matrix<float, 1, 2> J_pi;
					J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);

					const float J_phi = alpha_sqrt / tau;
					const Matrix<float, 1, 2> J_mult_back = J_phi*J_pi*T_inv;

					float Jx, Jy;
					if (w_contverts[i](u)[cv] > 0.f)
					{
						Jx = J_mult_back(0)*w_contverts[i](u)[cv];
						Jy = J_mult_back(1)*w_contverts[i](u)[cv];
					}			
				
				conect->appendLine(mx[i](u), my[i](u), 0.f, mx[i](u) + fact_norm*Jx*res_d1[i](u), my[i](u) + fact_norm*Jy*res_d1[i](u), 0.f);
				//printf("\n res_d[%d](%d) = %f", i, u, res_d1[i](u));
			}
	}

	for (unsigned int v=0; v<num_verts; v++)
		printf("\n coordinates of the vertex %d: %f, %f", v, vert_coords(0,v), vert_coords(1,v));


	window.unlockAccess3DScene();
	window.repaint();
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

	//Compute the second order derivatives
	Array<float, 2, 2> vert_dif2;
	vert_dif2.col(0) = vert_dif.col(1) - vert_dif.col(0);
	vert_dif2.col(1) = vert_dif.col(2) - vert_dif.col(1);
	vert_sums.col(5) = coef[5] * vert_dif2.col(1) + (1.f - coef[5])*vert_dif2.col(0);
	u1_der2[i](u)[0] = vert_sums(0, 5);
	u1_der2[i](u)[1] = vert_sums(1, 5);

	//Compute the normals
	nx[i](u) = u1_der[i](u)[1];
	ny[i](u) = -u1_der[i](u)[0];
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


	//Create the new mesh from them
	num_verts *= 2;
	num_faces *= 2;

	//Resize the weights
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
		{
			w_contverts[i](u) = new float[num_verts];
			w_derverts[i](u) = new float[num_verts];
		}

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

void Mod2DfromRGBD::computeCameraTransfandPosesFromTwist()
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

void Mod2DfromRGBD::solveSK_GradientDescent()
{
	sz_x = 0.001f; sz_xi = 0.0005f;
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


	evaluateSubDivSurface();
	computeTransCoordAndResiduals();
	optimizeUDataterm_LM();
	optimizeUBackground_LM();
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
					//Data alignment
					Matrix<float, 1, 2> res; res << res_x[i](u), res_y[i](u);
					Matrix<float, 1, 2> J_mult = -2.f*res*T_inv;

					//Normal alignment
					Matrix<float, 1, 2> res_n; res_n << res_nx[i](u), res_ny[i](u);
					const float inv_norm = 1.f/sqrtf(square(nx[i](u)) + square(ny[i](u)));
					Matrix2f J_nu; J_nu << square(ny[i](u)), -nx[i](u)*ny[i](u), -nx[i](u)*ny[i](u), square(nx[i](u));
					J_nu *= inv_norm*square(inv_norm);
					Matrix2f J_rot90; J_rot90 << 0.f, 1.f, -1.f, 0.f;
					Matrix<float, 1, 2> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu*J_rot90;
					
					//Control vertices
					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						const float ww = w_contverts[i](u)[cp];
						const float wn = w_derverts[i](u)[cp];
						vert_incrs(0, cp) += J_mult(0)*ww + J_mult_norm(0)*wn;
						vert_incrs(1, cp) += J_mult(1)*ww + J_mult_norm(1)*wn;
					}

					//Camera pose	
					Vector4f t_point; t_point << mx_t[i](u), my_t[i](u), 0.f, 1.f;
					for (unsigned int l = 0; l < 6; l++)
					{
						MatrixXf prod = mat_der_xi[l] * t_point;
						cam_incrs[i](l) += -2.f*(res_x[i](u)*prod(0) + res_y[i](u)*prod(1));
					}

					Vector2f normal; normal << nx[i](u), ny[i](u);
					Vector2f n_t = T_inv*inv_norm*normal;
					for (unsigned int l = 3; l < 6; l++)
					{
						Vector2f prod = mat_der_xi[l].block<2, 2>(0, 0)*n_t;
						cam_incrs[i](l) += -2.f*Kn*(res_nx[i](u)*prod(0) + res_ny[i](u)*prod(1));
					}
				}

				//Background
				else if (valid[i](u) && (abs(res_d1[i](u)) < tau)) 
				{
					Matrix<float, 1, 2> J_pi;
					J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);
					float J_phi;

					if (abs(res_d1[i](u)) > eps)	J_phi = -2.f*alpha*(1.f - abs(res_d1[i](u))/tau)*utils::sign(res_d1[i](u))/tau;	
					else							J_phi = -2.f*alpha*(1.f - eps/tau)*res_d1[i](u)/(eps*tau);

					const Matrix<float, 1, 2> J_mult = J_phi*J_pi*T_inv;

					//Control vertices
					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						const float ww = w_contverts[i](u)[cp];
						vert_incrs(0, cp) += J_mult(0)*ww;
						vert_incrs(1, cp) += J_mult(1)*ww;
					}

					//Camera pose
					Vector4f m_t; m_t << mx_t[i](u), my_t[i](u), 0.f , 1.f;
					for (unsigned int l = 0; l < 6; l++)
					{
						Vector2f aux_prod = (mat_der_xi[l] * m_t).block<2, 1>(0, 0);
						cam_incrs[i](l) += (J_phi*J_pi*aux_prod).value();
					}
				}
			}

			//Camera prior - Keep close to the initial pose
			for (unsigned int l = 0; l < 6; l++)
				cam_incrs[i](l) += K_cam_prior*cam_mfold_old[i](l);
		}


		energy_increasing = true;
		unsigned int cont = 0;

		//Update
		while (energy_increasing)
		{
			//Control vertices
			vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
				cam_mfold[i] = cam_mfold_old[i] - adap_mult*sz_xi*cam_incrs[i];
			computeCameraTransfandPosesFromTwist();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				uface[i] = uface_old_outer[i];
			}

			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			optimizeUDataterm_LM();
			optimizeUBackground_LM();
			new_energy = computeEnergyOverall();

			if ((new_energy <= last_energy) && (!model_behind_camera))
			{
				energy_increasing = false;
				adap_mult *= 1.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			}
			else
			{
				adap_mult *= 0.5f;
				model_behind_camera = false;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			}

			cont++;

			//We cannot decrease the energy
			if (cont > 5)
			{
				//Recover old variables
				vert_coords = vert_coords_old;
				cam_mfold = cam_mfold_old;
				computeCameraTransfandPosesFromTwist();
				new_energy = last_energy;
				energy_increasing = true;
				break;			
			}
			//energy_increasing = false;
		}

		const float runtime = 1000.f*clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();
		system::sleep(10);


		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		if (new_energy > 0.9999f*last_energy)
		{
			printf("\n Optimization finished because the energy does not decrease anymore.");
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime/max_iter);
}

void Mod2DfromRGBD::optimizeUBackground_LM()
{
	//float aver_lambda = 0.f;
	//int cont = 0;
	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float limit_uincr = 0.05f*num_faces;
	const float lambda_limit = 10000000.f;
	float lambda_mult = 3.f;

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
				u1_incr[i](u) = 1.f;
				lambda = 10.f;

				while (energy_ratio > 1.0002f)
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
						u1_incr[i](u) = J*res_d1[i](u) / (square(J)*(1.f + lambda));

						if (abs(u1_incr[i](u)) > limit_uincr)
						{
							lambda *= lambda_mult;
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
							lambda *= lambda_mult;
							//printf("\n Energy is higher than before");
						}
						else
						{
							energy_increasing = false;

							//aver_lambda += lambda;
							//cont++;

							lambda /= lambda_mult;
							//printf("\n Energy is lower than before");
						}

						//Keep the last solution and finish
						if (lambda > lambda_limit)
						{
							u1[i](u) = u1_old[i](u);
							uface[i](u) = uface_old[i](u);
							energy_increasing = false;
							energy = energy_old;
						}
					}

					energy_ratio = energy_old / energy;
				}
			}
	}

	//printf("\n Aver lambda = %f", aver_lambda / cont);
}

void Mod2DfromRGBD::optimizeUDataterm_LM()
{
	//float aver_lambda = 0.f;
	//int cont = 0;
	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float limit_uincr = 0.05f*num_faces;
	const float lambda_limit = 10000000.f;
	float lambda_mult = 3.f;
	const float Kn_sqrt = sqrtf(Kn);

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
			energy = square(res_x[i](u)) + square(res_y[i](u)) + Kn*(square(res_nx[i](u)) + square(res_ny[i](u)));
			energy_ratio = 2.f;
			u1_incr[i](u) = 1.f;
			lambda = 1.f;

			while (energy_ratio > 1.0005f)
			{		
				//Old equal to new for the next iteration
				u1_old[i](u) = u1[i](u);
				uface_old[i](u) = uface[i](u);
				energy_old = energy;
				//int cont = 0;

				//Fill the Jacobian with the gradients with respect to the internal points
				Vector2f u_der; u_der << u1_der[i](u)[0], u1_der[i](u)[1];
				Vector4f J; J.topRows(2) = -T_inv*u_der;

				const float inv_norm = 1.f/sqrtf(square(nx[i](u)) + square(ny[i](u)));
				Vector2f u_der2_rot; u_der2_rot << u1_der2[i](u)[1], -u1_der2[i](u)[0];
				Matrix2f J_nu; J_nu << square(ny[i](u)), -nx[i](u)*ny[i](u), -nx[i](u)*ny[i](u), square(nx[i](u));
				J_nu *= inv_norm*square(inv_norm);
				J.bottomRows(2) = -Kn_sqrt*T_inv*J_nu*u_der2_rot;

				Vector4f R; R << res_x[i](u), res_y[i](u), Kn_sqrt*res_nx[i](u), Kn_sqrt*res_ny[i](u);

				bool energy_increasing = true;

				while (energy_increasing)
				{
					//Solve with LM
					u1_incr[i](u) = -(J.transpose()*R).value() / ((J.transpose()*J).value()*(1.f + lambda));
					if (abs(u1_incr[i](u)) > limit_uincr)
					{
						lambda *= lambda_mult;
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
					res_x[i](u) = depth[i](u) - mx_t[i](u);
					res_y[i](u) = x_image[i](u) - my_t[i](u);

					nx_t[i](u) = mytrans_inv(0, 0)*nx[i](u) + mytrans_inv(0, 1)*ny[i](u);
					ny_t[i](u) = mytrans_inv(1, 0)*nx[i](u) + mytrans_inv(1, 1)*ny[i](u);
					const float inv_norm = 1.f/sqrtf(square(nx_t[i](u)) + square(ny_t[i](u)));
					res_nx[i](u) = nx_image[i](u) - inv_norm*nx_t[i](u);
					res_ny[i](u) = ny_image[i](u) - inv_norm*ny_t[i](u);

					//if (abs(u1[i](u) - u1_old_outer[i](u)) > 0.05f)
					//	printf("\n Energy = %f, i = %d, u = %d", energy, i, u);

					//Compute the energy associated to this pixel
					energy = square(res_x[i](u)) + square(res_y[i](u)) + Kn*(square(res_nx[i](u)) + square(res_ny[i](u)));

					if (energy > energy_old)
					{
						lambda *= lambda_mult;
						//printf("\n Energy is higher than before. Lambda = %f, energy = %f", lambda, energy);
					}
					else
					{
						energy_increasing = false;

						//aver_lambda += lambda;
						//cont++;

						lambda /= lambda_mult;
						//printf("\n Energy is lower than before. Lambda = %f, energy = %f", lambda, energy);
					}

					//Keep the last solution and finish
					if (lambda > lambda_limit)
					{
						u1[i](u) = u1_old[i](u);
						uface[i](u) = uface_old[i](u);
						energy_increasing = false;
						energy = energy_old;
					}

					//cont++;
					//if (cont > 50) energy_increasing = false;
					//energy_increasing = false;
				}

				energy_ratio = energy_old / energy;
				//printf("\n Pixel %d, energy = %f, u1 = %f, energy_old = %f", u, energy, u1[i](u), energy_old);
			}
		}
	}

	//printf("\n Aver lambda foreground = %f", aver_lambda / cont);
}

void Mod2DfromRGBD::test_FiniteDifferences()
{
	ArrayXXf grad_X; grad_X.resize(2, num_verts);
	const float incr = 0.001f;
	float grad_sz = 0.0001f;
	tau = 1.f; alpha = 4e-4f; //Truncated quadratic
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
				optimizeUDataterm_LM();
				optimizeUBackground_LM();

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
			computeInitialUDataterm();
			computeInitialUBackground();
			evaluateSubDivSurface();
			optimizeUDataterm_LM();
			optimizeUBackground_LM();
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

		printf("\n New_energy = %f, last_energy = %f", new_energy, last_energy);
		if (new_energy > 0.999f*last_energy)
		{
			printf("\n Optimization finished because the energy does not decrease anymore.");
			break;
		}
	}
}

void Mod2DfromRGBD::solveSK_LM()
{
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float alpha_sqrt = sqrtf(alpha);

	//Variables for Levenberg-Marquardt
	unsigned int J_rows = 0, J_cols = 2*num_verts + 6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
		{
			if (is_object[i](u))	J_rows += 4;
			else if (valid[i](u))	J_rows += 1;
		}

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);


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


	//Prev computations
	evaluateSubDivSurface();
	computeTransCoordAndResiduals();
	optimizeUDataterm_LM();
	optimizeUBackground_LM();
	new_energy = computeEnergyOverall();

	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		evaluateSubDivSurface(); //To compute the new weights associated to the control vertices
		computeTransCoordAndResiduals();


		//								Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			uface_old_outer[i] = uface[i];

			//Fast access to camera matrices
			const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				if (valid[i](u))
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
						//										Data alignment
						//----------------------------------------------------------------------------------------------
						//Control vertices
						const Matrix2f J_mult = -T_inv;
					
						float v_weight;
						for (unsigned int cp = 0; cp < num_verts; cp++)
							if ((v_weight = w_contverts[i](u)[cp]) > 0.f)
							{
								j_elem.push_back(Tri(cont, 2*cp, J_mult(0,0)*v_weight));
								j_elem.push_back(Tri(cont, 2*cp+1, J_mult(0,1)*v_weight));
								j_elem.push_back(Tri(cont+1, 2*cp, J_mult(1,0)*v_weight));
								j_elem.push_back(Tri(cont+1, 2*cp + 1, J_mult(1,1)*v_weight));
							}

						//Camera poses
						Vector4f t_point; t_point << mx_t[i](u), my_t[i](u), 0.f, 1.f;
						for (unsigned int l = 0; l < 6; l++)
						{
							Vector4f prod = -mat_der_xi[l] * t_point;
							j_elem.push_back(Tri(cont, 2*num_verts + 6*i + l, prod(0)));
							j_elem.push_back(Tri(cont+1, 2*num_verts + 6*i + l, prod(1)));
						}

						//Fill the residuals
						R(cont) = res_x[i](u);
						R(cont+1) = res_y[i](u);
						cont += 2;


						//										Normal alignment
						//------------------------------------------------------------------------------------------------
						//Control vertices
						const float inv_norm = 1.f/sqrtf(square(nx[i](u)) + square(ny[i](u)));
						Matrix2f J_nu; J_nu << square(ny[i](u)), -nx[i](u)*ny[i](u), -nx[i](u)*ny[i](u), square(nx[i](u));
						J_nu *= inv_norm*square(inv_norm);
						Matrix2f J_rot90; J_rot90 << 0.f, 1.f, -1.f, 0.f;
						const Matrix2f J_mult_norm = -sqrtf(Kn)*T_inv*J_nu*J_rot90;

						float n_weight;
						for (unsigned int cp = 0; cp < num_verts; cp++)
							if ((n_weight = w_derverts[i](u)[cp]) > 0.f)
							{
								j_elem.push_back(Tri(cont, 2*cp, J_mult_norm(0,0)*n_weight));
								j_elem.push_back(Tri(cont, 2*cp+1, J_mult_norm(0,1)*n_weight));
								j_elem.push_back(Tri(cont+1, 2*cp, J_mult_norm(1,0)*n_weight));
								j_elem.push_back(Tri(cont+1, 2*cp+1, J_mult_norm(1,1)*n_weight));
							}

						//Camera poses
						Vector2f normal; normal << nx[i](u), ny[i](u);
						Vector2f n_t = sqrtf(Kn)*T_inv*inv_norm*normal;
						for (unsigned int l = 3; l < 6; l++)
						{
							Vector2f prod = -mat_der_xi[l].block<2, 2>(0, 0)*n_t;
							j_elem.push_back(Tri(cont, 2*num_verts + 6*i + l, prod(0)));
							j_elem.push_back(Tri(cont+1, 2*num_verts + 6*i + l, prod(1)));
						}

						//Fill the residuals
						R(cont) = sqrtf(Kn)*res_nx[i](u);
						R(cont+1) = sqrtf(Kn)*res_ny[i](u);
						cont += 2;
					}

					//Background
					else
					{
						if ((abs(res_d1[i](u)) < tau) && (abs(res_d1[i](u)) > eps))
						{
							Matrix<float, 1, 2> J_pi;
							J_pi << fx*my_t[i](u) / square(mx_t[i](u)), -fx / mx_t[i](u);

							const float J_phi = -alpha_sqrt*mrpt::utils::sign(res_d1[i](u))/tau;
							const Matrix<float, 1, 2> J_mult_back = J_phi*J_pi*T_inv;

							//Control vertices
							float v_weight;
							for (unsigned int cp = 0; cp < num_verts; cp++)
								if ((v_weight = w_contverts[i](u)[cp]) > 0.f)
								{
									j_elem.push_back(Tri(cont, 2*cp, J_mult_back(0)*v_weight));
									j_elem.push_back(Tri(cont, 2*cp+1, J_mult_back(1)*v_weight));
								}

							//Camera pose
							Vector4f m_t; m_t << mx_t[i](u), my_t[i](u), 0.f , 1.f;
							for (unsigned int l = 0; l < 6; l++)
							{
								const float prod = (J_phi*J_pi*(mat_der_xi[l] * m_t).block<2, 1>(0, 0)).value();
								j_elem.push_back(Tri(cont, 2*num_verts + 6*i + l, prod));
							}

							//Fill the residuals
							R(cont) = alpha_sqrt*(1.f - abs(res_d1[i](u))/tau);
						}

						cont++;				
					}
				}

			//Camera prior - Keep close to the initial pose
			for (unsigned int l = 0; l < 6; l++)
				cam_incrs[i](l) += K_cam_prior*cam_mfold_old[i](l);
		}


		//Prepare Levenberg solver
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		MatrixXf JtJ = MatrixXf(JtJ_sparse);
		VectorXf b = -J.transpose()*R;
		MatrixXf JtJ_lm;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		//Update
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			//JtJ_lm = JtJ_f + JtJ_g + adap_mult*MatrixXf::Identity(J_cols, J_cols);	//Levenberg
			JtJ_lm = JtJ;
			//JtJ_lm.diagonal() += adap_mult*JtJ_lm.diagonal();					//Levenberg-Marquardt
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);


			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);


			//Update variables
			cont = 0;

			//control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 2; c++)
					vert_coords(c, k) += increments(cont++);

			//Update the camera poses
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int k = 0; k < 6; k++)
					cam_mfold[i](k) += increments(cont++);
			computeCameraTransfandPosesFromTwist();


			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				uface[i] = uface_old_outer[i];
			}

			evaluateSubDivSurface();
			computeTransCoordAndResiduals();

			if (model_behind_camera)
			{
				adap_mult *= 4.f;
				model_behind_camera = false;
				continue;
			}

			optimizeUDataterm_LM();
			optimizeUBackground_LM();
			new_energy = computeEnergyOverall();

			if (new_energy > last_energy)
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, grad_sz = %f", new_energy, last_energy, grad_sz);
			}
			else
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;

			//We cannot decrease the energy
			if (cont_inner > 5)
			{
				//Recover old variables
				vert_coords = vert_coords_old;
				cam_mfold = cam_mfold_old;
				computeCameraTransfandPosesFromTwist();
				new_energy = last_energy;
				//energy_increasing = true;
				break;			
			}
			//energy_increasing = false;
		}

		const float runtime = 1000.f*clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();
		//showJacobiansBackground();
		system::sleep(10);

		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		if (new_energy > 0.9999f*last_energy)
		{
			printf("\n Optimization finished because the energy does not decrease anymore.");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime/max_iter);
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

			//Compute the normals
			nx[i](u) = u1_der[i](u)[1];
			ny[i](u) = -u1_der[i](u)[0];

			//Compute the second order derivatives
			Array<float, 2, 2> vert_dif2;
			vert_dif2.col(0) = vert_dif.col(1) - vert_dif.col(0);
			vert_dif2.col(1) = vert_dif.col(2) - vert_dif.col(1);
			vert_sums.col(5) = coef[5] * vert_dif2.col(1) + (1.f - coef[5])*vert_dif2.col(0);
			u1_der2[i](u)[0] = vert_sums(0, 5);
			u1_der2[i](u)[1] = vert_sums(1, 5);

			//=======================================================================================================
			////Check the 2nd order derivatives
			//const float u1_incr = u1[i](u) + 0.001f;
			//const float coef2[6] = {(u1_incr + 2.f) / 3.f, (u1_incr + 1.f) / 3.f, u1_incr / 3.f, (u1_incr + 1.f) / 2.f, u1_incr / 2.f, u1_incr};
			//vert_sums.col(3) = coef2[3] * vert_dif.col(1) + (1.f - coef2[3])*vert_dif.col(0);
			//vert_sums.col(4) = coef2[4] * vert_dif.col(2) + (1.f - coef2[4])*vert_dif.col(1);
			//vert_sums.col(5) = coef2[5] * vert_sums.col(4) + (1.f - coef2[5])*vert_sums.col(3);
			//const float uderx_fin = vert_sums(0, 5);
			//const float udery_fin = vert_sums(1, 5);
			//printf("\n Analytic 2nd der = %.4f, %.4f, fin_dif 2nd der = %.4f, %.4f", u1_der2[i](u)[0], u1_der2[i](u)[1], 1000.f*(uderx_fin - u1_der[i](u)[0]), 1000.f*(udery_fin - u1_der[i](u)[1]));
			//=======================================================================================================


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

			//Compute the weights associated to the normals
			for (unsigned int k = 0; k < num_verts; k++)
				w_derverts[i](u)[k] = 0.f;
			w_derverts[i](u)[face_verts(0, face_l)] = -(1.f - coef[5])*(1.f - coef[3]);
			w_derverts[i](u)[face_verts(0, face_m)] = -coef[5]*(1.f - coef[4]) - coef[3]*(1.f - coef[5]) + (1.f - coef[5])*(1.f - coef[3]);
			w_derverts[i](u)[face_verts(1, face_m)] = -coef[5]*coef[4] + coef[5]*(1.f - coef[4]) + coef[3]*(1.f - coef[5]);
			w_derverts[i](u)[face_verts(1, face_r)] = coef[5]*coef[4];
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

void Mod2DfromRGBD::solveDT_GradientDescent()
{
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	sz_x = 0.001f; sz_xi = 0.0005f;
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


	evaluateSubDivSurface();
	computeTransCoordAndResiduals();
	sampleSurfaceForDTBackground();
	optimizeUDataterm_LM();
	new_energy = computeEnergyDTOverall();

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
		sampleSurfaceForDTBackground();


		//Compute the gradients
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices and clean increments
			const Matrix2f T_inv = cam_trans_inv[i].block<2, 2>(0, 0);
			cam_incrs[i].fill(0.f);

			//Foreground
			for (unsigned int u = 0; u < cols; u++)
				if (is_object[i](u))
				{
					//Warning
					if (mx_t[i](u) <= 0.f)
					{
						model_behind_camera = true;
						printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					}
					
					Matrix<float, 1, 2> res; res << res_x[i](u), res_y[i](u);
					Matrix<float, 1, 2> J_mult = -2.f*res*T_inv;

					//Control vertices
					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						const float ww = w_contverts[i](u)[cp];
						vert_incrs(0, cp) += J_mult(0)*ww;
						vert_incrs(1, cp) += J_mult(1)*ww;
					}

					//Camera pose	
					Vector4f t_point(4, 1); t_point << mx_t[i](u), my_t[i](u), 0.f, 1.f;
					for (unsigned int l = 0; l < 6; l++)
					{
						MatrixXf prod = mat_der_xi[l] * t_point;
						cam_incrs[i](l) += -2.f*(res_x[i](u)*prod(0) + res_y[i](u)*prod(1));
					}
				}

			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point(4, 1); t_point << mx_DT(s), my_DT(s), 0.f, 1.f;
				const float mx_t_DT = cam_trans_inv[i].row(0)*t_point;
				const float my_t_DT = cam_trans_inv[i].row(1)*t_point;

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 1, 2> J_pi;
				J_pi << fx*my_t_DT / square(mx_t_DT), -fx / mx_t_DT;

				const Matrix<float, 1, 2> J_mult = DT_grad[i](int(pixel_DT[i](s)))*J_pi*T_inv;

				//Control vertices
				for (unsigned int cp = 0; cp < num_verts; cp++)
				{
					const float ww = w_DT(s)[cp];
					vert_incrs(0, cp) += -alpha*J_mult(0)*ww;
					vert_incrs(1, cp) += -alpha*J_mult(1)*ww;
				}

				//Camera pose
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, 0.f, 1.f;
				for (unsigned int l = 0; l < 6; l++)
				{
					Vector2f aux_prod = (mat_der_xi[l] * m_t).block<2, 1>(0, 0);
					cam_incrs[i](l) += -alpha*(DT_grad[i](int(pixel_DT[i](s)))*J_pi*aux_prod).value();
				}
			}
		}

		energy_increasing = true;
		unsigned int cont = 0.f;

		//Update the control vertices
		while (energy_increasing)
		{
			//Update
			vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

			for (unsigned int i = 0; i < num_images; i++)
				cam_mfold[i] = cam_mfold_old[i] - adap_mult*sz_xi*cam_incrs[i];
			computeCameraTransfandPosesFromTwist();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			optimizeUDataterm_LM();
			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDTOverall();

			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 1.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 0.5f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			cont++;

			//We cannot decrease the energy
			if (cont > 5)
			{
				//Recover old variables
				vert_coords = vert_coords_old;
				cam_mfold = cam_mfold_old;
				computeCameraTransfandPosesFromTwist();
				new_energy = last_energy;
				//energy_increasing = true;
				break;			
			}
			//energy_increasing = false;
		}

		const float runtime = 1000.f*clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();
		system::sleep(10);

		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		if (new_energy > 0.9999f*last_energy)
		{
			printf("\n Optimization finished because the energy does not decrease anymore.");
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime / max_iter);
}

float Mod2DfromRGBD::computeEnergyDTOverall()
{
	float energy = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			if (is_object[i](u))
				energy += square(res_x[i](u)) + square(res_y[i](u));

		for (unsigned int s = 0; s < nsamples; s++)
			energy += alpha*DT[i](pixel_DT[i](s));
	}

	return energy;
}

void Mod2DfromRGBD::sampleSurfaceForDTBackground()
{
	//Compute the number of samples according to "nsamples_approx"
	const unsigned int nsamples_pface = nsamples_approx / num_faces;
	nsamples = nsamples_pface*num_faces;

	//Camera parameters
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float disp_u = 0.5f*float(cols - 1);

	//Resize DT variables
	w_DT.resize(nsamples);	
	u1_der_DT.resize(nsamples);
	mx_DT.resize(nsamples); my_DT.resize(nsamples);
	u1_DT.resize(nsamples); uface_DT.resize(nsamples);
	pixel_DT.resize(num_images);
	
	const float fact = 1.f / float(nsamples_pface); //It could also be .../float(nsamples_pface - 1), but I think that it does not make sense in 2D (repeated samples)
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
		}
	}
}



