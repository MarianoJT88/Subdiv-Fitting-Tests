// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "jacobian_comparison.h"


Mod3DfromRGBD::Mod3DfromRGBD()
{
	num_images = 5;
	fovh_i = utils::DEG2RAD(77.f); fovv_i = utils::DEG2RAD(57.75f);
	fovh_d = utils::DEG2RAD(90.f); fovv_d = utils::DEG2RAD(67.5f);
	downsample = 2; //I can be set to 1, 2, 4, etc.
	rows = 480/downsample; cols = 640/downsample;

	cam_poses.resize(num_images);
	cam_trans.resize(num_images);
	cam_mfold.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	intensity.resize(num_images);
	depth.resize(num_images); x_image.resize(num_images); y_image.resize(num_images);
	x_t.resize(num_images); y_t.resize(num_images); z_t.resize(num_images);
	d_labeling.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		intensity[i].resize(rows, cols);
		depth[i].resize(rows, cols); 
		x_image[i].resize(rows, cols);
		y_image[i].resize(rows, cols);
		x_t[i].resize(rows, cols); y_t[i].resize(rows, cols); z_t[i].resize(rows, cols);
		d_labeling[i].resize(rows, cols);
	}

	//Internal points
	u1.resize(num_images); u2.resize(num_images);
	uface.resize(num_images);
	u1_incr.resize(num_images); u2_incr.resize(num_images);
	res_x.resize(num_images); res_y.resize(num_images); res_z.resize(num_images);
	mx.resize(num_images); my.resize(num_images); mz.resize(num_images);
	u1_der.resize(num_images); u2_der.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1[i].resize(rows, cols); u2[i].resize(rows, cols);
		uface[i].resize(rows, cols);
		u1_incr[i].resize(rows, cols); u2_incr[i].resize(rows, cols);
		res_x[i].resize(rows, cols); res_y[i].resize(rows, cols); res_z[i].resize(rows, cols);
		mx[i].resize(rows, cols); my[i].resize(rows, cols); mz[i].resize(rows, cols);
		u1_der[i].resize(rows, cols); u2_der[i].resize(rows, cols);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				u1_der[i](v, u) = new float[3];
				u2_der[i](v, u) = new float[3];
			}
	}
}

void Mod3DfromRGBD::loadImagesFromDisc()
{
	string dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/images banana1/";
	//string dir = "../images banana1/";
	string name;
	char aux[30];
	const float norm_factor = 1.f/255.f;

	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	for (unsigned int i = 1; i <= num_images; i++)
	{
		//Intensity
		sprintf_s(aux, "i%d.png", i);
		name = dir + aux;

		cv::Mat im_i = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				intensity[i - 1](v,u) = norm_factor*im_i.at<unsigned char>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);


		//Depth
		sprintf_s(aux, "d%d.png", i);
		name = dir + aux;
		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		const float disp_v = 0.5f*(rows - 1);

		cv::Mat im_d = cv::imread(name, -1);
		cv::Mat depth_float;

		im_d.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);
				depth[i - 1](v, u) = d;
				x_image[i - 1](v, u) = (u - disp_u)*d*inv_fd;
				y_image[i - 1](v, u) = (v - disp_v)*d*inv_fd;
			}
	}
}

void Mod3DfromRGBD::loadInitialMesh()
{
	//Initial mesh - A cube...
	num_verts = 8;
	num_faces = 6;

	//Resize the weights
	w_contverts.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		w_contverts[i].resize(rows, cols);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
				w_contverts[i](v, u) = new float[num_verts];
	}

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1, 3, 2;
	face_verts.col(1) << 2, 3, 5, 4;
	face_verts.col(2) << 4, 5, 7, 6;
	face_verts.col(3) << 6, 7, 1, 0;
	face_verts.col(4) << 1, 7, 5, 3;
	face_verts.col(5) << 6, 0, 2, 4;

	//cout << endl << "Face vertices: " << endl << face_verts;

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

	//Fill the 3D coordinates of the vertices
	//Place the cube in the right place - Get the bounding box of the 3D point cloud
	float min_x = 10.f, min_y = 10.f, min_z = 10.f;
	float max_x = -10.f, max_y = -10.f, max_z = -10.f;

	for (unsigned int i = 0; i < num_images; i++)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates according to the camera pose
					const float x_t = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y_t = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z_t = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);

					if (x_t < min_x) 	min_x = x_t;
					if (x_t > max_x)	max_x = x_t;
					if (y_t < min_y) 	min_y = y_t;
					if (y_t > max_y)	max_y = y_t;
					if (z_t < min_z) 	min_z = z_t;
					if (z_t > max_z)	max_z = z_t;
				}
	}

	const float x_margin = 0.3f*(max_x - min_x);
	const float y_margin = 0.3f*(max_y - min_y);
	const float z_margin = 0.3f*(max_z - min_z);

	vert_coords.resize(3, num_verts);
	vert_coords.col(0) << min_x - x_margin, min_y - y_margin, max_z + z_margin;
	vert_coords.col(1) << max_x + x_margin, min_y - y_margin, max_z + z_margin;
	vert_coords.col(2) << min_x - x_margin, max_y + y_margin, max_z + z_margin;
	vert_coords.col(3) << max_x + x_margin, max_y + y_margin, max_z + z_margin;
	vert_coords.col(4) << min_x - x_margin, max_y + y_margin, min_z - z_margin;
	vert_coords.col(5) << max_x + x_margin, max_y + y_margin, min_z - z_margin;
	vert_coords.col(6) << min_x - x_margin, min_y - y_margin, min_z - z_margin;
	vert_coords.col(7) << max_x + x_margin, min_y - y_margin, min_z - z_margin;

	//Show the mesh on the 3D Scene
	showMesh();
}


void Mod3DfromRGBD::computeDepthSegmentation()
{
	vector<Array2i> buffer;
	Array<bool,Dynamic,Dynamic> checked(rows,cols);
	const float dist_thres = 0.0001f; //It is actually a threshold for the distance square(^2)
	
	//For every image, we find the closest area and expand it. 
	//We also check it afterwards and reject it if it does not fulfill certain conditions
	for (unsigned int i = 0; i < num_images; i++)
	{
		checked.fill(false);
		
		//Find the closest point
		float min_dist = 10.f;
		Array2i min_coords; min_coords.fill(0);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float dist_origin = square(depth[i](v, u)) + square(x_image[i](v, u)) + square(y_image[i](v, u));
				if ((dist_origin < min_dist) && (dist_origin > 0.f))
				{
					min_dist = dist_origin;
					min_coords(0) = v; min_coords(1) = u;
				}
			}

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
		d_labeling[i].swap(checked);

	}
}

void Mod3DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(max(int(num_images), 5));
	cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	cam_poses[1].setFromValues(0.f, -0.19f, 0.03f, utils::DEG2RAD(20.f), 0.f, 0.f);
	cam_poses[2].setFromValues(0.35f, 0.35f, -0.11f, utils::DEG2RAD(-65.8f), utils::DEG2RAD(-11.44f), utils::DEG2RAD(-20.1f));
	cam_poses[3].setFromValues(0.61f, 0.33f, -0.19f, utils::DEG2RAD(-97.f), utils::DEG2RAD(-20.f), utils::DEG2RAD(-43.f));
	cam_poses[4].setFromValues(0.71f, -0.33f, -0.28f, utils::DEG2RAD(158.f), utils::DEG2RAD(-49.f), utils::DEG2RAD(11.5f));

	//Store the transformation matrices and the manifold values
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Transformation
		CMatrixDouble44 aux;
		cam_poses[i].getHomogeneousMatrix(aux);
		cam_trans[i] = aux.cast<float>();
		cam_ini[i] = aux.cast<float>();
		cam_mfold[i].assign(0.f);


		//cout << endl << "Pose: " << endl << cam_poses[i];
		//cout << endl << "Manifold values" << endl << cam_mfold[i].transpose();
		//cout << endl << "Transformation matrix" << endl << cam_trans[i];
	}

	showCamPoses();


	//char c = 'y';
	//while (c == 'y')
	//{
	//	printf("\n Do you want to change the pose of any banana [y/n]? ");
	//	cin >> c;

	//	if (c != 'y')
	//		break;

	//	int num_banana;
	//	printf("\n Which banana do you want to move? (from 1 to num_bananas-1, the first one cannot be moved): ");
	//	cin >> num_banana;

	//	if (num_banana >= num_images)
	//		printf("\n We don't have so many bananas");
	//	
	//	else
	//	{
	//		printf("\n Use characters q w e r t y for positive increments");
	//		printf("\n Use characters a s d f g h for negative increments");
	//		printf("\n Push p to finish with the pose of this banana");

	//		int pushed_key = 0, stop = 0;
	//		const float incr_t = 0.01f, incr_r = 0.05f;
	//		CPose3D &pose = cam_poses[num_banana];
	//		Matrix<float, 6, 1> &mfold = cam_mfold[num_banana];
	//		Matrix4f &trans = cam_trans[num_banana];
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
	//				pose.z_incr(incr_t);
	//				break;
	//			case 'r':
	//				pose.setYawPitchRoll(pose.yaw() + incr_r, pose.pitch(), pose.roll());
	//				break;
	//			case 't':
	//				pose.setYawPitchRoll(pose.yaw(), pose.pitch() + incr_r, pose.roll());
	//				break;
	//			case 'y':
	//				pose.setYawPitchRoll(pose.yaw(), pose.pitch(), pose.roll() + incr_r);
	//				break;

	//			//negative increments
	//			case  'a':
	//				pose.x_incr(-incr_t);
	//				break;
	//			case 's':
	//				pose.y_incr(-incr_t);
	//				break;
	//			case 'd':
	//				pose.z_incr(-incr_t);
	//				break;
	//			case 'f':
	//				pose.setYawPitchRoll(pose.yaw() - incr_r, pose.pitch(), pose.roll());
	//				break;
	//			case 'g':
	//				pose.setYawPitchRoll(pose.yaw(), pose.pitch() - incr_r, pose.roll());
	//				break;
	//			case 'h':
	//				pose.setYawPitchRoll(pose.yaw(), pose.pitch(), pose.roll() - incr_r);
	//				break;

	//			case 'p':
	//				stop = 1;
	//				break;
	//			}

	//			if (pushed_key)
	//				updateCamPosesScene();
	//		}

	//		cout << endl << "Final pose for banana " << num_banana << ":" << endl << pose;
	//	}
	//}
}

void Mod3DfromRGBD::computeInitialIntPointsPinHole()
{
	TObject3D inters;
	TSegment3D ray;
	vector<TPlane3D> planes; planes.resize(2 * num_faces);

	//First, store the planes of the mesh
	for (unsigned int f = 0; f < num_faces; f++)
	{
		const unsigned int vert1 = face_verts(0, f); TPoint3D v1(vert_coords(0, vert1), vert_coords(1, vert1), vert_coords(2, vert1));
		const unsigned int vert2 = face_verts(1, f); TPoint3D v2(vert_coords(0, vert2), vert_coords(1, vert2), vert_coords(2, vert2));
		const unsigned int vert3 = face_verts(2, f); TPoint3D v3(vert_coords(0, vert3), vert_coords(1, vert3), vert_coords(2, vert3));
		const unsigned int vert4 = face_verts(3, f); TPoint3D v4(vert_coords(0, vert4), vert_coords(1, vert4), vert_coords(2, vert4));
		planes[2 * f] = TPlane3D(v1, v2, v3);
		planes[2 * f + 1] = TPlane3D(v3, v4, v1);
	}

	for (unsigned int i = 0; i < num_images; ++i)
	{
		ray.point1 = TPoint3D(cam_poses[i][0], cam_poses[i][1], cam_poses[i][2]);
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);

					//Create segment
					ray.point2 = TPoint3D(x, y, z);

					//Evaluate intersection between the mesh faces and the segment
					bool intersect = false;
					for (unsigned int f = 0; f < 2 * num_faces; f++)
					{
						intersect = math::intersect(planes[f], ray, inters);
						if (intersect)
						{
							TPoint3D p_inter; inters.getPoint(p_inter);
							const unsigned int face = f / 2;

							const TPoint3D v1f(vert_coords(0, face_verts(0, face)), vert_coords(1, face_verts(0, face)), vert_coords(2, face_verts(0, face)));
							const TPoint3D v2f(vert_coords(0, face_verts(1, face)), vert_coords(1, face_verts(1, face)), vert_coords(2, face_verts(1, face)));
							const TPoint3D v3f(vert_coords(0, face_verts(2, face)), vert_coords(1, face_verts(2, face)), vert_coords(2, face_verts(2, face)));
							const TPoint3D v4f(vert_coords(0, face_verts(3, face)), vert_coords(1, face_verts(3, face)), vert_coords(2, face_verts(3, face)));

							TPoint3D v1, v2, v3; //For the triangle

							if (f % 2 == 0)	{ v1 = v1f; v2 = v2f; v3 = v3f; }
							else			{ v1 = v3f; v2 = v4f, v3 = v1f; }

							//Check whether it intersected the plane within the boundaries or outside of the boundaries
							const float vect1[3] = {v1.x - p_inter.x, v1.y - p_inter.y, v1.z - p_inter.z};
							const float vect2[3] = {v2.x - p_inter.x, v2.y - p_inter.y, v2.z - p_inter.z};
							const float vect3[3] = {v3.x - p_inter.x, v3.y - p_inter.y, v3.z - p_inter.z};

							const float vect12[3] = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
							const float vect23[3] = {v2.x - v3.x, v2.y - v3.y, v2.z - v3.z};
							const float vect31[3] = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z};

							float c12[3]; cross_prod(vect12, vect1, c12);
							float c23[3]; cross_prod(vect23, vect2, c23);
							float c31[3]; cross_prod(vect31, vect3, c31);

							if (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c23, c31)) && (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c12, c31))))
							{
								//Find the parametric values from the 3D intersection
								float d[4];

								//Compute distances between this point and the four edges of the face
								TSegment3D edge1(v1f, v2f), edge2(v2f, v3f), edge3(v3f, v4f), edge4(v4f, v1f);
								d[0] = edge1.distance(p_inter); d[1] = edge2.distance(p_inter); d[2] = edge3.distance(p_inter); d[3] = edge4.distance(p_inter);

								u1[i](v, u) = d[3] / (d[1] + d[3]); u2[i](v, u) = d[0] / (d[0] + d[2]);
								uface[i](v, u) = face;

								break;
							}
						}
						else { ; } //Fill with any default value if they don't intersect ****************			
					}
				}
	}


	//Evaluate the surface
	evaluateSubDivSurface();

	//Draw points
	showSubSurface();
}

void Mod3DfromRGBD::computeInitialIntPointsOrtographic()
{
	TObject3D inters;
	TSegment3D ray;
	vector<TPlane3D> planes; planes.resize(2*num_faces);

	//First, store the planes of the mesh
	for (unsigned int f = 0; f < num_faces; f++)
	{
		const unsigned int vert1 = face_verts(0, f); TPoint3D v1(vert_coords(0, vert1), vert_coords(1, vert1), vert_coords(2, vert1));
		const unsigned int vert2 = face_verts(1, f); TPoint3D v2(vert_coords(0, vert2), vert_coords(1, vert2), vert_coords(2, vert2));
		const unsigned int vert3 = face_verts(2, f); TPoint3D v3(vert_coords(0, vert3), vert_coords(1, vert3), vert_coords(2, vert3));
		const unsigned int vert4 = face_verts(3, f); TPoint3D v4(vert_coords(0, vert4), vert_coords(1, vert4), vert_coords(2, vert4));
		planes[2*f] = TPlane3D(v1, v2, v3);
		planes[2*f + 1] = TPlane3D(v3, v4, v1);
	}

	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		Matrix4f &mytrans = cam_trans[i];
		TPoint3D dir(-mytrans(0, 0)*1.f, -mytrans(1, 0)*1.f, -mytrans(2, 0)*1.f);


		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);

					//Create segment
					ray.point1 = TPoint3D(x, y, z);
					ray.point2 = ray.point1 + dir;

					//Evaluate intersection between the mesh faces and the segment
					bool intersect = false;
					for (unsigned int f = 0; f < 2*num_faces; f++)
					{
						intersect = math::intersect(planes[f], ray, inters);
						if (intersect)
						{
							TPoint3D p_inter; inters.getPoint(p_inter);
							const unsigned int face = f / 2;

							const TPoint3D v1f(vert_coords(0, face_verts(0, face)), vert_coords(1, face_verts(0, face)), vert_coords(2, face_verts(0, face)));
							const TPoint3D v2f(vert_coords(0, face_verts(1, face)), vert_coords(1, face_verts(1, face)), vert_coords(2, face_verts(1, face)));
							const TPoint3D v3f(vert_coords(0, face_verts(2, face)), vert_coords(1, face_verts(2, face)), vert_coords(2, face_verts(2, face)));
							const TPoint3D v4f(vert_coords(0, face_verts(3, face)), vert_coords(1, face_verts(3, face)), vert_coords(2, face_verts(3, face)));

							TPoint3D v1, v2, v3; //For the triangle

							if (f%2 == 0)	{ v1 = v1f; v2 = v2f; v3 = v3f; }
							else			{ v1 = v3f; v2 = v4f, v3 = v1f; }
							
							//Check whether it intersected the plane within the boundaries or outside of the boundaries
							const float vect1[3] = {v1.x - p_inter.x, v1.y - p_inter.y, v1.z - p_inter.z};
							const float vect2[3] = {v2.x - p_inter.x, v2.y - p_inter.y, v2.z - p_inter.z};
							const float vect3[3] = {v3.x - p_inter.x, v3.y - p_inter.y, v3.z - p_inter.z};

							const float vect12[3] = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
							const float vect23[3] = {v2.x - v3.x, v2.y - v3.y, v2.z - v3.z};
							const float vect31[3] = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z};

							float c12[3]; cross_prod(vect12, vect1, c12);
							float c23[3]; cross_prod(vect23, vect2, c23);
							float c31[3]; cross_prod(vect31, vect3, c31);

							if (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c23, c31)) && (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c12, c31))))
							{
								//Find the parametric values from the 3D intersection
								float d[4];

								//Compute distances between this point and the four edges of the face
								TSegment3D edge1(v1f, v2f), edge2(v2f, v3f), edge3(v3f, v4f), edge4(v4f, v1f);
								d[0] = edge1.distance(p_inter); d[1] = edge2.distance(p_inter); d[2] = edge3.distance(p_inter); d[3] = edge4.distance(p_inter);

								u1[i](v, u) = d[3] / (d[1] + d[3]); u2[i](v, u) = d[0] / (d[0] + d[2]);
								uface[i](v, u) = face;

								break;
							}

						}
						else {; } //Fill with any default value if they don't intersect ****************			
					}
				}
	}

	//Evaluate the surface
	evaluateSubDivSurface();

	//Draw points
	showSubSurface();
}

void Mod3DfromRGBD::computeInitialIntPointsExpanding()
{
	//Possibility 3: Special projection that, instead of converge, like in the Pin-Hole model, expands using the Pin Hole model for this expansion
	//--------------------------------------------------------------------------------------------------------------------------------------------

	TObject3D inters;
	TSegment3D ray;
	vector<TPlane3D> planes; planes.resize(2*num_faces);

	//First, store the planes of the mesh
	for (unsigned int f = 0; f < num_faces; f++)
	{
		const unsigned int vert1 = face_verts(0, f); TPoint3D v1(vert_coords(0, vert1), vert_coords(1, vert1), vert_coords(2, vert1));
		const unsigned int vert2 = face_verts(1, f); TPoint3D v2(vert_coords(0, vert2), vert_coords(1, vert2), vert_coords(2, vert2));
		const unsigned int vert3 = face_verts(2, f); TPoint3D v3(vert_coords(0, vert3), vert_coords(1, vert3), vert_coords(2, vert3));
		const unsigned int vert4 = face_verts(3, f); TPoint3D v4(vert_coords(0, vert4), vert_coords(1, vert4), vert_coords(2, vert4));
		planes[2 * f] = TPlane3D(v1, v2, v3);
		planes[2 * f + 1] = TPlane3D(v3, v4, v1);
	}

	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);

					//Create segment
					TPoint3D dir(-mytrans(0, 0)*depth[i](v, u), -mytrans(1, 0)*depth[i](v, u), -mytrans(2, 0)*depth[i](v, u));
					ray.point1 = TPoint3D(x, y, z);
					TPoint3D ortog = ray.point1 + dir;
					TPoint3D expand(3.f*(ortog.x - cam_poses[i][0]), 3.f*(ortog.y - cam_poses[i][1]), 3.f*(ortog.z - cam_poses[i][2]));
					ray.point2 = ortog + expand;

					//Evaluate intersection between the mesh faces and the segment
					bool intersect = false;
					for (unsigned int f = 0; f < 2 * num_faces; f++)
					{
						intersect = math::intersect(planes[f], ray, inters);
						if (intersect)
						{
							TPoint3D p_inter; inters.getPoint(p_inter);
							const unsigned int face = f / 2;

							const TPoint3D v1f(vert_coords(0, face_verts(0, face)), vert_coords(1, face_verts(0, face)), vert_coords(2, face_verts(0, face)));
							const TPoint3D v2f(vert_coords(0, face_verts(1, face)), vert_coords(1, face_verts(1, face)), vert_coords(2, face_verts(1, face)));
							const TPoint3D v3f(vert_coords(0, face_verts(2, face)), vert_coords(1, face_verts(2, face)), vert_coords(2, face_verts(2, face)));
							const TPoint3D v4f(vert_coords(0, face_verts(3, face)), vert_coords(1, face_verts(3, face)), vert_coords(2, face_verts(3, face)));

							TPoint3D v1, v2, v3; //For the triangle

							if (f % 2 == 0)	{ v1 = v1f; v2 = v2f; v3 = v3f; }
							else			{ v1 = v3f; v2 = v4f, v3 = v1f; }

							//Check whether it intersected the plane within the boundaries or outside of the boundaries
							const float vect1[3] = {v1.x - p_inter.x, v1.y - p_inter.y, v1.z - p_inter.z};
							const float vect2[3] = {v2.x - p_inter.x, v2.y - p_inter.y, v2.z - p_inter.z};
							const float vect3[3] = {v3.x - p_inter.x, v3.y - p_inter.y, v3.z - p_inter.z};

							const float vect12[3] = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
							const float vect23[3] = {v2.x - v3.x, v2.y - v3.y, v2.z - v3.z};
							const float vect31[3] = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z};

							float c12[3]; cross_prod(vect12, vect1, c12);
							float c23[3]; cross_prod(vect23, vect2, c23);
							float c31[3]; cross_prod(vect31, vect3, c31);

							if (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c23, c31)) && (signbit(dot_prod(c12, c23)) == signbit(dot_prod(c12, c31))))
							{
								//Find the parametric values from the 3D intersection
								float d[4];

								//Compute distances between this point and the four edges of the face
								TSegment3D edge1(v1f, v2f), edge2(v2f, v3f), edge3(v3f, v4f), edge4(v4f, v1f);
								d[0] = edge1.distance(p_inter); d[1] = edge2.distance(p_inter); d[2] = edge3.distance(p_inter); d[3] = edge4.distance(p_inter);

								u1[i](v, u) = d[3] / (d[1] + d[3]); u2[i](v, u) = d[0] / (d[0] + d[2]);
								uface[i](v, u) = face;

								break;
							}

						}
						else { ; } //Fill with any default value if they don't intersect ****************			
					}
				}
	}

	//Evaluate subdivision surface
	evaluateSubDivSurface();

	//Draw points
	showSubSurface();
}

void Mod3DfromRGBD::computeInitialIntPointsClosest()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXXf> u1_ini, u2_ini;
	u1_ini.resize(num_faces); u2_ini.resize(num_faces);
	const unsigned int num_samp = 10;
	const float fact = 1.f / float(num_samp); //**********************
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		u1_ini[f].resize(num_samp, num_samp);
		u2_ini[f].resize(num_samp, num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				u1_ini[f](v, u) = float(u)*fact;
				u2_ini[f](v, u) = float(v)*fact;
			}
	}

	//Evaluate the surface
	vector<ArrayXXf> x_ini, y_ini, z_ini;
	x_ini.resize(num_faces); y_ini.resize(num_faces); z_ini.resize(num_faces);

	Far::PatchMap patchmap(*patchTable);
	//Far::PtexIndices ptexIndices(*refiner);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp, num_samp);
		y_ini[f].resize(num_samp, num_samp);
		z_ini[f].resize(num_samp, num_samp);

		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				// Locate the patch corresponding to the face ptex idx and (s,t)
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1_ini[f](v, u), u2_ini[f](v, u)); assert(handle);

				// Evaluate the patch weights, identify the CVs and compute the limit frame:
				patchTable->EvaluateBasis(*handle, u1_ini[f](v, u), u2_ini[f](v, u), pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Save the 3D coordinates
				x_ini[f](v, u) = eval.point[0];
				y_ini[f](v, u) = eval.point[1];
				z_ini[f](v, u) = eval.point[2];
			}
	}

	//Find the closest point to each of the observed with the cameras
	//----------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);


					float min_dist = 100.f, u1_min = 0.f, u2_min = 0.f, dist;
					unsigned int f_min = 0;

					for (unsigned int f = 0; f < num_faces; ++f)
						for (unsigned int uu = 0; uu < num_samp; uu++)
							for (unsigned int vv = 0; vv < num_samp; vv++)
								if ((dist = square(x - x_ini[f](vv, uu)) + square(y - y_ini[f](vv, uu)) + square(z - z_ini[f](vv, uu))) < min_dist)
								{
									min_dist = dist;
									u1_min = u1_ini[f](vv, uu);
									u2_min = u2_ini[f](vv, uu);
									f_min = f;
								}

					u1[i](v, u) = u1_min;
					u2[i](v, u) = u2_min;
					uface[i](v, u) = f_min;
				}
	}

	//scene = window.get3DSceneAndLock();

	//CPointCloudPtr points = opengl::CPointCloud::Create();
	//points->setColor(0.f, 0.f, 0.f);
	//points->setPointSize(2.f);
	//scene->insert(points);

	//for (unsigned int f = 0; f<num_faces; ++f)
	//	for (unsigned int u = 0; u < num_samp; u++)
	//		for (unsigned int v = 0; v < num_samp; v++)
	//				points->insertPoint(x_ini[f](v, u), y_ini[f](v, u), z_ini[f](v, u));

	//window.unlockAccess3DScene();
	//window.repaint();

	//Evaluate the surface
	evaluateSubDivSurface();

	//Draw points
	showSubSurface();

}

void Mod3DfromRGBD::computeTransCoordAndResiduals()
{
	for (unsigned int i = 0; i < num_images; i++)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					x_t[i](v,u) = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					y_t[i](v,u) = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					z_t[i](v,u) = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);
					
					res_x[i](v, u) = x_t[i](v, u) - mx[i](v, u);
					res_y[i](v, u) = y_t[i](v, u) - my[i](v, u);
					res_z[i](v, u) = z_t[i](v, u) - mz[i](v, u);
				}
	}
}


void Mod3DfromRGBD::fillGradControlVertices()
{
	unsigned int cont = 0;

	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					//Fill the columns for a given point (3 rows - x,y,z)
					float v_weight;
					for (unsigned int vert = 0; vert < num_verts; vert++)
						if ((v_weight = w_contverts[i](v, u)[vert]) > 0.f)
						{
							j_elem.push_back(Tri(cont, 3 * vert, -v_weight));
							j_elem.push_back(Tri(cont + 1, 3 * vert + 1, -v_weight));
							j_elem.push_back(Tri(cont + 2, 3 * vert + 2, -v_weight));
						}
					cont += 3;
				}
}

void Mod3DfromRGBD::fillGradCameraPoses()
{
	unsigned int cont = 0;
	const unsigned int col_offset = 3 * num_verts;

	//Create the matrices templates of the derivatives
	Matrix4f mat_der_xi[6];
	for (unsigned int l = 0; l < 6; l++)
		mat_der_xi[l].assign(0.f);

	mat_der_xi[0](0, 3) = 1.f;
	mat_der_xi[1](1, 3) = 1.f;
	mat_der_xi[2](2, 3) = 1.f;
	mat_der_xi[3](1, 2) = -1.f; mat_der_xi[3](2, 1) = 1.f;
	mat_der_xi[4](0, 2) = 1.f; mat_der_xi[4](2, 0) = -1.f;
	mat_der_xi[5](0, 1) = -1.f; mat_der_xi[5](1, 0) = 1.f;

	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					const Vector4f t_point(x_t[i](v, u), y_t[i](v, u), z_t[i](v, u), 1.f);
					for (unsigned int l = 0; l < 6; l++)
					{
						MatrixXf prod = mat_der_xi[l] * t_point;

						j_elem.push_back(Tri(cont, 6 * i + l + col_offset, prod(0)));				
						j_elem.push_back(Tri(cont + 1, 6 * i + l + col_offset, prod(1)));	
						j_elem.push_back(Tri(cont + 2, 6 * i + l + col_offset, prod(2)));
					}

					cont += 3;
				}
}

void Mod3DfromRGBD::fillGradInternalPoints()
{
	unsigned int cont_r = 0, cont_c = 3 * num_verts + 6 * num_images;

	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					j_elem.push_back(Tri(cont_r, cont_c, -u1_der[i](v,u)[0]));
					j_elem.push_back(Tri(cont_r++, cont_c + 1, -u2_der[i](v,u)[0]));
					j_elem.push_back(Tri(cont_r, cont_c, -u1_der[i](v,u)[1]));
					j_elem.push_back(Tri(cont_r++, cont_c + 1, -u2_der[i](v,u)[1]));
					j_elem.push_back(Tri(cont_r, cont_c, -u1_der[i](v,u)[2]));
					j_elem.push_back(Tri(cont_r++, cont_c + 1, -u2_der[i](v,u)[2]));

					cont_c += 2;
				}
}


void Mod3DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1000, 900);
	window.setPos(900, 0);
	window.setCameraZoom(3);
	window.setCameraAzimuthDeg(0);
	window.setCameraElevationDeg(45);
	window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	scene = window.get3DSceneAndLock();

	// Lights:
	//scene->getViewport()->setNumberOfLights(1);
	//mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	//light0.light_ID = 0;
	//light0.setPosition(2.5f,0,0.f,1.f);
	//light0.setDirection(0.f, 0.f, -1.f);

	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	control_mesh->setLineWidth(1.f);
	//control_mesh->setFaceColor(1.f, 0.f, 0.f);
	//control_mesh->setEdgeColor(0.f, 1.f, 0.f);
	//control_mesh->setVertColor(0.f, 0.f, 1.f);
	scene->insert(control_mesh);

	//Vertex numbers
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
	reference->setScale(0.2f);
	scene->insert(reference);

	//Frustums
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CFrustumPtr frustum = opengl::CFrustum::Create(0.01f, 0.1f, utils::RAD2DEG(fovh_d), utils::RAD2DEG(fovv_d), 1.5f, true, false);
		frustum->setColor(0.4f, 0.f, 0.f);
		scene->insert(frustum);
	}

	//Points
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
		points->setPointSize(3.f);
		points->enablePointSmooth(true);
		scene->insert(points);
	}

	//Internal model (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(3.f);
	points->enablePointSmooth(true);
	scene->insert(points);

	//Whole subdivision surface
	opengl::CPointCloudPtr subsurface = opengl::CPointCloud::Create();
	subsurface->setPointSize(1.f);
	subsurface->enablePointSmooth(true);
	subsurface->setPose(CPose3D(0.f, 0.8f, 0.f, 0.f, 0.f, 0.f));
	scene->insert(subsurface);


	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showCamPoses()
{
	scene = window.get3DSceneAndLock();

	for (unsigned int i = 0; i < num_images; i++)
	{
		//Points (bananas)
		opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(i);
		points->setPose(cam_poses[i]);
		points->clear();

		float r, g, b;
		utils::colormap(mrpt::utils::cmJET, float(i)/float(num_images), r, g, b);

		for (unsigned int v = 0; v < rows; v++)
			for (unsigned int u = 0; u < cols; u++)
				if (d_labeling[i](v,u))
					points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), r, g, b);

		//Cameras
		opengl::CFrustumPtr frustum = scene->getByClass<CFrustum>(i);
		frustum->setPose(cam_poses[i]);
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showMesh()
{
	scene = window.get3DSceneAndLock();

	//Control mesh
	opengl::CMesh3DPtr control_mesh = scene->getByClass<CMesh3D>(0);
	//control_mesh->loadMesh(g_nverts, g_nfaces, g_vertsperface, g_faceverts, g_verts);
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);

	////Show vertex numbers
	//for (unsigned int v = 0; v < num_verts; v++)
	//{
	//	opengl::CText3DPtr vert_nums = scene->getByClass<CText3D>(v);
	//	vert_nums->setLocation(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));
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

void Mod3DfromRGBD::showSubSurface()
{
	scene = window.get3DSceneAndLock();

	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(num_images);
	points->clear();

	for (unsigned int i = 0; i<num_images; ++i)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
					points->push_back(mx[i](v, u), my[i](v, u), mz[i](v, u), 0.f, 0.f, 0.f);


	//Show the whole surface
	const unsigned int sampl = 20.f;
	const float fact = 1.f / float(sampl-1);
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	CPointCloudPtr subsurface = scene->getByClass<CPointCloud>(0); subsurface->clear();

	for (unsigned int f = 0; f<num_faces; f++)
		for (unsigned int u = 0; u < sampl; u++)
			for (unsigned int v = 0; v < sampl; v++)
			{
				// Locate the patch corresponding to the face ptex idx and (s,t)
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u*fact, v*fact); assert(handle);

				// Evaluate the patch weights, identify the CVs and compute the limit frame:
				patchTable->EvaluateBasis(*handle, u*fact, v*fact, pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Insert the point
				subsurface->insertPoint(eval.point[0], eval.point[1], eval.point[2]);
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

	int *vertsperface; vertsperface = new int[num_faces];
	for (unsigned int i = 0; i < num_faces; i++)
	{
		if (is_quad(i))
			vertsperface[i] = 4;
		else
			vertsperface[i] = 3;
	}

	desc.numVertsPerFace = vertsperface; // g_vertsperface;
	desc.vertIndicesPerFace = face_verts.data();  //g_faceverts;

	//---------------------------------------------------------------
	//desc.numCreases = g_ncreases;
	//desc.creaseVertexIndexPairs = g_creaseverts;
	//desc.creaseWeights = g_creaseweights;
	//---------------------------------------------------------------

	//Instantiate a FarTopologyRefiner from the descriptor.
	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
				Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

	const int maxIsolation = 0; //Don't change it!
	refiner->RefineAdaptive( Far::TopologyRefiner::AdaptiveOptions(maxIsolation));


	// Generate a set of Far::PatchTable that we will use to evaluate the surface limit
	Far::PatchTableFactory::Options patchOptions;
	patchOptions.endCapType = Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

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
	//int ncontvert = st->GetNumControlVertices(); //printf("\n Num of control vert - %d", ncontvert);

	Far::Stencil *st = new Far::Stencil[nstencils];
	for (unsigned int i = 0; i < nstencils; i++)
	{
		st[i] = stenciltab->GetStencil(i);
		//unsigned int size_st = st[i].GetSize();
		//Far::Index const *ind = st[i].GetVertexIndices();
		//float const *wei = st[i].GetWeights();
		//printf("\n Stencil %d: ", i);
		//for (unsigned int i = 0; i < size_st; i++)
		//	printf("V=%d, W=%0.3f ,", ind[i], wei[i]);
	}
	
	
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	//Far::PtexIndices ptexIndices(*refiner);  // Far::PtexIndices helps to find indices of ptex faces.

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int i = 0; i<num_images; ++i)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
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

					//Save the derivatives
					u1_der[i](v, u)[0] = eval.deriv1[0];
					u1_der[i](v, u)[1] = eval.deriv1[1];
					u1_der[i](v, u)[2] = eval.deriv1[2];
					u2_der[i](v, u)[0] = eval.deriv2[0];
					u2_der[i](v, u)[1] = eval.deriv2[1];
					u2_der[i](v, u)[2] = eval.deriv2[2];

					//Compute the weights for the gradient with respect to the control vertices
					//Set all weights to zero
					for (unsigned int k = 0; k < num_verts; k++)
						w_contverts[i](v, u)[k] = 0.f;

					for (int cv = 0; cv < cvs.size(); ++cv)
					{						
						if (cvs[cv] < num_verts)
						{
							w_contverts[i](v, u)[cvs[cv]] += pWeights[cv];
							//printf("\n Warning!! It links to the original vertex directly!! vert = %d, pweight = %f", cvs[cv], pWeights[cv]);
						}
						else
						{
							const unsigned int ind_offset = cvs[cv] - num_verts;
							//Look at the stencil associated to this local point and distribute its weight over the control vertices
							unsigned int size_st = st[ind_offset].GetSize();
							Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
							float const *st_weights = st[ind_offset].GetWeights();
							for (unsigned int s = 0; s < size_st; s++)
								w_contverts[i](v, u)[st_ind[s]] += pWeights[cv] * st_weights[s];
						}
					}
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

	//Resize the weights
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
				w_contverts[i](v, u) = new float[num_verts];
	}

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
		for (unsigned int v = 0; v < face_v.size(); v++)
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
	showMesh();
}


void Mod3DfromRGBD::computeJacobianFiniteDifferences()
{
	//Make sure that everything was already evaluated ****************************

	//Resize the jacobian and the residual vector
	unsigned int jac_rows = 0, jac_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					jac_rows += 3;
					jac_cols += 2;
				}

	J_findif.resize(jac_rows, jac_cols);

	//Compute the residuals
	computeTransCoordAndResiduals();
	
	//Fill the Jacobian with the gradients with respect to the control vertices
	//================================================================================================
	const float contr_vert_incr = 0.1f;
	const float inv_vert_incr = 1.f / contr_vert_incr;
	unsigned int cont;

	for (unsigned int vert = 0; vert < num_verts; vert++)
		for (unsigned int c = 0; c < 3; c++)
		{
			//Move the coordinate "c" of the vertex "vert" forward
			vert_coords(c, vert) += contr_vert_incr;

			//Re-evaluate the subdivision surface
			createTopologyRefiner();
			Far::PatchMap patchmap(*patchTable);

			float pWeights[20], dsWeights[20], dtWeights[20];
			cont = 0;

			//Evaluate the surface with parametric coordinates
			for (unsigned int i = 0; i<num_images; ++i)
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (d_labeling[i](v, u))
						{
							// Locate the patch corresponding to the face ptex idx and (s,t)
							Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface[i](v, u), u1[i](v, u), u2[i](v, u)); assert(handle);

							// Evaluate the patch weights, identify the CVs and compute the limit frame:
							patchTable->EvaluateBasis(*handle, u1[i](v, u), u2[i](v, u), pWeights, dsWeights, dtWeights);

							Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

							LimitFrame eval; eval.Clear();
							for (int cv = 0; cv < cvs.size(); ++cv)
								eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

							//Save the 3D coordinates
							float grad[3];
							grad[0]	= inv_vert_incr*(eval.point[0] - mx[i](v, u));
							grad[1] = inv_vert_incr*(eval.point[1] - my[i](v, u));
							grad[2] = inv_vert_incr*(eval.point[2] - mz[i](v, u));

							if (grad[0] != 0.f)
								j_elem_fd.push_back(Tri(cont + c, 3 * vert, -grad[0]));
							if (grad[1] != 0.f)
								j_elem_fd.push_back(Tri(cont + c, 3 * vert + 1, -grad[1]));
							if (grad[2] != 0.f)
								j_elem_fd.push_back(Tri(cont + c, 3 * vert + 2, -grad[2]));

							cont += 3;
						}

			//Undo the change
			vert_coords(c, vert) -= contr_vert_incr;
		}
	createTopologyRefiner();


	//Fill the Jacobian with the gradients with respect to the camera parameters
	//==================================================================================================
	const unsigned int col_offset = 3 * num_verts;
	const float cam_pose_incr = 0.002f;
	const float inv_cam_incr = 1.f / cam_pose_incr;
	Matrix4f kai_mat;

	for (unsigned int c = 0; c < 6; c++)
	{
		cont = 0;
		for (unsigned int i = 0; i < num_images; i++)
		{
			cam_mfold[i](c) += cam_pose_incr;
			kai_mat << 0.f, -cam_mfold[i](5), cam_mfold[i](4), cam_mfold[i](0),
				cam_mfold[i](5), 0.f, -cam_mfold[i](3), cam_mfold[i](1),
				-cam_mfold[i](4), cam_mfold[i](3), 0.f, cam_mfold[i](2),
				0.f, 0.f, 0.f, 0.f;
			Matrix4f new_trans = kai_mat.exp();
			MatrixXf prod = new_trans*cam_ini[i];
			cam_trans[i] = prod.topLeftCorner<4, 4>();

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (d_labeling[i](v, u))
					{
						const Vector4f t_point(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 1.f);
						const MatrixXf p_trans = cam_trans[i] * t_point;

						float grad[3];
						grad[0] = inv_cam_incr*(p_trans(0) - x_t[i](v, u));
						grad[1] = inv_cam_incr*(p_trans(1) - y_t[i](v, u));
						grad[2] = inv_cam_incr*(p_trans(2) - z_t[i](v, u));

						if (grad[0] != 0.f)
							j_elem_fd.push_back(Tri(cont, 6 * i + c + col_offset, grad[0]));
						if (grad[1] != 0.f)
							j_elem_fd.push_back(Tri(cont + 1, 6 * i + c + col_offset, grad[1]));
						if (grad[2] != 0.f)
							j_elem_fd.push_back(Tri(cont + 2, 6 * i + c + col_offset, grad[2]));

						cont += 3;
					}

			cam_mfold[i](c) -= cam_pose_incr;
			kai_mat << 0.f, -cam_mfold[i](5), cam_mfold[i](4), cam_mfold[i](0),
				cam_mfold[i](5), 0.f, -cam_mfold[i](3), cam_mfold[i](1),
				-cam_mfold[i](4), cam_mfold[i](3), 0.f, cam_mfold[i](2),
				0.f, 0.f, 0.f, 0.f;
			new_trans = kai_mat.exp();
			prod = new_trans*cam_ini[i];
			cam_trans[i] = prod.topLeftCorner<4, 4>();
		}
	}

	//Fill the Jacobian with the gradients with respect to the internal points
	//==============================================================================================
	unsigned int cont_r = 0, cont_c = 3 * num_verts + 6 * num_images;
	const float u_incr = 0.002f;
	const float inv_u_incr = 1.f / u_incr;
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					float grad[3];

					//Change in u1
					Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(uface[i](v, u), u1[i](v, u) + u_incr, u2[i](v, u)); assert(handle);
					patchTable->EvaluateBasis(*handle1, u1[i](v, u) + u_incr, u2[i](v, u), pWeights, dsWeights, dtWeights);
					Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);

					LimitFrame eval; eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);
					
					grad[0] = inv_u_incr*(eval.point[0] - mx[i](v, u));
					grad[1] = inv_u_incr*(eval.point[1] - my[i](v, u));
					grad[2] = inv_u_incr*(eval.point[2] - mz[i](v, u));

					if (grad[0] != 0.f)
						j_elem_fd.push_back(Tri(cont_r, cont_c, -grad[0]));
					if (grad[1] != 0.f)
						j_elem_fd.push_back(Tri(cont_r + 1, cont_c, -grad[1]));
					if (grad[2] != 0.f)
						j_elem_fd.push_back(Tri(cont_r + 2, cont_c, -grad[2]));
					
					cont_c++;

					//Change in u2
					Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(uface[i](v, u), u1[i](v, u), u2[i](v, u) + u_incr); assert(handle);
					patchTable->EvaluateBasis(*handle2, u1[i](v, u), u2[i](v, u) + u_incr, pWeights, dsWeights, dtWeights);
					cvs = patchTable->GetPatchVertices(*handle2);

					eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

					grad[0] = inv_u_incr*(eval.point[0] - mx[i](v, u));
					grad[1] = inv_u_incr*(eval.point[1] - my[i](v, u));
					grad[2] = inv_u_incr*(eval.point[2] - mz[i](v, u));

					if (grad[0] != 0.f)
						j_elem_fd.push_back(Tri(cont_r, cont_c, -grad[0]));
					if (grad[1] != 0.f)
						j_elem_fd.push_back(Tri(cont_r + 1, cont_c, -grad[1]));
					if (grad[2] != 0.f)
						j_elem_fd.push_back(Tri(cont_r + 2, cont_c, -grad[2]));

					cont_c++;
					cont_r += 3;
				}

	//Build the Jacobian
	J_findif.setFromTriplets(j_elem_fd.begin(), j_elem_fd.end()); j_elem_fd.clear();
}

void Mod3DfromRGBD::compareJacobians()
{
	//Compare them
	const float sum_dif_block1 = (J - J_findif).leftCols(3 * num_verts).cwiseAbs().sum();

	printf("\n sum differences jacobian block 1 = %f", sum_dif_block1);

	SparseMatrix<float> block = J_findif.block(0, 0, 20, 3 * num_verts);
	//cout << endl << "Block of J_findif: " << endl << block;

}

void Mod3DfromRGBD::computeJacobianAnalytical()
{
	//Resize the jacobian and the residual vector
	unsigned int jac_rows = 0, jac_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (d_labeling[i](v, u))
				{
					jac_rows += 3;
					jac_cols += 2;
				}

	J.resize(jac_rows, jac_cols);

	//Compute the residuals
	computeTransCoordAndResiduals();

	//Fill the Jacobian with the gradients with respect to the control vertices
	fillGradControlVertices();

	//Fill the Jacobian with the gradients with respect to the camera parameters 
	fillGradCameraPoses();

	//Fill the Jacobian with the gradients with respect to the internal points
	fillGradInternalPoints();

	//Build the Jacobian
	J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
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



