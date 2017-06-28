// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "uShape_test.h"


Mod3DfromRGBD::Mod3DfromRGBD()
{	
	num_images = 1;
	//image_set = 2;
	//fovh_d = utils::DEG2RAD(78.2f);
	fovh_d = utils::DEG2RAD(49.35f);
	fovv_d = utils::DEG2RAD(49.35f);
	downsample = 8; //It can be set to 1, 2, 4, etc.
	rows = 1080/downsample; cols = 1080/downsample; //cols = 1920/downsample;
	cam_prior = 0.f;
	max_iter = 100;
	Kn = 0.01f;
	tau = 2.f; 
	alpha_raycast = 2e-4f/tau;
	peak_margin = 0.05f;
	nsamples_approx = 5000;
	alpha_DT = 0.5f/float(nsamples_approx);
	Kn = 0.005f;
	s_reg = 4;
	ctf_level = 1;

	cam_poses.resize(num_images);
	cam_incrs.resize(num_images);
	cam_trans.resize(num_images);
	cam_trans_inv.resize(num_images);
	cam_mfold.resize(num_images); cam_mfold_old.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	intensity.resize(num_images);
	depth.resize(num_images); x_image.resize(num_images); y_image.resize(num_images);
	nx_image.resize(num_images); ny_image.resize(num_images); nz_image.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images);
	DT.resize(num_images); DT_grad_u.resize(num_images), DT_grad_v.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		intensity[i].resize(rows, cols);
		depth[i].resize(rows, cols); x_image[i].resize(rows, cols); y_image[i].resize(rows, cols);
		nx_image[i].resize(rows, cols); ny_image[i].resize(rows, cols); nz_image[i].resize(rows, cols);
		is_object[i].resize(rows, cols); valid[i].resize(rows, cols);
		DT[i].resize(rows, cols); DT_grad_u[i].resize(rows, cols); DT_grad_v[i].resize(rows, cols);
	}

	//Internal points
	u1.resize(num_images); u2.resize(num_images);
	u1_old.resize(num_images); u2_old.resize(num_images);
	u1_old_outer.resize(num_images); u2_old_outer.resize(num_images);
	uface.resize(num_images); uface_old.resize(num_images); uface_old_outer.resize(num_images);
	u1_incr.resize(num_images); u2_incr.resize(num_images);
	res_x.resize(num_images); res_y.resize(num_images); res_z.resize(num_images);
	res_nx.resize(num_images); res_ny.resize(num_images); res_nz.resize(num_images);
	res_d1.resize(num_images); res_d2.resize(num_images);
	mx.resize(num_images); my.resize(num_images); mz.resize(num_images);
	mx_t.resize(num_images); my_t.resize(num_images); mz_t.resize(num_images);
	nx.resize(num_images); ny.resize(num_images); nz.resize(num_images);
	nx_t.resize(num_images); ny_t.resize(num_images); nz_t.resize(num_images);
	u1_der.resize(num_images); u2_der.resize(num_images);
	n_der_u1.resize(num_images); n_der_u2.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		u1[i].resize(rows, cols); u2[i].resize(rows, cols);
		u1_old[i].resize(rows, cols); u2_old[i].resize(rows, cols);
		u1_old_outer[i].resize(rows, cols); u2_old_outer[i].resize(rows, cols);
		uface[i].resize(rows, cols); uface_old[i].resize(rows, cols); uface_old_outer[i].resize(rows, cols);
		u1_incr[i].resize(rows, cols); u2_incr[i].resize(rows, cols);
		res_x[i].resize(rows, cols); res_y[i].resize(rows, cols); res_z[i].resize(rows, cols);
		res_nx[i].resize(rows, cols); res_ny[i].resize(rows, cols); res_nz[i].resize(rows, cols);
		res_d1[i].resize(rows, cols); res_d2[i].resize(rows, cols);
		mx[i].resize(rows, cols); my[i].resize(rows, cols); mz[i].resize(rows, cols);
		mx_t[i].resize(rows, cols); my_t[i].resize(rows, cols); mz_t[i].resize(rows, cols);
		nx[i].resize(rows, cols); ny[i].resize(rows, cols); nz[i].resize(rows, cols);
		nx_t[i].resize(rows, cols); ny_t[i].resize(rows, cols); nz_t[i].resize(rows, cols);
		u1_der[i].resize(rows, cols); u2_der[i].resize(rows, cols);
		n_der_u1[i].resize(rows,cols); n_der_u2[i].resize(rows,cols);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				u1_der[i](v,u) = new float[3];
				u2_der[i](v,u) = new float[3];
				n_der_u1[i](v,u) = new float[3];
				n_der_u2[i](v,u) = new float[3];
			}
	}

	//Resize the weights
	w_indices.resize(num_images);
	w_contverts.resize(num_images);
	w_u1.resize(num_images); w_u2.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		w_contverts[i].resize(max_num_w, rows*cols);
		w_u1[i].resize(max_num_w, rows*cols);
		w_u2[i].resize(max_num_w, rows*cols);
		w_indices[i].resize(max_num_w, rows*cols);
	}

	//Fill the matrices for the regularization terms
	Q_m_sqrt << 0.0074,    0.0090,   -0.0031,    0.0008,   0.0090,    0.0040,   -0.0075,   -0.0001,   -0.0031,   -0.0075,   -0.0045,   -0.0022,    0.0008,   -0.0001,   -0.0022,   -0.0004,
				0.0090,    0.0749,    0.0283,   -0.0031,   -0.0016,    0.0344,   -0.0097,   -0.0132,   -0.0132,   -0.0537,   -0.0418,   -0.0105,   -0.0001,    0.0053,   -0.0026,   -0.0022,
				-0.0031,    0.0283,    0.0749,    0.0090,   -0.0132,   -0.0097,    0.0344,   -0.0016,   -0.0105,   -0.0418,   -0.0537,   -0.0132,   -0.0022,   -0.0026,    0.0053,   -0.0001,
				 0.0008,  -0.0031,    0.0090,    0.0074,   -0.0001,   -0.0075,    0.0040,    0.0090,   -0.0022,   -0.0045,   -0.0075,   -0.0031,   -0.0004,   -0.0022,   -0.0001,    0.0008,
				 0.0090,   -0.0016,   -0.0132,   -0.0001,    0.0749,    0.0344,   -0.0537,    0.0053,    0.0283,   -0.0097,   -0.0418,   -0.0026,   -0.0031,   -0.0132,   -0.0105,   -0.0022,
				 0.0040,    0.0344,   -0.0097,   -0.0075,    0.0344,    0.3211,   -0.0146,   -0.0537,   -0.0097,   -0.0146,   -0.1344,   -0.0418,   -0.0075,   -0.0537,   -0.0418,   -0.0045,
				-0.0075,   -0.0097,    0.0344,    0.0040,   -0.0537,   -0.0146,    0.3211,    0.0344,   -0.0418,   -0.1344,   -0.0146,   -0.0097,   -0.0045,   -0.0418,   -0.0537,   -0.0075,
				-0.0001,   -0.0132,   -0.0016,    0.0090,    0.0053,   -0.0537,   0.0344,    0.0749,   -0.0026,   -0.0418,   -0.0097,   0.0283,   -0.0022,   -0.0105,   -0.0132,   -0.0031,
				-0.0031,   -0.0132,   -0.0105,   -0.0022,    0.0283,   -0.0097,   -0.0418,   -0.0026,    0.0749,    0.0344,   -0.0537,    0.0053,    0.0090,   -0.0016,   -0.0132,   -0.0001,
				-0.0075,   -0.0537,   -0.0418,   -0.0045,   -0.0097,   -0.0146,   -0.1344,   -0.0418,    0.0344,    0.3211,   -0.0146,   -0.0537,    0.0040,    0.0344,   -0.0097,   -0.0075,
				-0.0045,   -0.0418,   -0.0537,   -0.0075,   -0.0418,   -0.1344,   -0.0146,   -0.0097,   -0.0537,   -0.0146,    0.3211,    0.0344,   -0.0075,   -0.0097,    0.0344,    0.0040,
				-0.0022,   -0.0105,   -0.0132,   -0.0031,   -0.0026,   -0.0418,   -0.0097,    0.0283,    0.0053,   -0.0537,    0.0344,    0.0749,   -0.0001,   -0.0132,   -0.0016,    0.0090,
				 0.0008,   -0.0001,  -0.0022,   -0.0004,  -0.0031,   -0.0075,   -0.0045,   -0.0022,    0.0090,    0.0040,   -0.0075,   -0.0001,    0.0074,    0.0090,   -0.0031,    0.0008,
				-0.0001,    0.0053,   -0.0026,   -0.0022,   -0.0132,   -0.0537,   -0.0418,   -0.0105,   -0.0016,    0.0344,   -0.0097,   -0.0132,    0.0090,    0.0749,    0.0283,   -0.0031,
				-0.0022,   -0.0026,    0.0053,   -0.0001,   -0.0105,   -0.0418,   -0.0537,   -0.0132,   -0.0132,   -0.0097,    0.0344,   -0.0016,   -0.0031,    0.0283,    0.0749,    0.0090,
				-0.0004,   -0.0022,   -0.0001,    0.0008,   -0.0022,   -0.0045,   -0.0075,   -0.0031,   -0.0001,   -0.0075,    0.0040,    0.0090,    0.0008,   -0.0031,    0.0090,    0.0074;
		
	Q_tp_sqrt << 0.0470,    0.0144,   -0.0336,    0.0053,    0.0144,   -0.0348,   -0.0205,    0.0057,   -0.0336,   -0.0205,    0.0178,    0.0104,    0.0053,    0.0057,    0.0104,    0.0077,
				0.0144,    0.2456,    0.0130,   -0.0336,   -0.0821,   -0.0828,   -0.1426,   -0.0374,   -0.0374,   -0.0557,    0.0363,    0.0314,    0.0057,    0.0636,    0.0523,    0.0104,
				-0.0336,    0.0130,    0.2456,    0.0144,   -0.0374,   -0.1426,   -0.0828,   -0.0821,    0.0314,    0.0363,   -0.0557,   -0.0374,    0.0104,    0.0523,    0.0636,    0.0057,
				0.0053,   -0.0336,    0.0144,    0.0470,    0.0057,   -0.0205,   -0.0348,    0.0144,    0.0104,    0.0178,   -0.0205,   -0.0336,    0.0077,    0.0104,    0.0057,    0.0053,
				0.0144,   -0.0821,   -0.0374,    0.0057,    0.2456,   -0.0828,   -0.0557,    0.0636,    0.0130,   -0.1426,    0.0363,    0.0523,   -0.0336,   -0.0374,    0.0314,    0.0104,
				-0.0348,   -0.0828,   -0.1426,   -0.0205,   -0.0828,    0.7441,   -0.0654,   -0.0557,   -0.1426,   -0.0654,   -0.0647,    0.0363,   -0.0205,   -0.0557,    0.0363,    0.0178,
				-0.0205,   -0.1426,   -0.0828,   -0.0348,   -0.0557,   -0.0654,    0.7441,   -0.0828,    0.0363,   -0.0647,   -0.0654,   -0.1426,    0.0178,    0.0363,   -0.0557,   -0.0205,
				0.0057,   -0.0374,   -0.0821,    0.0144,    0.0636,   -0.0557,   -0.0828,    0.2456,    0.0523,    0.0363,   -0.1426,    0.0130,    0.0104,    0.0314,   -0.0374,   -0.0336,
				-0.0336,   -0.0374,    0.0314,    0.0104,    0.0130,   -0.1426,    0.0363,    0.0523,    0.2456,   -0.0828,   -0.0557,    0.0636,    0.0144,   -0.0821,   -0.0374,    0.0057,
				-0.0205,   -0.0557,    0.0363,    0.0178,   -0.1426,   -0.0654,   -0.0647,    0.0363,   -0.0828,    0.7441,   -0.0654,   -0.0557,   -0.0348,   -0.0828,   -0.1426,   -0.0205,
				0.0178,    0.0363,   -0.0557,   -0.0205,    0.0363,   -0.0647,   -0.0654,   -0.1426,   -0.0557,   -0.0654,    0.7441,   -0.0828,   -0.0205,   -0.1426,   -0.0828,   -0.0348,
				0.0104,    0.0314,   -0.0374,   -0.0336,    0.0523,    0.0363,   -0.1426,    0.0130,    0.0636,   -0.0557,   -0.0828,    0.2456,    0.0057,   -0.0374,   -0.0821,    0.0144,
				0.0053,    0.0057,    0.0104,    0.0077,   -0.0336,   -0.0205,    0.0178,    0.0104,    0.0144,   -0.0348,   -0.0205,    0.0057,    0.0470,    0.0144,   -0.0336,    0.0053,
				0.0057,    0.0636,    0.0523,    0.0104,   -0.0374,   -0.0557,    0.0363,    0.0314,   -0.0821,   -0.0828,   -0.1426,   -0.0374,    0.0144,    0.2456,    0.0130,   -0.0336,
				0.0104,    0.0523,    0.0636,    0.0057,    0.0314,    0.0363,   -0.0557,   -0.0374,   -0.0374,   -0.1426,   -0.0828,   -0.0821,   -0.0336,    0.0130,    0.2456,    0.0144,
				0.0077,    0.0104,    0.0057,    0.0053,    0.0104,    0.0178,   -0.0205,   -0.0336,    0.0057,   -0.0205,   -0.0348,    0.0144,    0.0053,   -0.0336,    0.0144,    0.0470;
}

void Mod3DfromRGBD::loadUShape()
{
	string dir;
	dir = "C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting/data/u shape/";

	string name;
	char aux[30];
	const float norm_factor = 1.f/255.f;
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	//Depth
	sprintf_s(aux, "d1.png");
	name = dir + aux;
	const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
	const float disp_u = 0.5f*(cols - 1);
	const float disp_v = 0.5f*(rows - 1);
	const float depth_offset = 1.f;

	cv::Mat im_d = cv::imread(name, -1);
	cv::Mat depth_float;

	im_d.convertTo(depth_float, CV_32FC1, 1.0 / 200000.0);
	for (unsigned int u = 0; u < cols; u++)
		for (unsigned int v = 0; v<rows; v++)
		{
			const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample) + depth_offset;
			depth[0](v, u) = d;
			x_image[0](v, u) = (u - disp_u)*d*inv_fd;
			y_image[0](v, u) = (v - disp_v)*d*inv_fd;
		}

}

void Mod3DfromRGBD::loadInitialMesh()
{
	//Initial mesh - A cube...
	num_verts = 8;
	num_faces = 6;

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
				if (is_object[i](v, u))
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

	const float x_margin = 0.1f*(max_x - min_x);
	const float y_margin = 0.1f*(max_y - min_y);
	const float z_margin = 0.1f*(max_z - min_z);

	vert_incrs.resize(3, num_verts);
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
	vert_coords.col(0) << min_x - x_margin, min_y - y_margin, max_z + z_margin;
	vert_coords.col(1) << max_x + x_margin, min_y - y_margin, max_z + z_margin;
	vert_coords.col(2) << min_x - x_margin, max_y + y_margin, max_z + z_margin;
	vert_coords.col(3) << max_x + x_margin, max_y + y_margin, max_z + z_margin;
	vert_coords.col(4) << min_x - x_margin, max_y + y_margin, min_z - z_margin;
	vert_coords.col(5) << max_x + x_margin, max_y + y_margin, min_z - z_margin;
	vert_coords.col(6) << min_x - x_margin, min_y - y_margin, min_z - z_margin;
	vert_coords.col(7) << max_x + x_margin, min_y - y_margin, min_z - z_margin;

	//Regularization
	if (with_reg_normals)
	{
		nx_reg.resize(num_faces); ny_reg.resize(num_faces); nz_reg.resize(num_faces); inv_reg_norm.resize(num_faces);
		u1_der_reg.resize(num_faces); u2_der_reg.resize(num_faces);
		w_u1_reg.resize(num_faces); w_u2_reg.resize(num_faces);
		for (unsigned int f=0; f<num_faces; f++)
		{
			nx_reg[f].resize(s_reg, s_reg); ny_reg[f].resize(s_reg, s_reg); nz_reg[f].resize(s_reg, s_reg); inv_reg_norm[f].resize(s_reg, s_reg);
			u1_der_reg[f].resize(s_reg, s_reg); u2_der_reg[f].resize(s_reg, s_reg);
			w_u1_reg[f].resize(num_verts, square(s_reg)); w_u2_reg[f].resize(num_verts, square(s_reg));

			for (unsigned int s1 = 0; s1 < s_reg; s1++)
				for (unsigned int s2 = 0; s2 < s_reg; s2++)
				{
					u1_der_reg[f](s1,s2) = new float[3];
					u2_der_reg[f](s1,s2) = new float[3];
				}
		}
	}

	//Show the mesh on the 3D Scene
	//showMesh();
}

void Mod3DfromRGBD::loadInitialMeshUShape()
{
	//Initial mesh - A cube...
	num_verts = 16;
	num_faces = 14;

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 1, 0, 3, 2;

	face_verts.col(1) << 0, 1, 5, 4;
	face_verts.col(2) << 1, 2, 6, 5;
	face_verts.col(3) << 2, 3, 7, 6;
	face_verts.col(4) << 3, 0, 4, 7;

	face_verts.col(5) << 5, 12, 13, 4;
	face_verts.col(6) << 12, 5, 6, 15;
	face_verts.col(7) << 15, 6, 7, 14;
	face_verts.col(8) << 4, 13, 14, 7;

	face_verts.col(9) << 8, 9, 13, 12;
	face_verts.col(10) << 9, 10, 14, 13;
	face_verts.col(11) << 10, 11, 15, 14;
	face_verts.col(12) << 11, 8, 12, 15;

	face_verts.col(13) << 9, 8, 11, 10;


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
				if (is_object[i](v, u))
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

	const float x_margin = 0.1f*(max_x - min_x);
	const float y_margin = 0.1f*(max_y - min_y);
	const float z_margin = 0.1f*(max_z - min_z);

	const float side = 0.5f*(max_y - min_y + max_z - min_z);

	vert_incrs.resize(3, num_verts);
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);

	vert_coords.col(0) << min_x, max_y, min_z;
	vert_coords.col(1) << min_x, max_y/3, min_z;
	vert_coords.col(2) << max_x, max_y/3, min_z;
	vert_coords.col(3) << max_x, max_y, min_z;

	vert_coords.col(4) << min_x, max_y, max_z;
	vert_coords.col(5) << min_x, max_y/3, max_z/3;
	vert_coords.col(6) << max_x, max_y/3, max_z/3;
	vert_coords.col(7) << max_x, max_y, max_z;

	vert_coords.col(8) << min_x, min_y/3, min_z;
	vert_coords.col(9) << min_x, min_y, min_z;
	vert_coords.col(10) << max_x, min_y, min_z;
	vert_coords.col(11) << max_x, min_y/3, min_z;

	vert_coords.col(12) << min_x, min_y/3, max_z/3;
	vert_coords.col(13) << min_x, min_y, max_z;
	vert_coords.col(14) << max_x, min_y, max_z;
	vert_coords.col(15) << max_x, min_y/3, max_z/3;


	//Regularization
	if (with_reg_normals)
	{
		nx_reg.resize(num_faces); ny_reg.resize(num_faces); nz_reg.resize(num_faces); inv_reg_norm.resize(num_faces);
		u1_der_reg.resize(num_faces); u2_der_reg.resize(num_faces);
		w_u1_reg.resize(num_faces); w_u2_reg.resize(num_faces);
		for (unsigned int f=0; f<num_faces; f++)
		{
			nx_reg[f].resize(s_reg, s_reg); ny_reg[f].resize(s_reg, s_reg); nz_reg[f].resize(s_reg, s_reg); inv_reg_norm[f].resize(s_reg, s_reg);
			u1_der_reg[f].resize(s_reg, s_reg); u2_der_reg[f].resize(s_reg, s_reg);
			w_u1_reg[f].resize(num_verts, square(s_reg)); w_u2_reg[f].resize(num_verts, square(s_reg));

			for (unsigned int s1 = 0; s1 < s_reg; s1++)
				for (unsigned int s2 = 0; s2 < s_reg; s2++)
				{
					u1_der_reg[f](s1,s2) = new float[3];
					u2_der_reg[f](s1,s2) = new float[3];
				}
		}
	}

	//Show the mesh on the 3D Scene
	showMesh();
}



void Mod3DfromRGBD::computeDepthSegmentationUShape()
{
	const float max_depth = 1.32f;
	valid[0].fill(true);
	is_object[0].fill(false);
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
			if (depth[0](v,u) < max_depth)
				is_object[0](v,u) = true;

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
		//Compute gaussian mask
		float v_mask[5] = {1, 4, 6, 4, 1};
		Matrix<float, 5, 5> mask;
		for (unsigned int k = 0; k<5; k++)
			for (unsigned int j = 0; j<5; j++)
				mask(k,j) = v_mask[k]*v_mask[j]/256.f;

		for (int u = 0; u < cols; u++)
			for (int v = 0; v < rows; v++)
			if (is_object[i](v,u))
			{
				float n_x = 0.f, n_y = 0.f, n_z = 0.f, sum = 0.f;
				for (int k = -2; k<3; k++)
				for (int l = -2; l<3; l++)
				{
					const int ind_u = u+k, ind_v = v+l;
					if ((ind_u >= 0)&&(ind_u < cols)&&(ind_v >= 0)&&(ind_v < rows)&&(is_object[i](ind_v, ind_u)))
					{
						n_x += mask(l+3, k+3)*nx_image[i](ind_v, ind_u);
						n_y += mask(l+3, k+3)*ny_image[i](ind_v, ind_u);
						n_z += mask(l+3, k+3)*nz_image[i](ind_v, ind_u);
						sum += mask(l+3, k+3);
					}
				}

				nx_image[i](v,u) = n_x/sum;
				ny_image[i](v,u) = n_y/sum;
				nz_image[i](v,u) = n_z/sum;
			}
	}
}

void Mod3DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(num_images);

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

	//showCamPoses();


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
	//				showCamPoses();
	//		}

	//		cout << endl << "Final pose for banana " << num_banana << ":" << endl << pose;
	//	}
	//}
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

void Mod3DfromRGBD::computeInitialU()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXXf> u1_ini, u2_ini;
	u1_ini.resize(num_faces); u2_ini.resize(num_faces);
	const unsigned int num_samp = max(3, int(round(float(1000)/square(num_faces))));
	const float fact = 1.f / float(num_samp-1);
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
	vector<ArrayXXf> nx_ini, ny_ini, nz_ini;
	nx_ini.resize(num_faces); ny_ini.resize(num_faces); nz_ini.resize(num_faces);

	Far::PatchMap patchmap(*patchTable);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp, num_samp); y_ini[f].resize(num_samp, num_samp); z_ini[f].resize(num_samp, num_samp);
		nx_ini[f].resize(num_samp, num_samp); ny_ini[f].resize(num_samp, num_samp); nz_ini[f].resize(num_samp, num_samp);

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

				//3D coordinates
				x_ini[f](v, u) = eval.point[0];
				y_ini[f](v, u) = eval.point[1];
				z_ini[f](v, u) = eval.point[2];

				//Normals
				const float nx_i = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
				const float ny_i = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
				const float nz_i = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
				const float inv_norm = 1.f/sqrtf(square(nx_i) + square(ny_i) + square(nz_i));
				nx_ini[f](v,u) = inv_norm*nx_i;
				ny_ini[f](v,u) = inv_norm*ny_i;
				nz_ini[f](v,u) = inv_norm*nz_i;
			}
	}

	//Find the closest point to each of the observed with the cameras
	//----------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);
					const float n_x = mytrans(0, 0)*nx_image[i](v, u) + mytrans(0, 1)*ny_image[i](v, u) + mytrans(0, 2)*nz_image[i](v, u);
					const float n_y = mytrans(1, 0)*nx_image[i](v, u) + mytrans(1, 1)*ny_image[i](v, u) + mytrans(1, 2)*nz_image[i](v, u);
					const float n_z = mytrans(2, 0)*nx_image[i](v, u) + mytrans(2, 1)*ny_image[i](v, u) + mytrans(2, 2)*nz_image[i](v, u);


					float min_dist = 100.f, u1_min = 0.f, u2_min = 0.f, dist;
					unsigned int f_min = 0;

					for (unsigned int f = 0; f < num_faces; ++f)
						for (unsigned int uu = 0; uu < num_samp; uu++)
							for (unsigned int vv = 0; vv < num_samp; vv++)
							{
								dist = square(x - x_ini[f](vv,uu)) + square(y - y_ini[f](vv,uu)) + square(z - z_ini[f](vv,uu))
										+ Kn*(square(n_x - nx_ini[f](vv,uu)) + square(n_y - ny_ini[f](vv,uu)) + square(n_z - nz_ini[f](vv,uu)));
								if (dist  < min_dist)
								{
									min_dist = dist;
									u1_min = u1_ini[f](vv, uu);
									u2_min = u2_ini[f](vv, uu);
									f_min = f;
								}
							}

					u1[i](v, u) = u1_min;
					u2[i](v, u) = u2_min;
					uface[i](v, u) = f_min;
				}
	}
}

void Mod3DfromRGBD::searchBetterU()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXXf> u1_ini, u2_ini;
	u1_ini.resize(num_faces); u2_ini.resize(num_faces);
	const unsigned int num_samp = max(3, int(round(float(1000)/square(num_faces))));
	const float fact = 1.f / float(num_samp-1);
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
	vector<ArrayXXf> nx_ini, ny_ini, nz_ini;
	nx_ini.resize(num_faces); ny_ini.resize(num_faces); nz_ini.resize(num_faces);

	Far::PatchMap patchmap(*patchTable);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f < num_faces; ++f)
	{
		x_ini[f].resize(num_samp, num_samp); y_ini[f].resize(num_samp, num_samp); z_ini[f].resize(num_samp, num_samp);
		nx_ini[f].resize(num_samp, num_samp); ny_ini[f].resize(num_samp, num_samp); nz_ini[f].resize(num_samp, num_samp);

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

				//3D coordinates
				x_ini[f](v, u) = eval.point[0];
				y_ini[f](v, u) = eval.point[1];
				z_ini[f](v, u) = eval.point[2];

				//Normals
				const float nx_i = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
				const float ny_i = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
				const float nz_i = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
				const float inv_norm = 1.f/sqrtf(square(nx_i) + square(ny_i) + square(nz_i));
				nx_ini[f](v,u) = inv_norm*nx_i;
				ny_ini[f](v,u) = inv_norm*ny_i;
				nz_ini[f](v,u) = inv_norm*nz_i;
			}
	}

	//Find the closest point to each of the observed with the cameras
	//----------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		Matrix4f &mytrans = cam_trans[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = mytrans(0, 0)*depth[i](v, u) + mytrans(0, 1)*x_image[i](v, u) + mytrans(0, 2)*y_image[i](v, u) + mytrans(0, 3);
					const float y = mytrans(1, 0)*depth[i](v, u) + mytrans(1, 1)*x_image[i](v, u) + mytrans(1, 2)*y_image[i](v, u) + mytrans(1, 3);
					const float z = mytrans(2, 0)*depth[i](v, u) + mytrans(2, 1)*x_image[i](v, u) + mytrans(2, 2)*y_image[i](v, u) + mytrans(2, 3);
					const float n_x = mytrans(0, 0)*nx_image[i](v, u) + mytrans(0, 1)*ny_image[i](v, u) + mytrans(0, 2)*nz_image[i](v, u);
					const float n_y = mytrans(1, 0)*nx_image[i](v, u) + mytrans(1, 1)*ny_image[i](v, u) + mytrans(1, 2)*nz_image[i](v, u);
					const float n_z = mytrans(2, 0)*nx_image[i](v, u) + mytrans(2, 1)*ny_image[i](v, u) + mytrans(2, 2)*nz_image[i](v, u);


					float min_dist = square(x - mx[i](v,u)) + square(y - my[i](v,u)) + square(z - mz[i](v,u))
										+ Kn*(square(n_x - nx[i](v,u)) + square(n_y - ny[i](v,u)) + square(n_z - nz[i](v,u)));
					float u1_min = u1[i](v,u), u2_min = u2[i](v,u), dist;
					unsigned int f_min = uface[i](v,u);

					for (unsigned int f = 0; f < num_faces; ++f)
						for (unsigned int uu = 0; uu < num_samp; uu++)
							for (unsigned int vv = 0; vv < num_samp; vv++)
							{
								dist = square(x - x_ini[f](vv,uu)) + square(y - y_ini[f](vv,uu)) + square(z - z_ini[f](vv,uu))
										+ Kn*(square(n_x - nx_ini[f](vv,uu)) + square(n_y - ny_ini[f](vv,uu)) + square(n_z - nz_ini[f](vv,uu)));
								if (dist  < min_dist)
								{
									min_dist = dist;
									u1_min = u1_ini[f](vv, uu);
									u2_min = u2_ini[f](vv, uu);
									f_min = f;
								}
							}

					u1[i](v, u) = u1_min;
					u2[i](v, u) = u2_min;
					uface[i](v, u) = f_min;
				}
	}
}

void Mod3DfromRGBD::computeInitialV()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXXf> u1_ini, u2_ini;
	u1_ini.resize(num_faces); u2_ini.resize(num_faces);
	const unsigned int num_samp = max(3, int(round(float(1000)/square(num_faces))));
	const float fact = 1.f / float(num_samp-1);
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

	//Find the one that projects closer to the corresponding pixel (using Pin-Hole model)
	//-----------------------------------------------------------------------------------
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		const Matrix4f &mytrans = cam_trans_inv[i];

		//Compute the transformed points
		vector<ArrayXXf> x_ini_t, y_ini_t, z_ini_t;
		x_ini_t.resize(num_faces); y_ini_t.resize(num_faces); z_ini_t.resize(num_faces);
		for (unsigned int f = 0; f < num_faces; f++)
		{
			x_ini_t[f].resize(num_samp, num_samp);
			y_ini_t[f].resize(num_samp, num_samp);
			z_ini_t[f].resize(num_samp, num_samp);

			for (unsigned int uu = 0; uu < num_samp; uu++)
				for (unsigned int vv = 0; vv < num_samp; vv++)
				{
					//Compute the 3D coordinates of the subdiv point after the relative transformation
					z_ini_t[f](vv, uu) = mytrans(0, 0)*x_ini[f](vv, uu) + mytrans(0, 1)*y_ini[f](vv, uu) + mytrans(0, 2)*z_ini[f](vv, uu) + mytrans(0, 3);
					x_ini_t[f](vv, uu) = mytrans(1, 0)*x_ini[f](vv, uu) + mytrans(1, 1)*y_ini[f](vv, uu) + mytrans(1, 2)*z_ini[f](vv, uu) + mytrans(1, 3);
					y_ini_t[f](vv, uu) = mytrans(2, 0)*x_ini[f](vv, uu) + mytrans(2, 1)*y_ini[f](vv, uu) + mytrans(2, 2)*z_ini[f](vv, uu) + mytrans(2, 3);

					if (z_ini_t[f](vv, uu) <= 0.f)
						printf("\n Problem computing the initial internal points for the background!! Model points too close to the cameras");
				}
		}

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v, u) && (valid[i](v,u)))
				{
					float min_dist = 100000.f, u1_min = 0.f, u2_min = 0.f;
					unsigned int f_min = 0;

					for (unsigned int f = 0; f < num_faces; f++)
						for (unsigned int uu = 0; uu < num_samp; uu++)
							for (unsigned int vv = 0; vv < num_samp; vv++)
							{
								//Project point onto the image plane (I assume that no point has z-coordinate == 0)
								const float u_pixel = fx*(x_ini_t[f](vv, uu) / z_ini_t[f](vv, uu)) + disp_u;
								const float v_pixel = fy*(y_ini_t[f](vv, uu) / z_ini_t[f](vv, uu)) + disp_v;

								const float pix_dist = square(u_pixel - float(u)) + square(v_pixel - float(v));

								if (pix_dist < min_dist)
								{
									min_dist = pix_dist;
									u1_min = u1_ini[f](vv, uu);
									u2_min = u2_ini[f](vv, uu);
									f_min = f;
								}
							}

					u1[i](v, u) = u1_min;
					u2[i](v, u) = u2_min;
					uface[i](v, u) = f_min;
				}
	}
}

void Mod3DfromRGBD::searchBetterV()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	vector<ArrayXXf> u1_ini, u2_ini;
	u1_ini.resize(num_faces); u2_ini.resize(num_faces);
	const unsigned int num_samp = max(3, int(round(float(1000)/square(num_faces))));
	const float fact = 1.f / float(num_samp-1);
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

	//Find the one that projects closer to the corresponding pixel (using Pin-Hole model)
	//-----------------------------------------------------------------------------------
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	for (unsigned int i = 0; i < num_images; ++i)
	{
		//Find direction of the projection
		const Matrix4f &mytrans = cam_trans_inv[i];

		//Compute the transformed points
		vector<ArrayXXf> x_ini_t, y_ini_t, z_ini_t;
		x_ini_t.resize(num_faces); y_ini_t.resize(num_faces); z_ini_t.resize(num_faces);
		for (unsigned int f = 0; f < num_faces; f++)
		{
			x_ini_t[f].resize(num_samp, num_samp);
			y_ini_t[f].resize(num_samp, num_samp);
			z_ini_t[f].resize(num_samp, num_samp);

			for (unsigned int uu = 0; uu < num_samp; uu++)
				for (unsigned int vv = 0; vv < num_samp; vv++)
				{
					//Compute the 3D coordinates of the subdiv point after the relative transformation
					z_ini_t[f](vv, uu) = mytrans(0, 0)*x_ini[f](vv, uu) + mytrans(0, 1)*y_ini[f](vv, uu) + mytrans(0, 2)*z_ini[f](vv, uu) + mytrans(0, 3);
					x_ini_t[f](vv, uu) = mytrans(1, 0)*x_ini[f](vv, uu) + mytrans(1, 1)*y_ini[f](vv, uu) + mytrans(1, 2)*z_ini[f](vv, uu) + mytrans(1, 3);
					y_ini_t[f](vv, uu) = mytrans(2, 0)*x_ini[f](vv, uu) + mytrans(2, 1)*y_ini[f](vv, uu) + mytrans(2, 2)*z_ini[f](vv, uu) + mytrans(2, 3);

					if (z_ini_t[f](vv, uu) <= 0.f)
						printf("\n Problem computing the initial internal points for the background!! Model points too close to the cameras");
				}
		}

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v, u) && (valid[i](v,u)))
				{
					float min_dist = square(res_d1[i](v, u)) + square(res_d2[i](v, u));
					float u1_min = u1[i](v,u), u2_min = u2[i](v,u);
					unsigned int f_min = uface[i](v,u);

					for (unsigned int f = 0; f < num_faces; f++)
						for (unsigned int uu = 0; uu < num_samp; uu++)
							for (unsigned int vv = 0; vv < num_samp; vv++)
							{
								//Project point onto the image plane (I assume that no point has z-coordinate == 0)
								const float u_pixel = fx*(x_ini_t[f](vv, uu) / z_ini_t[f](vv, uu)) + disp_u;
								const float v_pixel = fy*(y_ini_t[f](vv, uu) / z_ini_t[f](vv, uu)) + disp_v;

								const float pix_dist = square(u_pixel - float(u)) + square(v_pixel - float(v));

								if (pix_dist < min_dist)
								{
									min_dist = pix_dist;
									u1_min = u1_ini[f](vv, uu);
									u2_min = u2_ini[f](vv, uu);
									f_min = f;
								}
							}

					u1[i](v, u) = u1_min;
					u2[i](v, u) = u2_min;
					uface[i](v, u) = f_min;
				}
	}
}

void Mod3DfromRGBD::computeInitialCorrespondences()
{
	//Compute initial internal points for the foreground
	computeInitialU();

	//Compute initial internal points for the background
	computeInitialV();
}

void Mod3DfromRGBD::computeTransCoordAndResiduals()
{
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix4f &mytrans_inv = cam_trans_inv[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (valid[i](v,u))
				{			
					mx_t[i](v, u) = mytrans_inv(0, 0)*mx[i](v, u) + mytrans_inv(0, 1)*my[i](v, u) + mytrans_inv(0, 2)*mz[i](v, u) + mytrans_inv(0, 3);
					my_t[i](v, u) = mytrans_inv(1, 0)*mx[i](v, u) + mytrans_inv(1, 1)*my[i](v, u) + mytrans_inv(1, 2)*mz[i](v, u) + mytrans_inv(1, 3);
					mz_t[i](v, u) = mytrans_inv(2, 0)*mx[i](v, u) + mytrans_inv(2, 1)*my[i](v, u) + mytrans_inv(2, 2)*mz[i](v, u) + mytrans_inv(2, 3);

					if (mx_t[i](v, u) <= 0.f)
						printf("\n Depth coordinate of the internal correspondence is equal or inferior to zero after the transformation!!!");

					if (is_object[i](v, u))
					{
						res_x[i](v, u) = depth[i](v, u) - mx_t[i](v, u);
						res_y[i](v, u) = x_image[i](v, u) - my_t[i](v, u);
						res_z[i](v, u) = y_image[i](v, u) - mz_t[i](v, u);

						nx_t[i](v,u) = mytrans_inv(0, 0)*nx[i](v, u) + mytrans_inv(0, 1)*ny[i](v, u) + mytrans_inv(0, 2)*nz[i](v, u);
						ny_t[i](v,u) = mytrans_inv(1, 0)*nx[i](v, u) + mytrans_inv(1, 1)*ny[i](v, u) + mytrans_inv(1, 2)*nz[i](v, u);
						nz_t[i](v,u) = mytrans_inv(2, 0)*nx[i](v, u) + mytrans_inv(2, 1)*ny[i](v, u) + mytrans_inv(2, 2)*nz[i](v, u);
						const float inv_norm = 1.f/sqrtf(square(nx_t[i](v,u)) + square(ny_t[i](v,u)) + square(nz_t[i](v,u)));
						res_nx[i](v,u) = nx_image[i](v,u) - inv_norm*nx_t[i](v,u);
						res_ny[i](v,u) = ny_image[i](v,u) - inv_norm*ny_t[i](v,u);
						res_nz[i](v,u) = nz_image[i](v,u) - inv_norm*nz_t[i](v,u);
					}
					else
					{
						const float u_proj = fx*(my_t[i](v, u) / mx_t[i](v, u)) + disp_u;
						const float v_proj = fy*(mz_t[i](v, u) / mx_t[i](v, u)) + disp_v;
						res_d1[i](v, u) = float(u) - u_proj;
						res_d2[i](v, u) = float(v) - v_proj;
					}
				}
	}
}

void Mod3DfromRGBD::updateInternalPointCrossingEdges(unsigned int i, unsigned int v, unsigned int u, bool adaptive)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	//Check if crossing borders
	Vector2f u_incr; u_incr << u1_incr[i](v, u), u2_incr[i](v, u);
	float u1_old, u2_old;
	unsigned int face;
	if (adaptive)
	{
		u1_old = this->u1_old[i](v, u);
		u2_old = this->u2_old[i](v, u);
		face = uface_old[i](v, u);
	}
	else
	{
		u1_old = u1[i](v, u);
		u2_old = u2[i](v, u);
		face = uface[i](v, u);
	}
	float u1_new = u1_old + u_incr(0);
	float u2_new = u2_old + u_incr(1);
	bool crossing = true;

	unsigned int cont = 0;

	while (crossing)
	{
		//Find the new face	and the coordinates of the crossing point within the old face and the new face
		unsigned int face_new;
		float aux, dif, u1_cross, u2_cross;
		bool face_found = false;

		if (u1_new < 0.f)
		{
			dif = u1_old;
			const float u2t = u2_old - u_incr(1)*dif / u_incr(0);
			if ((u2t >= 0.f) && (u2t <= 1.f))
			{
				face_new = face_adj(3, face); aux = u2t; face_found = true;
				u1_cross = 0.f; u2_cross = u2t;
			}
		}
		if ((u1_new > 1.f) && (!face_found))
		{
			dif = 1.f - u1_old;
			const float u2t = u2_old + u_incr(1)*dif / u_incr(0);
			if ((u2t >= 0.f) && (u2t <= 1.f))
			{
				face_new = face_adj(1, face); aux = 1.f - u2t; face_found = true;
				u1_cross = 1.f; u2_cross = u2t;
			}
		}
		if ((u2_new < 0.f) && (!face_found))
		{
			dif = u2_old;
			const float u1t = u1_old - u_incr(0)*dif / u_incr(1);
			if ((u1t >= 0.f) && (u1t <= 1.f))
			{
				face_new = face_adj(0, face); aux = 1.f - u1t; face_found = true;
				u1_cross = u1t; u2_cross = 0.f;
			}
		}
		if ((u2_new > 1.f) && (!face_found))
		{
			dif = 1.f - u2_old;
			const float u1t = u1_old + u_incr(0)*dif / u_incr(1);
			if ((u1t >= 0.f) && (u1t <= 1.f))
			{
				face_new = face_adj(2, face); aux = u1t; face_found = true;
				u1_cross = u1t; u2_cross = 1.f;
			}
		}

		//Evaluate the subdivision surface at the edge (with respect to the original face)
		float pWeights[20], dsWeights[20], dtWeights[20];
		Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(face, u1_cross, u2_cross); assert(handle1);
		patchTable->EvaluateBasis(*handle1, u1_cross, u2_cross, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);
		LimitFrame eval;  eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		Matrix<float, 3, 2> J_Sa; J_Sa << eval.deriv1[0], eval.deriv2[0], eval.deriv1[1], eval.deriv2[1], eval.deriv1[2], eval.deriv2[2];

		//Find the coordinates of the crossing point as part of the new face
		unsigned int conf;
		for (unsigned int f = 0; f < 4; f++)
			if (face_adj(f, face_new) == face) { conf = f; }

		switch (conf)
		{
		case 0: u1_old = aux; u2_old = 0.f; break;
		case 1: u1_old = 1.f; u2_old = aux; break;
		case 2:	u1_old = 1.f - aux; u2_old = 1.f; break;
		case 3:	u1_old = 0.f; u2_old = 1.f - aux; break;
		}

		//Evaluate the subdivision surface at the edge (with respect to the new face)
		Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(face_new, u1_old, u2_old); assert(handle2);
		patchTable->EvaluateBasis(*handle2, u1_old, u2_old, pWeights, dsWeights, dtWeights);
		cvs = patchTable->GetPatchVertices(*handle2);
		eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		Matrix<float, 3, 2> J_Sb; J_Sb << eval.deriv1[0], eval.deriv2[0], eval.deriv1[1], eval.deriv2[1], eval.deriv1[2], eval.deriv2[2];


		//Compute the new u increments
		Vector2f du_remaining; du_remaining << u1_new - u1_cross, u2_new - u2_cross;
		MatrixXf prod = J_Sa*du_remaining;
		MatrixXf AtA, AtB;
		AtA.multiply_AtA(J_Sb);
		AtB.multiply_AtB(J_Sb, prod);
		//Vector2f du_new = AtA.ldlt().solve(AtB);
		u_incr = AtA.inverse()*AtB;

		u1_new = u1_old + u_incr(0);
		u2_new = u2_old + u_incr(1);
		face = face_new;

		crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

		cont++;
		if (cont > 5)
		{
			printf("\n Problem!!! Many jumps between the mesh faces for the update of one of the correspondences. I remove the remaining u_increment!");
			u1_new = u1_old;
			u2_new = u2_old;
			break;
		}
	}

	u1[i](v, u) = u1_new;
	u2[i](v, u) = u2_new;
	uface[i](v, u) = face;
}

float Mod3DfromRGBD::computeEnergyOverall()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if (is_object[i](v, u))
				{
					const float res = sqrtf(square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u)));
					if (res < truncated_res)
						energy_d += square(res);
					else
						energy_d += square(truncated_res);

					const float resn = sqrtf(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
					if (resn < truncated_resn)
						energy_d += Kn*square(resn);
					
					else
						energy_d += Kn*square(truncated_resn);	
				}
				else if (valid[i](v,u))
				{
					const float res_d_squared = square(res_d1[i](v, u)) + square(res_d2[i](v, u));
					
					if (robust_kernel == 0)
					{		
						//Truncated quadratic
						if (res_d_squared < square(tau))	energy_b += alpha_raycast*(1.f - res_d_squared / square(tau));
					}
					else if (robust_kernel == 1)
					{
						//2 parabolas with peak
						if (res_d_squared < square(tau))	energy_b += alpha_raycast*square(1.f - sqrtf(res_d_squared)/tau);					
					}
				}
			}

		//Camera prior - Keep close to the initial pose
		//for (unsigned int l = 0; l < 6; l++)
		//	energy += 0.5f*cam_prior*square(cam_mfold[i](l));
	}

	//Regularization
	if (with_reg_normals)		energy_r += computeEnergyRegNormals();
	if (with_reg_edges)			energy_r += computeEnergyRegEdges();
	if (with_reg_membrane) 		energy_r += computeEnergyRegMembrane();
	if (with_reg_thin_plate) 	energy_r += computeEnergyRegThinPlate();

	//Save?
	float new_energy = energy_d + energy_r + energy_b;
	if (new_energy < last_energy)
	{
		energy_data.push_back(energy_d);
		energy_background.push_back(energy_b);
		energy_reg.push_back(energy_r);
	}

	return (new_energy);
}


void Mod3DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1200, 1200); //window.resize(1000, 900);
	window.setPos(100, 100); //window.setPos(900, 0);
	window.setCameraZoom(1.8f); //window.setCameraZoom(3);
	window.setCameraAzimuthDeg(180);//window.setCameraAzimuthDeg(0);	//pers - 150, front - 180
	window.setCameraElevationDeg(0);//window.setCameraElevationDeg(45);	//pers - 30, front - 0
	window.setCameraPointingToPoint(1.2f, 0.f, 0.f);//window.setCameraPointingToPoint(0.f, 0.f, 0.f);
	window.getDefaultViewport()->setCustomBackgroundColor(utils::TColorf(1.f, 1.f, 1.f));

	window.captureImagesStart();

	scene = window.get3DSceneAndLock();

	// Lights:
	scene->getViewport()->setNumberOfLights(2);
	mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	light0.light_ID = 0;
	//light0.setPosition(2.5f,0,0.f,1.f);
	light0.setDirection(0.f, 0.f, -1.f);

	mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	light1.light_ID = 1;
	light1.setPosition(0.0f, 0, 0.f, 1.f);
	light1.setDirection(0.f, 1.f, 0.f);

	////Extra viewport for the distance transform
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
	//control_mesh->setFaceColor(1.f, 0.f, 0.f);
	control_mesh->setEdgeColor(0.9f, 0.f, 0.f);
	control_mesh->setVertColor(0.6f, 0.f, 0.f);
	scene->insert(control_mesh);

	//////Vertex numbers
	////for (unsigned int v = 0; v < 2000; v++)
	////{
	////	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	////	vert_nums->setString(std::to_string(v));
	////	vert_nums->setScale(0.02f);
	////	vert_nums->setColor(0.5, 0, 0);
	////	scene->insert(vert_nums);
	////}

	////Reference
	//opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	//reference->setScale(0.2f);
	//scene->insert(reference);

	////Frustums
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CFrustumPtr frustum = opengl::CFrustum::Create(0.01f, 0.1f, utils::RAD2DEG(fovh_d), utils::RAD2DEG(fovv_d), 1.5f, true, false);
	//	frustum->setColor(0.4f, 0.f, 0.f);
	//	scene->insert(frustum);
	//}

	////Points
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//	points->setPointSize(6.f);
	//	points->enablePointSmooth(true);
	//	scene->insert(points);

	//	//Insert points (they don't change through the optimization process)
	//	float r, g, b;
	//	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//	r = 0.f; g = 0.6; b = 0.f;
	//	for (unsigned int v = 0; v < rows; v++)
	//		for (unsigned int u = 0; u < cols; u++)
	//		if (is_object[i](v, u))
	//			points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), r, g, b);
	//		//else
	//		//	points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 0.8f, 0.f, 0.f);
	//}


	////Internal model (subdivision surface)
	//opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//points->setPointSize(3.f);
	//points->enablePointSmooth(true);
	//scene->insert(points);

	////Whole subdivision surface
	//opengl::CPointCloudPtr subsurface = opengl::CPointCloud::Create();
	//subsurface->setPointSize(1.f);
	//subsurface->setColor(0.4f, 0.4f, 0.4f);
	//subsurface->enablePointSmooth(true);
	//subsurface->setPose(CPose3D(0.f, 1.f, 0.f, 0.f, 0.f, 0.f));
	//scene->insert(subsurface);

	////Surface normals
	//const float fact = 0.01f;
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CSetOfLinesPtr normals = opengl::CSetOfLines::Create();
	//	normals->setColor(0, 0.8f, 0);
	//	normals->setLineWidth(1.f);
	//	scene->insert(normals);

	//	////Insert points (they don't change through the optimization process)
	//	//float r, g, b;
	//	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//	//normals->setColor(r, g, b);
	//	//for (unsigned int v = 0; v < rows; v++)
	//	//	for (unsigned int u = 0; u < cols; u++)
	//	//	if (is_object[i](v, u))
	//	//		normals->appendLine(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 
	//	//		depth[i](v, u) + fact*nx_image[i](v, u), x_image[i](v, u) + fact*ny_image[i](v, u), y_image[i](v, u) + fact*nz_image[i](v, u));
	//}


	//3D Model
	opengl::CMesh3DPtr model = opengl::CMesh3D::Create();
	//model->setPose(CPose3D(0.f, 2.f, 0.f, 0.f, 0.f, 0.f));
	model->enableShowVertices(false);
	model->enableShowEdges(false);
	model->enableShowFaces(true);
	model->enableFaceNormals(true);
	scene->insert(model);

	window.unlockAccess3DScene();
	window.repaint();

	//system::sleep(100);
	//utils::CImage screenshot;
	//window.getLastWindowImage(screenshot);
	//screenshot.saveToFile("test_image.png");
}

void Mod3DfromRGBD::showRenderedModel()
{
	scene = window.get3DSceneAndLock();

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	model->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);

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

		//Cameras
		opengl::CFrustumPtr frustum = scene->getByClass<CFrustum>(i);
		frustum->setPose(cam_poses[i]);

		//Normals
		opengl::CSetOfLinesPtr normals = scene->getByClass<CSetOfLines>(i);
		normals->setPose(cam_poses[i]);
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

	//Show correspondences and samples for DT (if solving with DT)
	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(num_images);
	points->clear();

	//for (unsigned int i = 0; i < num_images; ++i)
	//	for (unsigned int u = 0; u < cols; u++)
	//		for (unsigned int v = 0; v < rows; v++)
	//		{
	//			if (is_object[i](v, u))
	//				points->push_back(mx[i](v, u), my[i](v, u), mz[i](v, u), 0.f, 0.f, 1.f);

	//			else if(!solve_DT)
	//				points->push_back(mx[i](v, u), my[i](v, u), mz[i](v, u), 0.f, 0.f, 0.f);
	//		}

	//if (solve_DT)
	//{
	//	for (unsigned int k = 0; k < nsamples; k++)
	//		points->push_back(mx_DT(k), my_DT(k), mz_DT(k), 0.5f, 0.f, 0.5f);
	//}

	//Show the normals to the surface
	//const float scale = 0.1f;
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CSetOfLinesPtr normals = scene->getByClass<CSetOfLines>(i);
	//	normals->setColor(0,0.8f,0);
	//	normals->setLineWidth(1.f);
	//	normals->clear();
	//	normals->setPose(CPose3D(0,0,0,0,0,0));

	//	//Insert points (they don't change through the optimization process)
	//	float r,g,b;
	//	utils::colormap(mrpt::utils::cmJET,float(i) / float(num_images),r,g,b);
	//	normals->setColor(r,g,b);
	//	for (unsigned int v = 0; v < rows; v++)
	//	for (unsigned int u = 0; u < cols; u++)
	//	if (is_object[i](v,u))
	//		normals->appendLine(mx[i](v,u), my[i](v,u), mz[i](v,u),
	//							mx[i](v,u) + scale*nx[i](v,u),my[i](v,u) + scale*ny[i](v,u),mz[i](v,u) + scale*nz[i](v,u));
	//}

	////Connecting lines
	//const float fact_norm = 0.03f;
	//float r,g,b;
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(i);
	//	conect->clear();
	//	conect->setPose(cam_poses[i]);
	//	utils::colormap(mrpt::utils::cmJET,float(i) / float(num_images),r,g,b);
	//	conect->setColor(r,g,b);
	//	for (unsigned int u = 0; u < cols; u++)
	//		for (unsigned int v = 0; v < rows; v++)
	//			if (is_object[i](v,u))
	//				conect->appendLine(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u));
	//}

	//Show the whole surface
	const unsigned int sampl = max(3, int(100.f/sqrtf(num_faces)));
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

void Mod3DfromRGBD::takePictureLimitSurface(bool last)
{
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;
	
	float num_ref;
	if (last)	num_ref = 8;
	else		num_ref = 7;
	
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

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	is_quad.resize(num_faces_now, 1); is_quad.fill(true);
	model->loadMesh(num_verts_now, num_faces_now, is_quad, face_verts_now, vert_coords_now);

	opengl::CMesh3DPtr mesh = scene->getByClass<CMesh3D>(0);

	vert_coords_now.resize(3, num_verts);
	is_quad.resize(num_faces, 1); is_quad.fill(true);
	if (!last)
		mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);
	
	else
		mesh->loadMesh(0, 0, is_quad, face_verts, vert_coords);

	window.unlockAccess3DScene();
	window.repaint();	

	system::sleep(10);

	// Open file: find the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;
	string  name;

	while (!free_name)
	{
		nFile++;
		sprintf(aux, "image_%03u.png", nFile );
		name = f_folder + aux;
		free_name = !system::fileExists(name);
	}

	//utils::CImage img;
	//window.getLastWindowImage(img);
	//img.saveToFile(name);
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
				if (valid[i](v,u))
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

					//Compute the normals
					nx[i](v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
					ny[i](v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
					nz[i](v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

					//Compute the weights for the coordinates and the derivatives wrt the control vertices
					//Set all weights to zero
					VectorXf vect_wc(num_verts), vect_wu1(num_verts), vect_wu2(num_verts);
					vect_wc.fill(0.f); vect_wu1.fill(0.f); vect_wu2.fill(0.f);

					for (int cv = 0; cv < cvs.size(); ++cv)
					{						
						if (cvs[cv] < num_verts)
						{			
							vect_wc(cvs[cv]) += pWeights[cv];
							vect_wu1(cvs[cv]) += dsWeights[cv];
							vect_wu2(cvs[cv]) += dtWeights[cv];
						}
						else
						{
							const unsigned int ind_offset = cvs[cv] - num_verts;
							//Look at the stencil associated to this local point and distribute its weight over the control vertices
							unsigned int size_st = st[ind_offset].GetSize();
							Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
							float const *st_weights = st[ind_offset].GetWeights();
							for (unsigned int s = 0; s < size_st; s++)
							{
								vect_wc(st_ind[s]) += pWeights[cv]*st_weights[s];
								vect_wu1(st_ind[s]) += dsWeights[cv]*st_weights[s];
								vect_wu2(st_ind[s]) += dtWeights[cv]*st_weights[s];
							}
						}
					}

					//Store the weights
					unsigned int cont = 0;
					const unsigned int col_weights = v + u*rows;
					//w_contverts[i].col(col_weights).fill(0.f); w_u1[i].col(col_weights).fill(0.f); w_u2[i].col(col_weights).fill(0.f);
					w_indices[i].col(col_weights).fill(-1);

					for (unsigned int cv=0; cv<num_verts; cv++)
						if (vect_wc(cv) != 0.f)
						{
							w_indices[i](cont, col_weights) = cv;
							w_contverts[i](cont, col_weights) = vect_wc(cv);
							w_u1[i](cont, col_weights) = vect_wu1(cv);
							w_u2[i](cont, col_weights) = vect_wu2(cv);
							cont++;
						}

					//cout << endl << "w_indices: " << w_indices[i].col(col_weights).transpose();

					//unsigned int num_cont_verts = 0;
					//for (int cv=0; cv<num_verts; cv++)
					//	if (w_contverts[i](cv, col_weights) > 0.f)
					//		num_cont_verts++;

					//printf(" ncv = %d", num_cont_verts);
				}
}

void Mod3DfromRGBD::evaluateSubDivSurfaceRegularization()
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils(); ///printf("\n Num of stencils - %d", nstencils);

	Far::Stencil *st = new Far::Stencil[nstencils];
	for (unsigned int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);
	
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;
	const float s2u = 1.f/float(s_reg-1);

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f<num_faces; f++)
		for (unsigned int s2 = 0; s2 < s_reg; s2++)
			for (unsigned int s1 = 0; s1 < s_reg; s1++)
			{
				const float u1 = float(s1)*s2u;
				const float u2 = float(s2)*s2u;
				
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1, u2); assert(handle);
				patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);
				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Save the derivatives
				u1_der_reg[f](s1,s2)[0] = eval.deriv1[0];
				u1_der_reg[f](s1,s2)[1] = eval.deriv1[1];
				u1_der_reg[f](s1,s2)[2] = eval.deriv1[2];
				u2_der_reg[f](s1,s2)[0] = eval.deriv2[0];
				u2_der_reg[f](s1,s2)[1] = eval.deriv2[1];
				u2_der_reg[f](s1,s2)[2] = eval.deriv2[2];

				//Compute the normals
				const float nx = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
				const float ny = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
				const float nz = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
				inv_reg_norm[f](s1,s2) = 1.f/sqrtf(square(nx) + square(ny) + square(nz));

				nx_reg[f](s1,s2) = nx*inv_reg_norm[f](s1,s2);
				ny_reg[f](s1,s2) = ny*inv_reg_norm[f](s1,s2);
				nz_reg[f](s1,s2) = nz*inv_reg_norm[f](s1,s2);

				//Compute the weights for the coordinates and the derivatives wrt the control vertices
				//Set all weights to zero
				const unsigned int col_weights = s1 + s2*s_reg;
				for (unsigned int k = 0; k < num_verts; k++)
				{
					w_u1_reg[f](k, col_weights) = 0.f;
					w_u2_reg[f](k, col_weights) = 0.f;
				}

				for (int cv = 0; cv < cvs.size(); ++cv)
				{						
					if (cvs[cv] < num_verts)
					{
						w_u1_reg[f](cvs[cv], col_weights) += dsWeights[cv];
						w_u2_reg[f](cvs[cv], col_weights) += dtWeights[cv];
					}
					else
					{
						const unsigned int ind_offset = cvs[cv] - num_verts;
						//Look at the stencil associated to this local point and distribute its weight over the control vertices
						unsigned int size_st = st[ind_offset].GetSize();
						Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
						float const *st_weights = st[ind_offset].GetWeights();
						for (unsigned int s = 0; s < size_st; s++)
						{
							w_u1_reg[f](st_ind[s], col_weights) += dsWeights[cv]*st_weights[s];
							w_u2_reg[f](st_ind[s], col_weights) += dtWeights[cv]*st_weights[s];
						}
					}
				}
			}
}


void Mod3DfromRGBD::evaluateSubDivSurfaceOnlyBackground()
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	//Far::PtexIndices ptexIndices(*refiner);  // Far::PtexIndices helps to find indices of ptex faces.

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int i = 0; i<num_images; ++i)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u)&&(valid[i](v,u)))
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
				}
}

void Mod3DfromRGBD::computeNormalDerivativesPixel(unsigned int i,unsigned int v,unsigned int u)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20],dsWeights[20],dtWeights[20];

	const float uincr = 0.001f; const float uincr_inv = 1.f/uincr;

	//Compute normal for small increment of u1
	//=================================================================================================================
	Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(uface[i](v,u), u1[i](v,u)+uincr, u2[i](v,u)); assert(handle1);
	patchTable->EvaluateBasis(*handle1, u1[i](v,u)+uincr, u2[i](v,u), pWeights, dsWeights, dtWeights);
	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);

	LimitFrame eval; eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);

	//Compute the normals
	const float nx_u1 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	const float ny_u1 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	const float nz_u1 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

	n_der_u1[i](v,u)[0] = uincr_inv*(nx_u1 - nx[i](v,u));
	n_der_u1[i](v,u)[1] = uincr_inv*(ny_u1 - ny[i](v,u));
	n_der_u1[i](v,u)[2] = uincr_inv*(nz_u1 - nz[i](v,u));

	//Compute normal for small increment of u2
	//=================================================================================================================
	Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(uface[i](v,u), u1[i](v,u), u2[i](v,u)+uincr); assert(handle2);
	patchTable->EvaluateBasis(*handle2, u1[i](v,u), u2[i](v,u)+uincr, pWeights, dsWeights, dtWeights);
	cvs = patchTable->GetPatchVertices(*handle2);

	eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);

	//Compute the normals
	const float nx_u2 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	const float ny_u2 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	const float nz_u2 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

	n_der_u2[i](v,u)[0] = uincr_inv*(nx_u2 - nx[i](v,u));
	n_der_u2[i](v,u)[1] = uincr_inv*(ny_u2 - ny[i](v,u));
	n_der_u2[i](v,u)[2] = uincr_inv*(nz_u2 - nz[i](v,u));
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
	vert_incrs.resize(3, num_verts);
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
	for (unsigned int v = 0; v < verts.size(); v++)
	{
		vert_coords(0, v) = verts[v].point[0];
		vert_coords(1, v) = verts[v].point[1];
		vert_coords(2, v) = verts[v].point[2];
	}

	//Resize regularization variables
	if (with_reg_normals)
	{
		nx_reg.resize(num_faces); ny_reg.resize(num_faces); nz_reg.resize(num_faces); inv_reg_norm.resize(num_faces);
		u1_der_reg.resize(num_faces); u2_der_reg.resize(num_faces);
		w_u1_reg.resize(num_faces); w_u2_reg.resize(num_faces);
		for (unsigned int f=0; f<num_faces; f++)
		{
			nx_reg[f].resize(s_reg, s_reg); ny_reg[f].resize(s_reg, s_reg); nz_reg[f].resize(s_reg, s_reg); inv_reg_norm[f].resize(s_reg, s_reg);
			u1_der_reg[f].resize(s_reg, s_reg); u2_der_reg[f].resize(s_reg, s_reg);
			w_u1_reg[f].resize(num_verts, square(s_reg)); w_u2_reg[f].resize(num_verts, square(s_reg));

			for (unsigned int s2 = 0; s2 < s_reg; s2++)
				for (unsigned int s1 = 0; s1 < s_reg; s1++)
				{
					u1_der_reg[f](s1,s2) = new float[3];
					u2_der_reg[f](s1,s2) = new float[3];
				}
		}
	}

	//Show the mesh on the 3D Scene
	//showMesh();
	ctf_level++;
	//takePictureLimitSurface(false);
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
		for (unsigned int v = 0; v < face_v.size(); v++)
			face_verts(v, f) = face_v[v];
	}

	//Fill the 3D coordinates of the vertices
	vert_incrs.resize(3, num_verts);
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
	for (unsigned int v = 0; v < verts.size(); v++)
	{
		vert_coords(0, v) = verts[v].point[0];
		vert_coords(1, v) = verts[v].point[1];
		vert_coords(2, v) = verts[v].point[2];
	}
}


void Mod3DfromRGBD::solveGradientDescent()
{
	//								Initialize
	//======================================================================================
	utils::CTicTac clock; 
	robust_kernel = 0;
	sz_x = 0.002f;
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));

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
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	//									Iterative solver
	//====================================================================================
	for (unsigned int i = 0; i < max_iter; i++)
	{
		clock.Tic();
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		vert_incrs.fill(0.f);
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();


		//							Compute the gradients
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices and clean increments
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);
			cam_incrs[i].fill(0.f);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v, u))
						{
							Matrix<float, 1, 3> res; res << res_x[i](v,u), res_y[i](v,u), res_z[i](v,u);
							Matrix<float, 1, 3> J_mult = -2.f*res*T_inv;

							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							Matrix<float, 1, 3> res_n; res_n << res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u);
							Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu;

							//Control vertices
							const unsigned int weights_col = v + u*rows;
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									const float ww = w_contverts[i](c, weights_col);
									vert_incrs(0, cp) += J_mult(0)*ww;
									vert_incrs(1, cp) += J_mult(1)*ww;
									vert_incrs(2, cp) += J_mult(2)*ww;

									//Normals
									const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
									J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
									J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
									J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
									J_nX(1,0) = -J_nX(0,1);
									J_nX(2,0) = -J_nX(0,2);
									J_nX(2,1) = -J_nX(1,2);

									vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
									vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
									vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
								}
							}
						}

						//Background
						else if ( square(res_d1[i](v, u)) + square(res_d2[i](v, u)) < square(tau))
						{

							Matrix<float, 2, 3> J_pi;
							const float inv_z = 1.f / mx_t[i](v, u);

							J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
									fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

							const float J_phi1 = 2.f*res_d1[i](v, u) / square(tau);
							const float J_phi2 = 2.f*res_d2[i](v, u) / square(tau);
							const Matrix<float, 1, 3> J_phi_pi = J_phi1*J_pi.row(0) + J_phi2*J_pi.row(1);
							const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi_pi*T_inv;

							//Control vertices
							const unsigned int weights_col = v + u*rows;
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c, weights_col);
								if (cp >= 0)
								{
									const float ww = w_contverts[i](c, weights_col);
									vert_incrs(0, cp) += -alpha_raycast*J_phi_pi_Tinv(0)*ww;
									vert_incrs(1, cp) += -alpha_raycast*J_phi_pi_Tinv(1)*ww;
									vert_incrs(2, cp) += -alpha_raycast*J_phi_pi_Tinv(2)*ww;
								}
							}
						}
					}

				////Camera prior - Keep close to the initial pose
				//for (unsigned int l = 0; l < 6; l++)
				//	cam_incrs[i](l) += -cam_prior*cam_mfold_old[i](l);
		}

		if (with_reg_normals)
			vertIncrRegularizationNormals();

		if (with_reg_edges)
			vertIncrRegularizationEdges();


		energy_increasing = true;
		unsigned int cont = 0;


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Update control vertices
			vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();
			rayCastingLMBackgroundPerPixel();
			rayCastingLMForegroundPerPixel();
			new_energy = computeEnergyOverall();

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
			if (cont > 10) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterU();
				searchBetterV();
				evaluateSubDivSurface();			
				if (with_reg_normals) evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				rayCastingLMBackgroundPerPixel();
				rayCastingLMForegroundPerPixel();
				new_energy = computeEnergyOverall();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
					energy_increasing = true;
					break;
				}
				else
					energy_increasing = false;		
			}
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

	//	showMesh();
	//	showCamPoses();
	//	showSubSurface();
	//	showRenderedModel();
		takePictureLimitSurface(false);

		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (energy_increasing ||(new_energy > 0.9999f*last_energy))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void Mod3DfromRGBD::solveLM()
{
	//								Initialize
	//======================================================================================
	robust_kernel = 1;
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float alpha_sqrt = sqrtf(alpha_raycast);
	const float Kn_sqrtf = sqrtf(Kn);

	//Variables for Levenberg-Marquardt
	MatrixXf J;
	VectorXf R,increments;
	unsigned int J_rows = 0, J_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (is_object[i](v,u))
					J_rows += 6;
				else if (valid[i](v,u))
					J_rows++;
			}
	if (with_reg_normals)
		J_rows += 6*num_faces*square(s_reg);

	printf("\n Jacobian size = (%d, %d)", J_rows, J_cols);

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

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	utils::CTicTac clock; 

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int i = 0; i < max_iter; i++)
	{
		clock.Tic();
		unsigned int cont = 0;
		J.fill(0.f);
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		evaluateSubDivSurface();
		//if (with_reg)	evaluateSubDivSurfaceRegularization();
		computeTransCoordAndResiduals();

		printf("\n It starts to compute the Jacobians");

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{
							//									Data alignment
							//----------------------------------------------------------------------------------------------------
							//Control vertices
							float v_weight;
							const unsigned int weights_col = v + u*rows;
							for (unsigned int cp=0; cp<num_verts; cp++)
								if ((v_weight = w_contverts[i](cp, weights_col)) > 0.f)
								{
									J.block<3,3>(cont, 3*cp) = -v_weight*T_inv;
								}

							//Camera poses
							Vector4f t_point; t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
							for (unsigned int l = 0; l < 6; l++)
							{
								const Vector3f prod = -mat_der_xi[l].block<3,4>(0,0)*t_point;
								J.block<3,1>(cont, 3*num_verts + 6*i + l) = prod;
							}

							//Fill the residuals
							R(cont) = res_x[i](v,u);
							R(cont+1) = res_y[i](v,u);
							R(cont+2) = res_z[i](v,u);
							cont += 3;

							//									Normal alignment
							//----------------------------------------------------------------------------------------------
							//Control vertices
							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							const Matrix3f J_mult_norm = -Kn_sqrtf*T_inv*J_nu;
						
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Normals
								const float wu1 = w_u1[i](cp, weights_col), wu2 = w_u2[i](cp, weights_col);
								J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
								J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
								J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
								J_nX(1,0) = -J_nX(0,1);
								J_nX(2,0) = -J_nX(0,2);
								J_nX(2,1) = -J_nX(1,2);

								J.block<3,3>(cont, 3*cp) = J_mult_norm*J_nX;
							}

							//Camera pose			
							Vector3f normal; normal << nx[i](v,u), ny[i](v,u), nz[i](v,u);
							const Vector3f n_t = sqrtf(Kn)*T_inv*inv_norm*normal;
							for (unsigned int l = 3; l < 6; l++)
							{
								const Vector3f prod = -mat_der_xi[l].block<3,3>(0,0)*n_t;
								J.block<3,1>(cont, 3*num_verts + 6*i + l) = prod;
							}

							//Fill the residuals
							R(cont) = Kn_sqrtf*res_nx[i](v,u);
							R(cont+1) = Kn_sqrtf*res_ny[i](v,u);
							R(cont+2) = Kn_sqrtf*res_nz[i](v,u);
							cont += 3;
						}

						//Background
						else
						{
							const float norm_proj_error = sqrtf(square(res_d1[i](v, u)) + square(res_d2[i](v, u)));
							if ((norm_proj_error < tau) && (norm_proj_error > peak_margin))
							{

								Matrix<float, 2, 3> J_pi;
								const float inv_z = 1.f / mx_t[i](v, u);

								J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
										fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

								Matrix<float, 1, 2> J_phi; J_phi << res_d1[i](v,u), res_d2[i](v,u);
								J_phi *= -alpha_sqrt/(tau*norm_proj_error);

								const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi*J_pi*T_inv;

								//Control vertices
								float v_weight;
								const unsigned int weights_col = v + u*rows;
								for (unsigned int cp = 0; cp < num_verts; cp++)
									if ((v_weight = w_contverts[i](cp, weights_col)) > 0.f)
									{
										J.block<3,1>(cont, 3*cp) = v_weight*J_phi_pi_Tinv;
									}

								//Camera pose
								Vector4f m_t; m_t << mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u), 1.f;
								for (unsigned int l = 0; l < 6; l++)
								{
									const float prod = (J_phi*J_pi*(mat_der_xi[l] * m_t).block<3, 1>(0, 0)).value();
									J(cont, 3*num_verts + 6*i + l) = prod;
								}

								//Fill the residuals
								R(cont) = alpha_sqrt*(1.f - norm_proj_error/tau);
							}

							cont++;	
						}
					}

				////Camera prior - Keep close to the initial pose
				//for (unsigned int l = 0; l < 6; l++)
				//	cam_incrs[i](l) += -cam_prior*cam_mfold_old[i](l);
		}

		printf("\n It finishes with Jacobians (without regularization)");

		//Include regularization
		if (with_reg_normals)
		{
			Matrix3f J_nX_here, J_nX_forward;
			J_nX_here.assign(0.f); J_nX_forward.assign(0.f);
			Kr = Kr_total/float(num_faces*square(s_reg));
			const float Kr_sqrt = sqrtf(Kr);
	
			for (unsigned int f=0; f<num_faces; f++)
			{
				//Compute the normalizing jacobians
				vector<Matrix3f> J_nu; J_nu.resize(square(s_reg));
				for (int s2=0; s2<s_reg; s2++)
					for (int s1=0; s1<s_reg; s1++)
					{	
						J_nu[s1 + s_reg*s2].row(0) << square(ny_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), -nx_reg[f](s1,s2)*nz_reg[f](s1,s2);
						J_nu[s1 + s_reg*s2].row(1) << -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2);
						J_nu[s1 + s_reg*s2].row(2) << -nx_reg[f](s1,s2)*nz_reg[f](s1,s2), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2));
						J_nu[s1 + s_reg*s2] *= inv_reg_norm[f](s1,s2);
					}


				//Include every equation into LM  - Number of new equations: 3*2*num_faces*s_reg*s_reg
				for (int s2=0; s2<s_reg; s2++)
					for (int s1=0; s1<s_reg; s1++)
					{						
						//Coefficients associated to the regularization
						const unsigned int s1for = min(s1+1, int(s_reg-1));
						const unsigned int s2for = min(s2+1, int(s_reg-1));

						if (s1for != s1)
						{
							const unsigned int weights_col = s1 + s_reg*s2;
							const unsigned int weights_col_for = s1for + s_reg*s2;
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Matrix of weights
								const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
								J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
								J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
								J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
								J_nX_here(1,0) = -J_nX_here(0,1);
								J_nX_here(2,0) = -J_nX_here(0,2);
								J_nX_here(2,1) = -J_nX_here(1,2);	

								//Matrix of weights
								const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);
								J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1for,s2)[2] - wu2_for*u1_der_reg[f](s1for,s2)[2];
								J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1for,s2)[1] - wu1_for*u2_der_reg[f](s1for,s2)[1];
								J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1for,s2)[0] - wu2_for*u1_der_reg[f](s1for,s2)[0];
								J_nX_forward(1,0) = -J_nX_forward(0,1);
								J_nX_forward(2,0) = -J_nX_forward(0,2);
								J_nX_forward(2,1) = -J_nX_forward(1,2);

								const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

								//Update increments
								J.block<3,3>(cont, 3*cp) = J_reg;
							}	

							//Fill the residuals
							R(cont) = Kr_sqrt*(nx_reg[f](s1for,s2) - nx_reg[f](s1,s2));
							R(cont+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2));
							R(cont+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2));
							cont += 3;
						}

						if (s2for != s2)
						{
							const unsigned int weights_col = s1 + s_reg*s2;
							const unsigned int weights_col_for = s1 + s_reg*s2for;
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Matrix of weights
								const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
								J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
								J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
								J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
								J_nX_here(1,0) = -J_nX_here(0,1);
								J_nX_here(2,0) = -J_nX_here(0,2);
								J_nX_here(2,1) = -J_nX_here(1,2);	

								//Matrix of weights
								const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);
								J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1,s2for)[2] - wu2_for*u1_der_reg[f](s1,s2for)[2];
								J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1,s2for)[1] - wu1_for*u2_der_reg[f](s1,s2for)[1];
								J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1,s2for)[0] - wu2_for*u1_der_reg[f](s1,s2for)[0];
								J_nX_forward(1,0) = -J_nX_forward(0,1);
								J_nX_forward(2,0) = -J_nX_forward(0,2);
								J_nX_forward(2,1) = -J_nX_forward(1,2);

								const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

								//Update increments
								J.block<3,3>(cont, 3*cp) = J_reg;
							}	

							//Fill the residuals
							R(cont) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2));
							R(cont+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2));
							R(cont+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2));
							cont += 3;
						}				
					}
				}
		}

		printf("\n It finishes with Jacobians (with regularization)");

		//Prepare Levenberg solver

		MatrixXf JtJ; JtJ.multiply_AtA(J); 
		MatrixXf JtJ_lm;
		VectorXf b = -J.transpose()*R;	//printf("\n It computes b = -Jt*R");


		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n It enters the loop solver-energy-check");

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			//JtJ_lm = JtJ_f + JtJ_g + adap_mult*MatrixXf::Identity(J_cols, J_cols);	//Levenberg
			JtJ_lm = JtJ;
			//JtJ_lm.diagonal() += adap_mult*JtJ_lm.diagonal();					//Levenberg-Marquardt
			for (unsigned int j=0; j<J_cols; j++)
			{
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);
				//if (j>=3*num_verts)
				//	JtJ_lm(j,j) *= 4.f;
			}

			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);

			printf("\n It solves with LM");
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c, k) + increments(cont++);

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int k = 0; k < 6; k++)
					cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);

			computeCameraTransfandPosesFromTwist();


			printf("\n It updates variables");

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n It computes the residuals");
			rayCastingLMBackgroundPerPixel();	printf("\n It solves raycasting background");
			rayCastingLMForegroundPerPixel();	printf("\n It solves closest correspondence foreground");
			new_energy = computeEnergyOverall();


			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;
			if (cont_inner > 10) energy_increasing = false;
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();
		showRenderedModel();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (cont_inner > 10)//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			
			//Recover old variables
			vert_coords = vert_coords_old;
			cam_mfold = cam_mfold_old;
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void Mod3DfromRGBD::solveLMWithoutCameraOptimization()
{
	//								Initialize
	//======================================================================================
	robust_kernel = 1;
	float new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	last_energy = 1000000000.f;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float alpha_sqrt = sqrtf(alpha_raycast);
	const float Kn_sqrtf = sqrtf(Kn);

	//Variables for Levenberg-Marquardt
	unsigned int J_rows = 0, J_cols = 3 * num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (is_object[i](v,u))
					J_rows += 6;
				else if (valid[i](v,u))
					J_rows++;
			}

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_membrane)		J_rows += 3*num_faces;
	if (with_reg_thin_plate)	J_rows += 3*num_faces;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	takePictureLimitSurface(false);

	utils::CTicTac clock; 

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		//J.fill(0.f);
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;

		//Occasional search for the correspondences
		if ((iter+1) % 5 == 0)
		{
			searchBetterU();
			searchBetterV();
			printf("\n Global search to avoid wrong correspondences that LM cannot solve");
		}

		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		printf("\n It starts to compute the Jacobians"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{
							//									Data alignment
							//----------------------------------------------------------------------------------------------------
							//Control vertices
							const unsigned int weights_col = v + u*rows;
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									const float v_weight = w_contverts[i](c, weights_col);
									j_elem.push_back(Tri(cont, 3*cp, -T_inv(0,0)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+1, -T_inv(0,1)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+2, -T_inv(0,2)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp, -T_inv(1,0)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+1, -T_inv(1,1)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+2, -T_inv(1,2)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp, -T_inv(2,0)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+1, -T_inv(2,1)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+2, -T_inv(2,2)*v_weight));
								}
							}

							//Fill the residuals
							R(cont) = res_x[i](v,u);
							R(cont+1) = res_y[i](v,u);
							R(cont+2) = res_z[i](v,u);
							cont += 3;

							//									Normal alignment
							//----------------------------------------------------------------------------------------------
							//Control vertices
							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							const Matrix3f J_mult_norm = -Kn_sqrtf*T_inv*J_nu;
						
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									//Normals
									const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
									J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
									J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
									J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
									J_nX(1,0) = -J_nX(0,1);
									J_nX(2,0) = -J_nX(0,2);
									J_nX(2,1) = -J_nX(1,2);

									const Matrix3f J_norm_fit = J_mult_norm*J_nX;
									j_elem.push_back(Tri(cont, 3*cp, J_norm_fit(0,0)));
									j_elem.push_back(Tri(cont, 3*cp+1, J_norm_fit(0,1)));
									j_elem.push_back(Tri(cont, 3*cp+2, J_norm_fit(0,2)));
									j_elem.push_back(Tri(cont+1, 3*cp, J_norm_fit(1,0)));
									j_elem.push_back(Tri(cont+1, 3*cp + 1, J_norm_fit(1,1)));
									j_elem.push_back(Tri(cont+1, 3*cp + 2, J_norm_fit(1,2)));
									j_elem.push_back(Tri(cont+2, 3*cp, J_norm_fit(2,0)));
									j_elem.push_back(Tri(cont+2, 3*cp + 1, J_norm_fit(2,1)));
									j_elem.push_back(Tri(cont+2, 3*cp + 2, J_norm_fit(2,2)));
								}
							}

							//Fill the residuals
							R(cont) = Kn_sqrtf*res_nx[i](v,u);
							R(cont+1) = Kn_sqrtf*res_ny[i](v,u);
							R(cont+2) = Kn_sqrtf*res_nz[i](v,u);
							cont += 3;
						}

						//Background
						else
						{
							const float norm_proj_error = sqrtf(square(res_d1[i](v, u)) + square(res_d2[i](v, u)));
							if ((norm_proj_error < tau) && (norm_proj_error > peak_margin))
							{

								Matrix<float, 2, 3> J_pi;
								const float inv_z = 1.f / mx_t[i](v, u);

								J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
										fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

								Matrix<float, 1, 2> J_phi; J_phi << res_d1[i](v,u), res_d2[i](v,u);
								J_phi *= -alpha_sqrt/(tau*norm_proj_error);

								const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi*J_pi*T_inv;

								//Control vertices
								const unsigned int weights_col = v + u*rows;

								for (unsigned int c = 0; c < max_num_w; c++)
								{
									const int cp = w_indices[i](c,weights_col);
									if (cp >= 0)
									{
										const float v_weight = w_contverts[i](c, weights_col);
										j_elem.push_back(Tri(cont, 3*cp, J_phi_pi_Tinv(0)*v_weight));
										j_elem.push_back(Tri(cont, 3*cp+1, J_phi_pi_Tinv(1)*v_weight));
										j_elem.push_back(Tri(cont, 3*cp+2, J_phi_pi_Tinv(2)*v_weight));
									}
								}

								//Fill the residuals
								R(cont) = alpha_sqrt*(1.f - norm_proj_error/tau);
							}

							cont++;	
						}
					}
		}

		printf("\n It finishes with Jacobians (without regularization). Time = %f", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)
			fillJacobianRegNormals(cont);

		if (with_reg_edges)
			fillJacobianRegEdges(cont);

		if (with_reg_membrane)
			fillJacobianRegMembrane(cont);

		if (with_reg_thin_plate)
			fillJacobianRegThinPlate(cont);


		printf("\n It finishes with Jacobians (with regularization). Time = %f", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		MatrixXf JtJ = MatrixXf(JtJ_sparse);
		VectorXf b = -J.transpose()*R;
		MatrixXf JtJ_lm;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n It enters the loop solver-energy-check. Time = %f", clock.Tac()); clock.Tic();

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			JtJ_lm = JtJ;
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);
			

			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);

			printf("\n It solves with LM. Time = %f", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			printf("\n It updates variables. Time = %f", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n It creates topology, evaluates the surface and computes the residuals. Time = %f", clock.Tac()); clock.Tic();
			rayCastingLMBackgroundPerPixel();	printf("\n It solves raycasting background. Time = %f", clock.Tac()); clock.Tic();
			rayCastingLMForegroundPerPixel();	printf("\n It solves closest correspondence foreground. Time = %f", clock.Tac()); clock.Tic();
			new_energy = computeEnergyOverall();


			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;
			if (cont_inner > 5) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterU();
				searchBetterV();
				evaluateSubDivSurface();			
				if (with_reg_normals) evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				rayCastingLMBackgroundPerPixel();
				rayCastingLMForegroundPerPixel();
				new_energy = computeEnergyOverall();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
					energy_increasing = true;
					break;
				}
				else
					energy_increasing = false;		
			}
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		//showMesh();
		//showCamPoses();
		//showSubSurface();
		//showRenderedModel();
		takePictureLimitSurface(false);

		printf("\n Time to finish everything else = %f", clock.Tac()); clock.Tic();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy > last_energy - 0.0001f))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}
}


void Mod3DfromRGBD::solveLMSparseJ()
{
	//								Initialize
	//======================================================================================
	robust_kernel = 1;
	last_energy = 1000000000.f;
	float new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float alpha_sqrt = sqrtf(alpha_raycast);
	const float Kn_sqrtf = sqrtf(Kn);

	//Variables for Levenberg-Marquardt
	unsigned int J_rows = 0, J_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (is_object[i](v,u))			J_rows += 6;
				else if (valid[i](v,u)) 		J_rows++;
			}

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_edges)			J_rows += 4*num_faces;
	if (with_reg_membrane)		J_rows += 3*num_faces;
	if (with_reg_thin_plate)	J_rows += 3*num_faces;

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

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	utils::CTicTac clock; 

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;

		//Occasional search for the correspondences
		if ((iter+1) % 5 == 0)
		{
			searchBetterU();
			searchBetterV();
			printf("\n Global search to avoid wrong correspondences that LM cannot solve");
		}

		evaluateSubDivSurface();
		//if (with_reg)	evaluateSubDivSurfaceRegularization();
		computeTransCoordAndResiduals();

		printf("\n It starts to compute the Jacobians"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{
							//									Data alignment
							//----------------------------------------------------------------------------------------------------
							//Control vertices
							const unsigned int weights_col = v + u*rows;
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									const float v_weight = w_contverts[i](c, weights_col);
									j_elem.push_back(Tri(cont, 3*cp, -T_inv(0,0)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+1, -T_inv(0,1)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+2, -T_inv(0,2)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp, -T_inv(1,0)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+1, -T_inv(1,1)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+2, -T_inv(1,2)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp, -T_inv(2,0)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+1, -T_inv(2,1)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+2, -T_inv(2,2)*v_weight));
								}
							}

							//Camera poses
							Vector4f t_point; t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
							for (unsigned int l = 0; l < 6; l++)
							{
								const Vector3f prod = -mat_der_xi[l].block<3,4>(0,0)*t_point;
								j_elem.push_back(Tri(cont, 3*num_verts + 6*i + l, prod(0)));
								j_elem.push_back(Tri(cont+1, 3*num_verts + 6*i + l, prod(1)));
								j_elem.push_back(Tri(cont+2, 3*num_verts + 6*i + l, prod(2)));
							}

							//Fill the residuals
							R(cont) = res_x[i](v,u);
							R(cont+1) = res_y[i](v,u);
							R(cont+2) = res_z[i](v,u);
							cont += 3;

							//									Normal alignment
							//----------------------------------------------------------------------------------------------
							//Control vertices
							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							const Matrix3f J_mult_norm = -Kn_sqrtf*T_inv*J_nu;
						
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									//Normals
									const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
									J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
									J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
									J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
									J_nX(1,0) = -J_nX(0,1);
									J_nX(2,0) = -J_nX(0,2);
									J_nX(2,1) = -J_nX(1,2);

									const Matrix3f J_norm_fit = J_mult_norm*J_nX;
									j_elem.push_back(Tri(cont, 3*cp, J_norm_fit(0,0)));
									j_elem.push_back(Tri(cont, 3*cp+1, J_norm_fit(0,1)));
									j_elem.push_back(Tri(cont, 3*cp+2, J_norm_fit(0,2)));
									j_elem.push_back(Tri(cont+1, 3*cp, J_norm_fit(1,0)));
									j_elem.push_back(Tri(cont+1, 3*cp + 1, J_norm_fit(1,1)));
									j_elem.push_back(Tri(cont+1, 3*cp + 2, J_norm_fit(1,2)));
									j_elem.push_back(Tri(cont+2, 3*cp, J_norm_fit(2,0)));
									j_elem.push_back(Tri(cont+2, 3*cp + 1, J_norm_fit(2,1)));
									j_elem.push_back(Tri(cont+2, 3*cp + 2, J_norm_fit(2,2)));
								}
							}

							//Camera pose			
							Vector3f normal; normal << nx[i](v,u), ny[i](v,u), nz[i](v,u);
							const Vector3f n_t = sqrtf(Kn)*T_inv*inv_norm*normal;
							for (unsigned int l = 3; l < 6; l++)
							{
								const Vector3f prod = -mat_der_xi[l].block<3,3>(0,0)*n_t;
								j_elem.push_back(Tri(cont, 3*num_verts + 6*i + l, prod(0)));
								j_elem.push_back(Tri(cont+1, 3*num_verts + 6*i + l, prod(1)));
								j_elem.push_back(Tri(cont+2, 3*num_verts + 6*i + l, prod(2)));
							}

							//Fill the residuals
							R(cont) = Kn_sqrtf*res_nx[i](v,u);
							R(cont+1) = Kn_sqrtf*res_ny[i](v,u);
							R(cont+2) = Kn_sqrtf*res_nz[i](v,u);
							cont += 3;
						}

						//Background
						else
						{
							const float norm_proj_error = sqrtf(square(res_d1[i](v, u)) + square(res_d2[i](v, u)));
							if ((norm_proj_error < tau) && (norm_proj_error > peak_margin))
							{

								Matrix<float, 2, 3> J_pi;
								const float inv_z = 1.f / mx_t[i](v, u);

								J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
										fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

								Matrix<float, 1, 2> J_phi; J_phi << res_d1[i](v,u), res_d2[i](v,u);
								J_phi *= -alpha_sqrt/(tau*norm_proj_error);

								const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi*J_pi*T_inv;

								//Control vertices
								const unsigned int weights_col = v + u*rows;

								for (unsigned int c = 0; c < max_num_w; c++)
								{
									const int cp = w_indices[i](c,weights_col);
									if (cp >= 0)
									{
										const float v_weight = w_contverts[i](c, weights_col);
										j_elem.push_back(Tri(cont, 3*cp, J_phi_pi_Tinv(0)*v_weight));
										j_elem.push_back(Tri(cont, 3*cp+1, J_phi_pi_Tinv(1)*v_weight));
										j_elem.push_back(Tri(cont, 3*cp+2, J_phi_pi_Tinv(2)*v_weight));
									}
								}

								//Camera pose
								Vector4f m_t; m_t << mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u), 1.f;
								for (unsigned int l = 0; l < 6; l++)
								{
									const float prod = (J_phi*J_pi*(mat_der_xi[l] * m_t).block<3, 1>(0, 0)).value();
									j_elem.push_back(Tri(cont, 3*num_verts + 6*i + l, prod));
								}

								//Fill the residuals
								R(cont) = alpha_sqrt*(1.f - norm_proj_error/tau);
							}

							cont++;	
						}
					}

				////Camera prior - Keep close to the initial pose
				//for (unsigned int l = 0; l < 6; l++)
				//	cam_incrs[i](l) += -cam_prior*cam_mfold_old[i](l);
		}

		printf("\n It finishes with Jacobians (without regularization). Time = %f", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)
			fillJacobianRegNormals(cont);

		if (with_reg_edges)
			fillJacobianRegEdges(cont);

		if (with_reg_membrane)
			fillJacobianRegMembrane(cont);

		if (with_reg_thin_plate)
			fillJacobianRegThinPlate(cont);

		printf("\n It finishes with Jacobians (with regularization). Time = %f", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		MatrixXf JtJ = MatrixXf(JtJ_sparse);
		VectorXf b = -J.transpose()*R;
		MatrixXf JtJ_lm;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n It enters the loop solver-energy-check. Time = %f", clock.Tac()); clock.Tic();

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
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

			printf("\n It solves with LM. Time = %f", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int k = 0; k < 6; k++)
					cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
			computeCameraTransfandPosesFromTwist();

			printf("\n It updates variables. Time = %f", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n It creates topology, evaluates the surface and computes the residuals. Time = %f", clock.Tac()); clock.Tic();
			rayCastingLMBackgroundPerPixel();	printf("\n It solves raycasting background. Time = %f", clock.Tac()); clock.Tic();
			rayCastingLMForegroundPerPixel();	printf("\n It solves closest correspondence foreground. Time = %f", clock.Tac()); clock.Tic();
			new_energy = computeEnergyOverall();


			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;
			if (cont_inner > 5) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterU();
				searchBetterV();
				evaluateSubDivSurface();			
				if (with_reg_normals) evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				rayCastingLMBackgroundPerPixel();
				rayCastingLMForegroundPerPixel();
				new_energy = computeEnergyOverall();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
					energy_increasing = true;
					break;
				}
				else
					energy_increasing = false;		
			}
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();
		//showRenderedModel();
		takePictureLimitSurface(false);

		printf("\n Time to finish everything else = %f", clock.Tac()); clock.Tic();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (energy_increasing)//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}


void Mod3DfromRGBD::solveLMOnlyForeground()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));

	//Variables for Levenberg-Marquardt
	MatrixXf J;
	unsigned int J_rows = 0, J_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
					J_rows += 6;

	if (with_reg_normals)	
		J_rows += 6*num_faces*square(s_reg);

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

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	utils::CTicTac clock; 

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		J.fill(0.f);
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		evaluateSubDivSurface();
		//if (with_reg)	evaluateSubDivSurfaceRegularization();
		computeTransCoordAndResiduals();

		printf("\n It starts to compute the Jacobians");

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{
							//									Data alignment
							//----------------------------------------------------------------------------------------------------
							//Control vertices
							Matrix3f J_mult = -T_inv;

							float v_weight;
							const unsigned int weights_col = v + u*rows;
							for (unsigned int cp=0; cp<num_verts; cp++)
								if ((v_weight = w_contverts[i](cp, weights_col)) > 0.f)
								{
									J(cont, 3*cp) = J_mult(0,0)*v_weight;
									J(cont, 3*cp+1) = J_mult(0,1)*v_weight;
									J(cont, 3*cp+2) = J_mult(0,2)*v_weight;
									J(cont+1, 3*cp) = J_mult(1,0)*v_weight;
									J(cont+1, 3*cp+1) = J_mult(1,1)*v_weight;
									J(cont+1, 3*cp+2) = J_mult(1,2)*v_weight;
									J(cont+2, 3*cp) = J_mult(2,0)*v_weight;
									J(cont+2, 3*cp+1) = J_mult(2,1)*v_weight;
									J(cont+2, 3*cp+2) = J_mult(2,2)*v_weight;
								}

							//Camera poses
							Vector4f t_point; t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
							for (unsigned int l = 0; l < 6; l++)
							{
								MatrixXf prod = -mat_der_xi[l] * t_point;
								J(cont, 3*num_verts + 6*i + l) = prod(0);
								J(cont+1, 3*num_verts + 6*i + l) = prod(1);
								J(cont+2, 3*num_verts + 6*i + l) = prod(2);
							}

							//Fill the residuals
							R(cont) = res_x[i](v,u);
							R(cont+1) = res_y[i](v,u);
							R(cont+2) = res_z[i](v,u);
							cont += 3;

							//									Normal alignment
							//----------------------------------------------------------------------------------------------
							//Control vertices
							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							Matrix3f J_mult_norm = -sqrtf(Kn)*T_inv*J_nu;
						
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Normals
								const float wu1 = w_u1[i](cp, weights_col), wu2 = w_u2[i](cp, weights_col);
								J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
								J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
								J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
								J_nX(1,0) = -J_nX(0,1);
								J_nX(2,0) = -J_nX(0,2);
								J_nX(2,1) = -J_nX(1,2);

								Matrix3f J_norm_fit = J_mult_norm*J_nX;
								J(cont, 3*cp) = J_norm_fit(0,0);
								J(cont, 3*cp+1) = J_norm_fit(0,1);
								J(cont, 3*cp+2) = J_norm_fit(0,2);
								J(cont+1, 3*cp) = J_norm_fit(1,0);
								J(cont+1, 3*cp+1) = J_norm_fit(1,1);
								J(cont+1, 3*cp+2) = J_norm_fit(1,2);
								J(cont+2, 3*cp) = J_norm_fit(2,0);
								J(cont+2, 3*cp+1) = J_norm_fit(2,1);
								J(cont+2, 3*cp+2) = J_norm_fit(2,2);
							}

							//Camera pose			
							Vector3f normal; normal << nx[i](v,u), ny[i](v,u), nz[i](v,u);
							Vector3f n_t = sqrtf(Kn)*T_inv*inv_norm*normal;
							for (unsigned int l = 3; l < 6; l++)
							{
								Vector3f prod = -mat_der_xi[l].block<3,3>(0,0)*n_t;
								J(cont, 3*num_verts + 6*i + l) = prod(0);
								J(cont+1, 3*num_verts + 6*i + l) = prod(1);
								J(cont+2, 3*num_verts + 6*i + l) = prod(2);
							}

							//Fill the residuals
							R(cont) = sqrtf(Kn)*res_nx[i](v,u);
							R(cont+1) = sqrtf(Kn)*res_ny[i](v,u);
							R(cont+2) = sqrtf(Kn)*res_nz[i](v,u);
							cont += 3;
						}
					}

				////Camera prior - Keep close to the initial pose
				//for (unsigned int l = 0; l < 6; l++)
				//	cam_incrs[i](l) += -cam_prior*cam_mfold_old[i](l);
		}

		printf("\n It finishes with Jacobians (without regularization)");

		//Include regularization
		if (with_reg_normals)
		{
			Matrix3f J_nX_here, J_nX_forward;
			J_nX_here.assign(0.f); J_nX_forward.assign(0.f);
			Kr = Kr_total/float(num_faces*square(s_reg));
			const float Kr_sqrt = sqrtf(Kr);
	
			for (unsigned int f=0; f<num_faces; f++)
			{
				//Compute the normalizing jacobians
				vector<Matrix3f> J_nu; J_nu.resize(square(s_reg));
				for (int s2=0; s2<s_reg; s2++)
					for (int s1=0; s1<s_reg; s1++)
					{	
						J_nu[s1 + s_reg*s2].row(0) << square(ny_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), -nx_reg[f](s1,s2)*nz_reg[f](s1,s2);
						J_nu[s1 + s_reg*s2].row(1) << -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2);
						J_nu[s1 + s_reg*s2].row(2) << -nx_reg[f](s1,s2)*nz_reg[f](s1,s2), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2));
						J_nu[s1 + s_reg*s2] *= inv_reg_norm[f](s1,s2);
					}


				//Include every equation into LM  - Number of new equations: 3*2*num_faces*s_reg*s_reg
				for (int s2=0; s2<s_reg; s2++)
					for (int s1=0; s1<s_reg; s1++)
					{						
						//Coefficients associated to the regularization
						const unsigned int s1for = min(s1+1, int(s_reg-1));
						const unsigned int s2for = min(s2+1, int(s_reg-1));

						if (s1for != s1)
						{
							const unsigned int weights_col = s1 + s_reg*s2;
							const unsigned int weights_col_for = s1for + s_reg*s2;
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Matrix of weights
								const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
								J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
								J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
								J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
								J_nX_here(1,0) = -J_nX_here(0,1);
								J_nX_here(2,0) = -J_nX_here(0,2);
								J_nX_here(2,1) = -J_nX_here(1,2);	

								//Matrix of weights
								const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);
								J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1for,s2)[2] - wu2_for*u1_der_reg[f](s1for,s2)[2];
								J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1for,s2)[1] - wu1_for*u2_der_reg[f](s1for,s2)[1];
								J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1for,s2)[0] - wu2_for*u1_der_reg[f](s1for,s2)[0];
								J_nX_forward(1,0) = -J_nX_forward(0,1);
								J_nX_forward(2,0) = -J_nX_forward(0,2);
								J_nX_forward(2,1) = -J_nX_forward(1,2);

								const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

								//Update increments
								J.block<3,3>(cont,3*cp) = J_reg;
							}	

							//Fill the residuals
							R(cont) = Kr_sqrt*(nx_reg[f](s1for,s2) - nx_reg[f](s1,s2));
							R(cont+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2));
							R(cont+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2));
							cont += 3;
						}

						if (s2for != s2)
						{
							const unsigned int weights_col = s1 + s_reg*s2;
							const unsigned int weights_col_for = s1 + s_reg*s2for;
							for (unsigned int cp = 0; cp < num_verts; cp++)
							{
								//Matrix of weights
								const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
								J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
								J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
								J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
								J_nX_here(1,0) = -J_nX_here(0,1);
								J_nX_here(2,0) = -J_nX_here(0,2);
								J_nX_here(2,1) = -J_nX_here(1,2);	

								//Matrix of weights
								const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);
								J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1,s2for)[2] - wu2_for*u1_der_reg[f](s1,s2for)[2];
								J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1,s2for)[1] - wu1_for*u2_der_reg[f](s1,s2for)[1];
								J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1,s2for)[0] - wu2_for*u1_der_reg[f](s1,s2for)[0];
								J_nX_forward(1,0) = -J_nX_forward(0,1);
								J_nX_forward(2,0) = -J_nX_forward(0,2);
								J_nX_forward(2,1) = -J_nX_forward(1,2);

								const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

								//Update increments
								J.block<3,3>(cont,3*cp) = J_reg;
							}	

							//Fill the residuals
							R(cont) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2));
							R(cont+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2));
							R(cont+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2));
							cont += 3;
						}				
					}
				}
		}

		printf("\n It finishes with Jacobians (with regularization)");

		//Prepare Levenberg solver
		MatrixXf JtJ; JtJ.multiply_AtA(J);
		MatrixXf JtJ_lm;
		VectorXf b = -J.transpose()*R;	//printf("\n It computes b = -Jt*R");

		//VectorXi mask_singularities(J_cols);
		//mask_singularities.fill(0);
		//for (unsigned int k=0; k<J_cols; k++)
		//	if (abs(J.col(k).sumAll()) < 0.5f)
		//		mask_singularities(k) = 1;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n It enters the loop solver-energy-check");

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			JtJ_lm = JtJ;
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);


			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);
			//for (unsigned int k=0; k<J_cols; k++)
			//	if (mask_singularities(k))
			//		increments(k) = 0.f;

			printf("\n It solves with LM");
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c, k) + increments(cont++);

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int k = 0; k < 6; k++)
					cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);

			computeCameraTransfandPosesFromTwist();


			printf("\n It updates variables");

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();			
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n It computes the residuals");
			rayCastingLMForegroundPerPixel();	printf("\n It solves closest correspondence foreground");
			new_energy = computeEnergyOverall();


			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;
			if (cont_inner > 10) energy_increasing = false;
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();
		showRenderedModel();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (cont_inner > 10)//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			
			//Recover old variables
			vert_coords = vert_coords_old;
			cam_mfold = cam_mfold_old;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}


void Mod3DfromRGBD::solveLMOnlyBackground()
{
	//								Initialize
	//======================================================================================
	robust_kernel = 1;
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float alpha_sqrt = sqrtf(alpha_raycast);

	//	//Variables for Levenberg-Marquardt
	SparseMatrix<float> Jf, Jg;
	VectorXf Rg, increments;
	vector<Tri> jg_elem;
	unsigned int Jg_rows = 0, J_cols = 3 * num_verts + 6 * num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (!is_object[i](v,u) && valid[i](v,u))
					Jg_rows++;
			}

	Jg.resize(Jg_rows, J_cols);
	Rg.resize(Jg_rows);
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

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyOverall();

	utils::CTicTac clock; 

	//									Iterative solver
	//====================================================================================
	for (unsigned int i = 0; i < max_iter; i++)
	{
		clock.Tic();
		unsigned int cont_f = 0, cont_g = 0;
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		evaluateSubDivSurface();
		//if (with_reg)	evaluateSubDivSurfaceRegularization();
		computeTransCoordAndResiduals();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
				{
					//Warning
					if (mx_t[i](v,u) <= 0.f)
						printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
					//Background
					if (!is_object[i](v,u) && valid[i](v,u))
					{
						const float norm_proj_error = sqrtf(square(res_d1[i](v, u)) + square(res_d2[i](v, u)));
						if ((norm_proj_error < tau) && (norm_proj_error > peak_margin))
						{

							if (mx_t[i](v, u) <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

							Matrix<float, 2, 3> J_pi;
							const float inv_z = 1.f / mx_t[i](v, u);

							J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
									fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

							Matrix<float, 1, 2> J_phi; J_phi << res_d1[i](v,u), res_d2[i](v,u);
							J_phi *= -alpha_sqrt/(tau*norm_proj_error);

							const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi*J_pi*T_inv;

							//Control vertices
							float v_weight;
							const unsigned int weights_col = v + u*rows;
							for (unsigned int cp = 0; cp < num_verts; cp++)
								if ((v_weight = w_contverts[i](cp, weights_col)) > 0.f)
								{
									jg_elem.push_back(Tri(cont_g, 3*cp, J_phi_pi_Tinv(0)*v_weight));
									jg_elem.push_back(Tri(cont_g, 3*cp+1, J_phi_pi_Tinv(1)*v_weight));
									jg_elem.push_back(Tri(cont_g, 3*cp+2, J_phi_pi_Tinv(2)*v_weight));
								}

							//Camera pose
							Vector4f m_t; m_t << mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u), 1.f;
							for (unsigned int l = 0; l < 6; l++)
							{
								const float prod = (J_phi*J_pi*(mat_der_xi[l] * m_t).block<3, 1>(0, 0)).value();
								jg_elem.push_back(Tri(cont_g, 3*num_verts + 6*i + l, prod));
							}

							//Fill the residuals
							Rg(cont_g) = alpha_sqrt*(1.f - norm_proj_error/tau);
						}
						else
							Rg(cont_g) = 0.f;

						cont_g++;	
					}
				}

				////Camera prior - Keep close to the initial pose
				//for (unsigned int l = 0; l < 6; l++)
				//	cam_incrs[i](l) += -cam_prior*cam_mfold_old[i](l);
		}


		//Prepare Levenberg solver
		Jg.setFromTriplets(jg_elem.begin(), jg_elem.end()); jg_elem.clear();
		SparseMatrix<float> JtJ_g_sparse = Jg.transpose()*Jg;
		MatrixXf JtJ_g = MatrixXf(JtJ_g_sparse);
		MatrixXf JtJ_lm;
		VectorXf Jgxg = -Jg.transpose()*Rg;
		VectorXf b = Jgxg;

		energy_increasing = true;
		unsigned int cont_inner = 0;


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			//JtJ_lm = JtJ_f + JtJ_g + adap_mult*MatrixXf::Identity(J_cols, J_cols);	//Levenberg
			JtJ_lm = JtJ_g;
			//JtJ_lm.diagonal() += adap_mult*JtJ_lm.diagonal();					//Levenberg-Marquardt
			for (unsigned int j=0; j<J_cols; j++)
			{
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);
				//if (j>=3*num_verts)
				//	JtJ_lm(j,j) *= 3.f;
			}

			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);
			
			//Update variables
			cont_f = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c, k) + increments(cont_f++);

			//Camera poses
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int k = 0; k < 6; k++)
					cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont_f++);

			computeCameraTransfandPosesFromTwist();


			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();
			rayCastingLMBackgroundPerPixel();
			rayCastingLMForegroundPerPixel();
			new_energy = computeEnergyOverall();

			//if (new_energy <= last_energy)
			//{
			//	energy_increasing = false;
			//	adap_mult *= 4.f;
			//	//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}
			//else
			//{
			//	adap_mult *= 0.5f;
			//	//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}

			//cont_inner++;
			//if (cont_inner > 20) energy_increasing = false;
			energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();
		showRenderedModel();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (cont_inner > 20)//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void Mod3DfromRGBD::rayCastingLMBackgroundPerPixel()
{
	//float aver_lambda = 0.f;
	//unsigned int cont = 0;

	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	const float limit_uincr = 0.05f*sqrtf(num_faces);
	const float lambda_limit = 10000000.f;
	float norm_uincr;
	float lambda_mult = 3.f;


	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix4f &mytrans_inv = cam_trans_inv[i];
		const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u) && valid[i](v,u))
				{
					energy = square(res_d1[i](v,u)) + square(res_d2[i](v,u));
					energy_ratio = 2.f;
					norm_uincr = 1.f;
					lambda = 10.f;

					while (energy_ratio > 1.0001f)
					{				
						//Old equal to new for the next iteration
						u1_old[i](v, u) = u1[i](v, u);
						u2_old[i](v, u) = u2[i](v, u);
						uface_old[i](v, u) = uface[i](v, u);
						energy_old = energy;

						//Fill the Jacobian with the gradients with respect to the internal points
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! The model is behind the camera. Problems with the projection");

						Matrix<float, 2, 3> J_pi;
						const float inv_z = 1.f / mx_t[i](v, u);

						J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
								fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;
	
						Matrix2f J;
						Vector3f u_der_vec;
						u_der_vec << u1_der[i](v, u)[0], u1_der[i](v, u)[1], u1_der[i](v, u)[2];
						J.col(0) = J_pi*T_inv*u_der_vec;
						u_der_vec << u2_der[i](v, u)[0], u2_der[i](v, u)[1], u2_der[i](v, u)[2];
						J.col(1) = J_pi*T_inv*u_der_vec;

						Vector2f R; R << -res_d1[i](v, u), -res_d2[i](v, u);
						const Vector2f b = J.transpose()*R;
						Matrix2f JtJ; JtJ.multiply_AtA(J);

						bool energy_increasing = true;

						while (energy_increasing)
						{							
																		//printf("\n raycast background im = %d, pixel = (%d, %d), lambda = %f", i, v, u, lambda);
							
							//Solve with LM
							Matrix2f LM = JtJ;// + lambda*Matrix2f::Identity();
							LM.diagonal() += lambda*LM.diagonal();
							const Vector2f sol = LM.inverse()*b;

							u1_incr[i](v, u) = sol(0);
							u2_incr[i](v, u) = sol(1);

							norm_uincr = sqrt(square(sol(0)) + square(sol(1)));
							if (norm_uincr > limit_uincr)
							{
								lambda *= lambda_mult;
								continue;
							}

							//Update variable
							const float u1_new = u1_old[i](v, u) + u1_incr[i](v, u);
							const float u2_new = u2_old[i](v, u) + u2_incr[i](v, u);
							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
								updateInternalPointCrossingEdges(i, v, u, true);
							else
							{
								u1[i](v, u) = u1_new;
								u2[i](v, u) = u2_new;
								uface[i](v, u) = uface_old[i](v, u);
							}

							//Re-evaluate the mesh with the new parametric coordinates
							evaluateSubDivSurfacePixel(i, v, u);

							//Compute the residuals
							mx_t[i](v, u) = mytrans_inv(0, 0)*mx[i](v, u) + mytrans_inv(0, 1)*my[i](v, u) + mytrans_inv(0, 2)*mz[i](v, u) + mytrans_inv(0, 3);
							my_t[i](v, u) = mytrans_inv(1, 0)*mx[i](v, u) + mytrans_inv(1, 1)*my[i](v, u) + mytrans_inv(1, 2)*mz[i](v, u) + mytrans_inv(1, 3);
							mz_t[i](v, u) = mytrans_inv(2, 0)*mx[i](v, u) + mytrans_inv(2, 1)*my[i](v, u) + mytrans_inv(2, 2)*mz[i](v, u) + mytrans_inv(2, 3);
							if (mx_t[i](v, u) <= 0.f)
								printf("\n Depth coordinate of the internal correspondence is equal or inferior to zero after the transformation!!!");
							const float u_proj = fx*(my_t[i](v, u) / mx_t[i](v, u)) + disp_u;
							const float v_proj = fy*(mz_t[i](v, u) / mx_t[i](v, u)) + disp_v;
							res_d1[i](v, u) = float(u) - u_proj;
							res_d2[i](v, u) = float(v) - v_proj;

							//Compute the energy associated to this pixel
							energy = square(res_d1[i](v,u)) + square(res_d2[i](v,u));

							if (energy > energy_old)
							{
								lambda *= lambda_mult;
								//printf("\n Energy is higher than before");
								//cout << endl << "Lambda updated: " << lambda;
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
								u1[i](v, u) = u1_old[i](v, u);
								u2[i](v, u) = u2_old[i](v, u);
								uface[i](v, u) = uface_old[i](v, u);
								energy_increasing = false;
								energy = energy_old;
							}
						}

						energy_ratio = energy_old / energy;
					}
				}
	}

	//printf("\n Average lambda = %f", aver_lambda / cont);

	//Compute energy of the ray-casting problem
	//computeEnergyMaximization();
	//printf("\n Final energy = %f", energy_vec.back());
}

void Mod3DfromRGBD::evaluateSubDivSurfacePixel(unsigned int i, unsigned int v, unsigned int u)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	float pWeights[20], dsWeights[20], dtWeights[20];

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

	//Compute the normals
	nx[i](v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	ny[i](v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	nz[i](v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
}

void Mod3DfromRGBD::rayCastingLMForegroundPerPixel()
{
	float aver_lambda = 0.f;
	unsigned int cont = 0;
	unsigned int inner_cont;
	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float limit_uincr = 0.05f*sqrtf(num_faces);
	const float lambda_limit = 10000000.f;
	float norm_uincr;
	float lambda_mult = 3.f;
	const float Kn_sqrt = sqrtf(Kn);

	//Solve with LM
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);
		const Matrix4f &mytrans_inv = cam_trans_inv[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			if (is_object[i](v, u))
			{
				energy = square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u))
						 + Kn*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
				energy_ratio = 2.f;
				norm_uincr = 1.f;
				lambda = 1.f;
				inner_cont = 0;

				while (energy_ratio > 1.0005f)
				{
					u1_old[i](v, u) = u1[i](v, u);
					u2_old[i](v, u) = u2[i](v, u);
					uface_old[i](v, u) = uface[i](v, u);
					energy_old = energy;

						//if ((i == 0)&&(v == 43)&&(u == 71))
						//{
						//	const float energy_coord = square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u));
						//	const float energy_norm = Kn*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
						//	printf("\n v = %d, u = %d", v, u);
						//	printf("\n energy_coord = %f, energy_norm = %f", energy_coord, energy_norm);
						//	printf("\n res_n = %f, %f, %f", res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u));
						//}

					//Re-evaluate the normal derivatives
					computeNormalDerivativesPixel(i,v,u);

					//Fill the Jacobian with the gradients with respect to the internal points
					Matrix<float, 6, 2> J;
					Matrix<float, 3, 2> u_der; 
					u_der << u1_der[i](v,u)[0], u2_der[i](v,u)[0], u1_der[i](v,u)[1], u2_der[i](v,u)[1], u1_der[i](v,u)[2], u2_der[i](v,u)[2];
					J.topRows(3) = -T_inv*u_der;

					const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
					Matrix<float, 3, 2> n_der_u;
					n_der_u << n_der_u1[i](v,u)[0], n_der_u2[i](v,u)[0], n_der_u1[i](v,u)[1], n_der_u2[i](v,u)[1], n_der_u1[i](v,u)[2], n_der_u2[i](v,u)[2];
					Matrix3f J_nu;
					J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
					J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
					J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
					J_nu *= inv_norm*square(inv_norm);
					J.bottomRows(3) = -Kn_sqrt*T_inv*J_nu*n_der_u;


					Matrix2f JtJ; JtJ.multiply_AtA(J);
					Matrix<float, 6, 1> R; R << res_x[i](v, u), res_y[i](v, u), res_z[i](v, u), 
												Kn_sqrt*res_nx[i](v,u), Kn_sqrt*res_ny[i](v,u), Kn_sqrt*res_nz[i](v,u);
					Vector2f b = -J.transpose()*R;

					bool energy_increasing = true;

					while (energy_increasing)
					{
						//Solve with LM
						Matrix2f LM = JtJ;// + lambda*Matrix2f::Identity();
						//LM.diagonal() += lambda*LM.diagonal();
						LM(0,0) *= (1.f + lambda);
						LM(1,1) *= (1.f + lambda);
						//Vector2f sol = LM.ldlt().solve(b);
						Vector2f sol = LM.inverse()*b;

						//if ((i == 0)&&(v == 43)&&(u == 71))
						//{
						//	printf("\n v = %d, u = %d", v, u);
						//	cout << endl << "LM: " << LM;
						//	cout << endl << "b: " << b.transpose();
						//	printf("\n u_incr = %f, %f, prev_energy = %f", sol(0), sol(1), energy);
						//	printf("\n res_n = %f, %f, %f", res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u));
						//}

						u1_incr[i](v, u) = sol(0);
						u2_incr[i](v, u) = sol(1);
						norm_uincr = sqrt(square(sol(0)) + square(sol(1)));

						if (norm_uincr > limit_uincr)
						{
							lambda *= lambda_mult;
							continue;
						}

						//Update variable
						const float u1_new = u1_old[i](v, u) + u1_incr[i](v, u);
						const float u2_new = u2_old[i](v, u) + u2_incr[i](v, u);
						if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							updateInternalPointCrossingEdges(i, v, u, true);
						else
						{
							u1[i](v, u) = u1_new;
							u2[i](v, u) = u2_new;
							uface[i](v, u) = uface_old[i](v, u);
						}

						//Re-evaluate the mesh with the new parametric coordinates
						evaluateSubDivSurfacePixel(i, v, u);

						//Compute residuals
						mx_t[i](v, u) = mytrans_inv(0, 0)*mx[i](v, u) + mytrans_inv(0, 1)*my[i](v, u) + mytrans_inv(0, 2)*mz[i](v, u) + mytrans_inv(0, 3);
						my_t[i](v, u) = mytrans_inv(1, 0)*mx[i](v, u) + mytrans_inv(1, 1)*my[i](v, u) + mytrans_inv(1, 2)*mz[i](v, u) + mytrans_inv(1, 3);
						mz_t[i](v, u) = mytrans_inv(2, 0)*mx[i](v, u) + mytrans_inv(2, 1)*my[i](v, u) + mytrans_inv(2, 2)*mz[i](v, u) + mytrans_inv(2, 3);
						res_x[i](v, u) = depth[i](v, u) - mx_t[i](v, u);
						res_y[i](v, u) = x_image[i](v, u) - my_t[i](v, u);
						res_z[i](v, u) = y_image[i](v, u) - mz_t[i](v, u);

						nx_t[i](v, u) = mytrans_inv(0, 0)*nx[i](v, u) + mytrans_inv(0, 1)*ny[i](v, u) + mytrans_inv(0, 2)*nz[i](v, u);
						ny_t[i](v, u) = mytrans_inv(1, 0)*nx[i](v, u) + mytrans_inv(1, 1)*ny[i](v, u) + mytrans_inv(1, 2)*nz[i](v, u);
						nz_t[i](v, u) = mytrans_inv(2, 0)*nx[i](v, u) + mytrans_inv(2, 1)*ny[i](v, u) + mytrans_inv(2, 2)*nz[i](v, u);
						const float inv_norm = 1.f/sqrtf(square(nx_t[i](v,u)) + square(ny_t[i](v,u)) + square(nz_t[i](v,u)));
						res_nx[i](v,u) = nx_image[i](v,u) - inv_norm*nx_t[i](v,u);
						res_ny[i](v,u) = ny_image[i](v,u) - inv_norm*ny_t[i](v,u);
						res_nz[i](v,u) = nz_image[i](v,u) - inv_norm*nz_t[i](v,u);

						//Compute the energy associated to this pixel
						energy = square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u))
								+ Kn*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));

						if (energy > energy_old)
						{
							lambda *= lambda_mult;
							//printf("\n Energy is higher than before. Lambda = %f, energy = %f, u_incr = %f", lambda, energy, norm_uincr);
						}
						else
						{
							energy_increasing = false;
							lambda /= lambda_mult;

							aver_lambda += lambda;
							cont++;
							//printf("\n Energy is lower than before");
						}

						//Keep the last solution and finish
						if (lambda > lambda_limit)
						{
							u1[i](v, u) = u1_old[i](v, u);
							u2[i](v, u) = u2_old[i](v, u);
							uface[i](v, u) = uface_old[i](v, u);
							energy_increasing = false;
							energy = energy_old;
						}

						inner_cont++;
						//if (inner_cont > 20)
						//	printf("\n energy = %f, energy_old = %f, cont = %d, energy_ratio = %f, norm_uincr = %f", 
						//			energy, energy_old, inner_cont, energy_ratio, norm_uincr);
					}

					energy_ratio = energy_old / energy;
				}
			}
	}

	//printf("\n Average lambda = %f", aver_lambda / cont);

	//Compute energy of the ray-casting problem
	//computeEnergyMinimization();
	//printf("\n Final energy = %f", energy_vec.back());
}

void Mod3DfromRGBD::vertIncrRegularizationNormals()
{
	Matrix3f J_nX; J_nX.assign(0.f);
	Kr = Kr_total/float(square(s_reg));
	
	for (unsigned int f=0; f<num_faces; f++)
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{						
				//Jacobian of the normalization function
				Matrix3f J_nu;
				J_nu.row(0) << square(ny_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), -nx_reg[f](s1,s2)*nz_reg[f](s1,s2);
				J_nu.row(1) << -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2);
				J_nu.row(2) << -nx_reg[f](s1,s2)*nz_reg[f](s1,s2), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2));
				J_nu *= inv_reg_norm[f](s1,s2);

				//Coefficients associated to the regularization
				const unsigned int s1for = min(s1+1, int(s_reg-1));
				const unsigned int s1bac = max(s1-1, 0);
				const unsigned int s2for = min(s2+1, int(s_reg-1));
				const unsigned int s2bac = max(s2-1, 0);

				Matrix<float, 1, 3> norm_coef;	
				norm_coef(0) = 4.f*nx_reg[f](s1,s2) - (nx_reg[f](s1for,s2) + nx_reg[f](s1bac,s2) + nx_reg[f](s1,s2for) + nx_reg[f](s1,s2bac));
				norm_coef(1) = 4.f*ny_reg[f](s1,s2) - (ny_reg[f](s1for,s2) + ny_reg[f](s1bac,s2) + ny_reg[f](s1,s2for) + ny_reg[f](s1,s2bac));
				norm_coef(2) = 4.f*nz_reg[f](s1,s2) - (nz_reg[f](s1for,s2) + nz_reg[f](s1bac,s2) + nz_reg[f](s1,s2for) + nz_reg[f](s1,s2bac));

				Matrix<float, 1, 3> J_mult = 2.f*Kr*norm_coef*J_nu;		
				
				//For each control vertex
				const unsigned int weights_col = s1 + s2*s_reg;
				for (unsigned int cp = 0; cp < num_verts; cp++)
				{
					//Matrix of weights
					const float wu1 = w_u1_reg[f](cp, weights_col), wu2 = w_u2_reg[f](cp, weights_col);
					J_nX(0,1) = wu1*u2_der_reg[f](s1,s2)[2] - wu2*u1_der_reg[f](s1,s2)[2];
					J_nX(0,2) = wu2*u1_der_reg[f](s1,s2)[1] - wu1*u2_der_reg[f](s1,s2)[1];
					J_nX(1,2) = wu1*u2_der_reg[f](s1,s2)[0] - wu2*u1_der_reg[f](s1,s2)[0];
					J_nX(1,0) = -J_nX(0,1);
					J_nX(2,0) = -J_nX(0,2);
					J_nX(2,1) = -J_nX(1,2);	

					//Update increments
					vert_incrs(0, cp) += (J_mult*J_nX)(0);
					vert_incrs(1, cp) += (J_mult*J_nX)(1);
					vert_incrs(2, cp) += (J_mult*J_nX)(2);
				}	
			}
}

void Mod3DfromRGBD::vertIncrRegularizationEdges()
{
	Ke = Ke_total*float(num_faces);
	
	for (unsigned int f=0; f<num_faces; f++)
		for (unsigned int e=0; e<4; e++)		//They seem to be always stored in order (I have checked it at least for some random faces)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;
			const unsigned int ind_e2 = (e+2)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);
			const unsigned int vert_e2 = face_verts(ind_e2,f);

			const Vector3f edge_0 = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_0 = sqrtf(edge_0.sumAll());
			const Vector3f edge_1 = (vert_coords.col(vert_e2) - vert_coords.col(vert_e1)).square().matrix();
			const float length_1 = sqrtf(edge_1.sumAll());
			const float rL = length_0 - length_1;

			const float mult = 2.f*Ke*rL;

			vert_incrs.col(vert_e0) += mult*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length_0;
			vert_incrs.col(vert_e1) += mult*((vert_coords.col(vert_e1) - vert_coords.col(vert_e0))/length_0
									  	    +(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1);
			vert_incrs.col(vert_e2) += -mult*(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1;
		}
}

void Mod3DfromRGBD::fillJacobianRegNormals(unsigned int &J_row)
{
	Matrix3f J_nX_here, J_nX_forward;
	J_nX_here.assign(0.f); J_nX_forward.assign(0.f);
	//Kr = Kr_total/float(num_faces*square(s_reg));
	Kr = Kr_total/float(square(s_reg));
	const float Kr_sqrt = sqrtf(Kr);
	
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Compute the normalizing jacobians
		vector<Matrix3f> J_nu; J_nu.resize(square(s_reg));
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{	
				if (regularize_unitary_normals)
				{		
					J_nu[s1 + s_reg*s2].row(0) << square(ny_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), -nx_reg[f](s1,s2)*nz_reg[f](s1,s2);
					J_nu[s1 + s_reg*s2].row(1) << -nx_reg[f](s1,s2)*ny_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(nz_reg[f](s1,s2)), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2);
					J_nu[s1 + s_reg*s2].row(2) << -nx_reg[f](s1,s2)*nz_reg[f](s1,s2), -ny_reg[f](s1,s2)*nz_reg[f](s1,s2), square(nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2));
					J_nu[s1 + s_reg*s2] *= inv_reg_norm[f](s1,s2);
				}
				else
				{
					J_nu[s1 + s_reg*s2].setIdentity();				
				}
			}

		//Include every equation into LM  - Number of new equations: 3*2*num_faces*s_reg*s_reg
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{						
				//Coefficients associated to the regularization
				const unsigned int s1for = min(s1+1, int(s_reg-1));
				const unsigned int s2for = min(s2+1, int(s_reg-1));

				if (s1for != s1)
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1for + s_reg*s2;
					const float dist = 1.f; //sqrtf(square(mx_reg[f](s1for,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1for,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1for,s2) - mz_reg[f](s1,s2)));

					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						//Matrices of weights
						const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
						const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);

						if ((wu1_here == 0.f) && (wu2_here == 0.f) && (wu1_for == 0.f) && (wu2_for == 0.f))
							continue;

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1for,s2)[2] - wu2_for*u1_der_reg[f](s1for,s2)[2];
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1for,s2)[1] - wu1_for*u2_der_reg[f](s1for,s2)[1];
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1for,s2)[0] - wu2_for*u1_der_reg[f](s1for,s2)[0];
						J_nX_forward(1,0) = -J_nX_forward(0,1);
						J_nX_forward(2,0) = -J_nX_forward(0,2);
						J_nX_forward(2,1) = -J_nX_forward(1,2);

						//const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);
						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist;

						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp + 1, J_reg(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp + 2, J_reg(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp + 1, J_reg(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp + 2, J_reg(2,2)));
					}	

					//Fill the residuals
					R(J_row) = Kr_sqrt*(nx_reg[f](s1for,s2) - nx_reg[f](s1,s2))/dist;
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2))/dist;
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2))/dist;
					J_row += 3;
				}

				if (s2for != s2)
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1 + s_reg*s2for;
					const float dist = 1.f; //sqrtf(square(mx_reg[f](s1,s2for) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2for) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2for) - mz_reg[f](s1,s2)));

					for (unsigned int cp = 0; cp < num_verts; cp++)
					{
						//Matrices of weights
						const float wu1_here = w_u1_reg[f](cp, weights_col), wu2_here = w_u2_reg[f](cp, weights_col);
						const float wu1_for = w_u1_reg[f](cp, weights_col_for), wu2_for = w_u2_reg[f](cp, weights_col_for);

						if ((wu1_here == 0.f) && (wu2_here == 0.f) && (wu1_for == 0.f) && (wu2_for == 0.f))
							continue;

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1,s2for)[2] - wu2_for*u1_der_reg[f](s1,s2for)[2];
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1,s2for)[1] - wu1_for*u2_der_reg[f](s1,s2for)[1];
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1,s2for)[0] - wu2_for*u1_der_reg[f](s1,s2for)[0];
						J_nX_forward(1,0) = -J_nX_forward(0,1);
						J_nX_forward(2,0) = -J_nX_forward(0,2);
						J_nX_forward(2,1) = -J_nX_forward(1,2);

						//const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);
						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist;

						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp + 1, J_reg(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp + 2, J_reg(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp + 1, J_reg(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp + 2, J_reg(2,2)));
					}	

					//Fill the residuals
					R(J_row) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2))/dist;
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2))/dist;
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2))/dist;
					J_row += 3;
				}				
			}
		}
}

void Mod3DfromRGBD::fillJacobianRegEdges(unsigned int &J_row)
{
	Ke = Ke_total*float(num_faces);
	const float Ke_sqrt = sqrtf(Ke);
	const float sqrt_2 = sqrtf(2.f);
	
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Sides of the faces
		for (unsigned int e=0; e<4; e++)		//They seem to be always stored in order (I have checked it at least for some random faces)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;
			const unsigned int ind_e2 = (e+2)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);
			const unsigned int vert_e2 = face_verts(ind_e2,f);

			const Vector3f edge_0 = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_0 = sqrtf(edge_0.sumAll());
			const Vector3f edge_1 = (vert_coords.col(vert_e2) - vert_coords.col(vert_e1)).square().matrix();
			const float length_1 = sqrtf(edge_1.sumAll());
			//const float rL = length_0 - length_1;

			const Vector3f J_vert_e0 = Ke_sqrt*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length_0;
			const Vector3f J_vert_e1 = Ke_sqrt*((vert_coords.col(vert_e1) - vert_coords.col(vert_e0))/length_0
									  			+(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1);
			const Vector3f J_vert_e2 = -Ke_sqrt*(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1;

			j_elem.push_back(Tri(J_row, 3*vert_e0, J_vert_e0(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e1, J_vert_e1(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e2, J_vert_e2(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+1, J_vert_e0(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+1, J_vert_e1(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+1, J_vert_e2(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+2, J_vert_e0(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+2, J_vert_e1(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+2, J_vert_e2(2)));


			//Fill the residual
			R(J_row) = Ke_sqrt*(length_0 - length_1);
			J_row++;
		}	

		//Diagonals of the faces	(4 equations more)
		for (unsigned int e=0; e<4; e++)
		{
			unsigned int vert_e0, vert_e1, vert_e2;
			switch (e) {
			case 0:
				vert_e0 = face_verts(0,f); vert_e1 = face_verts(1,f); vert_e2 = face_verts(3,f);
				break;
			case 1:
				vert_e0 = face_verts(2,f); vert_e1 = face_verts(3,f); vert_e2 = face_verts(1,f);
				break;
			case 2:
				vert_e0 = face_verts(1,f); vert_e1 = face_verts(2,f); vert_e2 = face_verts(0,f);
				break;
			case 3:
				vert_e0 = face_verts(3,f); vert_e1 = face_verts(0,f); vert_e2 = face_verts(2,f);
				break;
			}

			const Vector3f edge = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_0 = sqrtf(edge.sumAll());
			const Vector3f diag = (vert_coords.col(vert_e2) - vert_coords.col(vert_e1)).square().matrix();
			const float length_1 = sqrtf(diag.sumAll());

			const Vector3f J_vert_e0 = Ke_sqrt*sqrt_2*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length_0;
			const Vector3f J_vert_e1 = Ke_sqrt*(sqrt_2*(vert_coords.col(vert_e1) - vert_coords.col(vert_e0))/length_0
									  			+(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1);
			const Vector3f J_vert_e2 = -Ke_sqrt*(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1;

			j_elem.push_back(Tri(J_row, 3*vert_e0, J_vert_e0(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e1, J_vert_e1(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e2, J_vert_e2(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+1, J_vert_e0(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+1, J_vert_e1(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+1, J_vert_e2(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+2, J_vert_e0(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+2, J_vert_e1(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+2, J_vert_e2(2)));

			//Fill the residual
			R(J_row) = Ke_sqrt*(sqrt_2*length_0 - length_1);
			J_row++;
		}
	}
}

void  Mod3DfromRGBD::fillJacobianRegMembrane(unsigned int &J_row)
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);

	//Other necessary steps...
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	const float K_m_sqrt = sqrtf(K_m);


	for (unsigned int f=0; f<num_faces; f++)
	{
		//Find the vertices associated to this face
		const float u1_eval = 0.5f, u2_eval = 0.5f;
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1_eval, u2_eval);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1_eval, u2_eval, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);	//Oook, the vertices in cvs are sorted in order!!

		//Add regularization to the system
		for (unsigned int k=0; k<3; k++)
		{
			Matrix<float, max_num_w, 1> verts_f; verts_f.fill(0.f);				
			for (unsigned int cv=0; cv<max_num_w; cv++)
			{
				if (cvs[cv] < num_verts)
					verts_f(cv) = vert_coords(k,cvs[cv]);
				else
				{
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						verts_f(cv) += st_weights[s]*vert_coords(k, st_ind[s]);
				}			
			}
					
			const Matrix<float, max_num_w, 1> w_f = Q_m_sqrt*verts_f;
			const float inv_norm = 1.f/w_f.norm();

			//Update the Jacobian
			for (unsigned int cv=0; cv<max_num_w; cv++)
			{
				const float J_Qm = K_m_sqrt*inv_norm*(w_f.transpose()*Q_m_sqrt.col(cv)).value();
				if (cvs[cv] < num_verts)
					j_elem.push_back(Tri(J_row, 3*cvs[cv]+k, J_Qm));
				else
				{
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						j_elem.push_back(Tri(J_row, 3*st_ind[s]+k, J_Qm*st_weights[s]));											
				}
			}

			//Update the residuals
			R(J_row) = K_m_sqrt/inv_norm;
			J_row++;
		}
	}
}

void  Mod3DfromRGBD::fillJacobianRegThinPlate(unsigned int &J_row)
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);

	//Other necessary steps...
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	const float K_tp_sqrt = sqrtf(K_tp);


	for (unsigned int f=0; f<num_faces; f++)
	{
		//Find the vertices associated to this face
		const float u1_eval = 0.5f, u2_eval = 0.5f;
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1_eval, u2_eval);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1_eval, u2_eval, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);	//Oook, the vertices in cvs are sorted in order!!

		//Add regularization to the system
		for (unsigned int k=0; k<3; k++)
		{
			Matrix<float, max_num_w, 1> verts_f; verts_f.fill(0.f);				
			for (unsigned int cv=0; cv<max_num_w; cv++)
			{
				if (cvs[cv] < num_verts)
					verts_f(cv) = vert_coords(k,cvs[cv]);
				else
				{
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						verts_f(cv) += st_weights[s]*vert_coords(k, st_ind[s]);
				}			
			}
					
			const Matrix<float, max_num_w, 1> w_f = Q_tp_sqrt*verts_f;
			const float inv_norm = 1.f/w_f.norm();

			//Update the Jacobian
			for (unsigned int cv=0; cv<max_num_w; cv++)
			{
				const float J_Qtp = K_tp_sqrt*inv_norm*(w_f.transpose()*Q_tp_sqrt.col(cv)).value();
				if (cvs[cv] < num_verts)
					j_elem.push_back(Tri(J_row, 3*cvs[cv]+k, J_Qtp));
				else
				{
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						j_elem.push_back(Tri(J_row, 3*st_ind[s]+k, J_Qtp*st_weights[s]));											
				}
			}

			//Update the residuals
			R(J_row) = K_tp_sqrt/inv_norm;
			J_row++;
		}
	}
}

float Mod3DfromRGBD::computeEnergyRegNormals()
{
	Kr = Kr_total/float(square(s_reg));		//***********************************************************************
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const float dist_s1 = 1.f; //square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
				const float dist_s2 = 1.f; //square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));

				energy += Kr*((square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1
								+(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2);
			}
		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const float dist_s1 = 1.f; //square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1;
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const float dist_s2 =  1.f; //square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2;	
		}
	}

	return energy;
}

float Mod3DfromRGBD::computeEnergyRegEdges()
{
	float energy = 0.f;
	const float sqrt_2 = sqrtf(2.f);
	Ke = Ke_total*float(num_faces);
	for (unsigned int f=0; f<num_faces; f++)
	{
		for (unsigned int e=0; e<4; e++)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;
			const unsigned int ind_e2 = (e+2)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);
			const unsigned int vert_e2 = face_verts(ind_e2,f);	

			const Vector3f edge_0 = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_0 = sqrtf(edge_0.sumAll());
			const Vector3f edge_1 = (vert_coords.col(vert_e2) - vert_coords.col(vert_e1)).square().matrix();
			const float length_1 = sqrtf(edge_1.sumAll());

			energy += Ke*square(length_0 - length_1);
		}

		//Diagonals
		for (unsigned int e=0; e<4; e++)
		{
			unsigned int vert_e0, vert_e1, vert_e2;
			switch (e) {
			case 0:
				vert_e0 = face_verts(0,f); vert_e1 = face_verts(1,f); vert_e2 = face_verts(3,f);
				break;
			case 1:
				vert_e0 = face_verts(2,f); vert_e1 = face_verts(3,f); vert_e2 = face_verts(1,f);
				break;
			case 2:
				vert_e0 = face_verts(1,f); vert_e1 = face_verts(2,f); vert_e2 = face_verts(0,f);
				break;
			case 3:
				vert_e0 = face_verts(3,f); vert_e1 = face_verts(0,f); vert_e2 = face_verts(2,f);
				break;
			}

			const Vector3f edge = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_0 = sqrtf(edge.sumAll());
			const Vector3f diag = (vert_coords.col(vert_e2) - vert_coords.col(vert_e1)).square().matrix();
			const float length_1 = sqrtf(diag.sumAll());

			energy += Ke*square(sqrt_2*length_0 - length_1);
		}
	}

	return energy;
}

float Mod3DfromRGBD::computeEnergyRegMembrane()
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);

	//Other necessary steps...
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	float energy_m = 0.f;

	for (unsigned int f=0; f<num_faces; f++)
	{
		//Find the vertices associated to this face
		const float u1_eval = 0.5f, u2_eval = 0.5f;
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1_eval, u2_eval);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1_eval, u2_eval, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);	//Oook, the vertices in cvs are sorted in order!!

		for (unsigned int k=0; k<3; k++)
		{
			Matrix<float, max_num_w, 1> verts_f; verts_f.fill(0.f);
			for (int cv = 0; cv < cvs.size(); ++cv)
			{	
				if (cvs[cv] < num_verts)
					verts_f(cv) = vert_coords(k,cvs[cv]);
				else
				{
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						verts_f(cv) += st_weights[s]*vert_coords(k, st_ind[s]);
				}
			}

			energy_m += K_m*(Q_m_sqrt*verts_f).squaredNorm();
			//energy_m += K_m*(verts_f.transpose()*Q_tp_sqrt*verts_f).value();	***** To check that the original formulation of X*Q*X is minimized as well.
		}
	}

	return energy_m;
}

float  Mod3DfromRGBD::computeEnergyRegThinPlate()
{
	//Get all the stencils from the patchTable (necessary to obtain the weights for the gradients)
	//--------------------------------------------------------------------------------------------------
	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);

	//Other necessary steps...
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	float energy_tp = 0.f;

	for (unsigned int f=0; f<num_faces; f++)
	{
		//Find the vertices associated to this face
		const float u1_eval = 0.5f, u2_eval = 0.5f;
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1_eval, u2_eval);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1_eval, u2_eval, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);	//Oook, the vertices in cvs are sorted in order!!

		for (unsigned int k=0; k<3; k++)
		{
			Matrix<float, max_num_w, 1> verts_f; verts_f.fill(0.f);
			for (int cv = 0; cv < cvs.size(); ++cv)
			{	
				if (cvs[cv] < num_verts)
					verts_f(cv) = vert_coords(k,cvs[cv]);
				else
				{	
					//Look at the stencil associated to this local point and distribute its weight over the control vertices
					const unsigned int ind_offset = cvs[cv] - num_verts;
					unsigned int size_st = st[ind_offset].GetSize();
					Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
					float const *st_weights = st[ind_offset].GetWeights();
					for (unsigned int s = 0; s < size_st; s++)
						verts_f(cv) += st_weights[s]*vert_coords(k, st_ind[s]);
				}
			}
				
			energy_tp += K_tp*(Q_tp_sqrt*verts_f).squaredNorm();
			//energy_m += K_m*(verts_f.transpose()*Q_tp_sqrt*verts_f).value();
		}
	}

	return energy_tp;
}



void Mod3DfromRGBD::computeDistanceTransform()
{
	for (unsigned int i = 0; i < num_images; i++)
	{
		//"Expand" the segmentation to its surrounding invalid pixels
		Array<bool, Dynamic, Dynamic> big_segment = is_object[i];
		vector<Array2i> buffer_vu;
		for (int u=0; u<cols-1; u++)
			for (int v=0; v<rows-1; v++)
			{
				//if (!valid[i](v,u))
				//	big_segment(v,u) = true;
				if ((big_segment(v,u) != big_segment(v,u+1))||(big_segment(v,u) != big_segment(v+1,u)))
					if (big_segment(v,u) == true)
					{
						Array2i vu; vu << v, u;
						buffer_vu.push_back(vu);
					}
			}

		while (!buffer_vu.empty())
		{
			const Array2i vu = buffer_vu.back();
			buffer_vu.pop_back();

			if ((vu(0) == 0) || (vu(0) == rows - 1)||(vu(1) == 0) || (vu(1) == cols - 1))
				continue;
			else
			{
				if ((valid[i](vu(0)-1, vu(1)) == false) && (big_segment(vu(0)-1, vu(1)) == false))
				{
					Array2i vu_new; vu_new << vu(0)-1, vu(1);
					buffer_vu.push_back(vu_new);
					big_segment(vu(0)-1, vu(1)) = true;
				}

				if ((valid[i](vu(0)+1, vu(1)) == false) && (big_segment(vu(0)+1, vu(1)) == false))
				{
					Array2i vu_new; vu_new << vu(0)+1, vu(1);
					buffer_vu.push_back(vu_new);
					big_segment(vu(0)+1, vu(1)) = true;
				}

				if ((valid[i](vu(0), vu(1)-1) == false) && (big_segment(vu(0), vu(1)-1) == false))
				{
					Array2i vu_new; vu_new << vu(0), vu(1)-1;
					buffer_vu.push_back(vu_new);
					big_segment(vu(0), vu(1)-1) = true;
				}

				if ((valid[i](vu(0), vu(1)+1) == false) && (big_segment(vu(0), vu(1)+1) == false))
				{
					Array2i vu_new; vu_new << vu(0), vu(1)+1;
					buffer_vu.push_back(vu_new);
					big_segment(vu(0), vu(1)+1) = true;
				}
			}
		}


		//Compute the distance tranform
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if (big_segment(v, u))
					DT[i](v, u) = 0.f;

				//Find the closest pixel which belongs to the object
				else
				{
					float min_dist_square = 1000000.f;
					unsigned int uc, vc; // c - closest

					for (unsigned int us = 0; us < cols; us++)	//s - search
						for (unsigned int vs = 0; vs < rows; vs++)
							if (big_segment(vs, us))
							{
								const float dist_square = square(us - u) + square(vs - v);
								if (dist_square < min_dist_square)
								{
									min_dist_square = dist_square;
									uc = us; vc = vs;
								}
							}
					DT[i](v, u) = sqrt(min_dist_square);
				}
			}

		//Compute the gradient of the distance transform
		for (unsigned int u = 1; u < cols - 1; u++)
			for (unsigned int v = 1; v < rows - 1; v++)
			{
				DT_grad_u[i](v, u) = 0.5f*(DT[i](v, u+1) - DT[i](v, u-1));
				DT_grad_v[i](v, u) = 0.5f*(DT[i](v+1, u) - DT[i](v-1, u));
			}

		for (unsigned int v = 0; v < rows; v++)
		{
			DT_grad_u[i](v, 0) = DT[i](v, 1) - DT[i](v, 0);
			DT_grad_u[i](v, cols-1) = DT[i](v, cols-1) - DT[i](v, cols-2);
		}
		for (unsigned int u = 0; u < rows; u++)
		{
			DT_grad_v[i](0, u) = DT[i](1, u) - DT[i](0, u);
			DT_grad_v[i](rows-1, u) = DT[i](rows-1, u) - DT[i](rows-2, u);
		}
	}
}

void Mod3DfromRGBD::sampleSurfaceForDTBackground()
{
	//Compute the number of samples according to "nsamples_approx"
	const unsigned int nsamples_per_edge = max(3, int(round(sqrtf(float(nsamples_approx) / float(num_faces)))));
	nsamples = square(nsamples_per_edge)*num_faces;

	//Camera parameters
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
	//Resize DT variables
	w_DT.resize(max_num_w, nsamples);
	w_indices_DT.resize(max_num_w, nsamples);
	u1_der_DT.resize(nsamples); u2_der_DT.resize(nsamples);
	mx_DT.resize(nsamples); my_DT.resize(nsamples); mz_DT.resize(nsamples);
	u1_DT.resize(nsamples); u2_DT.resize(nsamples);  uface_DT.resize(nsamples);
	pixel_DT_u.resize(num_images); pixel_DT_v.resize(num_images);

	const float fact = 1.f / float(nsamples_per_edge-1); //**********************
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int u1 = 0; u1 < nsamples_per_edge; u1++)
			for (unsigned int u2 = 0; u2 < nsamples_per_edge; u2++)
			{
				const unsigned int ind = f*square(nsamples_per_edge) + u1*nsamples_per_edge + u2;
				u1_DT(ind) = float(u1)*fact;
				u2_DT(ind) = float(u2)*fact;
				uface_DT(ind) = f;
				u1_der_DT(ind) = new float[3];
				u2_der_DT(ind) = new float[3];
			}

	//Evaluate the surface
	//-----------------------------------------------------------
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (unsigned int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);
	

	for (unsigned int s = 0; s < nsamples; s++)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface_DT(s), u1_DT(s), u2_DT(s)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1_DT(s), u2_DT(s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);
		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//Save the 3D coordinates
		mx_DT(s) = eval.point[0];
		my_DT(s) = eval.point[1];
		mz_DT(s) = eval.point[2];

		//Save the derivatives
		u1_der_DT(s)[0] = eval.deriv1[0];
		u1_der_DT(s)[1] = eval.deriv1[1];
		u1_der_DT(s)[2] = eval.deriv1[2];
		u2_der_DT(s)[0] = eval.deriv2[0];
		u2_der_DT(s)[1] = eval.deriv2[1];
		u2_der_DT(s)[2] = eval.deriv2[2];

		//Compute the weights for the gradient with respect to the control vertices
		VectorXf vect_wc(num_verts); vect_wc.fill(0.f); 

		for (int cv = 0; cv < cvs.size(); ++cv)
		{						
			if (cvs[cv] < num_verts)			
				vect_wc(cvs[cv]) += pWeights[cv];

			else
			{
				const unsigned int ind_offset = cvs[cv] - num_verts;
				unsigned int size_st = st[ind_offset].GetSize();
				Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
				float const *st_weights = st[ind_offset].GetWeights();
				for (unsigned int s = 0; s < size_st; s++)
					vect_wc(st_ind[s]) += pWeights[cv]*st_weights[s];

			}
		}

		//Store the weights
		unsigned int cont = 0;
		w_indices_DT.col(s).fill(-1);

		for (unsigned int cv=0; cv<num_verts; cv++)
			if (vect_wc(cv) != 0.f)
			{
				w_indices_DT(cont, s) = cv;
				w_DT(cont, s) = vect_wc(cv);
				cont++;
			}
	}

	//Compute the pixel to which the samples project
	for (unsigned int i = 0; i < num_images; i++)
	{
		pixel_DT_u[i].resize(nsamples);
		pixel_DT_v[i].resize(nsamples);
		const Matrix4f &T_inv = cam_trans_inv[i];

		for (unsigned int s = 0; s < nsamples; s++)
		{
			//Camera pose
			Vector4f m_DT; m_DT << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
			Vector3f m_t_DT = T_inv.topRows(3)*m_DT;
			const float u_proj = fx*m_t_DT(1) / m_t_DT(0) + disp_u;
			const float v_proj = fy*m_t_DT(2) / m_t_DT(0) + disp_v;
			pixel_DT_u[i](s) = roundf(min(float(cols - 1), max(0.f, u_proj)));
			pixel_DT_v[i](s) = roundf(min(float(rows - 1), max(0.f, v_proj)));
			//printf("\n Pixel proj = %f", pixel_DT[i](s));
		}
	}
}

void Mod3DfromRGBD::solveWithDT()
{
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	float adap_mult = 0.5f;
	sz_x = 0.002f; sz_xi = min(0.00002f*sqrtf(num_faces), 0.0005f);
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);

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
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	sampleSurfaceForDTBackground();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyDTOverall();

	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();

		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		vert_incrs.fill(0.f);
		for (unsigned int i = 0; i < num_images; i++)
			cam_incrs[i].fill(0.f);
		createTopologyRefiner();
		evaluateSubDivSurface(); //To compute the new weights associated to the control vertices
		computeTransCoordAndResiduals();
		sampleSurfaceForDTBackground();


		//Compute the gradients
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];

			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			//Foreground
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v, u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

						Matrix<float, 1, 3> res; res << res_x[i](v,u), res_y[i](v,u), res_z[i](v,u);
						Matrix<float, 1, 3> J_mult = -2.f*res*T_inv;

						const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
						Matrix3f J_nu, J_nX;
						J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
						J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
						J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
						J_nu *= inv_norm*square(inv_norm);
						J_nX.assign(0.f);
						Matrix<float, 1, 3> res_n; res_n << res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u);
						Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu;

						
						//Control vertices
						const unsigned int weights_col = v + u*rows;
						for (unsigned int c = 0; c < max_num_w; c++)
						{
							const int cp = w_indices[i](c,weights_col);
							if (cp >= 0)
							{
								const float ww = w_contverts[i](c, weights_col);
								vert_incrs(0, cp) += J_mult(0)*ww;
								vert_incrs(1, cp) += J_mult(1)*ww;
								vert_incrs(2, cp) += J_mult(2)*ww;

								//Normals
								const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
								J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
								J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
								J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
								J_nX(1,0) = -J_nX(0,1);
								J_nX(2,0) = -J_nX(0,2);
								J_nX(2,1) = -J_nX(1,2);

								vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
								vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
								vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
							}
						}

						//Camera pose			
						Vector4f t_point(4, 1); t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
						for (unsigned int l = 0; l < 6; l++)
						{
							MatrixXf prod = -mat_der_xi[l] * t_point;
							cam_incrs[i](l) += 2.f*(res*prod).value();
						}

						Vector3f normal; normal << nx[i](u), ny[i](u), nz[i](v,u);
						Vector3f n_t = T_inv*inv_norm*normal;
						for (unsigned int l = 3; l < 6; l++)
						{
							Vector3f prod = -mat_der_xi[l].block<3, 3>(0, 0)*n_t;
							cam_incrs[i](l) += 2.f*Kn*(res_n*prod).value();
						}

					}

			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point; t_point << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
				const float mx_t_DT = cam_trans_inv[i].row(0)*t_point;
				const float my_t_DT = cam_trans_inv[i].row(1)*t_point;
				const float mz_t_DT = cam_trans_inv[i].row(2)*t_point;

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 2, 3> J_pi;
				const float inv_z = 1.f / mx_t_DT;

				J_pi << fx*my_t_DT*square(inv_z), -fx*inv_z, 0.f,
						fy*mz_t_DT*square(inv_z), 0.f, -fy*inv_z;

				const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s))), DT_grad_v[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s)))};
				const Matrix<float, 1, 3> J_mult = J_DT*J_pi*T_inv;

				//Control vertices
				for (unsigned int c = 0; c < max_num_w; c++)
				{
					const int cp = w_indices_DT(c, s);
					if (cp >= 0)
					{
						const float ww = w_DT(c, s);
						vert_incrs(0, cp) += -alpha_DT*J_mult(0)*ww;
						vert_incrs(1, cp) += -alpha_DT*J_mult(1)*ww;
						vert_incrs(2, cp) += -alpha_DT*J_mult(2)*ww;
					}
				}

				//Camera pose
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, mz_t_DT, 1.f;
				for (unsigned int l = 0; l < 6; l++)
				{
					Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0, 0);
					cam_incrs[i](l) += -alpha_DT*(J_DT*J_pi*aux_prod).value();
				}
			}
		}

		if (with_reg_normals)
			vertIncrRegularizationNormals();

		energy_increasing = true;
		unsigned int cont = 0.f;

		//Update the control vertices
		while (energy_increasing)
		{
			//Update
			vert_coords = vert_coords_old - adap_mult*sz_x*vert_incrs;

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
			}

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();
			rayCastingLMForegroundPerPixel();
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
			if (cont > 10) energy_increasing = false;
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();

		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		if (cont > 10)	//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime / max_iter);
}

float Mod3DfromRGBD::computeEnergyDTOverall()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					energy_d += square(res_x[i](v, u)) + square(res_y[i](v, u)) + square(res_z[i](v, u));
					energy_d += Kn*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
				}

		for (unsigned int s = 0; s < nsamples; s++)
			energy_b += alpha_DT*DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s));
	}

	//Regularization
	if (with_reg_normals)
		energy_r += computeEnergyRegNormals();

	if (with_reg_edges)
		energy_r += computeEnergyRegEdges();

	if (with_reg_membrane)
		energy_r += computeEnergyRegMembrane();

	if (with_reg_thin_plate)
		energy_r += computeEnergyRegThinPlate();

	energy_data.push_back(energy_d);
	energy_background.push_back(energy_b);
	energy_reg.push_back(energy_r);

	return (energy_d + energy_b + energy_r);
}

float Mod3DfromRGBD::computeEnergyDT2Overall()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = sqrtf(square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u)));
					if (res < truncated_res)
						energy_d += square(res);
					else
						energy_d += square(truncated_res);

					const float resn = sqrtf(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
					if (resn < truncated_resn)
						energy_d += Kn*square(resn);
					
					else
						energy_d += Kn*square(truncated_resn);
				}

		for (unsigned int s = 0; s < nsamples; s++)
			energy_b += alpha_DT*square(DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s)));
	}

	//Regularization
	if (with_reg_normals) 		energy_r += computeEnergyRegNormals();
	if (with_reg_edges) 		energy_r += computeEnergyRegEdges();
	if (with_reg_membrane) 		energy_r += computeEnergyRegMembrane();
	if (with_reg_thin_plate)	energy_r += computeEnergyRegThinPlate();

	//Save?
	float new_energy = energy_d + energy_r + energy_b;
	if (new_energy < last_energy)
	{
		energy_data.push_back(energy_d);
		energy_background.push_back(energy_b);
		energy_reg.push_back(energy_r);
	}

	return new_energy;
}

void Mod3DfromRGBD::runRaycastFromDisplacedMesh()
{
	const float max_dev = 0.35f;
	const unsigned int num_val = 3;
	energy_disp_raycast.resize(num_val, num_val);
	energy_disp_DT.resize(num_val, num_val);

	//Raycast
	for (unsigned int dz=0; dz<num_val; dz++)
		for (unsigned int dy=0; dy<num_val; dy++)
		{
			const float cam_y = max_dev*(float(dy)/float(num_val - 1) - 0.5f);
			const float cam_z = max_dev*(float(dz)/float(num_val - 1) - 0.5f);

			//Set the initial camera pose
			cam_mfold[0].fill(0.f);
			cam_mfold[0](1) = cam_y;
			cam_mfold[0](2) = cam_z;
			computeCameraTransfandPosesFromTwist();
			printf("\n Camera displacement = %f, %f", cam_y, cam_z);
		
			//Run the solver
			adap_mult = 0.0001f;
			energy_disp_raycast(dz, dy) = solveGradientDescentBackgroundCam2D();
		}

	cout << endl << "Energies disp raycast: " << endl << energy_disp_raycast;

	//DT
	for (unsigned int dz=0; dz<num_val; dz++)
		for (unsigned int dy=0; dy<num_val; dy++)
		{
			const float cam_y = max_dev*(float(dy)/float(num_val - 1) - 0.5f);
			const float cam_z = max_dev*(float(dz)/float(num_val - 1) - 0.5f);

			//Set the initial camera pose
			cam_mfold[0].fill(0.f);
			cam_mfold[0](1) = cam_y;
			cam_mfold[0](2) = cam_z;
			computeCameraTransfandPosesFromTwist();
			printf("\n Camera displacement = %f, %f", cam_y, cam_z);
		
			//Run the solver
			adap_mult = 0.01f;
			energy_disp_DT(dz, dy) = solveWithDTBackgroundCam2D();
		}

	cout << endl << "Energies disp DT: " << endl << energy_disp_DT;
}

float Mod3DfromRGBD::solveGradientDescentBackgroundCam2D()			//****************************************** Use the other kernel **************************************
{
	//								Initialize
	//======================================================================================
	utils::CTicTac clock; 
	robust_kernel = 0;
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));

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
	rayCastingLMBackgroundPerPixel();
	rayCastingLMForegroundPerPixel();
	searchBetterU();
	searchBetterV();
	new_energy = computeEnergyOverall();

	//									Iterative solver
	//====================================================================================
	for (unsigned int i = 0; i < max_iter; i++)
	{
		clock.Tic();
		
		//Update old variables
		last_energy = new_energy;
		cam_mfold_old = cam_mfold;
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();


		//							Compute the gradients
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices and clean increments
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);
			cam_incrs[i].fill(0.f);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					

						//Background
						if ((!is_object[i](v, u)) && ( square(res_d1[i](v, u)) + square(res_d2[i](v, u)) < square(tau)))
						{

							Matrix<float, 2, 3> J_pi;
							const float inv_z = 1.f / mx_t[i](v, u);

							J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
									fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

							const float J_phi1 = 2.f*res_d1[i](v, u) / square(tau);
							const float J_phi2 = 2.f*res_d2[i](v, u) / square(tau);
							const Matrix<float, 1, 3> J_phi_pi = J_phi1*J_pi.row(0) + J_phi2*J_pi.row(1);


							//Camera pose
							Vector4f m_t; m_t << mx_t[i](u), my_t[i](u), mz_t[i](u), 1.f;
							for (unsigned int l = 1; l < 3; l++)
							{
								Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0, 0);
								cam_incrs[i](l) += -alpha_raycast*(J_phi_pi*aux_prod).value();
							}
						}
					}
		}

		energy_increasing = true;
		unsigned int cont = 0;


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Update the camera poses
			for (unsigned int i = 0; i < num_images; i++)
				cam_mfold[i] = cam_mfold_old[i] - adap_mult*cam_incrs[i];
			computeCameraTransfandPosesFromTwist();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();
			rayCastingLMBackgroundPerPixel();
			rayCastingLMForegroundPerPixel();
			searchBetterU();
			searchBetterV();
			new_energy = computeEnergyOverall();

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
			if (cont > 10) energy_increasing = false;
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (cont > 10)//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			new_energy = last_energy;
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime / max_iter);

	return new_energy;
}

float Mod3DfromRGBD::solveWithDTBackgroundCam2D()
{
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);

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
	rayCastingLMForegroundPerPixel();
	searchBetterU();
	new_energy = computeEnergyDTOverall();

	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();

		//Update old variables
		last_energy = new_energy;
		cam_mfold_old = cam_mfold;
		for (unsigned int i = 0; i < num_images; i++)
			cam_incrs[i].fill(0.f);
		createTopologyRefiner();
		evaluateSubDivSurface(); //To compute the new weights associated to the control vertices
		computeTransCoordAndResiduals();
		sampleSurfaceForDTBackground();


		//Compute the gradients
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];

			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);


			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point; t_point << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
				const float mx_t_DT = cam_trans_inv[i].row(0)*t_point;
				const float my_t_DT = cam_trans_inv[i].row(1)*t_point;
				const float mz_t_DT = cam_trans_inv[i].row(2)*t_point;

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 2, 3> J_pi;
				const float inv_z = 1.f / mx_t_DT;

				J_pi << fx*my_t_DT*square(inv_z), -fx*inv_z, 0.f,
						fy*mz_t_DT*square(inv_z), 0.f, -fy*inv_z;

				const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s))), DT_grad_v[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s)))};

				//Camera pose
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, mz_t_DT, 1.f;
				for (unsigned int l = 1; l < 3; l++)
				{
					Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0, 0);
					cam_incrs[i](l) += -alpha_DT*(J_DT*J_pi*aux_prod).value();		//J_pi has a - and it shouldn't so this is to compensate ************************************
				}
			}
		}

		energy_increasing = true;
		unsigned int cont = 0;

		//Update the control vertices
		while (energy_increasing)
		{

			for (unsigned int i = 0; i < num_images; i++)
				cam_mfold[i] = cam_mfold_old[i] - adap_mult*cam_incrs[i];
			computeCameraTransfandPosesFromTwist();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			rayCastingLMForegroundPerPixel();
			searchBetterU();
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
			if (cont > 10) energy_increasing = false;
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showCamPoses();
		showMesh();
		showSubSurface();

		printf("\n New_energy = %f, last_energy = %f, runtime = %f", new_energy, last_energy, runtime);
		if (cont > 10)	//(new_energy > last_energy - 0.0000001f)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			new_energy = last_energy;
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime / max_iter);

	return new_energy;
}

void Mod3DfromRGBD::solveDTwithLMNoCameraOptimization()
{
	//								Initialize
	//======================================================================================
	float new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	last_energy = 100000000.f;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float alpha_sqrt = sqrtf(alpha_DT);
	const float Kn_sqrtf = sqrtf(Kn);

	sampleSurfaceForDTBackground();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for Levenberg-Marquardt
	unsigned int J_rows = 0, J_cols = 3 * num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
					J_rows += 6;

	J_rows += num_images*nsamples;

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_membrane)		J_rows += 3*num_faces;
	if (with_reg_thin_plate)	J_rows += 3*num_faces;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	rayCastingLMForegroundPerPixel();
	new_energy = computeEnergyDT2Overall();

	takePictureLimitSurface(false);

	utils::CTicTac clock; 

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;

		//Occasional search for the correspondences
		if ((iter+1) % 5 == 0)
		{
			searchBetterU();
			printf("\n Global search to avoid wrong correspondences that LM cannot solve");
		}

		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		printf("\n It starts to compute the Jacobians"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];
			
			//Fast access to camera matrices
			const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0, 0);

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{
							//									Data alignment
							//----------------------------------------------------------------------------------------------------
							//Control vertices
							const unsigned int weights_col = v + u*rows;
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									const float v_weight = w_contverts[i](c, weights_col);
									j_elem.push_back(Tri(cont, 3*cp, -T_inv(0,0)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+1, -T_inv(0,1)*v_weight));
									j_elem.push_back(Tri(cont, 3*cp+2, -T_inv(0,2)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp, -T_inv(1,0)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+1, -T_inv(1,1)*v_weight));
									j_elem.push_back(Tri(cont+1, 3*cp+2, -T_inv(1,2)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp, -T_inv(2,0)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+1, -T_inv(2,1)*v_weight));
									j_elem.push_back(Tri(cont+2, 3*cp+2, -T_inv(2,2)*v_weight));
								}
							}

							//Fill the residuals
							R(cont) = res_x[i](v,u);
							R(cont+1) = res_y[i](v,u);
							R(cont+2) = res_z[i](v,u);
							cont += 3;

							//									Normal alignment
							//----------------------------------------------------------------------------------------------
							//Control vertices
							const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
							Matrix3f J_nu, J_nX;
							J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
							J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
							J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
							J_nu *= inv_norm*square(inv_norm);
							J_nX.assign(0.f);
							const Matrix3f J_mult_norm = -Kn_sqrtf*T_inv*J_nu;
						
							for (unsigned int c = 0; c < max_num_w; c++)
							{
								const int cp = w_indices[i](c,weights_col);
								if (cp >= 0)
								{
									//Normals
									const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
									J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
									J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
									J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
									J_nX(1,0) = -J_nX(0,1);
									J_nX(2,0) = -J_nX(0,2);
									J_nX(2,1) = -J_nX(1,2);

									const Matrix3f J_norm_fit = J_mult_norm*J_nX;
									j_elem.push_back(Tri(cont, 3*cp, J_norm_fit(0,0)));
									j_elem.push_back(Tri(cont, 3*cp+1, J_norm_fit(0,1)));
									j_elem.push_back(Tri(cont, 3*cp+2, J_norm_fit(0,2)));
									j_elem.push_back(Tri(cont+1, 3*cp, J_norm_fit(1,0)));
									j_elem.push_back(Tri(cont+1, 3*cp + 1, J_norm_fit(1,1)));
									j_elem.push_back(Tri(cont+1, 3*cp + 2, J_norm_fit(1,2)));
									j_elem.push_back(Tri(cont+2, 3*cp, J_norm_fit(2,0)));
									j_elem.push_back(Tri(cont+2, 3*cp + 1, J_norm_fit(2,1)));
									j_elem.push_back(Tri(cont+2, 3*cp + 2, J_norm_fit(2,2)));
								}
							}

							//Fill the residuals
							R(cont) = Kn_sqrtf*res_nx[i](v,u);
							R(cont+1) = Kn_sqrtf*res_ny[i](v,u);
							R(cont+2) = Kn_sqrtf*res_nz[i](v,u);
							cont += 3;
						}
					}

			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point; t_point << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
				const float mx_t_DT = cam_trans_inv[i].row(0)*t_point;
				const float my_t_DT = cam_trans_inv[i].row(1)*t_point;
				const float mz_t_DT = cam_trans_inv[i].row(2)*t_point;

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 2, 3> J_pi;
				const float inv_z = 1.f / mx_t_DT;

				J_pi << fx*my_t_DT*square(inv_z), -fx*inv_z, 0.f,
						fy*mz_t_DT*square(inv_z), 0.f, -fy*inv_z;

				const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s))), DT_grad_v[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s)))};
				const Matrix<float, 1, 3> J_mult = -alpha_sqrt*J_DT*J_pi*T_inv;

				//Control vertices
				for (unsigned int c = 0; c < max_num_w; c++)
				{
					const int cp = w_indices_DT(c, s);
					if (cp >= 0)
					{
						const float ww = w_DT(c, s);
						j_elem.push_back(Tri(cont, 3*cp, J_mult(0)*ww));
						j_elem.push_back(Tri(cont, 3*cp+1, J_mult(1)*ww));
						j_elem.push_back(Tri(cont, 3*cp+2, J_mult(2)*ww));
					}
				}

				//Fill the residuals
				R(cont) = alpha_sqrt*DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s));
				cont++;
			}
		}

		printf("\n It finishes with Jacobians (without regularization). Time = %f", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)
			fillJacobianRegNormals(cont);

		if (with_reg_edges)
			fillJacobianRegEdges(cont);

		if (with_reg_membrane)
			fillJacobianRegMembrane(cont);

		if (with_reg_thin_plate)
			fillJacobianRegThinPlate(cont);


		printf("\n It finishes with Jacobians (with regularization). Time = %f", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		MatrixXf JtJ = MatrixXf(JtJ_sparse);
		VectorXf b = -J.transpose()*R;
		MatrixXf JtJ_lm;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n It enters the loop solver-energy-check. Time = %f", clock.Tac()); clock.Tic();

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Set the lambdas for each variable
			JtJ_lm = JtJ;
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);


			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);

			printf("\n It solves with LM. Time = %f", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			printf("\n It updates variables. Time = %f", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n It creates topology, evaluates the surface and computes the residuals. Time = %f", clock.Tac()); clock.Tic();
			rayCastingLMForegroundPerPixel();	printf("\n It solves closest correspondence foreground. Time = %f", clock.Tac()); clock.Tic();
			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDT2Overall();


			if (new_energy <= last_energy)
			{
				energy_increasing = false;
				adap_mult *= 0.5f;
				//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}
			else
			{
				adap_mult *= 4.f;
				//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			}

			cont_inner++;
			if (cont_inner > 5) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterU();
				evaluateSubDivSurface();			
				if (with_reg_normals) evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				rayCastingLMForegroundPerPixel();
				sampleSurfaceForDTBackground();
				new_energy = computeEnergyDT2Overall();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
					energy_increasing = true;
					break;
				}
				else
					energy_increasing = false;		
			}
			//energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		//showMesh();
		//showCamPoses();
		//showSubSurface();
		//showRenderedModel();
		takePictureLimitSurface(false);

		printf("\n Time to finish everything else = %f", clock.Tac()); clock.Tic();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy > last_energy - 0.0001f))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void Mod3DfromRGBD::saveResults()
{
	try
	{
		// Open file: find the first free file-name.
		char	aux[100];
		int     nFile = 0;
		bool    free_name = false;

		system::createDirectory("./background results");

		while (!free_name)
		{
			nFile++;
			sprintf(aux, "./background results/experiment_%03u.txt", nFile );
			free_name = !system::fileExists(aux);
		}

		// Open log file:
		f_res.open(aux);
		printf(" Saving results to file: %s \n", aux);

		for (unsigned int k=0; k<energy_data.size(); k++)
		{
			f_res << energy_data[k] << " ";
			f_res << energy_reg[k] << " ";
			f_res << energy_background[k] << endl;
		}

		f_res.close();
	}
	catch (...)
	{
		printf("Exception found trying to create the 'results file' !!\n");
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



