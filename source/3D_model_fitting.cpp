// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"


Mod3DfromRGBD::Mod3DfromRGBD(unsigned int num_im, unsigned int downsamp, unsigned int im_set)
{
	num_images = num_im;
	image_set = im_set;
	downsample = downsamp; //It can be set to 1, 2, 4, etc.

	switch (image_set) {
	case 1:
		fovh_d = utils::DEG2RAD(49.35f); fovv_d = utils::DEG2RAD(49.35f);
		rows = 100/downsample; cols = 100/downsample;
		break;
	case 5: 
		fovh_d = utils::DEG2RAD(49.35f); fovv_d = utils::DEG2RAD(49.35f);
		rows = 1080/downsample; cols = 1080/downsample; 
		break;
	default:
		fovh_d = utils::DEG2RAD(58.6f); fovv_d = utils::DEG2RAD(45.6f); //58.6, 45.6
		rows = 480/downsample; cols = 640/downsample;
		break;
	}

		
	fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	fy = float(rows) / (2.f*tan(0.5f*fovv_d));

	s_reg = 4;
	trunc_threshold_DT = 5;
	behind_cameras = false;
	ctf_level = 1;

	//Default settings
	fix_first_camera = false;	
	paper_visualization = false;
	paper_vis_no_mesh = true;
	solve_DT = false;
	optimize_cameras = true;
	vis_errors = false;
	small_initialization = false;
	save_energy = false;
	adaptive_tau = true;

	with_reg_normals = false; with_reg_normals_good = false; with_reg_normals_4dir = false;
	with_reg_edges = false; with_reg_edges_iniShape = false;
	with_reg_ctf = false; with_reg_atraction = false;
	with_reg_arap = false; with_reg_rot_arap = false;


	//Cameras
	cam_poses.resize(num_images);
	cam_incrs.resize(num_images);
	cam_trans.resize(num_images);
	cam_trans_inv.resize(num_images);
	cam_mfold.resize(num_images); cam_mfold_old.resize(num_images);
	cam_ini.resize(num_images);
	
	//Images
	depth_background.resize(rows,cols);
	intensity.resize(num_images);
	depth.resize(num_images); x_image.resize(num_images); y_image.resize(num_images);
	nx_image.resize(num_images); ny_image.resize(num_images); nz_image.resize(num_images); n_weights.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images); tau_pixel.resize(num_images);
	DT.resize(num_images); DT_grad_u.resize(num_images), DT_grad_v.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
		intensity[i].resize(rows, cols);
		depth[i].resize(rows, cols); x_image[i].resize(rows, cols); y_image[i].resize(rows, cols);
		nx_image[i].resize(rows, cols); ny_image[i].resize(rows, cols); nz_image[i].resize(rows, cols); n_weights[i].resize(rows, cols);
		is_object[i].resize(rows, cols); valid[i].resize(rows, cols); tau_pixel[i].resize(rows, cols);
		DT[i].resize(rows, cols); DT_grad_u[i].resize(rows, cols); DT_grad_v[i].resize(rows, cols);
	}

	//Correspondences
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

	//Jacobian wrt the control vertices
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

	//Generators to compute the jacobians wrt the twists (camera poses)
	for (unsigned int l = 0; l < 6; l++)
		mat_der_xi[l].assign(0.f);

	mat_der_xi[0](0, 3) = 1.f;
	mat_der_xi[1](1, 3) = 1.f;
	mat_der_xi[2](2, 3) = 1.f;
	mat_der_xi[3](1, 2) = -1.f; mat_der_xi[3](2, 1) = 1.f;
	mat_der_xi[4](0, 2) = 1.f; mat_der_xi[4](2, 0) = -1.f;
	mat_der_xi[5](0, 1) = -1.f; mat_der_xi[5](1, 0) = 1.f;	
}

void Mod3DfromRGBD::chooseParameterSet(unsigned int exp_ID)
{
	//To be done

	//for the angel:
	//-----------------------
	////Reg parameters
	//mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
	//mod3D.regularize_unitary_normals = true;
	//mod3D.with_reg_normals_good = false;		mod3D.Kr_total = 0.000002f; //0.003f
	//mod3D.with_reg_normals_4dir = true;
	//mod3D.with_reg_edges = false;				mod3D.Ke_total = 0.001f; //0.02f*mod3D.num_images

	//mod3D.with_reg_ctf = false;					mod3D.K_ctf_total = 50.f; //500.f;
	//mod3D.with_reg_atraction = true;			mod3D.K_atrac_total = 0.0001f; //0.5

	////Dataterm parameters
	//mod3D.Kp = 0.3f*float(square(downsample))/float(num_images); //1.f
	//mod3D.Kn = 0.03f*0.0015f*float(square(downsample))/float(num_images); //0.005f
	//mod3D.truncated_res = 0.05f; //0.1f
	//mod3D.truncated_resn = 1.f; 

	////Background
	//mod3D.tau = 20.f/float(downsample); //3.5f
	//mod3D.eps = 0.5f/float(downsample); //0.1f
	//mod3D.alpha = 0.2*0.0004f*float(square(downsample))/float(mod3D.tau*mod3D.num_images);  //0.002f*mod3D.num_images/mod3D.tau;

	//For the unicorn
	//-------------------------
	////Reg parameters
	//mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
	//mod3D.regularize_unitary_normals = true;
	//mod3D.with_reg_normals_good = false;		mod3D.Kr_total = 0.00001f; //0.003f
	//mod3D.with_reg_normals_4dir = true;
	//mod3D.with_reg_edges = false;				mod3D.Ke_total = 0.001f; //0.02f*mod3D.num_images
	//mod3D.with_reg_ctf = false;					mod3D.K_ctf_total = 50.f; //500.f;
	//mod3D.with_reg_atraction = true;			mod3D.K_atrac_total = 0.05f; //0.5
	//
	////Dataterm parameters
	//mod3D.Kp = 0.3f*float(square(downsample))/float(num_images); //1.f
	//mod3D.Kn = 0.03f*0.0015f*float(square(downsample))/float(num_images); //0.005f
	//mod3D.truncated_res = 0.05f; //0.1f
	//mod3D.truncated_resn = 1.f; 
	//
	//
	//mod3D.tau = 20.f/float(downsample); //3.5f
	//mod3D.eps = 0.5f/float(downsample); //0.1f
	//mod3D.alpha = 0.0004f*float(square(downsample))/float(mod3D.tau*mod3D.num_images);  //0.002f*mod3D.num_images/mod3D.tau;

	//For the arch
	//------------------------
	//const unsigned int refine_levels =  3;
	//const unsigned int num_images = 1; //5
	//const unsigned int downsample = 8; //4
	//const unsigned int im_set = 5;
	//mod3D.convergence_ratio = 0.999f;

	//Reg parameters
	//mod3D.with_reg_normals = false;				//mod3D.Kr_total = 1000.f; //0.08f*mod3D.num_images
	//mod3D.with_reg_normals_good = true;			mod3D.Kr_total = 0.001f; //0.003f
	//mod3D.with_reg_normals_4dir = false;
	//mod3D.with_reg_edges = false;				mod3D.Ke_total = 0.001f; //0.02f*mod3D.num_images
	//mod3D.with_reg_ctf = false;					mod3D.K_ctf_total = 50.f; //500.f;
	//mod3D.with_reg_atraction = true;			mod3D.K_atrac_total = 0.1f; //0.5
	//
	////Dataterm parameters
	//mod3D.Kp = 0.3f*float(square(downsample))/float(num_images); //1.f
	//mod3D.Kn = 0.00005f*float(square(downsample))/float(num_images); //0.0015f
	//mod3D.truncated_res = 0.1f; //0.1f
	//mod3D.truncated_resn = 1.f; //1.f
	//
	////Background term parameters
	//if (solve_DT)
	//{
	//	mod3D.nsamples_approx = 5000;		
	//	mod3D.alpha = 0.05f*float(square(downsample))/float(mod3D.nsamples_approx*mod3D.num_images);
	//}
	//else
	//{
	//	mod3D.adaptive_tau = true;
	//	mod3D.tau_max = 32.f/float(downsample);
	//	mod3D.eps_rel = 0.1; //0.1f
	//	mod3D.alpha = 0.0016f*float(square(downsample))/float(mod3D.tau_max*mod3D.num_images); //0.0016 adaptive tau, 0.0015 fixed tau
	//}
}

void Mod3DfromRGBD::initializeRegForFitting()
{
	nx_reg.resize(num_faces); ny_reg.resize(num_faces); nz_reg.resize(num_faces); inv_reg_norm.resize(num_faces);
	mx_reg.resize(num_faces); my_reg.resize(num_faces); mz_reg.resize(num_faces);
	u1_der_reg.resize(num_faces); u2_der_reg.resize(num_faces);
	w_u1_reg.resize(num_faces); w_u2_reg.resize(num_faces);
	w_contverts_reg.resize(num_faces); w_indices_reg.resize(num_faces);
	for (unsigned int f=0; f<num_faces; f++)
	{
		nx_reg[f].resize(s_reg, s_reg); ny_reg[f].resize(s_reg, s_reg); nz_reg[f].resize(s_reg, s_reg); inv_reg_norm[f].resize(s_reg, s_reg);
		mx_reg[f].resize(s_reg, s_reg); my_reg[f].resize(s_reg, s_reg); mz_reg[f].resize(s_reg, s_reg);
		u1_der_reg[f].resize(s_reg, s_reg); u2_der_reg[f].resize(s_reg, s_reg);
		w_u1_reg[f].resize(max_num_w, square(s_reg)); w_u2_reg[f].resize(max_num_w, square(s_reg));
		w_contverts_reg[f].resize(max_num_w, square(s_reg)); w_indices_reg[f].resize(max_num_w);

		for (unsigned int s2 = 0; s2 < s_reg; s2++)
			for (unsigned int s1 = 0; s1 < s_reg; s1++)
			{
				u1_der_reg[f](s1,s2) = new float[3];
				u2_der_reg[f](s1,s2) = new float[3];
			}
	}
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
						const float norm = sqrtf(square(nx_image[i](v,u)) + square(ny_image[i](v,u)) + square(nz_image[i](v,u)));
						if (norm > 0)
						{
							nx_image[i](v,u) /= norm;
							ny_image[i](v,u) /= norm;
							nz_image[i](v,u) /= norm;
						}
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
	{
		if (image_set == 1)
			n_weights[i].fill(1.f);

		else
		{	
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

						n_weights[i](v,u) = exp(-w_constant*sum_dist); 
					}
		}
	}
}

void Mod3DfromRGBD::computeInitialCameraPoses()
{	
	cam_poses.resize(max(int(num_images), 20));

	if (image_set == 1) //cube
	{	
		const float displ = 1.5f;
		cam_poses[0].setFromValues(-displ, 0.f, 0.f, utils::DEG2RAD(0.f), utils::DEG2RAD(0.f), utils::DEG2RAD(0.f));
		cam_poses[1].setFromValues(0.f, -displ, 0.f, utils::DEG2RAD(90.f), utils::DEG2RAD(0.f), utils::DEG2RAD(0.f));
		cam_poses[2].setFromValues(displ, 0.f, 0.f, utils::DEG2RAD(180.f), utils::DEG2RAD(0.f), utils::DEG2RAD(0.f));
		cam_poses[3].setFromValues(0.f, displ, 0.f, utils::DEG2RAD(270.f), utils::DEG2RAD(0.f), utils::DEG2RAD(0.f));
		cam_poses[4].setFromValues(0.f, 0.f, displ, utils::DEG2RAD(0.f), utils::DEG2RAD(90.f), utils::DEG2RAD(0.f));
		cam_poses[5].setFromValues(0.f, 0.f, -displ, utils::DEG2RAD(0.f), utils::DEG2RAD(-90.f), utils::DEG2RAD(0.f));
	}
	else if (image_set == 2) //Teddy
	{
		if (optimize_cameras)
		{		
			cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
			cam_poses[1].setFromValues(0.2f, -0.37f, 0.15f, utils::DEG2RAD(60.2f), utils::DEG2RAD(20.1f), utils::DEG2RAD(-20.1f));
			cam_poses[2].setFromValues(0.47f, 0.26f, 0.47f, utils::DEG2RAD(-62.9f), utils::DEG2RAD(60.2f), utils::DEG2RAD(71.6f));
			cam_poses[3].setFromValues(0.23f, 0.48f, 0.06f, utils::DEG2RAD(-51.6f), utils::DEG2RAD(5.7f), utils::DEG2RAD(17.2f));
			cam_poses[4].setFromValues(0.77f, 0.4f, 0.29f, utils::DEG2RAD(-114.6f), utils::DEG2RAD(31.5f), utils::DEG2RAD(25.8f));
		}
		else
		{
			cam_poses[0].setFromValues(0.01f, -0.135f, 0.126f, utils::DEG2RAD(12.1f), utils::DEG2RAD(11.9f), utils::DEG2RAD(-1.11f));
			cam_poses[1].setFromValues(0.25f, -0.425f, 0.196f, utils::DEG2RAD(66.26f), utils::DEG2RAD(23.73f), utils::DEG2RAD(-21.51f));
			cam_poses[2].setFromValues(0.50f, 0.311f, 0.444f, utils::DEG2RAD(-70.4f), utils::DEG2RAD(54.7f), utils::DEG2RAD(49.3f));
			cam_poses[3].setFromValues(0.08f, 0.326f, 0.052f, utils::DEG2RAD(-29.1f), utils::DEG2RAD(4.4f), utils::DEG2RAD(7.87f));
			cam_poses[4].setFromValues(0.65f, 0.47f, 0.232f, utils::DEG2RAD(-93.3f), utils::DEG2RAD(23.1f), utils::DEG2RAD(23.1f));			
		}
	}
	else if (image_set == 3)
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.48f, -0.58f, 0.22f, utils::DEG2RAD(86.f), utils::DEG2RAD(23.8f), utils::DEG2RAD(-37.3f));
		cam_poses[2].setFromValues(0.7f, 0.1f, 0.59f, utils::DEG2RAD(-117.3f), utils::DEG2RAD(63.1f), utils::DEG2RAD(77.33f));
		cam_poses[3].setFromValues(0.58f, 0.43f, 0.29f, utils::DEG2RAD(-86.f), utils::DEG2RAD(25.8f), utils::DEG2RAD(37.2f));
	}
	else if (image_set == 4)
	{
		cam_poses[0].setFromValues(-1.65f, -0.05f, -0.05f, 0.f, -0.25f, 0.f);
	}
	else if (image_set == 5)
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
	}
	else if (image_set == 8)
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.03f, -0.08f, 0.01f, utils::DEG2RAD(27.8f), utils::DEG2RAD(2.9f), utils::DEG2RAD(-11.5f));
		cam_poses[2].setFromValues(0.19f, -0.19f, 0.08f, utils::DEG2RAD(90.f), utils::DEG2RAD(22.9f), utils::DEG2RAD(-20.f));
		cam_poses[3].setFromValues(0.32f, -0.13f, 0.11f, utils::DEG2RAD(135.f), utils::DEG2RAD(34.4f), utils::DEG2RAD(-11.5f));	
		cam_poses[4].setFromValues(0.38f, 0.f, 0.13f, utils::DEG2RAD(-174.f), utils::DEG2RAD(37.2f), utils::DEG2RAD(0.f));
		cam_poses[5].setFromValues(0.31f, 0.12f, 0.11f, utils::DEG2RAD(-132.f), utils::DEG2RAD(31.5f), utils::DEG2RAD(17.f));
		cam_poses[6].setFromValues(0.19f, 0.19f, 0.08f, utils::DEG2RAD(-90.f), utils::DEG2RAD(23.f), utils::DEG2RAD(20.f));
		cam_poses[7].setFromValues(0.02f, 0.1f, 0.f, utils::DEG2RAD(-33.5f), utils::DEG2RAD(0.f), utils::DEG2RAD(5.7f));	
	}

	else if (image_set == 9)
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.05f, 0.08f, 0.06f, utils::DEG2RAD(-6.6f), utils::DEG2RAD(8.6f), utils::DEG2RAD(14.3f));
		cam_poses[2].setFromValues(0.15f, 0.21f, 0.18f, utils::DEG2RAD(-47.5f), utils::DEG2RAD(28.6f), utils::DEG2RAD(31.6f));
		cam_poses[3].setFromValues(0.42f, 0.14f, 0.34f, utils::DEG2RAD(166.5f), utils::DEG2RAD(91.7f), utils::DEG2RAD(-37.3f));	
		cam_poses[4].setFromValues(0.28f, -0.03f, 0.2f, utils::DEG2RAD(140.f), utils::DEG2RAD(74.4f), utils::DEG2RAD(-68.8f));
		cam_poses[5].setFromValues(0.32f, 0.02f, 0.24f, utils::DEG2RAD(-152.f), utils::DEG2RAD(103.f), utils::DEG2RAD(5.5f));
		cam_poses[6].setFromValues(0.12f, -0.16f, 0.08f, utils::DEG2RAD(61.8f), utils::DEG2RAD(28.7f), utils::DEG2RAD(-71.7f));
		cam_poses[7].setFromValues(0.09f, -0.04f, 0.13f, utils::DEG2RAD(32.4f), utils::DEG2RAD(34.4f), utils::DEG2RAD(-83.1f));	
	}

	else if (image_set == 10)
	{
		cam_poses[0].setFromValues(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
		cam_poses[1].setFromValues(0.08f, -0.27f, 0.04f, utils::DEG2RAD(31.5f), utils::DEG2RAD(5.7f), utils::DEG2RAD(-17.2f));
		cam_poses[2].setFromValues(0.14f, -0.41f, 0.1f, utils::DEG2RAD(57.3f), utils::DEG2RAD(17.2f), utils::DEG2RAD(-34.4f));
		cam_poses[3].setFromValues(0.36f, -0.36f, 0.18f, utils::DEG2RAD(109.f), utils::DEG2RAD(46.f), utils::DEG2RAD(-26.f));	
		cam_poses[4].setFromValues(0.48f, -0.31f, 0.22f, utils::DEG2RAD(154.f), utils::DEG2RAD(51.6f), utils::DEG2RAD(-8.6f));
		cam_poses[5].setFromValues(0.5f, -0.04f, 0.21f, utils::DEG2RAD(-169.f), utils::DEG2RAD(46.f), utils::DEG2RAD(8.6f));
		cam_poses[6].setFromValues(0.38f, 0.18f, 0.19f, utils::DEG2RAD(-120.3f), utils::DEG2RAD(34.4f), utils::DEG2RAD(14.3f));
		cam_poses[7].setFromValues(0.17f, 0.25f, 0.07f, utils::DEG2RAD(-71.6f), utils::DEG2RAD(14.3f), utils::DEG2RAD(25.8f));	
	}
	else if (image_set == 11)
	{
		cam_poses[0].setFromValues(0.01f, 0.f, 0.3f, utils::DEG2RAD(-11.5f), utils::DEG2RAD(45.8f), utils::DEG2RAD(0.f));
		cam_poses[1].setFromValues(0.12f, 0.09f, 0.31f, utils::DEG2RAD(-51.6f), utils::DEG2RAD(51.6f), utils::DEG2RAD(-3.f));
		cam_poses[2].setFromValues(0.28f, 0.14f, 0.24f, utils::DEG2RAD(-97.4f), utils::DEG2RAD(40.1f), utils::DEG2RAD(-11.5f));
		cam_poses[3].setFromValues(0.37f, 0.14f, 0.21f, utils::DEG2RAD(-94.5f), utils::DEG2RAD(51.6f), utils::DEG2RAD(5.7f));
		cam_poses[4].setFromValues(0.49f, 0.09f, 0.22f, utils::DEG2RAD(-154.7f), utils::DEG2RAD(40.1f), utils::DEG2RAD(0.f));
		cam_poses[5].setFromValues(0.45f, -0.12f, 0.21f, utils::DEG2RAD(134.7f), utils::DEG2RAD(54.4f), utils::DEG2RAD(17.2f));
		cam_poses[6].setFromValues(0.28f, -0.11f, 0.23f, utils::DEG2RAD(80.3f), utils::DEG2RAD(45.8f), utils::DEG2RAD(-3.f));
		cam_poses[7].setFromValues(0.19f, -0.11f, 0.21f, utils::DEG2RAD(50.f), utils::DEG2RAD(54.4f), utils::DEG2RAD(-20.f));
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

	//char c = 'y';
	//while (c == 'y')
	//{
	//	printf("\n Do you want to change the pose of any banana [y/n]? ");
	//	cin >> c;

	//	if (c != 'y')
	//		break;

	//	int num_banana;
	//	printf("\n Which banana do you want to move? (from 0 to num_bananas-1): ");
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

void Mod3DfromRGBD::perturbInitialCameraPoses(float per_trans, float per_rot)
{	
	cam_pert.resize(num_images);
	cam_pert[0].fill(0.f);

	std::default_random_engine generator(std::clock());
	std::normal_distribution<float> trans_distribution(0.f, per_trans);
	std::normal_distribution<float> rot_distribution(0.f, per_rot);

	
	//Get and store the initial transformation matrices
	for (unsigned int i = 1; i < num_images; i++)
	{
		//Perturb the poses randomly
		cam_pert[i](0) = trans_distribution(generator);
		cam_pert[i](1) = trans_distribution(generator);
		cam_pert[i](2) = trans_distribution(generator);

		cam_pert[i](3) = rot_distribution(generator);
		cam_pert[i](4) = rot_distribution(generator);
		cam_pert[i](5) = rot_distribution(generator);

		//Option 1 - Perturb the initial cam poses
		//cam_poses[i].x_incr(pertx); cam_poses[i].y_incr(perty); cam_poses[i].z_incr(pertz);
		//cam_poses[i].setYawPitchRoll(cam_poses[i].yaw() + pertyaw, cam_poses[i].pitch() + pertpitch, cam_poses[i].roll() + pertroll);

		//CMatrixDouble44 aux;
		//cam_poses[i].getHomogeneousMatrix(aux);
		//cam_trans[i] = aux.cast<float>();

		//const Matrix3f rot_mat = cam_trans[i].block<3, 3>(0, 0).transpose();
		//const Vector3f tra_vec = cam_trans[i].block<3, 1>(0, 3);

		//cam_trans_inv[i].topLeftCorner<3, 3>() = rot_mat;
		//cam_trans_inv[i].block<3, 1>(0, 3) = -rot_mat*tra_vec;
		//cam_trans_inv[i].row(3) << 0.f, 0.f, 0.f, 1.f;
		//cam_ini[i] = cam_trans_inv[i].topLeftCorner<4, 4>();

		//Option 2 - Perturb cam_mfold over with we optimize
		cam_mfold[i] = cam_pert[i];	
	}

	computeCameraTransfandPosesFromTwist();
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

void Mod3DfromRGBD::computeInitialUDataterm()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	Array<float, 3, Dynamic> u_triplet;
	const unsigned int num_samp = min(10, max(3, int(round(float(5000)/square(num_faces)))));
	const unsigned int ini_samp_size = num_faces*square(num_samp);
	u_triplet.resize(3, ini_samp_size);

	const float fact = 1.f / float(num_samp-1);
	unsigned int cont = 0;
	for (unsigned int f = 0; f < num_faces; ++f)
		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				u_triplet(0,cont) = float(u)*fact;
				u_triplet(1,cont) = float(v)*fact;
				u_triplet(2,cont) = f;
				cont++;
			}

	//Evaluate the surface
	Matrix<float, 3, Dynamic> xyz_ini; xyz_ini.resize(3, ini_samp_size);
	Matrix<float, 3, Dynamic> n_ini; n_ini.resize(3, ini_samp_size);

	//// Create a Far::PtexIndices to help find indices of ptex faces.
    //Far::PtexIndices ptexIndices(*refiner);
    //int nfaces = ptexIndices.GetNumFaces();
	//printf("\n num_faces = %d, nfaces_ptex = %d", num_faces, nfaces);

	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	//Evaluate the surface with parametric coordinates
	for (unsigned int s = 0; s < ini_samp_size; s++)
	{
		const unsigned int f = u_triplet(2,s);
		
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u_triplet(0,s), u_triplet(1,s)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u_triplet(0,s), u_triplet(1,s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//3D coordinates
		xyz_ini.col(s) << eval.point[0], eval.point[1], eval.point[2];

		//Normals
		const float nx_i = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
		const float ny_i = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
		const float nz_i = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
		const float inv_norm = 1.f/sqrtf(square(nx_i) + square(ny_i) + square(nz_i));
		n_ini.col(s) << inv_norm*nx_i, inv_norm*ny_i, inv_norm*nz_i;
	}

	//Find the closest point to each of the observed with the cameras - Brute force
	//-----------------------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		const Matrix<float, 3, 4> &mytrans = cam_trans[i].topRows(3);

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const Vector4f xyz_data = {depth[i](v,u), x_image[i](v,u), y_image[i](v,u), 1.f};
					const Vector3f xyz_trans = mytrans*xyz_data;
					const Vector4f n_data = {nx_image[i](v,u), ny_image[i](v,u), nz_image[i](v,u), 0.f};
					const Vector3f n_trans = mytrans*n_data;

					float min_dist = 100.f, dist;
					unsigned int s_min = 0;

					for (unsigned int s = 0; s < ini_samp_size; s++)
					{
						//Slower than element-wise!!!!
						//const Vector3f xyz_dif = xyz_trans - xyz_ini.col(s);
						//const Vector3f n_dif = n_trans - n_ini.col(s);
						//dist = Kp*xyz_dif.squaredNorm() + Kn*n_dif.squaredNorm();
						//dist = Kp*(square(xyz_dif(0)) + square(xyz_dif(1)) + square(xyz_dif(2))) + 
						//	 + Kn*(square(n_dif(0)) + square(n_dif(1)) + square(n_dif(2)));
						
						dist = Kp*(square(xyz_trans(0) - xyz_ini(0,s)) + square(xyz_trans(1) - xyz_ini(1,s)) + square(xyz_trans(2) - xyz_ini(2,s)))
							 + Kn*n_weights[i](v,u)*(square(n_trans(0) - n_ini(0,s)) + square(n_trans(1) - n_ini(1,s)) + square(n_trans(2) - n_ini(2,s)));

						if (dist  < min_dist)
						{
							min_dist = dist;
							s_min = s;
						}
					}

					u1[i](v,u) = u_triplet(0,s_min);
					u2[i](v,u) = u_triplet(1,s_min);
					uface[i](v,u) = u_triplet(2,s_min);
				}
	}
}

void Mod3DfromRGBD::searchBetterUDataterm()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	Array<float, 3, Dynamic> u_triplet;
	const unsigned int num_samp = min(10, max(3, int(round(float(5000)/square(num_faces)))));
	const unsigned int ini_samp_size = num_faces*square(num_samp);
	u_triplet.resize(3, ini_samp_size);

	const float fact = 1.f / float(num_samp-1);
	unsigned int cont = 0;
	for (unsigned int f = 0; f < num_faces; ++f)
		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				u_triplet(0,cont) = float(u)*fact;
				u_triplet(1,cont) = float(v)*fact;
				u_triplet(2,cont) = f;
				cont++;
			}

	//Evaluate the surface
	Matrix<float, 3, Dynamic> xyz_ini; xyz_ini.resize(3, ini_samp_size);
	Matrix<float, 3, Dynamic> n_ini; n_ini.resize(3, ini_samp_size);

	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	//Evaluate the surface with parametric coordinates
	for (unsigned int s = 0; s < ini_samp_size; s++)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(u_triplet(2,s), u_triplet(0,s), u_triplet(1,s)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u_triplet(0,s), u_triplet(1,s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//3D coordinates
		xyz_ini.col(s) << eval.point[0], eval.point[1], eval.point[2];

		//Normals
		const float nx_i = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
		const float ny_i = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
		const float nz_i = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
		const float inv_norm = 1.f/sqrtf(square(nx_i) + square(ny_i) + square(nz_i));
		n_ini.col(s) << inv_norm*nx_i, inv_norm*ny_i, inv_norm*nz_i;
	}

	//Search for a better correspondence - Brute force
	//-----------------------------------------------------------------------------
	for (unsigned int i = 0; i < num_images; ++i)
	{
		const Matrix<float, 3, 4> &mytrans = cam_trans[i].topRows(3);

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const Vector4f xyz_data = {depth[i](v,u), x_image[i](v,u), y_image[i](v,u), 1.f};
					const Vector3f xyz_trans = mytrans*xyz_data;
					const Vector4f n_data = {nx_image[i](v,u), ny_image[i](v,u), nz_image[i](v,u), 0.f};
					const Vector3f n_trans = mytrans*n_data;

					float min_dist = Kp*(square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u)))
								   + Kn*n_weights[i](v,u)*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));

					float dist;
					int s_min = -1;

					for (unsigned int s = 0; s < ini_samp_size; s++)
					{
												
						//dist = Kp*(square(xyz_trans(0) - xyz_ini(0,s)) + square(xyz_trans(1) - xyz_ini(1,s)) + square(xyz_trans(2) - xyz_ini(2,s)))
						//	 + Kn*n_weights[i](v,u)*(square(n_trans(0) - n_ini(0,s)) + square(n_trans(1) - n_ini(1,s)) + square(n_trans(2) - n_ini(2,s)));
						
						//I compute it in 3 steps to save time
						dist = Kp*square(xyz_trans(0) - xyz_ini(0,s)); 
						if (dist > min_dist)	continue;

						dist += Kp*square(xyz_trans(1) - xyz_ini(1,s)); 
						if (dist > min_dist)	continue;

						dist += Kp*square(xyz_trans(2) - xyz_ini(2,s));
						if (dist > min_dist)	continue;

						dist += Kn*n_weights[i](v,u)*(square(n_trans(0) - n_ini(0,s)) + square(n_trans(1) - n_ini(1,s)) + square(n_trans(2) - n_ini(2,s)));

						if (dist < min_dist)
						{
							//printf("\n Better dist. dist = %f, min_dist = %f", dist, min_dist);				
							min_dist = dist;
							s_min = s;
						}
					}

					if (s_min >= 0)
					{
						u1[i](v,u) = u_triplet(0,s_min);
						u2[i](v,u) = u_triplet(1,s_min);
						uface[i](v,u) = u_triplet(2,s_min);
					}
				}
	}
}

void Mod3DfromRGBD::computeInitialUBackground()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	Array<float, 3, Dynamic> u_triplet;
	const unsigned int num_samp = min(10, max(3, int(round(float(5000)/square(num_faces)))));
	const unsigned int ini_samp_size = num_faces*square(num_samp);
	u_triplet.resize(3, ini_samp_size);

	const float fact = 1.f / float(num_samp-1);
	unsigned int cont = 0;
	for (unsigned int f = 0; f < num_faces; ++f)
		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				u_triplet(0,cont) = float(u)*fact;
				u_triplet(1,cont) = float(v)*fact;
				u_triplet(2,cont) = f;
				cont++;
			}

	//Evaluate the surface
	Matrix<float, 4, Dynamic> xyz_ini; xyz_ini.resize(4, ini_samp_size);

	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	//Evaluate the surface with parametric coordinates
	for (unsigned int s = 0; s < ini_samp_size; s++)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(u_triplet(2,s), u_triplet(0,s), u_triplet(1,s)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u_triplet(0,s), u_triplet(1,s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//3D coordinates
		xyz_ini.col(s) << eval.point[0], eval.point[1], eval.point[2], 1.f;
	}

	//Find the one that projects closer to the corresponding pixel (using Pin-Hole model) - Brute force
	//-------------------------------------------------------------------------------------------------
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Find direction of the projection
		const Matrix<float, 3, 4> &mytrans = cam_trans_inv[i].topRows(3);

		//Compute the transformed points of the surface samples
		const Matrix<float, 3, Dynamic> xyz_ini_t = mytrans*xyz_ini;

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u) && (valid[i](v,u)))
				{
					float min_dist = 100000.f, u1_min = 0.f, u2_min = 0.f;
					unsigned int s_min = 0;

					for (unsigned int s = 0; s < ini_samp_size; s++)
					{
						//Project point onto the image plane (I assume that no point has depth <= 0)
						const float inv_d = 1.f/xyz_ini_t(0,s);
						const float u_pixel = fx*xyz_ini_t(1,s)*inv_d + disp_u;
						const float v_pixel = fy*xyz_ini_t(2,s)*inv_d + disp_v;
						const float pix_dist = square(u_pixel - float(u)) + square(v_pixel - float(v));

						if (pix_dist < min_dist)
						{
							min_dist = pix_dist;
							s_min = s;
						}
					}

					u1[i](v, u) = u_triplet(0,s_min);
					u2[i](v, u) = u_triplet(1,s_min);
					uface[i](v, u) = u_triplet(2,s_min);
				}
	}
}

void Mod3DfromRGBD::searchBetterUBackground()
{
	//First sample the subdivision surface uniformly
	//------------------------------------------------------------------------------------
	//Create the parametric values
	Array<float, 3, Dynamic> u_triplet;
	const unsigned int num_samp = min(10, max(3, int(round(float(5000)/square(num_faces)))));
	const unsigned int ini_samp_size = num_faces*square(num_samp);
	u_triplet.resize(3, ini_samp_size);

	const float fact = 1.f / float(num_samp-1);
	unsigned int cont = 0;
	for (unsigned int f = 0; f < num_faces; ++f)
		for (unsigned int u = 0; u < num_samp; u++)
			for (unsigned int v = 0; v < num_samp; v++)
			{
				u_triplet(0,cont) = float(u)*fact;
				u_triplet(1,cont) = float(v)*fact;
				u_triplet(2,cont) = f;
				cont++;
			}

	//Evaluate the surface
	Matrix<float, 4, Dynamic> xyz_ini; xyz_ini.resize(4, ini_samp_size);

	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	//Evaluate the surface with parametric coordinates
	for (unsigned int s = 0; s < ini_samp_size; s++)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(u_triplet(2,s), u_triplet(0,s), u_triplet(1,s)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u_triplet(0,s), u_triplet(1,s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//3D coordinates
		xyz_ini.col(s) << eval.point[0], eval.point[1], eval.point[2], 1.f;
	}

	//Search for a correspondence that projects closer to the corresponding pixel - Brute force
	//-----------------------------------------------------------------------------------------
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Find direction of the projection
		const Matrix<float, 3, 4> &mytrans = cam_trans_inv[i].topRows(3);

		//Compute the transformed points of the surface samples
		const Matrix<float, 3, Dynamic> xyz_ini_t = mytrans*xyz_ini;

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u) && (valid[i](v,u)))
				{
					float min_dist = square(res_d1[i](v, u)) + square(res_d2[i](v, u));
					int s_min = -1;

					for (unsigned int s = 0; s < ini_samp_size; s++)
					{
						//Project point onto the image plane (I assume that no point has depth <= 0)
						const float inv_d = 1.f/xyz_ini_t(0,s);
						const float u_pixel = fx*xyz_ini_t(1,s)*inv_d + disp_u;
						const float v_pixel = fy*xyz_ini_t(2,s)*inv_d + disp_v;
						const float pix_dist = square(u_pixel - float(u)) + square(v_pixel - float(v));

						if (pix_dist < min_dist)
						{
							min_dist = pix_dist;
							s_min = s;
						}
					}

					if (s_min >= 0)
					{
						u1[i](v, u) = u_triplet(0,s_min);
						u2[i](v, u) = u_triplet(1,s_min);
						uface[i](v, u) = u_triplet(2,s_min);
					}
				}
	}
}

void Mod3DfromRGBD::computeTransCoordAndResiduals()
{
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Refs
		const Matrix4f &mytrans_inv = cam_trans_inv[i];
		const ArrayXXf &mx_ref = mx[i], &my_ref = my[i], &mz_ref = mz[i];
		const ArrayXXf &nx_ref = nx[i], &ny_ref = ny[i], &nz_ref = nz[i];

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (valid[i](v,u))
				{			
					mx_t[i](v,u) = mytrans_inv(0,0)*mx_ref(v,u) + mytrans_inv(0,1)*my_ref(v,u) + mytrans_inv(0,2)*mz_ref(v,u) + mytrans_inv(0,3);
					my_t[i](v,u) = mytrans_inv(1,0)*mx_ref(v,u) + mytrans_inv(1,1)*my_ref(v,u) + mytrans_inv(1,2)*mz_ref(v,u) + mytrans_inv(1,3);
					mz_t[i](v,u) = mytrans_inv(2,0)*mx_ref(v,u) + mytrans_inv(2,1)*my_ref(v,u) + mytrans_inv(2,2)*mz_ref(v,u) + mytrans_inv(2,3);

					if (mx_t[i](v, u) <= 0.f)
					{
						printf("\n Depth coordinate of the internal correspondence is equal or inferior to zero after the transformation!!!");
						behind_cameras = true;
						return;
					}

					if (is_object[i](v, u))
					{
						res_x[i](v,u) = depth[i](v,u) - mx_t[i](v,u);
						res_y[i](v,u) = x_image[i](v,u) - my_t[i](v,u);
						res_z[i](v,u) = y_image[i](v,u) - mz_t[i](v,u);

						nx_t[i](v,u) = mytrans_inv(0,0)*nx_ref(v,u) + mytrans_inv(0,1)*ny_ref(v,u) + mytrans_inv(0,2)*nz_ref(v,u);
						ny_t[i](v,u) = mytrans_inv(1,0)*nx_ref(v,u) + mytrans_inv(1,1)*ny_ref(v,u) + mytrans_inv(1,2)*nz_ref(v,u);
						nz_t[i](v,u) = mytrans_inv(2,0)*nx_ref(v,u) + mytrans_inv(2,1)*ny_ref(v,u) + mytrans_inv(2,2)*nz_ref(v,u);
						const float inv_norm = 1.f/sqrtf(square(nx_t[i](v,u)) + square(ny_t[i](v,u)) + square(nz_t[i](v,u)));
						res_nx[i](v,u) = nx_image[i](v,u) - inv_norm*nx_t[i](v,u);
						res_ny[i](v,u) = ny_image[i](v,u) - inv_norm*ny_t[i](v,u);
						res_nz[i](v,u) = nz_image[i](v,u) - inv_norm*nz_t[i](v,u);
					}
					else
					{
						const float u_proj = fx*(my_t[i](v,u) / mx_t[i](v,u)) + disp_u;
						const float v_proj = fy*(mz_t[i](v,u) / mx_t[i](v,u)) + disp_v;
						res_d1[i](v,u) = float(u) - u_proj;
						res_d2[i](v,u) = float(v) - v_proj;					
					}
				}
	}
}

bool Mod3DfromRGBD::updateInternalPointCrossingEdges(unsigned int i, unsigned int v, unsigned int u)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	//Check if crossing borders
	Vector2f u_incr; u_incr << u1_incr[i](v,u), u2_incr[i](v,u);
	float u1_old = this->u1_old[i](v,u);
	float u2_old= this->u2_old[i](v,u);
	unsigned int face = uface_old[i](v,u);

	float u1_new = u1_old + u_incr(0);
	float u2_new = u2_old + u_incr(1);
	bool crossing = true;
	bool alarm = false;
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
		float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
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
		const MatrixXf prod = J_Sa*du_remaining;
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
		if (cont > 3)
		{
			//printf("\n Problem!!! Many jumps between the mesh faces for the update of one correspondence. I remove the remaining u_increment!");
			//printf("\n Info about the jumps: max_incr = %f, u_new = (%f,%f,%d), i = %d, (v,u) = (%d,%d)", max(abs(u1_incr[i](v,u)), abs(u2_incr[i](v,u))), u1_new, u2_new, face, i, v, u);

			//if (cont > 15)
			//{
				u1_new = u1_old;
				u2_new = u2_old;
				alarm = true;
				break;
			//}
		}
	}

	u1[i](v,u) = u1_new;
	u2[i](v,u) = u2_new;
	uface[i](v,u) = face;
	return alarm;
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
	//desc.vertIndicesPerFace = face_verts.data();

	int *f_vertices = new int[4*num_faces];
	int cont = 0;
	for (unsigned int i = 0; i < num_faces; i++)
		for (unsigned int k=0; k<vertsperface[i]; k++)
			f_vertices[cont++] = face_verts(k,i);

	desc.vertIndicesPerFace = f_vertices;

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

	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	unsigned int cont = 0;

	//Evaluate the surface with parametric coordinates
	for (unsigned int i = 0; i<num_images; ++i)
	{
		//Refs
		const ArrayXXi &u_face_ref = uface[i];
		const ArrayXXf &u1_ref = u1[i], &u2_ref = u2[i];
		ArrayXXf &mx_ref = mx[i], &my_ref = my[i], &mz_ref = mz[i];
		Array<float*, Dynamic, Dynamic> &u1_der_ref = u1_der[i], &u2_der_ref = u2_der[i];
		ArrayXXf &nx_ref = nx[i], &ny_ref = ny[i], &nz_ref = nz[i];
		ArrayXXi &w_indices_ref = w_indices[i];
		ArrayXXf &w_contverts_ref = w_contverts[i];
		ArrayXXf &w_u1_ref = w_u1[i], &w_u2_ref = w_u2[i];
	
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (valid[i](v,u))
				{
					// Locate the patch corresponding to the face ptex idx and (s,t)
					Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(u_face_ref(v,u), u1_ref(v,u), u2_ref(v,u)); assert(handle);

					// Evaluate the patch weights, identify the CVs and compute the limit frame:
					patchTable->EvaluateBasis(*handle, u1_ref(v,u), u2_ref(v,u), pWeights, dsWeights, dtWeights);

					Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

					LimitFrame eval; eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

					//Save the 3D coordinates
					mx_ref(v,u) = eval.point[0];
					my_ref(v,u) = eval.point[1];
					mz_ref(v,u) = eval.point[2];

					//Save the derivatives
					u1_der_ref(v,u)[0] = eval.deriv1[0];
					u1_der_ref(v,u)[1] = eval.deriv1[1];
					u1_der_ref(v,u)[2] = eval.deriv1[2];
					u2_der_ref(v,u)[0] = eval.deriv2[0];
					u2_der_ref(v,u)[1] = eval.deriv2[1];
					u2_der_ref(v,u)[2] = eval.deriv2[2];

					//Compute the normals
					nx_ref(v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
					ny_ref(v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
					nz_ref(v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

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
					w_indices[i].col(col_weights).fill(-1);

					for (unsigned int cv=0; cv<num_verts; cv++)
						if (vect_wc(cv) != 0.f)
						{
							w_indices_ref(cont, col_weights) = cv;
							w_contverts_ref(cont, col_weights) = vect_wc(cv);
							w_u1_ref(cont, col_weights) = vect_wu1(cv);
							w_u2_ref(cont, col_weights) = vect_wu2(cv);
							cont++;
						}
					//cout << endl << "w_indices: " << w_indices[i].col(col_weights).transpose();
				}
	}
}

void Mod3DfromRGBD::evaluateSubDivSurfaceRegularization()
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

	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	const float s2u = 1.f/float(s_reg-1);

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f<num_faces; f++)
	{
		//Refs
		ArrayXXf &mx_ref = mx_reg[f], &my_ref = my_reg[f], &mz_ref = mz_reg[f];
		Array<float*, Dynamic, Dynamic> &u1_der_ref = u1_der_reg[f], &u2_der_ref = u2_der_reg[f];
		ArrayXXf &nx_ref = nx_reg[f], &ny_ref = ny_reg[f], &nz_ref = nz_reg[f];
		ArrayXi &w_indices_ref = w_indices_reg[f];
		ArrayXXf &w_contverts_ref = w_contverts_reg[f];
		ArrayXXf &w_u1_ref = w_u1_reg[f], &w_u2_ref = w_u2_reg[f];
		
		
		//Find the indices associated to this face
		//----------------------------------------
		const float u1 = 0.5f, u2 = 0.5f;			
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1, u2); assert(handle);
		patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);
		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);
		
		vector<int> indices;
		for (int cv = 0; cv < cvs.size(); ++cv)
		{						
			if (cvs[cv] < num_verts)
			{			
				if ( std::find(indices.begin(), indices.end(), cvs[cv]) == indices.end())
				   indices.push_back(cvs[cv]);
			}
			else
			{
				const unsigned int ind_offset = cvs[cv] - num_verts;
				unsigned int size_st = st[ind_offset].GetSize();
				Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
				for (unsigned int s = 0; s < size_st; s++)
					if ( std::find(indices.begin(), indices.end(), st_ind[s]) == indices.end())
						indices.push_back(st_ind[s]);
			}
		}

		//Copy result to vector
		w_indices_ref.fill(-1);
		for (unsigned int k=0; k<indices.size(); k++)
			w_indices_ref(k) = indices.at(k);

		//Clean the weights before updating them
		w_contverts_ref.fill(0.f); w_u1_ref.fill(0.f); w_u2_ref.fill(0.f);

		//Compute the coordinates, derivatives and weights
		//---------------------------------------------
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

				//Save the 3D coordinates
				mx_ref(s1,s2) = eval.point[0];
				my_ref(s1,s2) = eval.point[1];
				mz_ref(s1,s2) = eval.point[2];

				//Save the derivatives
				u1_der_ref(s1,s2)[0] = eval.deriv1[0];
				u1_der_ref(s1,s2)[1] = eval.deriv1[1];
				u1_der_ref(s1,s2)[2] = eval.deriv1[2];
				u2_der_ref(s1,s2)[0] = eval.deriv2[0];
				u2_der_ref(s1,s2)[1] = eval.deriv2[1];
				u2_der_ref(s1,s2)[2] = eval.deriv2[2];

				//Compute the normals
				const float nx = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
				const float ny = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
				const float nz = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
				inv_reg_norm[f](s1,s2) = 1.f/sqrtf(square(nx) + square(ny) + square(nz));

				nx_ref(s1,s2) = nx*inv_reg_norm[f](s1,s2);
				ny_ref(s1,s2) = ny*inv_reg_norm[f](s1,s2);
				nz_ref(s1,s2) = nz*inv_reg_norm[f](s1,s2);

				//Compute the weights for the coordinates and the derivatives wrt the control vertices
				const unsigned int col_weights = s1 + s2*s_reg;

				for (int cv = 0; cv < cvs.size(); ++cv)
				{						
					if (cvs[cv] < num_verts)
					{			
						//Find the position of cvs[cv] in the "w_index_reg" vector
						unsigned int index_pos = std::find(indices.begin(), indices.end(), cvs[cv]) - indices.begin();
						
						w_contverts_ref(index_pos, col_weights) += pWeights[cv];
						w_u1_ref(index_pos, col_weights) += dsWeights[cv];
						w_u2_ref(index_pos, col_weights) += dtWeights[cv];
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
							unsigned int index_pos = std::find(indices.begin(), indices.end(), st_ind[s]) - indices.begin();

							w_contverts_ref(index_pos, col_weights) += pWeights[cv]*st_weights[s];
							w_u1_ref(index_pos, col_weights) += dsWeights[cv]*st_weights[s];
							w_u2_ref(index_pos, col_weights) += dtWeights[cv]*st_weights[s];
						}
					}
				}
			}
	}
}

//void Mod3DfromRGBD::computeNormalDerivatives_FinDif(unsigned int i,unsigned int v,unsigned int u)
//{
//	// Create a Far::PatchMap to help locating patches in the table
//	Far::PatchMap patchmap(*patchTable);
//	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
//
//	const float uincr = 0.001f; const float uincr_inv = 1.f/uincr;
//
//	//Compute normal for small increment of u1
//	//=================================================================================================================
//	Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(uface[i](v,u), u1[i](v,u)+uincr, u2[i](v,u)); assert(handle1);
//	patchTable->EvaluateBasis(*handle1, u1[i](v,u)+uincr, u2[i](v,u), pWeights, dsWeights, dtWeights);
//	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);
//
//	LimitFrame eval; eval.Clear();
//	for (int cv = 0; cv < cvs.size(); ++cv)
//		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);
//
//	//Compute the normals
//	const float nx_u1 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
//	const float ny_u1 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
//	const float nz_u1 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
//
//	n_der_u1[i](v,u)[0] = uincr_inv*(nx_u1 - nx[i](v,u));
//	n_der_u1[i](v,u)[1] = uincr_inv*(ny_u1 - ny[i](v,u));
//	n_der_u1[i](v,u)[2] = uincr_inv*(nz_u1 - nz[i](v,u));
//
//
//	//Compute normal for small increment of u2
//	//=================================================================================================================
//	Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(uface[i](v,u), u1[i](v,u), u2[i](v,u)+uincr); assert(handle2);
//	patchTable->EvaluateBasis(*handle2, u1[i](v,u), u2[i](v,u)+uincr, pWeights, dsWeights, dtWeights);
//	cvs = patchTable->GetPatchVertices(*handle2);
//
//	eval.Clear();
//	for (int cv = 0; cv < cvs.size(); ++cv)
//		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);
//
//	//Compute the normals
//	const float nx_u2 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
//	const float ny_u2 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
//	const float nz_u2 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
//
//	n_der_u2[i](v,u)[0] = uincr_inv*(nx_u2 - nx[i](v,u));
//	n_der_u2[i](v,u)[1] = uincr_inv*(ny_u2 - ny[i](v,u));
//	n_der_u2[i](v,u)[2] = uincr_inv*(nz_u2 - nz[i](v,u));
//}

void Mod3DfromRGBD::computeNormalDerivatives_Analyt(unsigned int i,unsigned int v,unsigned int u)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], ds_w[max_num_w], dt_w[max_num_w], dss_w[max_num_w], dst_w[max_num_w], dtt_w[max_num_w];

	//Evaluate the surface
	Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(uface[i](v,u), u1[i](v,u), u2[i](v,u)); assert(handle1);
	patchTable->EvaluateBasis(*handle1, u1[i](v,u), u2[i](v,u), pWeights, ds_w, dt_w, dss_w, dst_w, dtt_w);
	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);

	LimitFrame2der eval; eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],ds_w[cv],dt_w[cv], dss_w[cv], dst_w[cv], dtt_w[cv]);

	//dn/du1
	n_der_u1[i](v,u)[0] = eval.der_ss[1]*eval.deriv2[2] + eval.deriv1[1]*eval.der_st[2]
					- eval.der_ss[2]*eval.deriv2[1] - eval.deriv1[2]*eval.der_st[1];

	n_der_u1[i](v,u)[1] = eval.der_ss[2]*eval.deriv2[0] + eval.deriv1[2]*eval.der_st[0]
					- eval.der_ss[0]*eval.deriv2[2] - eval.deriv1[0]*eval.der_st[2];

	n_der_u1[i](v,u)[2] = eval.der_ss[0]*eval.deriv2[1] + eval.deriv1[0]*eval.der_st[1]
					- eval.der_ss[1]*eval.deriv2[0] - eval.deriv1[1]*eval.der_st[0];

	//dn/du2
	n_der_u2[i](v,u)[0] = eval.der_st[1]*eval.deriv2[2] + eval.deriv1[1]*eval.der_tt[2]
					- eval.der_st[2]*eval.deriv2[1] - eval.deriv1[2]*eval.der_tt[1];

	n_der_u2[i](v,u)[1] = eval.der_st[2]*eval.deriv2[0] + eval.deriv1[2]*eval.der_tt[0]
					- eval.der_st[0]*eval.deriv2[2] - eval.deriv1[0]*eval.der_tt[2];

	n_der_u2[i](v,u)[2] = eval.der_st[0]*eval.deriv2[1] + eval.deriv1[0]*eval.der_tt[1]
					- eval.der_st[1]*eval.deriv2[0] - eval.deriv1[1]*eval.der_tt[0];
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
	//desc.vertIndicesPerFace = face_verts.data();

	int *f_vertices = new int[4*num_faces];
	int cont = 0;
	for (unsigned int i = 0; i < num_faces; i++)
		for (unsigned int k=0; k<vertsperface[i]; k++)
			f_vertices[cont++] = face_verts(k,i);
	desc.vertIndicesPerFace = f_vertices;

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
	//------------------------------------------------------------------------------------------------
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
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
	for (unsigned int v = 0; v < verts.size(); v++)
	{
		vert_coords(0, v) = verts[v].point[0];
		vert_coords(1, v) = verts[v].point[1];
		vert_coords(2, v) = verts[v].point[2];
	}
	vert_coords_reg = vert_coords;

	//Resize regularization variables
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
	{
		nx_reg.resize(num_faces); ny_reg.resize(num_faces); nz_reg.resize(num_faces); inv_reg_norm.resize(num_faces);
		mx_reg.resize(num_faces); my_reg.resize(num_faces); mz_reg.resize(num_faces);
		u1_der_reg.resize(num_faces); u2_der_reg.resize(num_faces);
		w_u1_reg.resize(num_faces); w_u2_reg.resize(num_faces);
		w_contverts_reg.resize(num_faces); w_indices_reg.resize(num_faces);
		for (unsigned int f=0; f<num_faces; f++)
		{
			nx_reg[f].resize(s_reg, s_reg); ny_reg[f].resize(s_reg, s_reg); nz_reg[f].resize(s_reg, s_reg); inv_reg_norm[f].resize(s_reg, s_reg);
			mx_reg[f].resize(s_reg, s_reg); my_reg[f].resize(s_reg, s_reg); mz_reg[f].resize(s_reg, s_reg);
			u1_der_reg[f].resize(s_reg, s_reg); u2_der_reg[f].resize(s_reg, s_reg);
			w_u1_reg[f].resize(max_num_w, square(s_reg)); w_u2_reg[f].resize(max_num_w, square(s_reg));
			w_contverts_reg[f].resize(max_num_w, square(s_reg)); w_indices_reg[f].resize(max_num_w);

			for (unsigned int s2 = 0; s2 < s_reg; s2++)
				for (unsigned int s1 = 0; s1 < s_reg; s1++)
				{
					u1_der_reg[f](s1,s2) = new float[3];
					u2_der_reg[f](s1,s2) = new float[3];
				}
		}
	}

	//Show the mesh on the 3D Scene
	ctf_level++;
	if (paper_visualization)		;//takePictureLimitSurface(false);
	else							showMesh();
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
	//desc.vertIndicesPerFace = face_verts.data();
	int *f_vertices = new int[4*num_faces];
	int cont = 0;
	for (unsigned int i = 0; i < num_faces; i++)
		for (unsigned int k=0; k<vertsperface[i]; k++)
			f_vertices[cont++] = face_verts(k,i);
	desc.vertIndicesPerFace = f_vertices;

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
	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
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
	//Refs
	float *u1_der_ref = u1_der[i](v,u);
	float *u2_der_ref = u2_der[i](v,u);
	
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

	//Save the derivatives
	u1_der_ref[0] = eval.deriv1[0];
	u1_der_ref[1] = eval.deriv1[1];
	u1_der_ref[2] = eval.deriv1[2];
	u2_der_ref[0] = eval.deriv2[0];
	u2_der_ref[1] = eval.deriv2[1];
	u2_der_ref[2] = eval.deriv2[2];

	//Compute the normals
	nx[i](v,u) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	ny[i](v,u) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	nz[i](v,u) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
}

void Mod3DfromRGBD::sampleSurfaceForDTBackground()
{
	//Compute the number of samples according to "nsamples_approx"
	const unsigned int nsamples_per_edge = max(3, int(round(sqrtf(float(nsamples_approx) / float(num_faces)))));
	nsamples = square(nsamples_per_edge)*num_faces;

	//Camera parameters
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
	for (int i = 0; i < nstencils; i++)
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

void Mod3DfromRGBD::sampleSurfaceForBSTerm()
{
	//Compute the number of samples according to "nsamples_approx"
	const unsigned int nsamples_per_edge = max(3, int(round(sqrtf(float(nsamples_approx) / float(num_faces)))));
	nsamples = square(nsamples_per_edge)*num_faces;

	//Resize DT variables
	w_DT.resize(max_num_w, nsamples);
	w_u1_DT.resize(max_num_w, nsamples); w_u2_DT.resize(max_num_w, nsamples);
	w_indices_DT.resize(max_num_w, nsamples);
	u1_der_DT.resize(nsamples); u2_der_DT.resize(nsamples);
	u1_dernorm_DT.resize(nsamples); u2_dernorm_DT.resize(nsamples);
	mx_DT.resize(nsamples); my_DT.resize(nsamples); mz_DT.resize(nsamples);
	nx_DT.resize(nsamples); ny_DT.resize(nsamples); nz_DT.resize(nsamples), norm_n_DT.resize(nsamples);
	u1_DT.resize(nsamples); u2_DT.resize(nsamples); uface_DT.resize(nsamples);
	pixel_DT_u.resize(num_images); pixel_DT_v.resize(num_images);

	const float fact = 1.f / float(nsamples_per_edge);
	for (unsigned int f = 0; f < num_faces; f++)
		for (unsigned int u1 = 0; u1 < nsamples_per_edge; u1++)
			for (unsigned int u2 = 0; u2 < nsamples_per_edge; u2++)
			{
				const unsigned int ind = f*square(nsamples_per_edge) + u1*nsamples_per_edge + u2;
				u1_DT(ind) = (0.5f + float(u1))*fact;
				u2_DT(ind) = (0.5f + float(u2))*fact;
				uface_DT(ind) = f;
				u1_der_DT(ind) = new float[3];
				u2_der_DT(ind) = new float[3];
			}
}

void Mod3DfromRGBD::evaluateSurfaceForBSSamples()
{
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);
	
	//Evaluate the surface
	//-----------------------------------------------------------
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

		//Compute the normals
		nx_DT(s) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
		ny_DT(s) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
		nz_DT(s) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
		norm_n_DT(s) = sqrt(square(nx_DT(s)) + square(ny_DT(s)) + square(nz_DT(s)));

		//Compute the weights for the gradient with respect to the control vertices
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
		w_indices_DT.col(s).fill(-1);

		for (unsigned int cv=0; cv<num_verts; cv++)
			if (vect_wc(cv) != 0.f)
			{
				w_indices_DT(cont,s) = cv;
				w_DT(cont,s) = vect_wc(cv);
				w_u1_DT(cont,s) = vect_wu1(cv);
				w_u2_DT(cont,s) = vect_wu2(cv);
				cont++;
			}
	}

	//Compute the pixel to which the samples project
	//-------------------------------------------------------
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
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
		}
	}
}

void Mod3DfromRGBD::evaluateSurfaceForBGSamples()
{
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	Far::StencilTable const *stenciltab = patchTable->GetLocalPointStencilTable();
	const int nstencils = stenciltab->GetNumStencils();
	Far::Stencil *st = new Far::Stencil[nstencils];
	for (int i = 0; i < nstencils; i++)
		st[i] = stenciltab->GetStencil(i);
	
	//Evaluate the surface
	//-----------------------------------------------------------
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

		//Compute the norms of the gradients
		u1_dernorm_DT(s) = sqrt(square(u1_der_DT(s)[0]) + square(u1_der_DT(s)[1]) + square(u1_der_DT(s)[2]));
		u2_dernorm_DT(s) = sqrt(square(u2_der_DT(s)[0]) + square(u2_der_DT(s)[1]) + square(u2_der_DT(s)[2]));

		//Compute the weights for the gradient with respect to the control vertices
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
		w_indices_DT.col(s).fill(-1);

		for (unsigned int cv=0; cv<num_verts; cv++)
			if (vect_wc(cv) != 0.f)
			{
				w_indices_DT(cont,s) = cv;
				w_DT(cont,s) = vect_wc(cv);
				w_u1_DT(cont,s) = vect_wu1(cv);
				w_u2_DT(cont,s) = vect_wu2(cv);
				cont++;
			}
	}

	//Compute the pixel to which the samples project
	//-------------------------------------------------------
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
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
		}
	}
}


//void Mod3DfromRGBD::computeDistanceTransform()
//{
//	for (unsigned int i = 0; i < num_images; i++)
//	{
//		//"Expand" the segmentation to its surrounding invalid pixels
//		Array<bool, Dynamic, Dynamic> big_segment = is_object[i];
//		vector<Array2i> buffer_vu;
//		for (int u=0; u<cols-1; u++)
//			for (int v=0; v<rows-1; v++)
//			{
//				//if (!valid[i](v,u))
//				//	big_segment(v,u) = true;
//				if ((big_segment(v,u) != big_segment(v,u+1))||(big_segment(v,u) != big_segment(v+1,u)))
//					if (big_segment(v,u) == true)
//					{
//						Array2i vu; vu << v, u;
//						buffer_vu.push_back(vu);
//					}
//			}
//
//		while (!buffer_vu.empty())
//		{
//			const Array2i vu = buffer_vu.back();
//			buffer_vu.pop_back();
//
//			if ((vu(0) == 0) || (vu(0) == rows - 1)||(vu(1) == 0) || (vu(1) == cols - 1))
//				continue;
//			else
//			{
//				if ((valid[i](vu(0)-1, vu(1)) == false) && (big_segment(vu(0)-1, vu(1)) == false))
//				{
//					Array2i vu_new; vu_new << vu(0)-1, vu(1);
//					buffer_vu.push_back(vu_new);
//					big_segment(vu(0)-1, vu(1)) = true;
//				}
//
//				if ((valid[i](vu(0)+1, vu(1)) == false) && (big_segment(vu(0)+1, vu(1)) == false))
//				{
//					Array2i vu_new; vu_new << vu(0)+1, vu(1);
//					buffer_vu.push_back(vu_new);
//					big_segment(vu(0)+1, vu(1)) = true;
//				}
//
//				if ((valid[i](vu(0), vu(1)-1) == false) && (big_segment(vu(0), vu(1)-1) == false))
//				{
//					Array2i vu_new; vu_new << vu(0), vu(1)-1;
//					buffer_vu.push_back(vu_new);
//					big_segment(vu(0), vu(1)-1) = true;
//				}
//
//				if ((valid[i](vu(0), vu(1)+1) == false) && (big_segment(vu(0), vu(1)+1) == false))
//				{
//					Array2i vu_new; vu_new << vu(0), vu(1)+1;
//					buffer_vu.push_back(vu_new);
//					big_segment(vu(0), vu(1)+1) = true;
//				}
//			}
//		}
//
//
//		//Compute the distance tranform
//		for (unsigned int u = 0; u < cols; u++)
//			for (unsigned int v = 0; v < rows; v++)
//			{
//				if (big_segment(v, u))
//					DT[i](v, u) = 0.f;
//
//				//Find the closest pixel which belongs to the object
//				else
//				{
//					float min_dist_square = 1000000.f;
//					unsigned int uc, vc; // c - closest
//
//					for (unsigned int us = 0; us < cols; us++)	//s - search
//						for (unsigned int vs = 0; vs < rows; vs++)
//							if (big_segment(vs, us))
//							{
//								const float dist_square = square(us - u) + square(vs - v);
//								if (dist_square < min_dist_square)
//								{
//									min_dist_square = dist_square;
//									uc = us; vc = vs;
//								}
//							}
//					DT[i](v, u) = sqrt(min_dist_square);
//				}
//			}
//
//		//Compute the gradient of the distance transform
//		for (unsigned int u = 1; u < cols - 1; u++)
//			for (unsigned int v = 1; v < rows - 1; v++)
//			{
//				DT_grad_u[i](v, u) = 0.5f*(DT[i](v, u+1) - DT[i](v, u-1));
//				DT_grad_v[i](v, u) = 0.5f*(DT[i](v+1, u) - DT[i](v-1, u));
//			}
//
//		for (unsigned int v = 0; v < rows; v++)
//		{
//			DT_grad_u[i](v, 0) = DT[i](v, 1) - DT[i](v, 0);
//			DT_grad_u[i](v, cols-1) = DT[i](v, cols-1) - DT[i](v, cols-2);
//		}
//		for (unsigned int u = 0; u < rows; u++)
//		{
//			DT_grad_v[i](0, u) = DT[i](1, u) - DT[i](0, u);
//			DT_grad_v[i](rows-1, u) = DT[i](rows-1, u) - DT[i](rows-2, u);
//		}
//	}
//}

void Mod3DfromRGBD::computeDistanceTransformOpenCV(bool safe_DT)
{
    utils::CTicTac clock; clock.Tic();
	
	for (unsigned int i=0; i<num_images; i++)
	{
		// Create binary image with opencv format
		cv::Mat bw, bw_8cu1; //(rows, cols, CV_8UC1);
		MatrixXi ones(rows,cols); ones.fill(1);
		MatrixXi mask;
		if (!safe_DT)	mask = ones - is_object[i].matrix().cast<int>(); //Choose between "invalid & object" or only "object"
		else			mask = valid[i].matrix().cast<int>() - is_object[i].matrix().cast<int>();
		cv::eigen2cv(mask, bw);
		bw.convertTo(bw_8cu1, CV_8UC1);	

		// Perform the distance transform algorithm
		cv::Mat dist;
		cv::distanceTransform(bw_8cu1, dist, CV_DIST_L2, 5);

		//Back to Eigen format
		MatrixXf dt_copy;
		cv::cv2eigen(dist, dt_copy);
		DT[i] = dt_copy.array();

		//Compute the gradient of the distance transform
		for (unsigned int u = 1; u<cols-1; u++)
			for (unsigned int v = 1; v<rows-1; v++)
			{
				DT_grad_u[i](v,u) = 0.5f*(DT[i](v,u+1) - DT[i](v,u-1));
				DT_grad_v[i](v,u) = 0.5f*(DT[i](v+1,u) - DT[i](v-1,u));
			}

		//The gradients are set to 0 at the boundaries to avoid strange behaviours of samples which project out of the image plane.
		for (unsigned int v = 0; v < rows; v++)
		{
			DT_grad_u[i](v,0) = 0.f; //DT[i](v, 1) - DT[i](v, 0); 
			DT_grad_u[i](v,cols-1) = 0.f; //DT[i](v, cols-1) - DT[i](v, cols-2);

			//To avoid having a huge energy when the model is bigger than the image
			DT[i](v,0) = 0.f;
			DT[i](v,cols-1) = 0.f;
		}
		for (unsigned int u = 0; u < cols; u++)
		{
			DT_grad_v[i](0,u) = 0.f; //DT[i](1, u) - DT[i](0, u);
			DT_grad_v[i](rows-1,u) = 0.f; //DT[i](rows-1, u) - DT[i](rows-2, u);
	
			//To avoid having a huge energy when the model is bigger than the image
			DT[i](0,u) = 0.f;
			DT[i](rows-1,u) = 0.f;
		}

		//cout << endl << "DT: " << DT[i];
		//cout << endl << "DT_grad_v: " << DT_grad_v[i];
		//cout << endl << "DT_grad_u: " << DT_grad_u[i];

		//Use the DT to compute the taus
		tau_pixel[i].fill(tau_max);
		if (adaptive_tau)
		{
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if ((DT[i](v,u) < tau_max)&&(DT[i](v,u) > 0.f))
						tau_pixel[i](v,u) = min(tau_max, DT[i](v,u)) - 0.5f;
		}
	}

	const float time_dt = clock.Tac();
	printf("\n Time to compute the DT = %f", time_dt);
}

void Mod3DfromRGBD::computeTruncatedDTOpenCV()
{
    utils::CTicTac clock; clock.Tic();
	
	for (unsigned int i=0; i<num_images; i++)
	{
		// Create binary image with opencv format
		cv::Mat bw, bw_8cu1; //(rows, cols, CV_8UC1);
		MatrixXi ones(rows,cols); ones.fill(1);
		MatrixXi mask = valid[i].matrix().cast<int>() - is_object[i].matrix().cast<int>();
		cv::eigen2cv(mask, bw);
		bw.convertTo(bw_8cu1, CV_8UC1);	

		// Perform the distance transform algorithm
		cv::Mat dist;
		cv::distanceTransform(bw_8cu1, dist, CV_DIST_L2, 5);

		//Back to Eigen format
		MatrixXf dt_copy;
		cv::cv2eigen(dist, dt_copy);
		DT[i] = dt_copy.array();

		//Truncate it
		for (unsigned int u = 0; u<cols; u++)
			for (unsigned int v = 0; v<rows; v++)
				if (DT[i](v,u) > trunc_threshold_DT)
					DT[i](v,u) = trunc_threshold_DT;

		//Compute the gradient of the distance transform
		for (unsigned int u = 1; u<cols-1; u++)
			for (unsigned int v = 1; v<rows-1; v++)
			{
				DT_grad_u[i](v,u) = 0.5f*(DT[i](v,u+1) - DT[i](v,u-1));
				DT_grad_v[i](v,u) = 0.5f*(DT[i](v+1,u) - DT[i](v-1,u));
			}

		//The gradients are set to 0 at the boundaries to avoid strange behaviours of samples which project out of the image plane.
		for (unsigned int v = 0; v < rows; v++)
		{
			DT_grad_u[i](v,0) = 0.f; //DT[i](v, 1) - DT[i](v, 0); 
			DT_grad_u[i](v,cols-1) = 0.f; //DT[i](v, cols-1) - DT[i](v, cols-2);

			//To avoid having a huge energy when the model is bigger than the image
			DT[i](v,0) = 0.f;
			DT[i](v,cols-1) = 0.f;
		}
		for (unsigned int u = 0; u < cols; u++)
		{
			DT_grad_v[i](0,u) = 0.f; //DT[i](1, u) - DT[i](0, u);
			DT_grad_v[i](rows-1,u) = 0.f; //DT[i](rows-1, u) - DT[i](rows-2, u);
	
			//To avoid having a huge energy when the model is bigger than the image
			DT[i](0,u) = 0.f;
			DT[i](rows-1,u) = 0.f;
		}

		//cout << endl << "DT: " << DT[i];
		//cout << endl << "DT_grad_v: " << DT_grad_v[i];
		//cout << endl << "DT_grad_u: " << DT_grad_u[i];
	}

	const float time_dt = clock.Tac();
	printf("\n Time to compute the truncated DT = %f", time_dt);
}

