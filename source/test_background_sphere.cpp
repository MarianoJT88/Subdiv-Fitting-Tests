// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "test_background_sphere.h"


Mod3DfromRGBD::Mod3DfromRGBD()
{
	num_images = 1;
	image_set = 2;
	fovh_d = utils::DEG2RAD(58.6f); fovv_d = utils::DEG2RAD(45.6f);	
	downsample = 4; //It can be set to 1, 2, 4, etc.
	rows = 480/downsample; cols = 640/downsample;
	cam_prior = 0.f;
	max_iter = 100;
	Kn = 0.01f;
	tau = 2.f; 
	alpha = 2e-4f/tau;
	Kn = 0.005f;

	
	//Images
	depth.resize(num_images); x_image.resize(num_images); y_image.resize(num_images);
	nx_image.resize(num_images); ny_image.resize(num_images); nz_image.resize(num_images);
	is_object.resize(num_images); valid.resize(num_images);
	DT.resize(num_images); DT_grad_u.resize(num_images), DT_grad_v.resize(num_images);
	for (unsigned int i = 0; i < num_images; i++)
	{
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
}

void Mod3DfromRGBD::createImageFromSphere()
{
	//Sphere location and radius, and background depth
	xs = 1.f; ys = 0.f; zs = 0.f; r = 0.25f;
	const float back_depth = 2.f;

	//Camera parameters
	const float inv_fu = 2.f*tan(0.5f*fovh_d) / float(cols);
	const float inv_fv = 2.f*tan(0.5f*fovv_d) / float(rows);
	const float um = 0.5f*(cols - 1);
	const float vm = 0.5f*(rows - 1);

	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
		{
			//Compute the intersection between the pixel rays (Pin-Hole) and the Sphere
			const float a = 1.f + square((u-um)*inv_fu) + square((v-vm)*inv_fv);
			const float b = -2.f*(xs + ys*(u-um)*inv_fu + zs*(v*vm)*inv_fv);
			const float c = square(xs) + square(ys) + square(zs) - square(r);
		
			if (b*b - 4.f*a*c >= 0.f)
				depth[0](v,u) = (-b - sqrtf(b*b - 4.f*a*c))/(2.f*a);

			else
				depth[0](v,u) = back_depth;

			x_image[0](v,u) = (u - um)*depth[0](v,u)*inv_fu;
			y_image[0](v,u) = (v - vm)*depth[0](v,u)*inv_fv;	
		}

	//Segment it
	valid[0].fill(true);
	is_object[0].fill(true);
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
			if (depth[0](v,u) > 1.5)
				is_object[0](v,u) = false;
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
	float min_x = xs-r, min_y = ys-r, min_z = zs-r;
	float max_x = xs+r, max_y = ys+r, max_z = zs+r;

	const float x_margin = 0.5*r;
	const float y_margin = 0.5f*r;
	const float z_margin = 0.5f*r;

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

void Mod3DfromRGBD::computeInitialUDataterm()
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
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					//Compute the 3D coordinates of the observed point after the relative transformation
					const float x = depth[i](v, u);
					const float y = x_image[i](v, u);
					const float z = y_image[i](v, u);
					const float n_x = nx_image[i](v, u);
					const float n_y = ny_image[i](v, u);
					const float n_z = nz_image[i](v, u);


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

void Mod3DfromRGBD::searchBetterUDataterm()
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
		const Matrix4f mytrans = Matrix4f::Identity();

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

void Mod3DfromRGBD::computeInitialUBackground()
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
		const Matrix4f mytrans = Matrix4f::Identity();

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

void Mod3DfromRGBD::searchBetterUBackground()
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
		const Matrix4f mytrans = Matrix4f::Identity();

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
	computeInitialUDataterm();

	//Compute initial internal points for the background
	computeInitialUBackground();
}

void Mod3DfromRGBD::computeTransCoordAndResiduals()
{
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix4f mytrans_inv = Matrix4f::Identity();

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
	float energy_f = 0.f, energy_b = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if (is_object[i](v, u))
				{
					energy_f += square(res_x[i](v, u)) + square(res_y[i](v, u)) + square(res_z[i](v, u));
					energy_f += Kn*(square(res_nx[i](v,u)) + square(res_ny[i](v,u)) + square(res_nz[i](v,u)));
				}
				else if (valid[i](v,u))
				{
					const float res_d_squared = square(res_d1[i](v, u)) + square(res_d2[i](v, u));
					
					if (robust_kernel == 0)
					{		
						//Truncated quadratic
						if (res_d_squared < square(tau))	energy_b += alpha*(1.f - res_d_squared / square(tau));
					}
					else if (robust_kernel == 1)
					{
						//2 parabolas with peak
						if (res_d_squared < square(tau))	energy_b += alpha*square(1.f - sqrtf(res_d_squared)/tau);						
					}
				}
			}
	}

	energy_foreground.push_back(energy_f);
	energy_background.push_back(energy_b);

	return (energy_f + energy_b);
}


void Mod3DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1000, 900);
	window.setPos(900, 0);
	window.setCameraZoom(3);
	window.setCameraAzimuthDeg(0);
	window.setCameraElevationDeg(45);
	window.setCameraPointingToPoint(0.f, 0.f, 0.f);

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
	control_mesh->setLineWidth(1.f);
	//control_mesh->setFaceColor(1.f, 0.f, 0.f);
	//control_mesh->setEdgeColor(0.f, 1.f, 0.f);
	//control_mesh->setVertColor(0.f, 0.f, 1.f);
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

		//Insert points (they don't change through the optimization process)
		for (unsigned int v = 0; v < rows; v++)
			for (unsigned int u = 0; u < cols; u++)
			{
				if (is_object[i](v, u))
					points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 0.f, 0.8f, 0.f);
				//else
				//	points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 1.f, 0.f, 0.f);
			}
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

	//Surface normals
	const float fact = 0.01f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		opengl::CSetOfLinesPtr normals = opengl::CSetOfLines::Create();
		normals->setColor(0, 0.8f, 0);
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
	model->setPose(CPose3D(0.f, 1.2f, 0.f, 0.f, 0.f, 0.f));
	model->enableShowVertices(false);
	model->enableShowEdges(false);
	model->enableShowFaces(true);
	model->enableFaceNormals(true);
	scene->insert(model);

	window.unlockAccess3DScene();
	window.repaint();
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

		//Cameras
		opengl::CFrustumPtr frustum = scene->getByClass<CFrustum>(i);

		//Normals
		opengl::CSetOfLinesPtr normals = scene->getByClass<CSetOfLines>(i);
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

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showSubSurface()
{
	scene = window.get3DSceneAndLock();

	//Show correspondences and samples for DT (if solving with DT)
	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(num_images);
	points->clear();

	for (unsigned int i = 0; i < num_images; ++i)
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if (is_object[i](v, u))
					points->push_back(mx[i](v, u), my[i](v, u), mz[i](v, u), 0.f, 0.f, 1.f);

				else if(!solve_DT)
					points->push_back(mx[i](v, u), my[i](v, u), mz[i](v, u), 0.f, 0.f, 0.f);
			}

	if (solve_DT)
	{
		for (unsigned int k = 0; k < nsamples; k++)
			points->push_back(mx_DT(k), my_DT(k), mz_DT(k), 0.5f, 0.f, 0.5f);
	}

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

	//Connecting lines
	//const float fact_norm = 0.03f;
	//float r,g,b;
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(i);
	//	conect->clear();
	//	utils::colormap(mrpt::utils::cmJET,float(i) / float(num_images),r,g,b);
	//	conect->setColor(r,g,b);
	//	for (unsigned int u = 0; u < cols; u++)
	//		for (unsigned int v = 0; v < rows; v++)
	//			if (is_object[i](v,u))
	//				conect->appendLine(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u));
	//}

	//Show the whole surface
	const unsigned int sampl = max(2, int(50.f/sqrtf(num_faces)));
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

	desc.numVertsPerFace = vertsperface;
	desc.vertIndicesPerFace = face_verts.data();

	//Instantiate a FarTopologyRefiner from the descriptor.
	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
				Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

	const int maxIsolation = 0; //Don't change it!
	refiner->RefineAdaptive( Far::TopologyRefiner::AdaptiveOptions(maxIsolation));


	// Generate a set of Far::PatchTable that we will use to evaluate the surface limit
	Far::PatchTableFactory::Options patchOptions;
	patchOptions.endCapType = Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

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
						if (vect_wc(cv) > 0.f)
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


	//Show the mesh on the 3D Scene
	showMesh();
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

	showRenderedModel();
}


void Mod3DfromRGBD::solveGradientDescent()
{
	//								Initialize
	//======================================================================================
	utils::CTicTac clock; 
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));


	evaluateSubDivSurface();
	computeTransCoordAndResiduals();
	optimizeUBackground_LM();
	optimizeUDataterm_LM();
	new_energy = computeEnergyOverall();

	//									Iterative solver
	//====================================================================================
	for (unsigned int i = 0; i < max_iter; i++)
	{
		clock.Tic();
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
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
			const Matrix3f T_inv = Matrix3f::Identity();

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (mx_t[i](v,u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v, u))					;
	//					{
	//						//Matrix<float, 1, 3> res; res << res_x[i](v,u), res_y[i](v,u), res_z[i](v,u);
	//						//Matrix<float, 1, 3> J_mult = -2.f*res*T_inv;


	//						//const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
	//						//Matrix3f J_nu, J_nX;
	//						//J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
	//						//J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
	//						//J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
	//						//J_nu *= inv_norm*square(inv_norm);
	//						//J_nX.assign(0.f);
	//						//Matrix<float, 1, 3> res_n; res_n << res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u);
	//						//Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu;

	//						////Control vertices
	//						//const unsigned int weights_col = v + u*rows;
	//						//for (unsigned int c = 0; c < max_num_w; c++)
	//						//{
	//						//	const int cp = w_indices[i](c,weights_col);
	//						//	if (cp >= 0)
	//						//	{
	//						//		const float ww = w_contverts[i](c, weights_col);
	//						//		vert_incrs(0, cp) += J_mult(0)*ww;
	//						//		vert_incrs(1, cp) += J_mult(1)*ww;
	//						//		vert_incrs(2, cp) += J_mult(2)*ww;

	//						//		//Normals
	//						//		const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
	//						//		J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
	//						//		J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
	//						//		J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
	//						//		J_nX(1,0) = -J_nX(0,1);
	//						//		J_nX(2,0) = -J_nX(0,2);
	//						//		J_nX(2,1) = -J_nX(1,2);

	//						//		vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
	//						//		vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
	//						//		vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
	//						//	}
	//						//}
	//					}

						//Background
						else if (robust_kernel == 0)
						{
							if ( square(res_d1[i](v, u)) + square(res_d2[i](v, u)) < square(tau))
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
										vert_incrs(0, cp) += -alpha*J_phi_pi_Tinv(0)*ww;
										vert_incrs(1, cp) += -alpha*J_phi_pi_Tinv(1)*ww;
										vert_incrs(2, cp) += -alpha*J_phi_pi_Tinv(2)*ww;
									}
								}
							}
						}
						else if (robust_kernel == 1)
						{
							const float norm_proj_error = sqrtf(square(res_d1[i](v, u)) + square(res_d2[i](v, u)));
							if ((norm_proj_error < tau) && (norm_proj_error > eps))
							{

								Matrix<float, 2, 3> J_pi;
								const float inv_z = 1.f / mx_t[i](v, u);

								J_pi << fx*my_t[i](v, u)*square(inv_z), -fx*inv_z, 0.f,
										fy*mz_t[i](v, u)*square(inv_z), 0.f, -fy*inv_z;

								const float J_phi = -(2.f*alpha/tau)*(1.f/norm_proj_error - 1.f/tau);
								Matrix<float, 1, 2> J_res; J_res << res_d1[i](v,u), res_d2[i](v,u);

								const Matrix<float, 1, 3> J_phi_res_pi = J_phi*J_res*J_pi;
								const Matrix<float, 1, 3> J_phi_res_pi_Tinv = J_phi_res_pi*T_inv;

								//Control vertices
								const unsigned int weights_col = v + u*rows;
								for (unsigned int c = 0; c < max_num_w; c++)
								{
									const int cp = w_indices[i](c, weights_col);
									if (cp >= 0)
									{
										const float ww = w_contverts[i](c, weights_col);
										vert_incrs(0, cp) += J_phi_res_pi_Tinv(0)*ww;
										vert_incrs(1, cp) += J_phi_res_pi_Tinv(1)*ww;
										vert_incrs(2, cp) += J_phi_res_pi_Tinv(2)*ww;
									}
								}
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
			//Update control vertices
			vert_coords = vert_coords_old - adap_mult*vert_incrs;


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
			optimizeUBackground_LM();
			optimizeUDataterm_LM();
			new_energy = computeEnergyOverall();

			energy_increasing = false;
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		showMesh();
		showCamPoses();
		showSubSurface();
		showRenderedModel();


		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}


void Mod3DfromRGBD::optimizeUBackground_LM()
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
	float norm_uincr;
	float lambda_mult = 3.f;


	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix4f mytrans_inv = Matrix4f::Identity();
		const Matrix3f T_inv = Matrix3f::Identity();

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u) && valid[i](v,u))
				{
					energy = square(res_d1[i](v,u)) + square(res_d2[i](v,u));
					energy_ratio = 2.f;
					norm_uincr = 1.f;
					lambda = 1.f;

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
						}

						energy_ratio = energy_old / energy;
					}
				}
	}
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

void Mod3DfromRGBD::optimizeUDataterm_LM()
{
	float aver_lambda = 0.f;
	unsigned int cont = 0;
	unsigned int inner_cont;
	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float limit_uincr = 0.05f*sqrtf(num_faces);
	float norm_uincr;
	float lambda_mult = 3.f;
	const float Kn_sqrt = sqrtf(Kn);

	//Solve with LM
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix3f T_inv = Matrix3f::Identity();
		const Matrix4f mytrans_inv = Matrix4f::Identity();

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


void Mod3DfromRGBD::computeDistanceTransform()
{
	for (unsigned int i = 0; i < num_images; i++)
	{
		//"Expand" the segmentation to its surrounding invalid pixels
		Array<bool, Dynamic, Dynamic> big_segment = is_object[i];


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
	const unsigned int nsamples_per_edge = round(sqrtf(float(nsamples_approx) / float(num_faces)));
	nsamples = square(nsamples_per_edge)*num_faces;

	//Camera parameters
	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	
	//Resize DT variables
	w_DT.resize(nsamples);
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
				w_DT(ind) = new float[num_verts];
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
		for (unsigned int k = 0; k < num_verts; k++)
			w_DT(s)[k] = 0.f;

		for (int cv = 0; cv < cvs.size(); ++cv)
		{
			if (cvs[cv] < num_verts)
				w_DT(s)[cvs[cv]] += pWeights[cv];

			else
			{
				const unsigned int ind_offset = cvs[cv] - num_verts;
				//Look at the stencil associated to this local point and distribute its weight over the control vertices
				unsigned int size_st = st[ind_offset].GetSize();
				Far::Index const *st_ind = st[ind_offset].GetVertexIndices();
				float const *st_weights = st[ind_offset].GetWeights();
				for (unsigned int st = 0; st < size_st; st++)
					w_DT(s)[st_ind[st]] += pWeights[cv] * st_weights[st];
			}
		}

	}

	//Compute the pixel to which the samples project
	for (unsigned int i = 0; i < num_images; i++)
	{
		pixel_DT_u[i].resize(nsamples);
		pixel_DT_v[i].resize(nsamples);
		const Matrix4f T_inv = Matrix4f::Identity();

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
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);


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
		vert_incrs.fill(0.f);
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

			const Matrix3f T_inv = Matrix3f::Identity();

			////Foreground
			//for (unsigned int u = 0; u < cols; u++)
			//	for (unsigned int v = 0; v < rows; v++)
			//		if (is_object[i](v, u))
			//		{
			//			//Warning
			//			if (mx_t[i](v,u) <= 0.f)
			//				printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

			//			Matrix<float, 1, 3> res; res << res_x[i](v,u), res_y[i](v,u), res_z[i](v,u);
			//			Matrix<float, 1, 3> J_mult = -2.f*res*T_inv;

			//			const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
			//			Matrix3f J_nu, J_nX;
			//			J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
			//			J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
			//			J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
			//			J_nu *= inv_norm*square(inv_norm);
			//			J_nX.assign(0.f);
			//			Matrix<float, 1, 3> res_n; res_n << res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u);
			//			Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu;

			//			
			//			//Control vertices
			//			const unsigned int weights_col = v + u*rows;
			//			for (unsigned int cp = 0; cp < num_verts; cp++)
			//			{
			//				const float ww = w_contverts[i](cp, weights_col);
			//				vert_incrs(0, cp) += J_mult(0)*ww;
			//				vert_incrs(1, cp) += J_mult(1)*ww;
			//				vert_incrs(2, cp) += J_mult(2)*ww;

			//				//Normals
			//				const float wu1 = w_u1[i](cp, weights_col), wu2 = w_u2[i](cp, weights_col);
			//				J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
			//				J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
			//				J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
			//				J_nX(1,0) = -J_nX(0,1);
			//				J_nX(2,0) = -J_nX(0,2);
			//				J_nX(2,1) = -J_nX(1,2);

			//				vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
			//				vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
			//				vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
			//			}
			//		}

			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point; t_point << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
				const float mx_t_DT = t_point(0);
				const float my_t_DT = t_point(1);
				const float mz_t_DT = t_point(2);

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 2, 3> J_pi;
				const float inv_z = 1.f / mx_t_DT;

				J_pi << fx*my_t_DT*square(inv_z), -fx*inv_z, 0.f,
						fy*mz_t_DT*square(inv_z), 0.f, -fy*inv_z;

				const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s))), DT_grad_v[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s)))};
				const Matrix<float, 1, 3> J_mult = J_DT*J_pi*T_inv;

				//Control vertices
				for (unsigned int cp = 0; cp < num_verts; cp++)
				{
					const float ww = w_DT(s)[cp];
					vert_incrs(0, cp) += -alpha*J_mult(0)*ww;
					vert_incrs(1, cp) += -alpha*J_mult(1)*ww;
					vert_incrs(2, cp) += -alpha*J_mult(2)*ww;
				}
			}
		}

		energy_increasing = true;
		unsigned int cont = 0.f;

		//Update the control vertices
		while (energy_increasing)
		{
			//Update
			vert_coords = vert_coords_old - adap_mult*vert_incrs;

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
			optimizeUDataterm_LM();
			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDTOverall();

			//if (new_energy <= last_energy)
			//{
			//	energy_increasing = false;
			//	adap_mult *= 1.5f;
			//	//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}
			//else
			//{
			//	adap_mult *= 0.5f;
			//	//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}

			//cont++;
			//if (cont > 10) energy_increasing = false;
			energy_increasing = false;
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

void Mod3DfromRGBD::solveWithDT2()
{
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;
	utils::CTicTac clock;

	const float fx = float(cols) / (2.f*tan(0.5f*fovh_d));
	const float fy = float(rows) / (2.f*tan(0.5f*fovv_d));
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);


	evaluateSubDivSurface();
	computeTransCoordAndResiduals();
	sampleSurfaceForDTBackground();
	optimizeUDataterm_LM();
	new_energy = computeEnergyDT2Overall();

	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();

		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		vert_incrs.fill(0.f);
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

			const Matrix3f T_inv = Matrix3f::Identity();

			////Foreground
			//for (unsigned int u = 0; u < cols; u++)
			//	for (unsigned int v = 0; v < rows; v++)
			//		if (is_object[i](v, u))
			//		{
			//			//Warning
			//			if (mx_t[i](v,u) <= 0.f)
			//				printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

			//			Matrix<float, 1, 3> res; res << res_x[i](v,u), res_y[i](v,u), res_z[i](v,u);
			//			Matrix<float, 1, 3> J_mult = -2.f*res*T_inv;

			//			const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));
			//			Matrix3f J_nu, J_nX;
			//			J_nu.row(0) << square(ny[i](v,u)) + square(nz[i](v,u)), -nx[i](v,u)*ny[i](v,u), -nx[i](v,u)*nz[i](v,u);
			//			J_nu.row(1) << -nx[i](v,u)*ny[i](v,u), square(nx[i](v,u)) + square(nz[i](v,u)), -ny[i](v,u)*nz[i](v,u);
			//			J_nu.row(2) << -nx[i](v,u)*nz[i](v,u), -ny[i](v,u)*nz[i](v,u), square(nx[i](v,u)) + square(ny[i](v,u));
			//			J_nu *= inv_norm*square(inv_norm);
			//			J_nX.assign(0.f);
			//			Matrix<float, 1, 3> res_n; res_n << res_nx[i](v,u), res_ny[i](v,u), res_nz[i](v,u);
			//			Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*res_n*T_inv*J_nu;

			//			
			//			//Control vertices
			//			const unsigned int weights_col = v + u*rows;
			//			for (unsigned int cp = 0; cp < num_verts; cp++)
			//			{
			//				const float ww = w_contverts[i](cp, weights_col);
			//				vert_incrs(0, cp) += J_mult(0)*ww;
			//				vert_incrs(1, cp) += J_mult(1)*ww;
			//				vert_incrs(2, cp) += J_mult(2)*ww;

			//				//Normals
			//				const float wu1 = w_u1[i](cp, weights_col), wu2 = w_u2[i](cp, weights_col);
			//				J_nX(0,1) = wu1*u2_der[i](v,u)[2] - wu2*u1_der[i](v,u)[2];
			//				J_nX(0,2) = wu2*u1_der[i](v,u)[1] - wu1*u2_der[i](v,u)[1];
			//				J_nX(1,2) = wu1*u2_der[i](v,u)[0] - wu2*u1_der[i](v,u)[0];
			//				J_nX(1,0) = -J_nX(0,1);
			//				J_nX(2,0) = -J_nX(0,2);
			//				J_nX(2,1) = -J_nX(1,2);

			//				vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
			//				vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
			//				vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
			//			}
			//		}

			//Background term with DT
			for (unsigned int s = 0; s < nsamples; s++)
			{
				Vector4f t_point; t_point << mx_DT(s), my_DT(s), mz_DT(s), 1.f;
				const float mx_t_DT = t_point(0);
				const float my_t_DT = t_point(1);
				const float mz_t_DT = t_point(2);

				if (mx_t_DT <= 0.f)  printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

				Matrix<float, 2, 3> J_pi;
				const float inv_z = 1.f / mx_t_DT;

				J_pi << fx*my_t_DT*square(inv_z), -fx*inv_z, 0.f,
						fy*mz_t_DT*square(inv_z), 0.f, -fy*inv_z;

				const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s))), DT_grad_v[i](int(pixel_DT_v[i](s)), int(pixel_DT_u[i](s)))};
				const Matrix<float, 1, 3> J_mult = 2.f*DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s))*J_DT*J_pi*T_inv;

				//Control vertices
				for (unsigned int cp = 0; cp < num_verts; cp++)
				{
					const float ww = w_DT(s)[cp];
					vert_incrs(0, cp) += -alpha*J_mult(0)*ww;
					vert_incrs(1, cp) += -alpha*J_mult(1)*ww;
					vert_incrs(2, cp) += -alpha*J_mult(2)*ww;
				}
			}
		}

		energy_increasing = true;
		unsigned int cont = 0.f;

		//Update the control vertices
		while (energy_increasing)
		{
			//Update
			vert_coords = vert_coords_old - adap_mult*vert_incrs;

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
			optimizeUDataterm_LM();
			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDT2Overall();

			//if (new_energy <= last_energy)
			//{
			//	energy_increasing = false;
			//	adap_mult *= 1.5f;
			//	//printf("\n Energy decreasing: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}
			//else
			//{
			//	adap_mult *= 0.5f;
			//	//printf("\n Energy increasing -> repeat: ne = %f, le = %f, adap_mult = %f", new_energy, last_energy, adap_mult);
			//}

			//cont++;
			//if (cont > 10) energy_increasing = false;
			energy_increasing = false;
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
	float energy_f = 0.f, energy_b = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
					energy_f += square(res_x[i](v, u)) + square(res_y[i](v, u)) + square(res_z[i](v, u));

		for (unsigned int s = 0; s < nsamples; s++)
			energy_b += alpha*DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s));
	}

	energy_foreground.push_back(energy_f);
	energy_background.push_back(energy_b);

	return (energy_f + energy_b);
}

float Mod3DfromRGBD::computeEnergyDT2Overall()
{
	float energy_f = 0.f, energy_b = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
					energy_f += square(res_x[i](v, u)) + square(res_y[i](v, u)) + square(res_z[i](v, u));

		for (unsigned int s = 0; s < nsamples; s++)
			energy_b += alpha*square(DT[i](pixel_DT_v[i](s), pixel_DT_u[i](s)));
	}

	energy_foreground.push_back(energy_f);
	energy_background.push_back(energy_b);

	return (energy_f + energy_b);
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

		for (unsigned int k=0; k<energy_foreground.size(); k++)
		{
			f_res << energy_foreground[k] << " ";
			f_res << energy_background[k] << endl;
		}

		f_res.close();
	}
	catch (...)
	{
		printf("Exception found trying to create the 'results file' !!\n");
	}
}





