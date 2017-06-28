// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "mesh_smoothing.h"


MeshSmoother::MeshSmoother() {}


void MeshSmoother::loadTargetMesh()
{
	//Open file
	cout << endl << "Filename: " << mesh_path;
	std::ifstream	f_mesh;

	f_mesh.open(mesh_path.c_str());
		
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
	num_points = num_faces*data_per_face;


	//Load the vertices
	vert_coords_target.resize(3, num_verts);

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
			iss >> aux_v >> vert_coords_target(0,cont) >> vert_coords_target(1,cont) >> vert_coords_target(2,cont);	
			cont++;
		}
		else
			is_vert = false;
	}

	//Load the faces
	face_verts.resize(4, num_faces);
	cont = 0;
	printf("\nReading faces...");

	//while (line.at(0) == 'v')
	//	std::getline(f_mesh, line);

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

void MeshSmoother::generateDataFromMesh()
{
	//Resize all variables
	//================================================================================================================

	//Images
	data_coords.resize(3,num_points); data_normals.resize(3,num_points); n_weights.resize(num_points); n_weights.fill(1.f); 

	//Correspondences
	u1.resize(num_points); u2.resize(num_points);
	u1_old.resize(num_points); u2_old.resize(num_points);
	u1_old_outer.resize(num_points); u2_old_outer.resize(num_points);
	uface.resize(num_points); uface_old.resize(num_points); uface_old_outer.resize(num_points);
	u1_incr.resize(num_points); u2_incr.resize(num_points);
	res_x.resize(num_points); res_y.resize(num_points); res_z.resize(num_points);
	res_nx.resize(num_points); res_ny.resize(num_points); res_nz.resize(num_points);

	mx.resize(num_points); my.resize(num_points); mz.resize(num_points);
	nx.resize(num_points); ny.resize(num_points); nz.resize(num_points);
	u1_der.resize(num_points); u2_der.resize(num_points);
	n_der_u1.resize(num_points); n_der_u2.resize(num_points);

	for (unsigned int v = 0; v < num_points; v++)
	{
		u1_der(v) = new float[3];
		u2_der(v) = new float[3];
		n_der_u1(v) = new float[3];
		n_der_u2(v) = new float[3];
	}


	//Jacobian wrt the control vertices
	w_indices.resize(max_num_w, num_points);
	w_contverts.resize(max_num_w, num_points);
	w_u1.resize(max_num_w, num_points);
	w_u2.resize(max_num_w, num_points);

	//Initialize the new mesh like the target mesh
	vert_coords = vert_coords_target;
	vert_coords_old = vert_coords;
	vert_incrs.resize(3,num_points);
	

	//Regularization
	s_reg = 4;
	vert_coords_reg = vert_coords; //For the ctf_regularization

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

	//Generate data by evaluating the target subdiv surface
	createTopologyRefiner();

	//Evaluate the surface to generate data
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];
	unsigned int cont = 0;
	const unsigned int data_sqrt = sqrt(data_per_face);

	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f<num_faces; f++)
		for (unsigned int v=0; v<data_sqrt; v++)
			for (unsigned int u=0; u<data_sqrt; u++)
			{
				const float u1 = (0.5f + u)/float(data_sqrt);
				const float u2 = (0.5f + v)/float(data_sqrt);
			
				// Locate the patch corresponding to the face ptex idx and (s,t)
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1, u2); assert(handle);

				// Evaluate the patch weights, identify the CVs and compute the limit frame:
				patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Save the 3D coordinates
				data_coords(0,cont) = eval.point[0];
				data_coords(1,cont) = eval.point[1];
				data_coords(2,cont) = eval.point[2];

				//Compute the normals
				data_normals(0,cont) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
				data_normals(1,cont) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
				data_normals(2,cont) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

				cont++;
			}	
}


void MeshSmoother::computeInitialUDataterm()
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

	//Find the closest point to each of the observed with the cameras - Brute force
	//-----------------------------------------------------------------------------
	for (unsigned int i = 0; i < num_points; ++i)
	{

		float min_dist = 100.f, dist;
		unsigned int s_min = 0;

		for (unsigned int s = 0; s < ini_samp_size; s++)
		{						
			dist = Kp*(square(data_coords(0,i) - xyz_ini(0,s)) + square(data_coords(1,i) - xyz_ini(1,s)) + square(data_coords(2,i) - xyz_ini(2,s)))
					+ Kn*n_weights(i)*(square(data_normals(0,i) - n_ini(0,s)) + square(data_normals(1,i) - n_ini(1,s)) + square(data_normals(2,i) - n_ini(2,s)));

			if (dist  < min_dist)
			{
				min_dist = dist;
				s_min = s;
			}
		}

		u1(i) = u_triplet(0,s_min);
		u2(i) = u_triplet(1,s_min);
		uface(i) = u_triplet(2,s_min);
	}
}

void MeshSmoother::searchBetterUDataterm()
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
	for (unsigned int i = 0; i < num_points; ++i)
	{
		float min_dist = Kp*(square(res_x(i)) + square(res_y(i)) + square(res_z(i)))
						+ Kn*n_weights(i)*(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));

		float dist;
		int s_min = -1;

		for (unsigned int s = 0; s < ini_samp_size; s++)
		{
			dist = Kp*(square(data_coords(0,i) - xyz_ini(0,s)) + square(data_coords(1,i) - xyz_ini(1,s)) + square(data_coords(2,i) - xyz_ini(2,s)))
					+ Kn*n_weights(i)*(square(data_normals(0,i) - n_ini(0,s)) + square(data_normals(1,i) - n_ini(1,s)) + square(data_normals(2,i) - n_ini(2,s)));

			if (dist  < min_dist)
			{
				//printf("\n Better dist. dist = %f, min_dist = %f", dist, min_dist);				
				min_dist = dist;
				s_min = s;
			}
		}

		if (s_min >= 0)
		{
			u1(i) = u_triplet(0,s_min);
			u2(i) = u_triplet(1,s_min);
			uface(i) = u_triplet(2,s_min);
		}
	}
}

void MeshSmoother::computeTransCoordAndResiduals()
{	
	for (unsigned int i = 0; i < num_points; i++)
	{
		res_x(i) = data_coords(0,i) - mx(i);
		res_y(i) = data_coords(1,i) - my(i);
		res_z(i) = data_coords(2,i) - mz(i);

		const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
		res_nx(i) = data_normals(0,i) - inv_norm*nx(i);
		res_ny(i) = data_normals(1,i) - inv_norm*ny(i);
		res_nz(i) = data_normals(2,i) - inv_norm*nz(i);
	}
}

void MeshSmoother::updateInternalPointCrossingEdges(unsigned int i)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	//Check if crossing borders
	Vector2f u_incr; u_incr << u1_incr(i), u2_incr(i);
	float u1_old = this->u1_old(i);
	float u2_old = this->u2_old(i);
	unsigned int face = uface_old(i);

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
		if (cont > 5)
		{
			printf("\n Problem!!! Many jumps between the mesh faces for the update of one correspondence. I remove the remaining u_increment!");
			u1_new = u1_old;
			u2_new = u2_old;
			break;
		}
	}

	u1(i) = u1_new;
	u2(i) = u2_new;
	uface(i) = face;
}


float MeshSmoother::computeEnergyNB()
{
	float energy_d = 0.f, energy_r = 0.f;

	for (unsigned int i = 0; i < num_points; i++)
	{
		const float res = sqrtf(square(res_x(i)) + square(res_y(i)) + square(res_z(i)));
		if (res < truncated_res)
			energy_d += Kp*square(res);
		else
			energy_d += Kp*square(truncated_res);

		const float resn = sqrtf(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));
		if (resn < truncated_resn)
			energy_d += Kn*n_weights(i)*square(resn);	
		else
			energy_d += Kn*n_weights(i)*square(truncated_resn);
	}

	//Regularization
	if (with_reg_normals)		energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)	energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)	energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)			energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)			energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)		energy_r += computeEnergyRegAtraction();

	const float energy_o = energy_d + energy_r;
	//printf("\n Energies: overall = %f, dataterm = %f, regularization = %f", energy_o, energy_d, energy_r);

	return energy_o;
}


float MeshSmoother::computeEnergyRegNormals()
{
	Kr = Kr_total/float(square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				energy += Kr*((square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2)) + square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))
							 +(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2)) + square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2))));
			}
		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			energy += Kr*(square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)));
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			energy += Kr*(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)));	
		}
	}

	return energy;
}

float MeshSmoother::computeEnergyRegNormalsGood()
{
	Kr = Kr_total/float(num_faces*square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const float dist_s1 = square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
				const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));

				energy += Kr*((square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1
								+(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2);
			}
		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const float dist_s1 = square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1;
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2;	
		}
	}

	return energy;
}

float MeshSmoother::computeEnergyRegAtraction()
{
	float energy = 0.f;
	const float sqrt_2 = sqrtf(2.f);
	K_atrac = K_atrac_total;
	for (unsigned int f=0; f<num_faces; f++)
		for (unsigned int e=0; e<4; e++)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);

			const Vector3f edge = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length_squared = (edge.sumAll());

			energy += K_atrac*length_squared;
		}

	return energy;
}

float MeshSmoother::computeEnergyRegNormals4dir()
{
	Kr = Kr_total/float(num_faces*square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Middle points
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const float dist_s1 = square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
				const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));

				energy += Kr*((square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1
								+(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2);
			}
		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const float dist_s1 = square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1;
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));
			energy += Kr*(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2;	
		}

		//Diagonals
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const float dist_s1 = square(mx_reg[f](s1+1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2+1) - mz_reg[f](s1,s2));
				const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1+1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1+1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1+1,s2));

				energy += Kr*((square(nx_reg[f](s1+1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2+1) - ny_reg[f](s1,s2)) + square(nz_reg[f](s1+1,s2+1) - nz_reg[f](s1,s2)))/dist_s1
							 +(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1+1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1+1,s2)) + square(nz_reg[f](s1,s2+1) - nz_reg[f](s1+1,s2)))/dist_s2);
			}
	}

	return energy;
}

float MeshSmoother::computeEnergyRegEdges()
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

float MeshSmoother::computeEnergyRegCTF()
{
	float energy = 0.f;
	K_ctf = K_ctf_total/num_verts;
	
	for (unsigned int cv=0; cv<num_verts; cv++)
		for (unsigned int k=0; k<3; k++)
			energy += K_ctf*square(vert_coords(k,cv) - vert_coords_reg(k,cv));

	return energy;
}


void MeshSmoother::initializeScene()
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

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	reference->setScale(0.05f);
	scene->insert(reference);

	//target mesh
	const float separation = 0.6f;
	opengl::CMesh3DPtr target_mesh = opengl::CMesh3D::Create();
	target_mesh->setPose(CPose3D(0.f, separation, 0.f, 0.f, 0.f, 0.f));
	target_mesh->enableShowEdges(true);
	target_mesh->enableShowFaces(false);
	target_mesh->enableShowVertices(true);
	target_mesh->setLineWidth(2.f);
	target_mesh->setPointSize(8.f);
	target_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords_target);
	scene->insert(target_mesh);

	//target surface
	opengl::CMesh3DPtr model = opengl::CMesh3D::Create();
	model->setPose(CPose3D(separation, separation, 0.f, 0.f, 0.f, 0.f));
	model->enableShowVertices(false);
	model->enableShowEdges(false);
	model->enableShowFaces(true);
	model->enableFaceNormals(true);
	model->setFaceColor(0.7f, 0.7f, 0.8f, 1.f);
	scene->insert(model);
	showOriginalSurface();

	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->setPose(CPose3D(0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	control_mesh->setLineWidth(2.f);
	control_mesh->setPointSize(8.f);
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);
	scene->insert(control_mesh);

	//Smooth surface
	opengl::CMesh3DPtr smooth_surface = opengl::CMesh3D::Create();
	smooth_surface->setPose(CPose3D(separation, 0.f, 0.f, 0.f, 0.f, 0.f));
	smooth_surface->enableShowEdges(false);
	smooth_surface->enableShowFaces(true);
	smooth_surface->enableShowVertices(false);
	smooth_surface->enableFaceNormals(true);
	smooth_surface->setFaceColor(0.7f, 0.7f, 0.8f, 1.f);
	smooth_surface->setLineWidth(2.f);
	smooth_surface->setPointSize(8.f);
	scene->insert(smooth_surface);

	////Vertex numbers
	//for (unsigned int v = 0; v < 2000; v++)
	//{
	//	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	//	vert_nums->setString(std::to_string(v));
	//	vert_nums->setScale(0.02f);
	//	vert_nums->setColor(0.5, 0, 0);
	//	scene->insert(vert_nums);
	//}


	//Data points
	opengl::CPointCloudColouredPtr datapoints = opengl::CPointCloudColoured::Create();
	datapoints->setPointSize(4.f);
	datapoints->enablePointSmooth(true);
	scene->insert(datapoints);
	for (unsigned int i = 0; i < num_points; i++)
		datapoints->push_back(data_coords(0,i), data_coords(1,i), data_coords(2,i), 0.f, 0.f, 1.f);


	//Correspondences (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(2.f);
	points->enablePointSmooth(true);
	scene->insert(points);


	window.unlockAccess3DScene();
	window.repaint();
}

void MeshSmoother::showOriginalSurface()
{
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;

	const unsigned int num_ref = 3;
	
	for (unsigned int r=0; r<num_ref; r++)
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

	//Update mesh
	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	Array<bool, Dynamic, 1>	is_quad_now(num_faces_now); is_quad_now.fill(true);
	model->loadMesh(num_verts_now, num_faces_now, is_quad_now, face_verts_now, vert_coords_now);
}


void MeshSmoother::showSubSurface()
{
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;

	const unsigned int num_ref = 3;
	
	for (unsigned int r=0; r<num_ref; r++)
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

	//Update mesh
	scene = window.get3DSceneAndLock();

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(3);
	Array<bool, Dynamic, 1>	is_quad_now(num_faces_now); is_quad_now.fill(true);
	model->loadMesh(num_verts_now, num_faces_now, is_quad_now, face_verts_now, vert_coords_now);

	window.unlockAccess3DScene();
	window.repaint();
}

void MeshSmoother::showMeshAndCorrespondences()
{
	scene = window.get3DSceneAndLock();

	//Show correspondences and samples for DT (if solving with DT)
	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(1);
	points->clear();

	for (unsigned int i = 0; i < num_points; ++i)
		points->push_back(mx(i), my(i), mz(i), 0.7f, 0.7f, 0.7f);

	//Control mesh
	opengl::CMesh3DPtr control_mesh = scene->getByClass<CMesh3D>(2);
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);


	window.unlockAccess3DScene();
	window.repaint();
}


void MeshSmoother::createTopologyRefiner()
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

void MeshSmoother::evaluateSubDivSurface()
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
	for (unsigned int i = 0; i<num_points; ++i)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface(i), u1(i), u2(i)); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1(i), u2(i), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		//Save the 3D coordinates
		mx(i) = eval.point[0];
		my(i) = eval.point[1];
		mz(i) = eval.point[2];

		//Save the derivatives
		u1_der(i)[0] = eval.deriv1[0];
		u1_der(i)[1] = eval.deriv1[1];
		u1_der(i)[2] = eval.deriv1[2];
		u2_der(i)[0] = eval.deriv2[0];
		u2_der(i)[1] = eval.deriv2[1];
		u2_der(i)[2] = eval.deriv2[2];

		//Compute the normals
		nx(i) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
		ny(i) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
		nz(i) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

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
		const unsigned int col_weights = i;
		w_indices.col(col_weights).fill(-1);

		for (unsigned int cv=0; cv<num_verts; cv++)
			if (vect_wc(cv) != 0.f)
			{
				w_indices(cont, col_weights) = cv;
				w_contverts(cont, col_weights) = vect_wc(cv);
				w_u1(cont, col_weights) = vect_wu1(cv);
				w_u2(cont, col_weights) = vect_wu2(cv);
				cont++;
			}
		//cout << endl << "w_indices: " << w_indices[i].col(col_weights).transpose();
	}
}

void MeshSmoother::evaluateSubDivSurfaceRegularization()
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
		w_indices_reg[f].fill(-1);
		for (unsigned int k=0; k<indices.size(); k++)
			w_indices_reg[f](k) = indices.at(k);

		//Clean the weights before updating them
		w_contverts_reg[f].fill(0.f); w_u1_reg[f].fill(0.f); w_u2_reg[f].fill(0.f);

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
				mx_reg[f](s1,s2) = eval.point[0];
				my_reg[f](s1,s2) = eval.point[1];
				mz_reg[f](s1,s2) = eval.point[2];

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
				const unsigned int col_weights = s1 + s2*s_reg;

				for (int cv = 0; cv < cvs.size(); ++cv)
				{						
					if (cvs[cv] < num_verts)
					{			
						//Find the position of cvs[cv] in the "w_index_reg" vector
						unsigned int index_pos = std::find(indices.begin(), indices.end(), cvs[cv]) - indices.begin();
						
						w_contverts_reg[f](index_pos, col_weights) += pWeights[cv];
						w_u1_reg[f](index_pos, col_weights) += dsWeights[cv];
						w_u2_reg[f](index_pos, col_weights) += dtWeights[cv];
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

							w_contverts_reg[f](index_pos, col_weights) += pWeights[cv]*st_weights[s];
							w_u1_reg[f](index_pos, col_weights) += dsWeights[cv]*st_weights[s];
							w_u2_reg[f](index_pos, col_weights) += dtWeights[cv]*st_weights[s];
						}
					}
				}
			}
	}
}

void MeshSmoother::computeNormalDerivativesPixel(unsigned int i)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	const float uincr = 0.001f; const float uincr_inv = 1.f/uincr;

	//Compute normal for small increment of u1
	//=================================================================================================================
	Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(uface(i), u1(i)+uincr, u2(i)); assert(handle1);
	patchTable->EvaluateBasis(*handle1, u1(i)+uincr, u2(i), pWeights, dsWeights, dtWeights);
	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);

	LimitFrame eval; eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);

	//Compute the normals
	const float nx_u1 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	const float ny_u1 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	const float nz_u1 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

	n_der_u1(i)[0] = uincr_inv*(nx_u1 - nx(i));
	n_der_u1(i)[1] = uincr_inv*(ny_u1 - ny(i));
	n_der_u1(i)[2] = uincr_inv*(nz_u1 - nz(i));

	//Compute normal for small increment of u2
	//=================================================================================================================
	Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(uface(i), u1(i), u2(i)+uincr); assert(handle2);
	patchTable->EvaluateBasis(*handle2, u1(i), u2(i)+uincr, pWeights, dsWeights, dtWeights);
	cvs = patchTable->GetPatchVertices(*handle2);

	eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]],pWeights[cv],dsWeights[cv],dtWeights[cv]);

	//Compute the normals
	const float nx_u2 = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	const float ny_u2 = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	const float nz_u2 = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];

	n_der_u2(i)[0] = uincr_inv*(nx_u2 - nx(i));
	n_der_u2(i)[1] = uincr_inv*(ny_u2 - ny(i));
	n_der_u2(i)[2] = uincr_inv*(nz_u2 - nz(i));
}

void MeshSmoother::refineMeshOneLevel()
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
	vert_incrs.resize(3, num_verts);
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
}

//void MeshSmoother::refineMeshToShow()
//{
//	typedef Far::TopologyDescriptor Descriptor;
//
//	Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
//	Sdc::Options options;
//	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_NONE);
//
//	//Fill the topology of the mesh
//	Descriptor desc;
//	desc.numVertices = num_verts;
//	desc.numFaces = num_faces;
//
//	int *vertsperface; vertsperface = new int[num_faces];
//	for (unsigned int i = 0; i < num_faces; i++)
//	{
//		if (is_quad(i)) vertsperface[i] = 4;
//		else			vertsperface[i] = 3;
//	}
//
//	desc.numVertsPerFace = vertsperface;
//	desc.vertIndicesPerFace = face_verts.data();
//
//	//Instantiate a FarTopologyRefiner from the descriptor.
//	refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
//		Far::TopologyRefinerFactory<Descriptor>::Options(type, options));
//
//	// Uniformly refine the topolgy once
//	refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(1));
//	const Far::TopologyLevel mesh_level = refiner->GetLevel(1);
//
//	// Allocate and fill a buffer for the old vertex primvar data
//	vector<Vertex> old_verts; old_verts.resize(num_verts);
//	for (unsigned int v = 0; v<num_verts; v++)
//		old_verts[v].SetPosition(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));
//
//	// Allocate and fill a buffer for the new vertex primvar data
//	verts.resize(mesh_level.GetNumVertices());
//	Vertex *src = &old_verts[0], *dst = &verts[0];
//	Far::PrimvarRefiner(*refiner).Interpolate(1, src, dst);
//
//
//	//									Create the new mesh from it
//	//---------------------------------------------------------------------------------------------------------
//	num_verts = mesh_level.GetNumVertices();
//	num_faces = mesh_level.GetNumFaces();
//
//	//Fill the type of poligons (triangles or quads)
//	is_quad.resize(num_faces, 1);
//	for (unsigned int f = 0; f < num_faces; f++)
//	{
//		if (mesh_level.GetFaceVertices(f).size() == 4)
//			is_quad(f) = true;
//		else
//		{
//			is_quad(f) = false;
//			printf("\n Warning!!!! Some faces are not Quad and the algorithm is more than likely to crash!!");
//		}
//	}
//
//	//Fill the vertices per face
//	face_verts.resize(4, num_faces);
//	for (unsigned int f = 0; f < num_faces; f++)
//	{
//		Far::ConstIndexArray face_v = mesh_level.GetFaceVertices(f);
//		for (int v = 0; v < face_v.size(); v++)
//			face_verts(v, f) = face_v[v];
//	}
//
//	//Fill the 3D coordinates of the vertices
//	vert_incrs.resize(3, num_verts);
//	vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
//	for (unsigned int v = 0; v < verts.size(); v++)
//	{
//		vert_coords(0, v) = verts[v].point[0];
//		vert_coords(1, v) = verts[v].point[1];
//		vert_coords(2, v) = verts[v].point[2];
//	}
//
//	showRenderedModel();
//}


void MeshSmoother::solveNB_GradientDescent()
{
	//								Initialize
	//======================================================================================
	utils::CTicTac clock; 
	sz_x = 0.0001f;
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	evaluateSubDivSurface();
	if (with_reg_normals)	evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	optimizeUDataterm_LM();
	new_energy = computeEnergyNB();

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
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
		for (unsigned int i = 0; i < num_points; i++)
		{
			//Keep the last solution for u
			u1_old_outer(i) = u1(i);
			u2_old_outer(i) = u2(i);
			uface_old_outer(i) = uface(i);

					
			//									Data alignment
			//----------------------------------------------------------------------------------------------------
			const float res = sqrtf(square(res_x(i)) + square(res_y(i)) + square(res_z(i)));
			if (res < truncated_res)
			{
				Matrix<float, 1, 3> res; res << res_x(i), res_y(i), res_z(i);
				Matrix<float, 1, 3> J_mult = -2.f*Kp*res;

				//Control vertices
				const unsigned int weights_col = i;
				for (unsigned int c = 0; c < max_num_w; c++)
				{
					const int cp = w_indices(c,weights_col);
					if (cp >= 0)
					{
						const float ww = w_contverts(c, weights_col);
						vert_incrs(0, cp) += J_mult(0)*ww;
						vert_incrs(1, cp) += J_mult(1)*ww;
						vert_incrs(2, cp) += J_mult(2)*ww;
					}
				}
			}

			//									Normal alignment
			//----------------------------------------------------------------------------------------------
			const float resn = sqrtf(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));
			if (resn < truncated_resn)
			{
				const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
				Matrix3f J_nu, J_nX;
				J_nu.row(0) << square(ny(i)) + square(nz(i)), -nx(i)*ny(i), -nx(i)*nz(i);
				J_nu.row(1) << -nx(i)*ny(i), square(nx(i)) + square(nz(i)), -ny(i)*nz(i);
				J_nu.row(2) << -nx(i)*nz(i), -ny(i)*nz(i), square(nx(i)) + square(ny(i));
				J_nu *= inv_norm*square(inv_norm);
				J_nX.assign(0.f);
				Matrix<float, 1, 3> res_n; res_n << res_nx(i), res_ny(i), res_nz(i);
				Matrix<float, 1, 3> J_mult_norm = -2.f*Kn*n_weights(i)*res_n*J_nu;

				//Control vertices
				const unsigned int weights_col = i;
				for (unsigned int c = 0; c < max_num_w; c++)
				{
					const int cp = w_indices(c,weights_col);
					if (cp >= 0)
					{
						const float wu1 = w_u1(c, weights_col), wu2 = w_u2(c, weights_col);
						J_nX(0,1) = wu1*u2_der(i)[2] - wu2*u1_der(i)[2];
						J_nX(0,2) = wu2*u1_der(i)[1] - wu1*u2_der(i)[1];
						J_nX(1,2) = wu1*u2_der(i)[0] - wu2*u1_der(i)[0];
						J_nX(1,0) = -J_nX(0,1);
						J_nX(2,0) = -J_nX(0,2);
						J_nX(2,1) = -J_nX(1,2);

						vert_incrs(0, cp) += (J_mult_norm*J_nX)(0);
						vert_incrs(1, cp) += (J_mult_norm*J_nX)(1);
						vert_incrs(2, cp) += (J_mult_norm*J_nX)(2);
					}
				}
			}
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
			u1 = u1_old_outer;
			u2 = u2_old_outer;
			uface = uface_old_outer;

			createTopologyRefiner();
			evaluateSubDivSurface();
			if (with_reg_normals) evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();
			optimizeUDataterm_LM();
			new_energy = computeEnergyNB();

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
			if (cont > 5)
			{					
				//Recover old variables
				vert_coords = vert_coords_old;
				energy_increasing = true;
				break;	
			}
		}

		const float runtime = clock.Tac();
		aver_runtime += runtime;

		//Visualize results
		showSubSurface(); 
		showMeshAndCorrespondences();

		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (energy_increasing)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void MeshSmoother::solveNB_LM_Joint()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts;
	for (unsigned int i = 0; i < num_points; i++)
	{
		J_rows += 6;
		J_cols += 2;
	}

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();

	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();

	computeTransCoordAndResiduals();
	new_energy = computeEnergyNB();


	utils::CTicTac clock; 
	printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if ((iter+1) % 5 == 0)
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyNB());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();


		//printf("\n Start to compute the Jacobian"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_points; i++)
		{						
			//Data alignment
			fillJacobianEpPixelJoint(i, cont);

			//Normal alignment
			fillJacobianEnPixelJoint(i, cont);										
		}

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)		fillJacobianRegNormals(cont);
		if (with_reg_normals_good)	fillJacobianRegNormalsGood(cont);
		if (with_reg_normals_4dir)	fillJacobianRegNormals4dir(cont);
		if (with_reg_edges)			fillJacobianRegEdges(cont);
		if (with_reg_ctf)			fillJacobianRegCTF(cont);
		if (with_reg_atraction)		fillJacobianRegAtraction(cont);

		//printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;

		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		//printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{

			JtJ_lm = JtJ_sparse;

			//cout << endl << "Diag: " << JtJ_lm.diagonal();

			for (unsigned int j=0; j<J_cols; j++)
			{
				const float lm_correction = (1.f + adap_mult)*JtJ_lm.coeffRef(j,j);
				if (lm_correction != 0.f)
					JtJ_lm.coeffRef(j,j) = lm_correction;
				else
				{
					JtJ_lm.insert(j,j) = 1.f;
					printf("\n Null value in the diagonal");
				}
				//printf("\n lm_correction = %f, adap_mult = %f", lm_correction, adap_mult);
			}


			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)
				printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success)
				printf("Solving failed");

			//printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_points; i++)
			{
				u1_incr(i) = increments(cont++);
				u2_incr(i) = increments(cont++);

				const float high_uincr = max(abs(u1_incr(i)), abs(u2_incr(i)));
				if (high_uincr > 10.f)		printf("\n warning high incr = %f.  point = %d", high_uincr, i);	

				//Saturate increments to avoid too many jumps between faces
				const float norm_incr = sqrtf(square(u1_incr(i)) + square(u2_incr(i)));
				if (norm_incr > max_incr)
				{
					u1_incr(i) *= max_incr/norm_incr;
					u2_incr(i) *= max_incr/norm_incr;
				}

				//Update variable
				const float u1_new = u1_old(i) + u1_incr(i);
				const float u2_new = u2_old(i) + u2_incr(i);

				if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
				{
					updateInternalPointCrossingEdges(i);
				}
				else
				{
					u1(i) = u1_new;
					u2(i) = u2_new;
					uface(i) = uface_old(i);
				}
			}

			//printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	


			new_energy = computeEnergyNB();

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
				searchBetterUDataterm();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				new_energy = computeEnergyNB();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					u1 = u1_old;
					u2 = u2_old;
					uface = uface_old;
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

		//Show the update
		showSubSurface(); 
		showMeshAndCorrespondences();

		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > 0.999f))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}



void MeshSmoother::fillJacobianEpPixel(unsigned int i, unsigned int &J_row)
{
	const float Kp_sqrtf = sqrtf(Kp);
	
	const float res = sqrtf(square(res_x(i)) + square(res_y(i)) + square(res_z(i)));
	if (res < truncated_res)
	{			
		//Control vertices
		const unsigned int weights_col = i;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices(c,weights_col);
			const float prod = -Kp_sqrtf;
			if (cp >= 0)
			{
				const float v_weight = w_contverts(c, weights_col);
				for (unsigned int l=0; l<3; l++)
					j_elem.push_back(Tri(J_row+l, 3*cp+l, prod*v_weight));
			}
		}

		//Fill the residuals
		R(J_row) = Kp_sqrtf*res_x(i);
		R(J_row+1) = Kp_sqrtf*res_y(i);
		R(J_row+2) = Kp_sqrtf*res_z(i);
		J_row += 3;
	}
	else
		J_row += 3;	
}

void MeshSmoother::fillJacobianEpPixelJoint(unsigned int i, unsigned int &J_row)
{
	const float Kp_sqrt = sqrtf(Kp);
	
	const float res = sqrtf(square(res_x(i)) + square(res_y(i)) + square(res_z(i)));
	if (res < truncated_res)
	{			
		//Control vertices
		const unsigned int weights_col = i;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices(c,weights_col);
			const float prod = -Kp_sqrt;
			if (cp >= 0)
			{
				const float v_weight = w_contverts(c, weights_col);
				for (unsigned int l=0; l<3; l++)
					j_elem.push_back(Tri(J_row+l, 3*cp+l, prod*v_weight));
			}
		}

		//Correspondence
		Matrix<float, 3, 2> u_der; 
		u_der << u1_der(i)[0], u2_der(i)[0], u1_der(i)[1], u2_der(i)[1], u1_der(i)[2], u2_der(i)[2];
		const Matrix<float, 3, 2> J_u = -Kp_sqrt*u_der;
		const unsigned int ind_bias = 3*num_verts;
		const unsigned int J_col = 2*(J_row/6);
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + J_col + k, J_u(l,k)));

		//Fill the residuals
		R(J_row) = Kp_sqrt*res_x(i);
		R(J_row+1) = Kp_sqrt*res_y(i);
		R(J_row+2) = Kp_sqrt*res_z(i);
		J_row += 3;
	}

	//Simplest (and probably bad) solution to the problem of underdetermined unknowns for the solver
	else
	{
		const unsigned int ind_bias = 3*num_verts; 
		const unsigned int J_col = 2*(J_row/6);
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + J_col + k, 10.001f));

		J_row += 3;		
	}
}

void MeshSmoother::fillJacobianEnPixel(unsigned int i, unsigned int &J_row)
{
	const float Kn_sqrt = sqrtf(Kn);
	const float wn_sqrt = sqrtf(n_weights(i));
	
	const float resn = sqrtf(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));
	if (resn < truncated_resn)
	{
		//Control vertices
		const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
		Matrix3f J_nu, J_nX;
		J_nu.row(0) << square(ny(i)) + square(nz(i)), -nx(i)*ny(i), -nx(i)*nz(i);
		J_nu.row(1) << -nx(i)*ny(i), square(nx(i)) + square(nz(i)), -ny(i)*nz(i);
		J_nu.row(2) << -nx(i)*nz(i), -ny(i)*nz(i), square(nx(i)) + square(ny(i));
		J_nu *= inv_norm*square(inv_norm);
		J_nX.assign(0.f);
		const Matrix3f J_mult_norm = -Kn_sqrt*wn_sqrt*J_nu;
						
		const unsigned int weights_col = i;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices(c,weights_col);
			if (cp >= 0)
			{
				//Normals
				const float wu1 = w_u1(c, weights_col), wu2 = w_u2(c, weights_col);
				J_nX(0,1) = wu1*u2_der(i)[2] - wu2*u1_der(i)[2];
				J_nX(0,2) = wu2*u1_der(i)[1] - wu1*u2_der(i)[1];
				J_nX(1,2) = wu1*u2_der(i)[0] - wu2*u1_der(i)[0];
				J_nX(1,0) = -J_nX(0,1);
				J_nX(2,0) = -J_nX(0,2);
				J_nX(2,1) = -J_nX(1,2);

				const Matrix3f J_norm_fit = J_mult_norm*J_nX;
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						j_elem.push_back(Tri(J_row+l, 3*cp+k, J_norm_fit(l,k)));
			}
		}

		//Fill the residuals
		R(J_row) = Kn_sqrt*wn_sqrt*res_nx(i);
		R(J_row+1) = Kn_sqrt*wn_sqrt*res_ny(i);
		R(J_row+2) = Kn_sqrt*wn_sqrt*res_nz(i);
		J_row += 3;
	}
	else
		J_row += 3;	
}

void MeshSmoother::fillJacobianEnPixelJoint(unsigned int i, unsigned int &J_row)
{
	const float Kn_sqrt = sqrtf(Kn);
	const float wn_sqrt = sqrtf(n_weights(i));
	
	const float resn = sqrtf(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));
	if (resn < truncated_resn)
	{
		//Control vertices
		const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
		Matrix3f J_nu, J_nX;
		J_nu.row(0) << square(ny(i)) + square(nz(i)), -nx(i)*ny(i), -nx(i)*nz(i);
		J_nu.row(1) << -nx(i)*ny(i), square(nx(i)) + square(nz(i)), -ny(i)*nz(i);
		J_nu.row(2) << -nx(i)*nz(i), -ny(i)*nz(i), square(nx(i)) + square(ny(i));
		J_nu *= inv_norm*square(inv_norm);
		J_nX.assign(0.f);
		const Matrix3f J_mult_norm = -Kn_sqrt*wn_sqrt*J_nu;
		
		//Control vertices
		const unsigned int weights_col = i;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices(c,weights_col);
			if (cp >= 0)
			{
				const float wu1 = w_u1(c, weights_col), wu2 = w_u2(c, weights_col);
				J_nX(0,1) = wu1*u2_der(i)[2] - wu2*u1_der(i)[2];
				J_nX(0,2) = wu2*u1_der(i)[1] - wu1*u2_der(i)[1];
				J_nX(1,2) = wu1*u2_der(i)[0] - wu2*u1_der(i)[0];
				J_nX(1,0) = -J_nX(0,1);
				J_nX(2,0) = -J_nX(0,2);
				J_nX(2,1) = -J_nX(1,2);

				const Matrix3f J_norm_fit = J_mult_norm*J_nX;
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						j_elem.push_back(Tri(J_row+l, 3*cp+k, J_norm_fit(l,k)));
			}
		}

		//Correspondence
		computeNormalDerivativesPixel(i);
		Matrix<float, 3, 2> n_der_u;
		n_der_u << n_der_u1(i)[0], n_der_u2(i)[0], n_der_u1(i)[1], n_der_u2(i)[1], n_der_u1(i)[2], n_der_u2(i)[2];
		const Matrix<float, 3, 2> J_u = -Kn_sqrt*wn_sqrt*J_nu*n_der_u;
		const unsigned int ind_bias = 3*num_verts; 
		const unsigned int J_col = 2*(J_row/6);
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + J_col + k, J_u(l,k)));

		//Fill the residuals
		R(J_row) = Kn_sqrt*wn_sqrt*res_nx(i);
		R(J_row+1) = Kn_sqrt*wn_sqrt*res_ny(i);
		R(J_row+2) = Kn_sqrt*wn_sqrt*res_nz(i);
		J_row += 3;
	}

	//Simplest (and probably bad) solution to the problem of underdetermined knowns for the solver
	else
	{
		//const unsigned int ind_bias = 3*num_verts + 6*num_images; 
		//const unsigned int J_col = 2*(J_row/6);
		//for (unsigned int k=0; k<2; k++)
		//	for (unsigned int l=0; l<3; l++)
		//		j_elem.push_back(Tri(J_row+l, ind_bias + J_col + k, 0.001f));	
		J_row += 3;		
	}
}

void MeshSmoother::fillJacobianRegNormals(unsigned int &J_row)
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

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0)	break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

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
					R(J_row) = Kr_sqrt*(nx_reg[f](s1for,s2) - nx_reg[f](s1,s2));
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2));
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2));
					J_row += 3;
				}

				if (s2for != s2)
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1 + s_reg*s2for;

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0)	break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here);

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
					R(J_row) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2));
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2));
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2));
					J_row += 3;
				}				
			}
		}
}

void MeshSmoother::fillJacobianRegNormalsGood(unsigned int &J_row)
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
				//Indices associated to the regularization
				const unsigned int s1for = min(s1+1, int(s_reg-1));
				const unsigned int s2for = min(s2+1, int(s_reg-1));

				if (s1for != s1)
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1for + s_reg*s2;
					const float dist = sqrtf(square(mx_reg[f](s1for,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1for,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1for,s2) - mz_reg[f](s1,s2)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;
						
						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist; 

						//Second part of the Jacobian: (n(s1for) - n(s1))*dif(norm(vec_s1for_s1))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1for,s2) - nx_reg[f](s1,s2),
														 ny_reg[f](s1for,s2) - ny_reg[f](s1,s2),
														 nz_reg[f](s1for,s2) - nz_reg[f](s1,s2);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1for,s2) - mx_reg[f](s1,s2),
										my_reg[f](s1for,s2) - my_reg[f](s1,s2),
										mz_reg[f](s1for,s2) - mz_reg[f](s1,s2);
						J_norm_dist /= dist;


						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 


						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
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
					const float dist = sqrtf(square(mx_reg[f](s1,s2for) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2for) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2for) - mz_reg[f](s1,s2)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0)	break;
						
						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist;

						//Second part of the Jacobian: (n(s2for) - n(s2))*dif(norm(vec_s2for_s2))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1,s2for) - nx_reg[f](s1,s2),
														 ny_reg[f](s1,s2for) - ny_reg[f](s1,s2),
														 nz_reg[f](s1,s2for) - nz_reg[f](s1,s2);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1,s2for) - mx_reg[f](s1,s2),
										my_reg[f](s1,s2for) - my_reg[f](s1,s2),
										mz_reg[f](s1,s2for) - mz_reg[f](s1,s2);
						J_norm_dist /= dist;

						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 

						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
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

void MeshSmoother::fillJacobianRegNormals4dir(unsigned int &J_row)
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

		//Include every equation into LM  - Number of new equations: 6*num_faces*(s_reg*s_reg + (s_reg-1)*(s_reg-1))
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{						
				//Indices associated to the regularization
				const unsigned int s1for = min(s1+1, int(s_reg-1));
				const unsigned int s2for = min(s2+1, int(s_reg-1));

				if (s1for != s1)
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1for + s_reg*s2;
					const float dist = sqrtf(square(mx_reg[f](s1for,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1for,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1for,s2) - mz_reg[f](s1,s2)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist; 

						//Second part of the Jacobian: (n(s1for) - n(s1))*dif(norm(vec_s1for_s1))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1for,s2) - nx_reg[f](s1,s2),
														 ny_reg[f](s1for,s2) - ny_reg[f](s1,s2),
														 nz_reg[f](s1for,s2) - nz_reg[f](s1,s2);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1for,s2) - mx_reg[f](s1,s2),
										my_reg[f](s1for,s2) - my_reg[f](s1,s2),
										mz_reg[f](s1for,s2) - mz_reg[f](s1,s2);
						J_norm_dist /= dist;

						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 


						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
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
					const float dist = sqrtf(square(mx_reg[f](s1,s2for) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2for) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2for) - mz_reg[f](s1,s2)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

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

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1 + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist;

						//Second part of the Jacobian: (n(s2for) - n(s2))*dif(norm(vec_s2for_s2))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1,s2for) - nx_reg[f](s1,s2),
														 ny_reg[f](s1,s2for) - ny_reg[f](s1,s2),
														 nz_reg[f](s1,s2for) - nz_reg[f](s1,s2);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1,s2for) - mx_reg[f](s1,s2),
										my_reg[f](s1,s2for) - my_reg[f](s1,s2),
										mz_reg[f](s1,s2for) - mz_reg[f](s1,s2);
						J_norm_dist /= dist;

						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 

						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
					}	

					//Fill the residuals
					R(J_row) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2))/dist;
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2))/dist;
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2))/dist;
					J_row += 3;
				}	

				//Include diagonals - (1,1) - (0,0)
				if ((s1for != s1)&&(s2for != s2))
				{
					const unsigned int weights_col = s1 + s_reg*s2;
					const unsigned int weights_col_for = s1for + s_reg*s2for;
					const float dist = sqrtf(square(mx_reg[f](s1for,s2for) - mx_reg[f](s1,s2)) + square(my_reg[f](s1for,s2for) - my_reg[f](s1,s2)) + square(mz_reg[f](s1for,s2for) - mz_reg[f](s1,s2)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

						if ((wu1_here == 0.f) && (wu2_here == 0.f) && (wu1_for == 0.f) && (wu2_for == 0.f))
							continue;

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2)[2] - wu2_here*u1_der_reg[f](s1,s2)[2];
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2)[1] - wu1_here*u2_der_reg[f](s1,s2)[1];
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2)[0] - wu2_here*u1_der_reg[f](s1,s2)[0];
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1for,s2for)[2] - wu2_for*u1_der_reg[f](s1for,s2for)[2];
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1for,s2for)[1] - wu1_for*u2_der_reg[f](s1for,s2for)[1];
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1for,s2for)[0] - wu2_for*u1_der_reg[f](s1for,s2for)[0];
						J_nX_forward(1,0) = -J_nX_forward(0,1);
						J_nX_forward(2,0) = -J_nX_forward(0,2);
						J_nX_forward(2,1) = -J_nX_forward(1,2);

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2for]*J_nX_forward - J_nu[s1 + s_reg*s2]*J_nX_here)/dist; 

						//Second part of the Jacobian: (n(s1for) - n(s1))*dif(norm(vec_s1for_s1))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1for,s2for) - nx_reg[f](s1,s2),
														 ny_reg[f](s1for,s2for) - ny_reg[f](s1,s2),
														 nz_reg[f](s1for,s2for) - nz_reg[f](s1,s2);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1for,s2for) - mx_reg[f](s1,s2),
										my_reg[f](s1for,s2for) - my_reg[f](s1,s2),
										mz_reg[f](s1for,s2for) - mz_reg[f](s1,s2);
						J_norm_dist /= dist;

						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 


						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
					}	

					//Fill the residuals
					R(J_row) = Kr_sqrt*(nx_reg[f](s1for,s2for) - nx_reg[f](s1,s2))/dist;
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1for,s2for) - ny_reg[f](s1,s2))/dist;
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1for,s2for) - nz_reg[f](s1,s2))/dist;
					J_row += 3;
				}

				//Include diagonals - (1,0) - (0,1)
				if ((s1for != s1)&&(s2for != s2))
				{
					const unsigned int weights_col = s1 + s_reg*s2for;
					const unsigned int weights_col_for = s1for + s_reg*s2;
					const float dist = sqrtf(square(mx_reg[f](s1for,s2) - mx_reg[f](s1,s2for)) + square(my_reg[f](s1for,s2) - my_reg[f](s1,s2for)) + square(mz_reg[f](s1for,s2) - mz_reg[f](s1,s2for)));

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;

						//Matrices of weights
						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

						if ((wu1_here == 0.f) && (wu2_here == 0.f) && (wu1_for == 0.f) && (wu2_for == 0.f))
							continue;

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](s1,s2for)[2] - wu2_here*u1_der_reg[f](s1,s2for)[2];
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](s1,s2for)[1] - wu1_here*u2_der_reg[f](s1,s2for)[1];
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](s1,s2for)[0] - wu2_here*u1_der_reg[f](s1,s2for)[0];
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](s1for,s2)[2] - wu2_for*u1_der_reg[f](s1for,s2)[2];
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](s1for,s2)[1] - wu1_for*u2_der_reg[f](s1for,s2)[1];
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](s1for,s2)[0] - wu2_for*u1_der_reg[f](s1for,s2)[0];
						J_nX_forward(1,0) = -J_nX_forward(0,1);
						J_nX_forward(2,0) = -J_nX_forward(0,2);
						J_nX_forward(2,1) = -J_nX_forward(1,2);

						const Matrix3f J_reg = Kr_sqrt*(J_nu[s1for + s_reg*s2]*J_nX_forward - J_nu[s1 + s_reg*s2for]*J_nX_here)/dist; 

						//Second part of the Jacobian: (n(s1for) - n(s1))*dif(norm(vec_s1for_s1))/dX
						Vector3f prev_coef; prev_coef << nx_reg[f](s1for,s2) - nx_reg[f](s1,s2for),
														 ny_reg[f](s1for,s2) - ny_reg[f](s1,s2for),
														 nz_reg[f](s1for,s2) - nz_reg[f](s1,s2for);
						prev_coef /= square(dist);

						Matrix<float, 1, 3> J_norm_dist; 
						J_norm_dist << mx_reg[f](s1for,s2) - mx_reg[f](s1,s2for),
										my_reg[f](s1for,s2) - my_reg[f](s1,s2for),
										mz_reg[f](s1for,s2) - mz_reg[f](s1,s2for);
						J_norm_dist /= dist;

						const Matrix3f J_dif_coords = Matrix3f::Identity()*(w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col));
						const Matrix3f J_reg_extra = Kr_sqrt*(-prev_coef*J_norm_dist*J_dif_coords); 


						//Update increments
						j_elem.push_back(Tri(J_row, 3*cp, J_reg(0,0) + J_reg_extra(0,0)));
						j_elem.push_back(Tri(J_row, 3*cp+1, J_reg(0,1) + J_reg_extra(0,1)));
						j_elem.push_back(Tri(J_row, 3*cp+2, J_reg(0,2) + J_reg_extra(0,2)));
						j_elem.push_back(Tri(J_row+1, 3*cp, J_reg(1,0) + J_reg_extra(1,0)));
						j_elem.push_back(Tri(J_row+1, 3*cp+1, J_reg(1,1) + J_reg_extra(1,1)));
						j_elem.push_back(Tri(J_row+1, 3*cp+2, J_reg(1,2) + J_reg_extra(1,2)));
						j_elem.push_back(Tri(J_row+2, 3*cp, J_reg(2,0) + J_reg_extra(2,0)));
						j_elem.push_back(Tri(J_row+2, 3*cp+1, J_reg(2,1) + J_reg_extra(2,1)));
						j_elem.push_back(Tri(J_row+2, 3*cp+2, J_reg(2,2) + J_reg_extra(2,2)));
					}	

					//Fill the residuals
					R(J_row) = Kr_sqrt*(nx_reg[f](s1for,s2) - nx_reg[f](s1,s2for))/dist;
					R(J_row+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2for))/dist;
					R(J_row+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2for))/dist;
					J_row += 3;
				}
			}
		}
}

void MeshSmoother::fillJacobianRegEdges(unsigned int &J_row)
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

void MeshSmoother::fillJacobianRegAtraction(unsigned int &J_row)
{
	K_atrac = K_atrac_total;
	const float K_atrac_sqrt = sqrtf(K_atrac);
	const float sqrt_2 = sqrtf(2.f);
	
	for (unsigned int f=0; f<num_faces; f++)
		for (unsigned int e=0; e<4; e++)		//They seem to be always stored in order (I have checked it at least for some random faces)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);

			const Vector3f edge = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length = sqrtf(edge.sumAll());

			const Vector3f J_vert_e0 = K_atrac_sqrt*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length;
			const Vector3f J_vert_e1 = -J_vert_e0;

			j_elem.push_back(Tri(J_row, 3*vert_e0, J_vert_e0(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e1, J_vert_e1(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+1, J_vert_e0(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+1, J_vert_e1(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+2, J_vert_e0(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+2, J_vert_e1(2)));


			//Fill the residual
			R(J_row) = K_atrac_sqrt*length;
			J_row++;
		}	
}

void MeshSmoother::fillJacobianRegCTF(unsigned int &J_row)
{
	K_ctf = K_ctf_total/num_verts;
	const float K_ctf_sqrt = sqrtf(K_ctf);
	
	for (unsigned int cv=0; cv<num_verts; cv++)
		for (unsigned int k=0; k<3; k++)
		{
			j_elem.push_back(Tri(J_row, 3*cv+k, K_ctf_sqrt));
			R(J_row) = K_ctf_sqrt*(vert_coords(k,cv) - vert_coords_reg(k,cv));
			J_row++;
		}
}


void MeshSmoother::evaluateSubDivSurfacePixel(unsigned int i)
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	float pWeights[max_num_w], dsWeights[max_num_w], dtWeights[max_num_w];

	// Locate the patch corresponding to the face ptex idx and (s,t)
	Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface(i), u1(i), u2(i)); assert(handle);

	// Evaluate the patch weights, identify the CVs and compute the limit frame:
	patchTable->EvaluateBasis(*handle, u1(i), u2(i), pWeights, dsWeights, dtWeights);

	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

	LimitFrame eval; eval.Clear();
	for (int cv = 0; cv < cvs.size(); ++cv)
		eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

	//Save the 3D coordinates
	mx(i) = eval.point[0];
	my(i) = eval.point[1];
	mz(i) = eval.point[2];

	//Save the derivatives
	u1_der(i)[0] = eval.deriv1[0];
	u1_der(i)[1] = eval.deriv1[1];
	u1_der(i)[2] = eval.deriv1[2];
	u2_der(i)[0] = eval.deriv2[0];
	u2_der(i)[1] = eval.deriv2[1];
	u2_der(i)[2] = eval.deriv2[2];

	//Compute the normals
	nx(i) = eval.deriv1[1] * eval.deriv2[2] - eval.deriv1[2] * eval.deriv2[1];
	ny(i) = eval.deriv1[2] * eval.deriv2[0] - eval.deriv1[0] * eval.deriv2[2];
	nz(i) = eval.deriv1[0] * eval.deriv2[1] - eval.deriv1[1] * eval.deriv2[0];
}

void MeshSmoother::optimizeUDataterm_LM()
{
	unsigned int cont = 0;
	unsigned int inner_cont;
	float aver_lambda = 0.f;
	float aver_num_iters = 0.f;
	float aver_first_uincr = 0.f;
	
	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float limit_uincr = 0.05f*sqrtf(num_faces);
	const float lambda_limit = 1000.f;
	const float u_incr_min = 0.005f;
	float norm_uincr;
	float lambda_mult = 3.f;
	const float Kp_sqrt = sqrtf(Kp);
	const float Kn_sqrt = sqrtf(Kn);

	//Solve with LM
	for (unsigned int i = 0; i < num_points; i++)
	{

		energy = Kp*(square(res_x(i)) + square(res_y(i)) + square(res_z(i)))
					+ Kn*(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));
		energy_ratio = 2.f;
		norm_uincr = 1.f;
		lambda = 0.1f;
		inner_cont = 0;

		while (energy_ratio > 1.0005f)
		{
			u1_old(i) = u1(i);
			u2_old(i) = u2(i);
			uface_old(i) = uface(i);
			energy_old = energy;

			//Re-evaluate the normal derivatives
			computeNormalDerivativesPixel(i);

			//Fill the Jacobian with the gradients with respect to the internal points
			Matrix<float, 6, 2> J;
			Matrix<float, 3, 2> u_der; 
			u_der << u1_der(i)[0], u2_der(i)[0], u1_der(i)[1], u2_der(i)[1], u1_der(i)[2], u2_der(i)[2];
			J.topRows(3) = -Kp_sqrt*u_der;

			const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
			Matrix<float, 3, 2> n_der_u;
			n_der_u << n_der_u1(i)[0], n_der_u2(i)[0], n_der_u1(i)[1], n_der_u2(i)[1], n_der_u1(i)[2], n_der_u2(i)[2];
			Matrix3f J_nu;
			J_nu.row(0) << square(ny(i)) + square(nz(i)), -nx(i)*ny(i), -nx(i)*nz(i);
			J_nu.row(1) << -nx(i)*ny(i), square(nx(i)) + square(nz(i)), -ny(i)*nz(i);
			J_nu.row(2) << -nx(i)*nz(i), -ny(i)*nz(i), square(nx(i)) + square(ny(i));
			J_nu *= inv_norm*square(inv_norm);
			J.bottomRows(3) = -Kn_sqrt*J_nu*n_der_u;


			Matrix2f JtJ; JtJ.multiply_AtA(J);
			Matrix<float, 6, 1> R; R << Kp_sqrt*res_x(i), Kp_sqrt*res_y(i), Kp_sqrt*res_z(i), 
										Kn_sqrt*res_nx(i), Kn_sqrt*res_ny(i), Kn_sqrt*res_nz(i);
			Vector2f b = -J.transpose()*R;

			bool energy_increasing = true;

			while (energy_increasing)
			{
				//Solve with LM
				Matrix2f LM = JtJ;
				LM.diagonal() += lambda*LM.diagonal();
				//LM(0,0) *= (1.f + lambda);
				//LM(1,1) *= (1.f + lambda);
				//Vector2f sol = LM.ldlt().solve(b);
				Vector2f sol = LM.inverse()*b;

				u1_incr(i) = sol(0);
				u2_incr(i) = sol(1);
				norm_uincr = sqrtf(square(sol(0)) + square(sol(1)));

				//Statistics --------------------------------------
				inner_cont++;
				if (inner_cont == 1)
					aver_first_uincr += norm_uincr;
				//-------------------------------------------------

				if (norm_uincr > limit_uincr)
				{
					lambda *= lambda_mult;
					continue;
				}

				//Update variable
				const float u1_new = u1_old(i) + u1_incr(i);
				const float u2_new = u2_old(i) + u2_incr(i);
				if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
					updateInternalPointCrossingEdges(i);
				else
				{
					u1(i) = u1_new;
					u2(i) = u2_new;
					uface(i) = uface_old(i);
				}

				//Re-evaluate the mesh with the new parametric coordinates
				evaluateSubDivSurfacePixel(i);

				//Compute residuals
				res_x(i) = data_coords(0,i) - mx(i);
				res_y(i) = data_coords(1,i) - my(i);
				res_z(i) = data_coords(2,i) - mz(i);

				const float inv_norm = 1.f/sqrtf(square(nx(i)) + square(ny(i)) + square(nz(i)));
				res_nx(i) = data_normals(0,i) - inv_norm*nx(i);
				res_ny(i) = data_normals(1,i) - inv_norm*ny(i);
				res_nz(i) = data_normals(2,i) - inv_norm*nz(i);

				//Compute the energy associated to this pixel
				energy = Kp*(square(res_x(i)) + square(res_y(i)) + square(res_z(i)))
						+ Kn*(square(res_nx(i)) + square(res_ny(i)) + square(res_nz(i)));

				//Increase lambda
				if (energy > energy_old)
				{
					lambda *= lambda_mult;
					//printf("\n Energy is higher than before. Lambda = %f, energy = %f, u_incr = %f", lambda, energy, norm_uincr);
				}
				else
				{
					energy_increasing = false;
					lambda /= lambda_mult;
					//printf("\n Energy is lower than before");
				}

				//Keep the last solution and finish
				if ((lambda > lambda_limit)||(norm_uincr < u_incr_min))
				{
					u1(i) = u1_old(i);
					u2(i) = u2_old(i);
					uface(i) = uface_old(i);
					energy_increasing = false;
					energy = energy_old;
				}

				//if (inner_cont > 20)
				//	printf("\n energy = %f, energy_old = %f, cont = %d, energy_ratio = %f, norm_uincr = %f", 
				//			energy, energy_old, inner_cont, energy_ratio, norm_uincr);
			}

			energy_ratio = energy_old / energy;
		}

		//printf("\n Num_iters = %d, lambda = %f, energy_ratio = %f, u_incr = %f", inner_cont, lambda, energy_ratio, norm_uincr);

		//Statistics
		aver_lambda += lambda;
		aver_num_iters += float(inner_cont);
		cont++;
	}

	//printf("\n **** Average lambda = %f, aver_iter_per_pixel = %f, aver_first_uincr = %f ****", aver_lambda/cont, aver_num_iters/cont, aver_first_uincr/cont);
}

void MeshSmoother::vertIncrRegularizationNormals()
{
	Matrix3f J_nX; J_nX.assign(0.f);
	Kr = Kr_total/float(num_faces*square(s_reg));
	
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

void MeshSmoother::vertIncrRegularizationEdges()
{
	Ke = Ke_total*float(num_faces);
	const float sqrt_2 = sqrtf(2.f);
	
	for (unsigned int f=0; f<num_faces; f++)
	{
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
			const float rL = length_0 - length_1;

			const float mult = 2.f*Ke*rL;

			vert_incrs.col(vert_e0) += mult*sqrt_2*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length_0;
			vert_incrs.col(vert_e1) += mult*(sqrt_2*(vert_coords.col(vert_e1) - vert_coords.col(vert_e0))/length_0
									  	    +(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1);
			vert_incrs.col(vert_e2) += -mult*(vert_coords.col(vert_e2) - vert_coords.col(vert_e1))/length_1;
		}
	}
}

