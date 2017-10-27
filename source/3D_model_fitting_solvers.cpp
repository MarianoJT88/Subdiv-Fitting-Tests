// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"


float Mod3DfromRGBD::computeEnergySK()
{
	float energy_p = 0.f, energy_n = 0.f, energy_b = 0.f, energy_r = 0.f;
	const float const_concave_back = 1.f - eps_rel; //(1.f - eps/tau);

	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if (is_object[i](v,u))
				{
					const float res = res_pos[i].col(v+u*rows).norm();
					energy_p += Kp*square(min(res, truncated_res));

					if (fit_normals_old)
					{
						const float resn = res_normals[i].col(v+rows*u).norm();
						energy_n += Kn*n_weights[i](v+rows*u)*square(min(resn, truncated_resn));
					}

				}
				else if (valid[i](v,u))
				{
					const float res_d_squared = res_pixels[i].col(v+rows*u).squaredNorm();
					const float tau_here = tau_pixel[i](v,u);
					const float tau_here_squared = square(tau_here);
					
					//2 parabolas with peak	
					if ((res_d_squared < tau_here_squared)&&(res_d_squared > square(eps_rel*tau_here)))
						energy_b += alpha*square(1.f - sqrtf(res_d_squared)/tau_here);

					else if (res_d_squared < square(eps_rel*tau_here))	
						energy_b += alpha*const_concave_back*(1.f - res_d_squared/(eps_rel*tau_here_squared));
				}
			}
	}

	//Regularization
	if (with_reg_normals)			energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)		energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)		energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)				energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)				energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)			energy_r += computeEnergyRegAtraction();
	if (with_reg_edges_iniShape)	energy_r += computeEnergyRegEdgesIniShape();
	if (with_reg_arap)				energy_r += computeEnergyRegArap();
	if (with_reg_rot_arap)			energy_r += computeEnergyRegRotArap();

	const float energy_o = energy_p + energy_n + energy_r + energy_b;
	//printf("\n Energies: overall = %.4f, dterm(p) = %.4f, dterm(n) = %.4f, reg = %.4f, backg = %.4f", energy_o, energy_p, energy_n, energy_r, energy_b);

	return energy_o;
}

float Mod3DfromRGBD::computeEnergyNB()
{
	float energy_d = 0.f, energy_r = 0.f;

	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = res_pos[i].col(v+rows*u).norm(); 
					energy_d += Kp*square(min(res, truncated_res));

					if (fit_normals_old)
					{
						const float resn = res_normals[i].col(v+rows*u).norm();
						energy_d += Kn*n_weights[i](v+rows*u)*square(min(resn, truncated_resn));
					}
				}
	}

	//Regularization
	if (with_reg_normals)			energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)		energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)		energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)				energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)				energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)			energy_r += computeEnergyRegAtraction();
	if (with_reg_edges_iniShape)	energy_r += computeEnergyRegEdgesIniShape();
	if (with_reg_arap)				energy_r += computeEnergyRegArap();
	if (with_reg_rot_arap)			energy_r += computeEnergyRegRotArap();

	const float energy_o = energy_d + energy_r;
	//printf("\n Energies: overall = %.4f, dataterm = %.4f, reg = %.4f", energy_o, energy_d, energy_r);

	return energy_o;
}

float Mod3DfromRGBD::computeEnergyRegNormals()
{
	Kr = Kr_total/float(square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Refs
		const MatrixXf &n_reg_f = normals_reg[f];
		
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const int c_ind = s1+s_reg*s2;
				energy += Kr*((square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind)) + square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)))
							+(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind))));
			}

		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const int c_ind = s1+s_reg*s2;
			energy += Kr*(square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)));
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const int c_ind = s1+s_reg*s2;
			energy += Kr*(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind)));
		}
	}


	return energy;
}

float Mod3DfromRGBD::computeEnergyRegNormalsGood()
{
	Kr = Kr_total/float(num_faces*square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Refs
		const MatrixXf &n_reg_f = normals_reg[f];
		const MatrixXf &surf_reg_f = surf_reg[f];
		
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const int c_ind = s1+s_reg*s2;
				const float dist_s1 = square(surf_reg_f(0,c_ind+1) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+1) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+1) - surf_reg_f(2,c_ind));
				const float dist_s2 = square(surf_reg_f(0,c_ind+s_reg) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+s_reg) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+s_reg) - surf_reg_f(2,c_ind));

				energy += Kr*((square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind)) + square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)))/dist_s1
							+(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind)))/dist_s2);
			}

		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const int c_ind = s1+s_reg*s2;
			const float dist_s1 = square(surf_reg_f(0,c_ind+1) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+1) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+1) - surf_reg_f(2,c_ind));
			energy += Kr*(square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)))/dist_s1;
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const int c_ind = s1+s_reg*s2;
			const float dist_s2 = square(surf_reg_f(0,c_ind+s_reg) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+s_reg) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+s_reg) - surf_reg_f(2,c_ind));
			energy += Kr*(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind)))/dist_s2;	
		}
	}

	return energy;
}

float Mod3DfromRGBD::computeEnergyRegNormals4dir()
{
	Kr = Kr_total/float(num_faces*square(s_reg));
	float energy = 0.f;
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Refs
		const MatrixXf &n_reg_f = normals_reg[f];	
		const MatrixXf &surf_reg_f = surf_reg[f];
		
		//Middle points
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const int c_ind = s1+s_reg*s2;
				const float dist_s1 = square(surf_reg_f(0,c_ind+1) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+1) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+1) - surf_reg_f(2,c_ind));
				const float dist_s2 = square(surf_reg_f(0,c_ind+s_reg) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+s_reg) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+s_reg) - surf_reg_f(2,c_ind));

				energy += Kr*((square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind)) + square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)))/dist_s1
							+(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind)))/dist_s2);
			}
		//Boundaries
		const float s2 = s_reg-1;
		for (unsigned int s1=0; s1<s_reg-1; s1++)
		{
			const int c_ind = s1+s_reg*s2;
			const float dist_s1 = square(surf_reg_f(0,c_ind+1) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+1) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+1) - surf_reg_f(2,c_ind));
			energy += Kr*(square(n_reg_f(0,c_ind+1) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+1) - n_reg_f(2,c_ind)))/dist_s1;
		}
					
		const float s1 = s_reg-1;
		for (unsigned int s2=0; s2<s_reg-1; s2++)
		{
			const int c_ind = s1+s_reg*s2;
			const float dist_s2 = square(surf_reg_f(0,c_ind+s_reg) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+s_reg) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+s_reg) - surf_reg_f(2,c_ind));
			energy += Kr*(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind)))/dist_s2;
		}

		//Diagonals
		for (unsigned int s2=0; s2<s_reg-1; s2++)
			for (unsigned int s1=0; s1<s_reg-1; s1++)
			{
				const int c_ind = s1+s_reg*s2;
				const float dist_s1 = square(surf_reg_f(0,c_ind+1+s_reg) - surf_reg_f(0,c_ind)) + square(surf_reg_f(1,c_ind+1+s_reg) - surf_reg_f(1,c_ind)) + square(surf_reg_f(2,c_ind+1+s_reg) - surf_reg_f(2,c_ind));
				const float dist_s2 = square(surf_reg_f(0,c_ind+s_reg) - surf_reg_f(0,c_ind+1)) + square(surf_reg_f(1,c_ind+s_reg) - surf_reg_f(1,c_ind+1)) + square(surf_reg_f(2,c_ind+s_reg) - surf_reg_f(2,c_ind+1));

				energy += Kr*((square(n_reg_f(0,c_ind+1+s_reg) - n_reg_f(0,c_ind)) + square(n_reg_f(1,c_ind+1+s_reg) - n_reg_f(1,c_ind)) + square(n_reg_f(2,c_ind+1+s_reg) - n_reg_f(2,c_ind)))/dist_s1
							+(square(n_reg_f(0,c_ind+s_reg) - n_reg_f(0,c_ind+1)) + square(n_reg_f(1,c_ind+s_reg) - n_reg_f(1,c_ind+1))	+ square(n_reg_f(2,c_ind+s_reg) - n_reg_f(2,c_ind+1)))/dist_s2);
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

float Mod3DfromRGBD::computeEnergyRegEdgesIniShape()
{
	float energy = 0.f;
	const float sqrt_2 = sqrtf(2.f);
	K_ini = K_ini_total*float(num_faces);
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Edges	
		for (unsigned int e=0; e<4; e++)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);

			const Vector3f edge = (vert_coords.col(vert_e1) - vert_coords.col(vert_e0)).square().matrix();
			const float length = sqrtf(edge.sumAll());
			const Vector3f edge_ini = (vert_coords_reg.col(vert_e1) - vert_coords_reg.col(vert_e0)).square().matrix();
			const float length_ini = sqrtf(edge_ini.sumAll());

			energy += K_ini*square(length - length_ini);
		}

		//opposite vertices of adjacent faces
		for (unsigned int k=0; k<4; k++)
		{							
			const unsigned int fadj = face_adj(k,f);
			const unsigned int vert_e0 = opposite_verts(0+4*k,f);
			const unsigned int vert_e1 = opposite_verts(1+4*k,f);
			const unsigned int vert_e2 = opposite_verts(2+4*k,f);
			const unsigned int vert_e3 = opposite_verts(3+4*k,f);

			//if ((vert_e2 > 2000)&&(f < 1000))
			//{
			//	printf("\n v0 = %d, v1 = %d, v2 = %d, v3 = %d", vert_e0, vert_e1, vert_e2, vert_e3);
			//	printf("\n face = %d, verts: %d %d %d %d",f, face_verts(0,f), face_verts(1,f), face_verts(2,f), face_verts(3,f));
			//	printf("\n face_adj = %d, verts: %d %d %d %d", fadj, face_verts(0,fadj), face_verts(1,fadj), face_verts(2,fadj), face_verts(3,fadj));
			//}

			const Vector3f edge = (0.5f*(vert_coords.col(vert_e2) + vert_coords.col(vert_e3))
								 - 0.5f*(vert_coords.col(vert_e0) + vert_coords.col(vert_e1))).square().matrix();
			//const float length = sqrtf(edge.sumAll());
			//const Vector3f edge_ini = (0.5f*(vert_coords_reg.col(vert_e2) + vert_coords_reg.col(vert_e3))
			//						 - 0.5f*(vert_coords_reg.col(vert_e0) + vert_coords_reg.col(vert_e1))).square().matrix();
			//const float length_ini = sqrtf(edge_ini.sumAll());

			//energy += K_ini*square(length - length_ini);
		}
	}

	return energy;
}

float Mod3DfromRGBD::computeEnergyRegCTF()
{
	float energy = 0.f;
	K_ctf = K_ctf_total/num_verts;
	
	for (unsigned int cv=0; cv<num_verts; cv++)
		for (unsigned int k=0; k<3; k++)
			energy += K_ctf*square(vert_coords(k,cv) - vert_coords_reg(k,cv));

	return energy;
}

float Mod3DfromRGBD::computeEnergyRegAtraction()
{
	float energy = 0.f;
	const float sqrt_2 = sqrtf(2.f);
	K_atrac = K_atrac_total; //*float(num_faces);
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

float Mod3DfromRGBD::computeEnergyRegVertColor()
{
	float energy = 0.f;
	const float sqrt_2 = sqrtf(2.f);
	for (unsigned int f=0; f<num_faces; f++)
		for (unsigned int e=0; e<4; e++)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);

			const float color_dif = vert_colors(vert_e1) - vert_colors(vert_e0);
			energy += K_color_reg*square(color_dif);
		}

	return energy;
}


void Mod3DfromRGBD::solveSK_LM()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (is_object[i](v,u))			J_rows += 6;
				else if (valid[i](v,u))			J_rows++;
			}

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	optimizeUDataterm_LM();
	optimizeUBackground_LM();
	new_energy = computeEnergySK();

	if (paper_visualization) takePictureLimitSurface(false);

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
		if (((iter+1) % 5 == 0)&&(ctf_level > 2))
		{
			searchBetterUDataterm();
			searchBetterUBackground();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergySK());
		}

		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;


		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		printf("\n Start to compute the Jacobian"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
					
						//Foreground
						if (is_object[i](v,u))
						{				
							//Data alignment
							fill_J_EpPixel(i, v, u, cont);

							//Normal alignment
							fill_J_EnPixel(i, v, u, cont);										
						}

						//Background
						else
						{
							fill_J_BackSKPixel(i, v, u, cont);
						}
					}
		}

		printf("\n Fill J (dataterm + background term) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)			fill_J_RegNormals(cont);
		if (with_reg_normals_good)		fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)		fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)				fill_J_RegEdges(cont);
		if (with_reg_ctf)				fill_J_RegCTF(cont);
		if (with_reg_edges_iniShape)	fill_J_RegEdgesIniShape(cont);

		printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;

		MatrixXf JtJ = MatrixXf(JtJ_sparse);
		MatrixXf JtJ_lm;

		//Tests
		//SparseMatrix<float, RowMajor> J_here(J_rows, J_cols);
		//J_here.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		//SparseMatrix<float> JtJ_sparse = J_here.transpose()*J_here;
		//VectorXf b = -J_here.transpose()*R;

		//MatrixXf J_dense = J.toDense();
		//MatrixXf JtJ; JtJ.multiply_AtA(J_dense);
		//VectorXf b; b.multiply_AtB(J_dense,R);
		//b = -b;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{

			JtJ_lm = JtJ;
			//JtJ_lm.diagonal() += adap_mult*JtJ_lm.diagonal();					//Levenberg-Marquardt
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);


			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);

			printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			printf("\n Updates variables - %f sec", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	printf("\n Create topology, evaluate the surface and computes residual - %f sec", clock.Tac()); clock.Tic();

			if (behind_cameras)
			{
				adap_mult *= 4.f;
				behind_cameras = false;
				continue;
			}

			optimizeUBackground_LM();	printf("\n Solve raycasting background - %f sec", clock.Tac()); clock.Tic();
			optimizeUDataterm_LM();		printf("\n Solve closest correspondence foreground - %f sec", clock.Tac()); clock.Tic();
			new_energy = computeEnergySK();


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
				searchBetterUBackground();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				optimizeUBackground_LM();
				optimizeUDataterm_LM();
				new_energy = computeEnergySK();

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

		//Show the update
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureLimitSurface(false);

		printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if (energy_increasing)
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			break;
		}
	}

	//printf("\n Average runtime = %f", aver_runtime / max_iter);
}

void Mod3DfromRGBD::solveNB_LM_Joint(bool verbose)
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
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
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;
	if (fix_first_camera)		J_rows += 6;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();

	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();

	computeTransCoordAndResiduals();
	new_energy = computeEnergyNB();

	if (paper_visualization)	takePictureLimitSurface(paper_vis_no_mesh);
	if (save_energy)			saveCurrentEnergyInFile(true);

	utils::CTicTac clock; 
	if (verbose) printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(num_faces > 10))
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyNB());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();


		if (verbose) {printf("\n Start to compute the Jacobian"); clock.Tic();}

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{

			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
						{
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							return;
						}
							
						//Data alignment
						fill_J_EpPixelJoint(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixelJoint(i, v, u, cont);										
					}
		}

		//Contrain motion of the 1st camera if necessary
		if (fix_first_camera) 	fill_J_fixFirstCamera(cont);

		if (verbose) {printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();}

		//Include regularization
		if (with_reg_normals)		fill_J_RegNormals(cont);
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)			fill_J_RegEdges(cont);
		if (with_reg_ctf)			fill_J_RegCTF(cont);
		if (with_reg_atraction)		fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape) fill_J_RegEdgesIniShape(cont);

		if (verbose) {printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();}

		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;

		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		if (verbose) {printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();}


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
					printf("\n Null value in the diagonal (unknown %d)", j);
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

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_images; i++)
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (is_object[i](v,u))
						{
							u1_incr[i](v,u) = increments(cont++);
							u2_incr[i](v,u) = increments(cont++);

							const float high_uincr = max(abs(u1_incr[i](v,u)), abs(u2_incr[i](v,u)));
							if (high_uincr > 10.f)		printf("\n warning high incr = %f.  image = %d, v = %d, u = %d", high_uincr, i, v, u);	

							//Saturate increments to avoid too many jumps between faces
							const float norm_incr = sqrtf(square(u1_incr[i](v,u)) + square(u2_incr[i](v,u)));
							if (norm_incr > max_incr)
							{
								u1_incr[i](v,u) *= max_incr/norm_incr;
								u2_incr[i](v,u) *= max_incr/norm_incr;
							}

							//Update variable
							const float u1_new = u1_old[i](v,u) + u1_incr[i](v,u);
							const float u2_new = u2_old[i](v,u) + u2_incr[i](v,u);

							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							{
								updateInternalPointCrossingEdges(i, v, u);
							}
							else
							{
								u1[i](v, u) = u1_new;
								u2[i](v, u) = u2_new;
								uface[i](v, u) = uface_old[i](v, u);
							}
						}

			if (verbose) {printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();}

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue; }

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
					cam_mfold = cam_mfold_old;
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
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureLimitSurface(paper_vis_no_mesh);

		//Save energy to file
		if (save_energy)	saveCurrentEnergyInFile(true);


		//Print info about energy and check convergence
		if (verbose) {printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();}
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > convergence_ratio))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}

void Mod3DfromRGBD::solveSK_LM_Joint(bool verbose)
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (is_object[i](v,u))	
				{
					J_rows += 6;
					J_cols += 2;
				}
				else if (valid[i](v,u))
					J_rows++;
			}

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;
	if (fix_first_camera)		J_rows += 6;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	optimizeUBackground_LM();
	new_energy = computeEnergySK();

	if (paper_visualization)	takePictureLimitSurface(paper_vis_no_mesh);
	if (save_energy)			saveCurrentEnergyInFile(true);

	utils::CTicTac clock; 
	if (verbose) printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(num_faces > 10))
		{
			searchBetterUDataterm();
			searchBetterUBackground();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergySK());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		u1_old_outer = u1_old;
		u2_old_outer = u2_old;
		uface_old_outer = uface_old;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		if (verbose)  {printf("\n Start to compute the Jacobian"); clock.Tic();}

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		//Foreground
		for (unsigned int i = 0; i < num_images; i++)
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
						{
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							return;
						}
					
						//Data alignment
						fill_J_EpPixelJoint(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixelJoint(i, v, u, cont);										

					}

		//Background
		for (unsigned int i = 0; i < num_images; i++)
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u) && !is_object[i](v,u))
						fill_J_BackSKPixel(i, v, u, cont);

		//Contrain motion of the 1st camera if necessary
		if (fix_first_camera) 	fill_J_fixFirstCamera(cont);

		if (verbose) {printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();}

		//Include regularization
		if (with_reg_normals)		fill_J_RegNormals(cont);
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)			fill_J_RegEdges(cont);
		if (with_reg_ctf)			fill_J_RegCTF(cont);
		if (with_reg_atraction)		fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape) fill_J_RegEdgesIniShape(cont);

		if (verbose) {printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();}

		//Analyzing the triplets
		//unsigned int num_zeros = 0;
		//for (unsigned int k=0; k<j_elem.size(); k++)
		//	if (j_elem[k].value() == 0.f)
		//		num_zeros++;
		//const float perct_zeros = 100*float(num_zeros)/float(j_elem.size());

		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()-1); j_elem.clear();

		//printf("\n Compute J from triplets - %f sec", clock.Tac()); clock.Tic();
		const SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		//JtJ_sparse.triangularView<Eigen::Upper>() = J.transpose()*J;
		
		//printf("\n Build JtJ - %f sec", clock.Tac()); clock.Tic();
		VectorXf b = -J.transpose()*R;
		//printf("\n Compute b - %f sec", clock.Tac()); clock.Tic();

		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		if (verbose) {printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();}


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
			if(solver.info()!=Success)			printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 			printf("Solving failed");

			if (verbose) {printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();}

			//if (iter == 0) 
			//{
				//printf("\n Debugging....");
				//cout << endl << "JtJ_lm: " << JtJ_lm;
				//cout << endl << "b: " << b.transpose();
				//cout << endl << "To avoid a 'Heisenbug' -> print increments" << increments.transpose();
			//}
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_images; i++)
			{
				//Refs
				ArrayXXf &u1_incr_ref = u1_incr[i];
				ArrayXXf &u2_incr_ref = u2_incr[i];
			
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (is_object[i](v,u))
						{
							u1_incr_ref(v,u) = increments(cont++);
							u2_incr_ref(v,u) = increments(cont++);

							//Saturate increments to avoid too many jumps between faces
							const float norm_incr = sqrtf(square(u1_incr_ref(v,u)) + square(u2_incr_ref(v,u)));
							if (norm_incr > max_incr)
							{
								u1_incr_ref(v,u) *= max_incr/norm_incr;
								u2_incr_ref(v,u) *= max_incr/norm_incr;
							}

							//Update variable
							const float u1_new = u1_old[i](v,u) + u1_incr_ref(v,u);
							const float u2_new = u2_old[i](v,u) + u2_incr_ref(v,u);

							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							{
								updateInternalPointCrossingEdges(i, v, u);
							}
							else
							{
								u1[i](v,u) = u1_new;
								u2[i](v,u) = u2_new;
								uface[i](v,u) = uface_old[i](v,u);
							}
						}
						else
						{
							u1[i](v,u) = u1_old_outer[i](v,u);
							u2[i](v,u) = u2_old_outer[i](v,u);
							uface[i](v,u) = uface_old_outer[i](v,u);
						}
			}

			if (verbose) {printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();}

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	
			
			if (verbose) {printf("\n Create topology, evaluate the surface and computes residual - %f sec", clock.Tac()); clock.Tic();}

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue;}

			optimizeUBackground_LM();
			new_energy = computeEnergySK();
			if (verbose) {printf("\n Optimize background and compute the energy - %f sec", clock.Tac()); clock.Tic();}

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
			if (cont_inner > 10) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
				searchBetterUBackground();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				optimizeUBackground_LM();
				new_energy = computeEnergySK();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
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
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureLimitSurface(paper_vis_no_mesh);

		//Save energy to file
		if (save_energy)	saveCurrentEnergyInFile(true);


		if (verbose) {printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();}
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > convergence_ratio))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}

void Mod3DfromRGBD::solveDT2_LM_Joint(bool verbose)
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForDTBackground();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += 6;
					J_cols += 2;
				}

	J_rows += num_images*nsamples; //Background DT

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;
	if (fix_first_camera)		J_rows += 6;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);


	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	new_energy = computeEnergyDT2();

	if (paper_visualization)	takePictureLimitSurface(paper_vis_no_mesh);
	if (save_energy)			saveCurrentEnergyInFile(true);

	utils::CTicTac clock; 
	if (verbose) printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(num_faces > 10))
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyDT2());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		u1_old_outer = u1_old;
		u2_old_outer = u2_old;
		uface_old_outer = uface_old;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		if (verbose) {printf("\n Start to compute the Jacobian"); clock.Tic();}

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		//Foreground
		for (unsigned int i = 0; i < num_images; i++)
		{
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
						{
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							return;
						}
					
						//Data alignment
						fill_J_EpPixelJoint(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixelJoint(i, v, u, cont);										
					}
		}

		//Background DT^2 - It must be done after the dataterm to avoid breaking the order of the elem of the jacobian (correspondences)
		for (unsigned int i = 0; i < num_images; i++)			
			fill_J_BackDT2(i, cont);

		//Contrain motion of the 1st camera if necessary
		if (fix_first_camera) 	fill_J_fixFirstCamera(cont);

		if (verbose) {printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();}

		//Include regularization
		if (with_reg_normals)		fill_J_RegNormals(cont);
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)			fill_J_RegEdges(cont);
		if (with_reg_ctf)			fill_J_RegCTF(cont);
		if (with_reg_atraction)		fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape) fill_J_RegEdgesIniShape(cont);

		if (verbose) {printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();}


		//Analyzing the triplets
		//unsigned int num_zeros = 0;
		//for (unsigned int k=0; k<j_elem.size(); k++)
		//	if (j_elem[k].value() == 0.f)
		//		num_zeros++;
		//const float perct_zeros = 100*float(num_zeros)/float(j_elem.size());



		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()-1); j_elem.clear();
		const SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;
		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		if (verbose) {printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();}


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			JtJ_lm = JtJ_sparse;
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
			}


			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)
				printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success)
				printf("Solving failed");

			if (verbose) {printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();}
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_images; i++)
			{
				//Refs
				ArrayXXf &u1_incr_ref = u1_incr[i];
				ArrayXXf &u2_incr_ref = u2_incr[i];
			
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (is_object[i](v,u))
						{
							u1_incr_ref(v,u) = increments(cont++);
							u2_incr_ref(v,u) = increments(cont++);

							//Saturate increments to avoid too many jumps between faces
							const float norm_incr = sqrtf(square(u1_incr_ref(v,u)) + square(u2_incr_ref(v,u)));
							if (norm_incr > max_incr)
							{
								u1_incr_ref(v,u) *= max_incr/norm_incr;
								u2_incr_ref(v,u) *= max_incr/norm_incr;
							}

							//Update variable
							const float u1_new = u1_old[i](v,u) + u1_incr_ref(v,u);
							const float u2_new = u2_old[i](v,u) + u2_incr_ref(v,u);

							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							{
								updateInternalPointCrossingEdges(i, v, u);
							}
							else
							{
								u1[i](v,u) = u1_new;
								u2[i](v,u) = u2_new;
								uface[i](v,u) = uface_old[i](v,u);
							}
						}
						else
						{
							u1[i](v,u) = u1_old_outer[i](v,u);
							u2[i](v,u) = u2_old_outer[i](v,u);
							uface[i](v,u) = uface_old_outer[i](v,u);
						}
			}

			if (verbose) {printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();}

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	
			
			if (verbose) {printf("\n Create topology, evaluate the surface and computes residual - %f sec", clock.Tac()); clock.Tic();}

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras)
			{
				adap_mult *= 4.f;
				behind_cameras = false;
				continue;
			}

			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDT2();
			if (verbose) {printf("\n Optimize background and compute the energy - %f sec", clock.Tac()); clock.Tic();}

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
			if (cont_inner > 10) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
				searchBetterUBackground();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				sampleSurfaceForDTBackground();
				new_energy = computeEnergyDT2();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
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
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureLimitSurface(paper_vis_no_mesh);

		//Save energy to file
		if (save_energy)	saveCurrentEnergyInFile(true);


		if (verbose) {printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();}
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > convergence_ratio))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}

void Mod3DfromRGBD::solveBS_LM_Joint(bool verbose)
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForBSTerm();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += 6;
					J_cols += 2;
				}

	J_rows += num_images*nsamples; //Background DT

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;
	if (fix_first_camera)		J_rows += 6;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);


	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	evaluateSurfaceForBSSamples();
	new_energy = computeEnergyBS();

	if (paper_visualization)	takePictureDataArch();  //takePictureLimitSurface(paper_vis_no_mesh);
	if (save_energy)			saveCurrentEnergyInFile(true);

	utils::CTicTac clock; 
	if (verbose) printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(num_faces > 10))
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyBS());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		u1_old_outer = u1_old;
		u2_old_outer = u2_old;
		uface_old_outer = uface_old;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		if (verbose) {printf("\n Start to compute the Jacobian"); clock.Tic();}

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		//Foreground
		for (unsigned int i = 0; i < num_images; i++)
		{
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
						{
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							return;
						}
					
						//Data alignment
						fill_J_EpPixelJoint(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixelJoint(i, v, u, cont);										
					}
		}

		//Background DT^2 - It must be done after the dataterm to avoid breaking the order of the elem of the jacobian (correspondences)
		for (unsigned int i = 0; i < num_images; i++)			
			fill_J_BackBS(i, cont);

		//Contrain motion of the 1st camera if necessary
		if (fix_first_camera) 	fill_J_fixFirstCamera(cont);

		if (verbose) {printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();}

		//Include regularization
		if (with_reg_normals)		fill_J_RegNormals(cont);
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)			fill_J_RegEdges(cont);
		if (with_reg_ctf)			fill_J_RegCTF(cont);
		if (with_reg_atraction)		fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape) fill_J_RegEdgesIniShape(cont);

		if (verbose) {printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();}


		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		J.setFromTriplets(j_elem.begin(), j_elem.end()-1); j_elem.clear();
		const SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;
		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		if (verbose) {printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();}


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			JtJ_lm = JtJ_sparse;
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
			}


			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)
				printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success)
				printf("Solving failed");

			if (verbose) {printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();}
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_images; i++)
			{
				//Refs
				ArrayXXf &u1_incr_ref = u1_incr[i];
				ArrayXXf &u2_incr_ref = u2_incr[i];
			
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (is_object[i](v,u))
						{
							u1_incr_ref(v,u) = increments(cont++);
							u2_incr_ref(v,u) = increments(cont++);

							//Saturate increments to avoid too many jumps between faces
							const float norm_incr = sqrtf(square(u1_incr_ref(v,u)) + square(u2_incr_ref(v,u)));
							if (norm_incr > max_incr)
							{
								u1_incr_ref(v,u) *= max_incr/norm_incr;
								u2_incr_ref(v,u) *= max_incr/norm_incr;
							}

							//Update variable
							const float u1_new = u1_old[i](v,u) + u1_incr_ref(v,u);
							const float u2_new = u2_old[i](v,u) + u2_incr_ref(v,u);

							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							{
								updateInternalPointCrossingEdges(i, v, u);
							}
							else
							{
								u1[i](v,u) = u1_new;
								u2[i](v,u) = u2_new;
								uface[i](v,u) = uface_old[i](v,u);
							}
						}
						else
						{
							u1[i](v,u) = u1_old_outer[i](v,u);
							u2[i](v,u) = u2_old_outer[i](v,u);
							uface[i](v,u) = uface_old_outer[i](v,u);
						}
			}

			if (verbose) {printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();}

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	
			
			if (verbose) {printf("\n Create topology, evaluate the surface and computes residual - %f sec", clock.Tac()); clock.Tic();}

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras)
			{
				adap_mult *= 4.f;
				behind_cameras = false;
				continue;
			}

			evaluateSurfaceForBSSamples();
			new_energy = computeEnergyBS();
			if (verbose) {printf("\n Optimize background and compute the energy - %f sec", clock.Tac()); clock.Tic();}

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
			if (cont_inner > 10) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				evaluateSurfaceForBSSamples();
				new_energy = computeEnergyBS();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
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
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureDataArch(); // takePictureLimitSurface(paper_vis_no_mesh);

		//Save energy to file
		if (save_energy)	saveCurrentEnergyInFile(true);


		if (verbose) {printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();}
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > convergence_ratio))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}

void Mod3DfromRGBD::solveBG_LM_Joint(bool verbose)
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForBSTerm();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	SimplicialLDLT<SparseMatrix<float>> solver;
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += 6;
					J_cols += 2;
				}

	J_rows += 2*num_images*nsamples; //Background DT

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;
	if (fix_first_camera)		J_rows += 6;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);


	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	evaluateSurfaceForBGSamples();
	new_energy = computeEnergyBG();

	if (paper_visualization)	takePictureDataArch(); // takePictureLimitSurface(paper_vis_no_mesh);
	if (save_energy)			saveCurrentEnergyInFile(true);

	utils::CTicTac clock; 
	if (verbose) printf("\n Enter the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(num_faces > 10))
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyBG());
		}
		
		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		u1_old_outer = u1_old;
		u2_old_outer = u2_old;
		uface_old_outer = uface_old;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		if (verbose) {printf("\n Start to compute the Jacobian"); clock.Tic();}

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		//Foreground
		for (unsigned int i = 0; i < num_images; i++)
		{
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
						{
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							return;
						}
					
						//Data alignment
						fill_J_EpPixelJoint(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixelJoint(i, v, u, cont);										
					}
		}

		//Background DT^2 - It must be done after the dataterm to avoid breaking the order of the elem of the jacobian (correspondences)
		for (unsigned int i = 0; i < num_images; i++)			
			fill_J_BackBG(i, cont);

		//Contrain motion of the 1st camera if necessary
		if (fix_first_camera) 	fill_J_fixFirstCamera(cont);

		if (verbose) {printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();}

		//Include regularization
		if (with_reg_normals)		fill_J_RegNormals(cont);
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges)			fill_J_RegEdges(cont);
		if (with_reg_ctf)			fill_J_RegCTF(cont);
		if (with_reg_atraction)		fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape) fill_J_RegEdgesIniShape(cont);

		if (verbose) {printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();}


		//Prepare Levenberg solver - It seems that creating J within the method makes it faster

		//Test
		const float max_j = 0.f; //10e-7f;
		vector<Tri> j_elem_trunc;
		for (unsigned int k=0; k<j_elem.size(); k++)
			if (abs(j_elem[k].value()) > max_j)
				j_elem_trunc.push_back(j_elem[k]);

		printf("\n percentage of used elements (J) = %f", 100.f*float(j_elem_trunc.size())/float(j_elem.size()));


		J.setFromTriplets(j_elem_trunc.begin(), j_elem_trunc.end()-1); j_elem.clear();
		const SparseMatrix<float> JtJ_sparse = J.transpose()*J;
		VectorXf b = -J.transpose()*R;
		SparseMatrix<float> JtJ_lm;

		energy_increasing = true;
		unsigned int cont_inner = 0;

		if (verbose) {printf("\n Compute J, JtJ and b - %f sec", clock.Tac()); clock.Tic();}


		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			JtJ_lm = JtJ_sparse;
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
			}


			//Solve the system
			solver.compute(JtJ_lm); //Equivalent to 	solver.analyzePattern(JtJ_lm); solver.factorize(JtJ_lm);
			if(solver.info()!=Success)
				printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success)
				printf("Solving failed");

			if (verbose) {printf("\n Solve with LM - %f sec", clock.Tac()); clock.Tic();}
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Correspondences
			const float max_incr = 2.f;
			for (unsigned int i = 0; i < num_images; i++)
			{
				//Refs
				ArrayXXf &u1_incr_ref = u1_incr[i];
				ArrayXXf &u2_incr_ref = u2_incr[i];
			
				for (unsigned int u = 0; u < cols; u++)
					for (unsigned int v = 0; v < rows; v++)
						if (is_object[i](v,u))
						{
							u1_incr_ref(v,u) = increments(cont++);
							u2_incr_ref(v,u) = increments(cont++);

							//Saturate increments to avoid too many jumps between faces
							const float norm_incr = sqrtf(square(u1_incr_ref(v,u)) + square(u2_incr_ref(v,u)));
							if (norm_incr > max_incr)
							{
								u1_incr_ref(v,u) *= max_incr/norm_incr;
								u2_incr_ref(v,u) *= max_incr/norm_incr;
							}

							//Update variable
							const float u1_new = u1_old[i](v,u) + u1_incr_ref(v,u);
							const float u2_new = u2_old[i](v,u) + u2_incr_ref(v,u);

							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							{
								updateInternalPointCrossingEdges(i, v, u);
							}
							else
							{
								u1[i](v,u) = u1_new;
								u2[i](v,u) = u2_new;
								uface[i](v,u) = uface_old[i](v,u);
							}
						}
						else
						{
							u1[i](v,u) = u1_old_outer[i](v,u);
							u2[i](v,u) = u2_old_outer[i](v,u);
							uface[i](v,u) = uface_old_outer[i](v,u);
						}
			}

			if (verbose) {printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();}

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	
			
			if (verbose) {printf("\n Create topology, evaluate the surface and computes residual - %f sec", clock.Tac()); clock.Tic();}

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras)
			{
				adap_mult *= 4.f;
				behind_cameras = false;
				continue;
			}

			evaluateSurfaceForBGSamples();
			new_energy = computeEnergyBG();
			if (verbose) {printf("\n Optimize background and compute the energy - %f sec", clock.Tac()); clock.Tic();}

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
			if (cont_inner > 10) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				evaluateSurfaceForBGSamples();
				new_energy = computeEnergyBG();

				if (new_energy > last_energy)
				{					
					//Recover old variables
					vert_coords = vert_coords_old;
					cam_mfold = cam_mfold_old;
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
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureDataArch(); //takePictureLimitSurface(paper_vis_no_mesh);

		//Save energy to file
		if (save_energy)	saveCurrentEnergyInFile(true);


		if (verbose) {printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();}
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy/last_energy > convergence_ratio))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}





void Mod3DfromRGBD::fill_J_EpPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const float Kp_sqrtf = sqrtf(Kp);
	const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0,0);
	
	const float res = res_pos[i].col(v+u*rows).norm();
	if (res < truncated_res)
	{			
		//Control vertices
		const unsigned int weights_col = v + u*rows;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices[i](c,weights_col);
			const Array33f prod = -Kp_sqrtf*T_inv;
			if (cp >= 0)
			{
				const float v_weight = w_contverts[i](c, weights_col);
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						j_elem.push_back(Tri(J_row+l, 3*cp+k, prod(l,k)*v_weight));
				//j_elem.push_back(Tri(J_row, 3*cp, prod(0,0)*v_weight));
				//j_elem.push_back(Tri(J_row, 3*cp+1, prod(0,1)*v_weight));
				//j_elem.push_back(Tri(J_row, 3*cp+2, prod(0,2)*v_weight));
				//j_elem.push_back(Tri(J_row+1, 3*cp, prod(1,0)*v_weight));
				//j_elem.push_back(Tri(J_row+1, 3*cp+1, prod(1,1)*v_weight));
				//j_elem.push_back(Tri(J_row+1, 3*cp+2, prod(1,2)*v_weight));
				//j_elem.push_back(Tri(J_row+2, 3*cp, prod(2,0)*v_weight));
				//j_elem.push_back(Tri(J_row+2, 3*cp+1, prod(2,1)*v_weight));
				//j_elem.push_back(Tri(J_row+2, 3*cp+2, prod(2,2)*v_weight));
			}
		}

		//Camera poses
		if (optimize_cameras)
		{
			Vector4f t_point; t_point << surf_t[i](0,v+rows*u), surf_t[i](1,v+rows*u), surf_t[i](2,v+rows*u), 1.f;
			for (unsigned int l = 0; l < 6; l++)
			{
				const Vector3f prod = -Kp_sqrtf*mat_der_xi[l].block<3,4>(0,0)*t_point;
				j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, prod(0)));
				j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + l, prod(1)));
				j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + l, prod(2)));
			}
		}

		//Fill the residuals
		R.middleRows(J_row,3) = Kp_sqrtf*res_pos[i].col(v+rows*u);
		//R(J_row) = Kp_sqrtf*res_x[i](v,u);
		//R(J_row+1) = Kp_sqrtf*res_y[i](v,u);
		//R(J_row+2) = Kp_sqrtf*res_z[i](v,u);
		J_row += 3;
	}
	else
		J_row += 3;	
}

void Mod3DfromRGBD::fill_J_EpPixelJoint(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const float Kp_sqrt = sqrtf(Kp);
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);
	
	const float res = res_pos[i].col(v+u*rows).norm();
	if (res < truncated_res)
	{			
		//Control vertices
		const unsigned int weights_col = v + u*rows;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices[i](c,weights_col);
			const Matrix3f prod = -Kp_sqrt*T_inv;
			if (cp >= 0)
			{
				const float v_weight = w_contverts[i](c, weights_col);
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						if (prod(l,k) != 0.f)
							j_elem.push_back(Tri(J_row+l, 3*cp+k, prod(l,k)*v_weight));
			}
		}

		//Camera poses
		if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
		{
			//Vector4f t_point; t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
			const Vector3f t_point = surf_t[i].col(v+rows*u);

			//translations
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i, -Kp_sqrt));
			j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + 1, -Kp_sqrt));
			j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + 2, -Kp_sqrt));

			//rotations
			j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + 3, Kp_sqrt*t_point(2)));
			j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + 3, -Kp_sqrt*t_point(1)));

			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 4, -Kp_sqrt*t_point(2)));
			j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + 4, Kp_sqrt*t_point(0)));

			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 5, Kp_sqrt*t_point(1)));
			j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + 5, -Kp_sqrt*t_point(0)));

			//General formula (less efficient)
			//for (unsigned int l = 0; l < 6; l++)
			//{
			//	const Vector3f prod = -Kp_sqrt*mat_der_xi[l].block<3,4>(0,0)*t_point;
			//	j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, prod(0)));
			//	j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + l, prod(1)));
			//	j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + l, prod(2)));
			//}
		}

		//Correspondence
		Matrix<float, 3, 2> u_der; 
		u_der << u1_der[i](0,v+rows*u), u2_der[i](0,v+rows*u), u1_der[i](1,v+rows*u), u2_der[i](1,v+rows*u), u1_der[i](2,v+rows*u), u2_der[i](2,v+rows*u);
		const Matrix<float, 3, 2> J_u = -Kp_sqrt*T_inv*u_der;
		const unsigned int ind_bias = 3*num_verts + optimize_cameras*6*num_images + 2*(J_row/6);
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + k, J_u(l,k)));

		//Fill the residuals
		R.middleRows(J_row,3) = Kp_sqrt*res_pos[i].col(v+rows*u);
		//R(J_row) = Kp_sqrt*res_x[i](v,u);
		//R(J_row+1) = Kp_sqrt*res_y[i](v,u);
		//R(J_row+2) = Kp_sqrt*res_z[i](v,u);
	}
	//Simplest (and probably bad) solution to the problem of underdetermined unknowns for the solver
	else
	{
		const unsigned int ind_bias = 3*num_verts + optimize_cameras*6*num_images + 2*(J_row/6); 
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + k, 10.001f));	
	}

	J_row += 3;	
}

void Mod3DfromRGBD::fill_J_fixFirstCamera(unsigned int &J_row)
{
	//To constrain the camera motion (to null motion)
	const float high_value = 10000.f;
	j_elem.push_back(Tri(J_row, 3*num_verts, high_value));
	j_elem.push_back(Tri(J_row+1, 3*num_verts + 1, high_value));
	j_elem.push_back(Tri(J_row+2, 3*num_verts + 2, high_value));
	j_elem.push_back(Tri(J_row+2, 3*num_verts + 3, high_value));
	j_elem.push_back(Tri(J_row+2, 3*num_verts + 4, high_value));
	j_elem.push_back(Tri(J_row+2, 3*num_verts + 5, high_value));
}

void Mod3DfromRGBD::fill_J_EnPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const int index = v+rows*u;
	const float Kn_sqrt = sqrtf(Kn);
	const float wn_sqrt = sqrtf(n_weights[i](v+rows*u));
	const Matrix3f T_inv = cam_trans_inv[i].block<3, 3>(0,0);
	const Vector3f n_vec = normals[i].col(index);
	
	const float resn = res_normals[i].col(v+rows*u).norm();
	if (resn < truncated_resn)
	{
		//Control vertices
		const float inv_norm = 1.f/n_vec.norm();
		Matrix3f J_nu, J_nX;
		J_nu.row(0) << square(n_vec(1)) + square(n_vec(2)), -n_vec(0)*n_vec(1), -n_vec(0)*n_vec(2);
		J_nu.row(1) << -n_vec(0)*n_vec(1), square(n_vec(0)) + square(n_vec(2)), -n_vec(1)*n_vec(2);
		J_nu.row(2) << -n_vec(0)*n_vec(2), -n_vec(1)*n_vec(2), square(n_vec(0)) + square(n_vec(1));
		J_nu *= inv_norm*square(inv_norm);
		J_nX.assign(0.f);
		const Matrix3f J_mult_norm = -Kn_sqrt*wn_sqrt*T_inv*J_nu;
						
		const unsigned int weights_col = v + u*rows;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices[i](c,weights_col);
			if (cp >= 0)
			{
				//Normals
				const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);
				J_nX(0,1) = wu1*u2_der[i](2,v+rows*u) - wu2*u1_der[i](2,v+rows*u);
				J_nX(0,2) = wu2*u1_der[i](1,v+rows*u) - wu1*u2_der[i](1,v+rows*u);
				J_nX(1,2) = wu1*u2_der[i](0,v+rows*u) - wu2*u1_der[i](0,v+rows*u);
				J_nX(1,0) = -J_nX(0,1);
				J_nX(2,0) = -J_nX(0,2);
				J_nX(2,1) = -J_nX(1,2);

				const Matrix3f J_norm_fit = J_mult_norm*J_nX;
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						j_elem.push_back(Tri(J_row+l, 3*cp+k, J_norm_fit(l,k)));
				//j_elem.push_back(Tri(J_row, 3*cp, J_norm_fit(0,0)));
				//j_elem.push_back(Tri(J_row, 3*cp+1, J_norm_fit(0,1)));
				//j_elem.push_back(Tri(J_row, 3*cp+2, J_norm_fit(0,2)));
				//j_elem.push_back(Tri(J_row+1, 3*cp, J_norm_fit(1,0)));
				//j_elem.push_back(Tri(J_row+1, 3*cp + 1, J_norm_fit(1,1)));
				//j_elem.push_back(Tri(J_row+1, 3*cp + 2, J_norm_fit(1,2)));
				//j_elem.push_back(Tri(J_row+2, 3*cp, J_norm_fit(2,0)));
				//j_elem.push_back(Tri(J_row+2, 3*cp + 1, J_norm_fit(2,1)));
				//j_elem.push_back(Tri(J_row+2, 3*cp + 2, J_norm_fit(2,2)));
			}
		}

		//Camera pose	
		if (optimize_cameras)
		{
			const Vector3f n_t = Kn_sqrt*wn_sqrt*T_inv*inv_norm*n_vec;
			for (unsigned int l = 3; l < 6; l++)
			{
				const Vector3f prod = -mat_der_xi[l].block<3,3>(0,0)*n_t;
				j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, prod(0)));
				j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + l, prod(1)));
				j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + l, prod(2)));
			}
		}

		//Fill the residuals
		R.middleRows(J_row,3) = Kn_sqrt*wn_sqrt*res_normals[i].col(v+rows*u);
		//R(J_row) = Kn_sqrt*wn_sqrt*res_nx[i](v,u);
		//R(J_row+1) = Kn_sqrt*wn_sqrt*res_ny[i](v,u);
		//R(J_row+2) = Kn_sqrt*wn_sqrt*res_nz[i](v,u);
		J_row += 3;
	}
	else
		J_row += 3;	
}

void Mod3DfromRGBD::fill_J_EnPixelJoint(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const int index = v+rows*u;
	const float Kn_sqrt = sqrtf(Kn);
	const float wn_sqrt = sqrtf(n_weights[i](v+rows*u));	
	const float resn = res_normals[i].col(v+rows*u).norm();
	const Vector3f n_vec = normals[i].col(index);

	if (resn < truncated_resn)
	{
		const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);
		
		Matrix3f J_nu, J_nX;
		const float inv_norm = 1.f/n_vec.norm();
		const float J_nu_xy = -n_vec(0)*n_vec(1);
		const float J_nu_xz = -n_vec(0)*n_vec(2);
		const float J_nu_yz = -n_vec(1)*n_vec(2);
		J_nu.row(0) << square(n_vec(1)) + square(n_vec(2)), J_nu_xy, J_nu_xz;
		J_nu.row(1) << J_nu_xy, square(n_vec(0)) + square(n_vec(2)), J_nu_yz;
		J_nu.row(2) << J_nu_xz, J_nu_yz, square(n_vec(0)) + square(n_vec(1));
		J_nu *= inv_norm*square(inv_norm);
		J_nX.assign(0.f);
		const Matrix3f J_mult_norm = -Kn_sqrt*wn_sqrt*T_inv*J_nu;
		
		//Control vertices
		const unsigned int weights_col = v + u*rows;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices[i](c,weights_col);
			if (cp >= 0)
			{
				const float wu1 = w_u1[i](c, weights_col), wu2 = w_u2[i](c, weights_col);

				J_nX(0,1) = wu1*u2_der[i](2,v+rows*u) - wu2*u1_der[i](2,v+rows*u);
				J_nX(0,2) = wu2*u1_der[i](1,v+rows*u) - wu1*u2_der[i](1,v+rows*u);
				J_nX(1,2) = wu1*u2_der[i](0,v+rows*u) - wu2*u1_der[i](0,v+rows*u);
				J_nX(1,0) = -J_nX(0,1);
				J_nX(2,0) = -J_nX(0,2);
				J_nX(2,1) = -J_nX(1,2);

				const Matrix3f J_norm_fit = J_mult_norm*J_nX;
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						j_elem.push_back(Tri(J_row+l, 3*cp+k, J_norm_fit(l,k)));
			}
		}

		//Camera pose	
		if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
		{
			const Vector3f n_t = Kn_sqrt*wn_sqrt*T_inv*inv_norm*n_vec;

			//rotations only for the normals
			j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + 3, n_t(2)));
			j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + 3, -n_t(1)));

			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 4, -n_t(2)));
			j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + 4, n_t(0)));

			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 5, n_t(1)));
			j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + 5, -n_t(0)));


			//General expression (inefficient)
			//for (unsigned int l = 3; l < 6; l++)
			//{
			//	const Vector3f prod = -mat_der_xi[l].block<3,3>(0,0)*n_t; 
			//	j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, prod(0)));
			//	j_elem.push_back(Tri(J_row+1, 3*num_verts + 6*i + l, prod(1)));
			//	j_elem.push_back(Tri(J_row+2, 3*num_verts + 6*i + l, prod(2)));
			//}
		}

		//Correspondence
		computeNormalDerivatives_Analyt(i, v, u);
		Matrix<float, 3, 2> n_der_u;
		n_der_u << n_der_u1[i](0,v+rows*u), n_der_u2[i](0,v+rows*u), n_der_u1[i](1,v+rows*u), n_der_u2[i](1,v+rows*u), n_der_u1[i](2,v+rows*u), n_der_u2[i](2,v+rows*u);
		const Matrix<float, 3, 2> J_u = -Kn_sqrt*wn_sqrt*T_inv*J_nu*n_der_u;
		const unsigned int ind_bias = 3*num_verts + optimize_cameras*6*num_images + 2*(J_row/6); 
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + k, J_u(l,k)));

		//Fill the residuals
		R.middleRows(J_row,3) = Kn_sqrt*wn_sqrt*res_normals[i].col(v+rows*u);
		//R(J_row) = Kn_sqrt*wn_sqrt*res_nx[i](v,u);
		//R(J_row+1) = Kn_sqrt*wn_sqrt*res_ny[i](v,u);
		//R(J_row+2) = Kn_sqrt*wn_sqrt*res_nz[i](v,u);
	}

	J_row += 3;		
}

void Mod3DfromRGBD::fill_J_BackSKPixel(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const float weight_sqrt = sqrtf(alpha);
	const float norm_proj_error = res_pixels[i].col(v+rows*u).norm(); 
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);	
	
	if ((norm_proj_error < tau_pixel[i](v,u)) && (norm_proj_error > eps_rel*tau_pixel[i](v,u)))
	{

		Matrix<float, 2, 3> J_pi;
		const float inv_z = 1.f / surf_t[i](0,v+rows*u);

		J_pi << fx*surf_t[i](1,v+rows*u)*square(inv_z), -fx*inv_z, 0.f,
				fy*surf_t[i](2,v+rows*u)*square(inv_z), 0.f, -fy*inv_z;

		Matrix<float, 1, 2> J_phi = res_pixels[i].col(v+rows*u).transpose(); 
		J_phi *= -weight_sqrt/(tau_pixel[i](v,u)*norm_proj_error);

		const Matrix<float, 1, 3> J_phi_pi = J_phi*J_pi;
		const Matrix<float, 1, 3> J_phi_pi_Tinv = J_phi_pi*T_inv;

		//Control vertices
		const unsigned int weights_col = v + u*rows;
		for (unsigned int c = 0; c < max_num_w; c++)
		{
			const int cp = w_indices[i](c,weights_col);
			if (cp >= 0)
			{
				const float v_weight = w_contverts[i](c, weights_col);
				j_elem.push_back(Tri(J_row, 3*cp, J_phi_pi_Tinv(0)*v_weight));
				j_elem.push_back(Tri(J_row, 3*cp+1, J_phi_pi_Tinv(1)*v_weight));
				j_elem.push_back(Tri(J_row, 3*cp+2, J_phi_pi_Tinv(2)*v_weight));
			}
		}

		//Camera pose
		if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
		{
			//Vector4f m_t; m_t << mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u), 1.f;
			const Vector3f surf_t_vec = surf_t[i].col(v+rows*u);

			//Translations
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i, J_phi_pi(0)));
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 1, J_phi_pi(1)));
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 2, J_phi_pi(2)));

			//Rotations
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 3, -J_phi_pi(1)*surf_t_vec(2) + J_phi_pi(2)*surf_t_vec(1)));
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 4, J_phi_pi(0)*surf_t_vec(2) - J_phi_pi(2)*surf_t_vec(0)));
			j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + 5, -J_phi_pi(0)*surf_t_vec(1) + J_phi_pi(1)*surf_t_vec(0)));

			//General expression (less efficient)
			//for (unsigned int l = 0; l < 6; l++)
			//{
			//	const float prod = (J_phi_pi*(mat_der_xi[l] * m_t).block<3, 1>(0,0)).value();
			//	j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, prod));
			//}
		}

		//Fill the residuals
		R(J_row) = weight_sqrt*(1.f - norm_proj_error/tau_pixel[i](v,u));
	}

	J_row++;	
}

void Mod3DfromRGBD::fill_J_BackDT2(unsigned int i, unsigned int &J_row)
{
	const float alpha_sqrt = sqrtf(alpha);
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);	
	
	for (unsigned int s = 0; s < nsamples; s++)
	{
		const int v = int(pixel_DT_v[i](s)), u = int(pixel_DT_u[i](s));
		if (DT[i](v,u) > 0.f)
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

			const Matrix<float, 1, 2> J_DT = {DT_grad_u[i](v,u), DT_grad_v[i](v,u)};
			const Matrix<float, 1, 3> J_mult = -alpha_sqrt*J_DT*J_pi*T_inv;

			//Control vertices
			for (unsigned int c = 0; c < max_num_w; c++)
			{
				const int cp = w_indices_DT(c, s);
				if (cp >= 0)
				{
					const float ww = w_DT(c, s);
					j_elem.push_back(Tri(J_row, 3*cp, J_mult(0)*ww));	
					j_elem.push_back(Tri(J_row, 3*cp+1, J_mult(1)*ww));	
					j_elem.push_back(Tri(J_row, 3*cp+2, J_mult(2)*ww));	 
				}
			}

			//Camera pose
			if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
			{
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, mz_t_DT, 1.f;
				for (unsigned int l = 0; l < 6; l++)
				{
					Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0,0);
					const float value = -alpha_sqrt*(J_DT*J_pi*aux_prod).value();
					j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, value));		//printf("\n cam: j_elem = %f", value);		printf("\n cam: j_elem_fromTri = %f", j_elem[j_elem.size()-1].value()); 
					//printf("\n CP: image %d: s = %d, jrow = %d, elem: %f", i, s, J_row, value);
				}
			}

			//Fill the residuals
			R(J_row) = alpha_sqrt*DT[i](v,u);
			J_row++;
		}
	}	
}

void Mod3DfromRGBD::fill_J_BackBS(unsigned int i, unsigned int &J_row)
{
	const float alpha_sqrt = sqrtf(alpha);
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);	
	Matrix3f J_nX;
	
	for (unsigned int s = 0; s < nsamples; s++)
	{
		const int v = roundf(pixel_DT_v[i](s)), u = roundf(pixel_DT_u[i](s));
		J_nX.assign(0.f); //Maybe necessary only once out

		if (DT[i](v,u) > 0.f)
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


			const Matrix<float, 1, 2> grad_DT = {DT_grad_u[i](v,u), DT_grad_v[i](v,u)};
			const Matrix<float, 1, 3> J_DT = -alpha_sqrt*grad_DT*J_pi*T_inv*norm_n_DT(s);
			const float inv_norm = 1.f/norm_n_DT(s);
			const Matrix<float, 1, 3> J_norm = {nx_DT(s)*inv_norm, ny_DT(s)*inv_norm, nz_DT(s)*inv_norm}; 
			const Matrix<float, 1, 3> DT_J_norm = -alpha_sqrt*DT[i](v,u)*J_norm;

			//Control vertices
			for (unsigned int c = 0; c < max_num_w; c++)
			{
				const int cp = w_indices_DT(c,s);
				if (cp >= 0)
				{
					const float ww = w_DT(c,s);
					const float wu1 = w_u1_DT(c,s), wu2 = w_u2_DT(c,s);
					J_nX(0,1) = wu1*u2_der_DT(s)[2] - wu2*u1_der_DT(s)[2];
					J_nX(0,2) = wu2*u1_der_DT(s)[1] - wu1*u2_der_DT(s)[1];
					J_nX(1,2) = wu1*u2_der_DT(s)[0] - wu2*u1_der_DT(s)[0];
					J_nX(1,0) = -J_nX(0,1);
					J_nX(2,0) = -J_nX(0,2);
					J_nX(2,1) = -J_nX(1,2);
					const Matrix<float, 1, 3> J_all = ww*J_DT - DT_J_norm*J_nX; 

					j_elem.push_back(Tri(J_row, 3*cp, J_all(0)));	
					j_elem.push_back(Tri(J_row, 3*cp+1, J_all(1)));	
					j_elem.push_back(Tri(J_row, 3*cp+2, J_all(2)));	 
				}
			}


			//Camera pose
			if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
			{
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, mz_t_DT, 1.f;
				for (unsigned int l = 0; l < 6; l++)
				{
					Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0,0);
					const float value = -alpha_sqrt*(grad_DT*J_pi*aux_prod).value()*norm_n_DT(s);
					j_elem.push_back(Tri(J_row, 3*num_verts + 6*i + l, value));		
					//printf("\n cam: j_elem = %f", value);		printf("\n cam: j_elem_fromTri = %f", j_elem[j_elem.size()-1].value()); 
					//printf("\n CP: image %d: s = %d, jrow = %d, elem: %f", i, s, J_row, value);
				}
			}

			//Fill the residuals
			R(J_row) = alpha_sqrt*DT[i](v,u)*norm_n_DT(s);
			J_row++;
		}
	}	
}

void Mod3DfromRGBD::fill_J_BackBG(unsigned int i, unsigned int &J_row)
{
	const float alpha_sqrt = sqrtf(alpha);
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);	
	
	for (unsigned int s = 0; s < nsamples; s++)
	{
		const int v = roundf(pixel_DT_v[i](s)), u = roundf(pixel_DT_u[i](s));

		if (DT[i](v,u) > 0.f)
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


			const Matrix<float, 1, 2> grad_DT = {DT_grad_u[i](v,u), DT_grad_v[i](v,u)};
			const Matrix<float, 1, 3> J_DT = -alpha_sqrt*grad_DT*J_pi*T_inv;
			const Matrix<float, 1, 3> J_DT_u1 = J_DT*u1_dernorm_DT(s);
			const Matrix<float, 1, 3> J_DT_u2 = J_DT*u2_dernorm_DT(s);
			const float inv_norm_u1 = 1.f/u1_dernorm_DT(s);
			const float inv_norm_u2 = 1.f/u2_dernorm_DT(s);
			const Matrix<float, 1, 3> J_u1_norm = {u1_der_DT(s)[0]*inv_norm_u1, u1_der_DT(s)[1]*inv_norm_u1, u1_der_DT(s)[2]*inv_norm_u1}; 
			const Matrix<float, 1, 3> J_u2_norm = {u2_der_DT(s)[0]*inv_norm_u2, u2_der_DT(s)[1]*inv_norm_u2, u2_der_DT(s)[2]*inv_norm_u2}; 
			const Matrix<float, 1, 3> DT_J_u1_norm = -alpha_sqrt*DT[i](v,u)*J_u1_norm;
			const Matrix<float, 1, 3> DT_J_u2_norm = -alpha_sqrt*DT[i](v,u)*J_u2_norm;

			//Control vertices
			for (unsigned int c = 0; c < max_num_w; c++)
			{
				const int cp = w_indices_DT(c,s);
				if (cp >= 0)
				{
					const float ww = w_DT(c,s);
					const float wu1 = w_u1_DT(c,s), wu2 = w_u2_DT(c,s);
					const Matrix<float, 1, 3> J_all_u1 = ww*J_DT_u1 - wu1*DT_J_u1_norm; 
					const Matrix<float, 1, 3> J_all_u2 = ww*J_DT_u2 - wu2*DT_J_u2_norm; 

					j_elem.push_back(Tri(J_row, unk_per_vertex*cp, J_all_u1(0)));	
					j_elem.push_back(Tri(J_row, unk_per_vertex*cp+1, J_all_u1(1)));	
					j_elem.push_back(Tri(J_row, unk_per_vertex*cp+2, J_all_u1(2)));	 
					j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp, J_all_u2(0)));	
					j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp+1, J_all_u2(1)));	
					j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp+2, J_all_u2(2)));	 
				}
			}


			//Camera pose
			if ((optimize_cameras)&&((i>0)||(!fix_first_camera)))
			{
				Vector4f m_t; m_t << mx_t_DT, my_t_DT, mz_t_DT, 1.f;
				for (unsigned int l = 0; l < 6; l++)
				{
					Vector3f aux_prod = (mat_der_xi[l] * m_t).block<3, 1>(0,0);
					const float grad_u1 = -alpha_sqrt*(grad_DT*J_pi*aux_prod).value()*u1_dernorm_DT(s);
					const float grad_u2 = -alpha_sqrt*(grad_DT*J_pi*aux_prod).value()*u2_dernorm_DT(s);
					j_elem.push_back(Tri(J_row, unk_per_vertex*num_verts + 6*i + l, grad_u1));
					j_elem.push_back(Tri(J_row+1, unk_per_vertex*num_verts + 6*i + l, grad_u2));
					//printf("\n cam: j_elem = %f", value);		printf("\n cam: j_elem_fromTri = %f", j_elem[j_elem.size()-1].value()); 
					//printf("\n CP: image %d: s = %d, jrow = %d, elem: %f", i, s, J_row, value);
				}
			}

			//Fill the residuals
			R(J_row) = alpha_sqrt*DT[i](v,u)*u1_dernorm_DT(s);
			R(J_row+1) = alpha_sqrt*DT[i](v,u)*u2_dernorm_DT(s);
			J_row += 2;
		}
	}	
}

void Mod3DfromRGBD::fill_J_RegNormals(unsigned int &J_row)
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
				const Vector3f n_vec_reg = normals_reg[f].col(s1+s_reg*s2);
				J_nu[s1 + s_reg*s2].row(0) << square(n_vec_reg(1)) + square(n_vec_reg(2)), -n_vec_reg(0)*n_vec_reg(1), -n_vec_reg(0)*n_vec_reg(2);
				J_nu[s1 + s_reg*s2].row(1) << -n_vec_reg(0)*n_vec_reg(1), square(n_vec_reg(0)) + square(n_vec_reg(2)), -n_vec_reg(1)*n_vec_reg(2);
				J_nu[s1 + s_reg*s2].row(2) << -n_vec_reg(0)*n_vec_reg(2), -n_vec_reg(1)*n_vec_reg(2), square(n_vec_reg(0)) + square(n_vec_reg(1));
				J_nu[s1 + s_reg*s2] *= inv_reg_norm[f](s1+s_reg*s2);
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

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](2,s1+s_reg*s2) - wu2_here*u1_der_reg[f](2,s1+s_reg*s2);
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](1,s1+s_reg*s2) - wu1_here*u2_der_reg[f](1,s1+s_reg*s2);
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](0,s1+s_reg*s2) - wu2_here*u1_der_reg[f](0,s1+s_reg*s2);
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](2,s1for+s_reg*s2) - wu2_for*u1_der_reg[f](2,s1for+s_reg*s2);
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](1,s1for+s_reg*s2) - wu1_for*u2_der_reg[f](1,s1for+s_reg*s2);
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](0,s1for+s_reg*s2) - wu2_for*u1_der_reg[f](0,s1for+s_reg*s2);
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
					R.middleRows(J_row,3) = Kr_sqrt*(normals_reg[f].col(s1for+s_reg*s2) - normals_reg[f].col(s1+s_reg*s2));
					//R(J_row) = Kr_sqrt*(normals_reg[f](0,s1for+s_reg*s2) - normals_reg[f](0,s1+s_reg*s2));
					//R(J_row+1) = Kr_sqrt*(ny_reg[f](s1for,s2) - ny_reg[f](s1,s2));
					//R(J_row+2) = Kr_sqrt*(nz_reg[f](s1for,s2) - nz_reg[f](s1,s2));
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

						J_nX_here(0,1) = wu1_here*u2_der_reg[f](2,s1+s_reg*s2) - wu2_here*u1_der_reg[f](2,s1+s_reg*s2);
						J_nX_here(0,2) = wu2_here*u1_der_reg[f](1,s1+s_reg*s2) - wu1_here*u2_der_reg[f](1,s1+s_reg*s2);
						J_nX_here(1,2) = wu1_here*u2_der_reg[f](0,s1+s_reg*s2) - wu2_here*u1_der_reg[f](0,s1+s_reg*s2);
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_reg[f](2,s1+s_reg*s2for) - wu2_for*u1_der_reg[f](2,s1+s_reg*s2for);
						J_nX_forward(0,2) = wu2_for*u1_der_reg[f](1,s1+s_reg*s2for) - wu1_for*u2_der_reg[f](1,s1+s_reg*s2for);
						J_nX_forward(1,2) = wu1_for*u2_der_reg[f](0,s1+s_reg*s2for) - wu2_for*u1_der_reg[f](0,s1+s_reg*s2for);
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
					R.middleRows(J_row,3) = Kr_sqrt*(normals_reg[f].col(s1+s_reg*s2for) - normals_reg[f].col(s1+s_reg*s2));
					//R(J_row) = Kr_sqrt*(nx_reg[f](s1,s2for) - nx_reg[f](s1,s2));
					//R(J_row+1) = Kr_sqrt*(ny_reg[f](s1,s2for) - ny_reg[f](s1,s2));
					//R(J_row+2) = Kr_sqrt*(nz_reg[f](s1,s2for) - nz_reg[f](s1,s2));
					J_row += 3;
				}				
			}
		}
}

void Mod3DfromRGBD::fill_J_RegNormalsCurvature(unsigned int &J_row)
{
	Matrix3f J_nX_here, J_nX_forward;
	J_nX_here.assign(0.f); J_nX_forward.assign(0.f);
	Kr = Kr_total/float(num_faces*square(s_reg));
	const float Kr_sqrt = sqrtf(Kr);
	const unsigned int max_k = with_reg_normals_4dir ? 4 : 2; //if "with_reg_normals_4dir == true" use diagonals and if "with_reg_normals_good == true" use only edges.
	
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Refs
		const MatrixXf &n_reg_f = normals_reg[f];
		const MatrixXf &surf_reg_f = surf_reg[f];
		const MatrixXf &u1_der_ref = u1_der_reg[f], &u2_der_ref = u2_der_reg[f];
		
		//Compute the normalizing jacobians (J_nu)
		vector<Matrix3f> J_nu; J_nu.resize(square(s_reg));
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{			
				//J_nu
				const Vector3f n_vec = n_reg_f.col(s1 + s_reg*s2);
				const float J_nu_xy = -n_vec(0)*n_vec(1);
				const float J_nu_xz = -n_vec(0)*n_vec(2);
				const float J_nu_yz = -n_vec(1)*n_vec(2);
				J_nu[s1 + s_reg*s2].row(0) << square(n_vec(1)) + square(n_vec(2)), J_nu_xy, J_nu_xz;
				J_nu[s1 + s_reg*s2].row(1) << J_nu_xy, square(n_vec(0)) + square(n_vec(2)), J_nu_yz;
				J_nu[s1 + s_reg*s2].row(2) << J_nu_xz, J_nu_yz, square(n_vec(0)) + square(n_vec(1));
				J_nu[s1 + s_reg*s2] *= inv_reg_norm[f](s1+s_reg*s2);
			}

		//Include every equation into LM  - Number of new equations: 6*num_faces*(s_reg*s_reg + (s_reg-1)*(s_reg-1))
		for (int s2=0; s2<s_reg; s2++)
			for (int s1=0; s1<s_reg; s1++)
			{						
				//Indices associated to the regularization
				const unsigned int s1for = min(s1+1, int(s_reg-1));
				const unsigned int s2for = min(s2+1, int(s_reg-1));

				//Create indices for all the combinations of edges and diagonals
				Array<unsigned int, 4, 2> s_index_a, s_index_b;
				s_index_b << s1for, s2,	 s1, s2for,  s1for, s2for,  s1for, s2;
				s_index_a << s1, s2,	 s1, s2,	 s1, s2,		s1, s2for;

				for (unsigned int k=0; k<max_k; k++)
				{
					const unsigned int s1_a = s_index_a(k,0);
					const unsigned int s1_b = s_index_b(k,0);
					const unsigned int s2_a = s_index_a(k,1);
					const unsigned int s2_b = s_index_b(k,1);
					
					//edges of the face
					if ((k == 0)&&(s1_b == s1_a)) continue;
					if ((k == 1)&&(s2_b == s2_a)) continue;
					if ((k >= 2)&&((s1_b == s1_a)||(s2_b == s2_a))) continue;

					//Solve
					const unsigned int weights_col = s1_a + s_reg*s2_a;
					const unsigned int weights_col_for = s1_b + s_reg*s2_b;

					const Vector3f segment = surf_reg_f.col(weights_col_for) - surf_reg_f.col(weights_col);
					const float dist = segment.norm();
					//const float dist = sqrtf(square(surf_reg_f(0,weights_col_for) - mx_reg_f(s1_a,s2_a)) + square(my_reg_f(s1_b,s2_b) - my_reg_f(s1_a,s2_a)) + square(mz_reg_f(s1_b,s2_b) - mz_reg_f(s1_a,s2_a)));
					const float inv_dist = 1.f/dist;

					Vector3f prev_coef = n_reg_f.col(s1_b+s_reg*s2_b) - n_reg_f.col(s1_a+s_reg*s2_a);
					//prev_coef << nx_reg_f(s1_b,s2_b) - nx_reg_f(s1_a,s2_a), ny_reg_f(s1_b,s2_b) - ny_reg_f(s1_a,s2_a), nz_reg_f(s1_b,s2_b) - nz_reg_f(s1_a,s2_a);
					prev_coef *= square(inv_dist);

					const Matrix<float, 1, 3> J_norm_dist = inv_dist*segment.transpose(); 

					for (unsigned int ind = 0; ind < max_num_w; ind++)
					{
						const int cp = w_indices_reg[f](ind);
						if (cp < 0) break;

						const float wu1_here = w_u1_reg[f](ind, weights_col), wu2_here = w_u2_reg[f](ind, weights_col);
						const float wu1_for = w_u1_reg[f](ind, weights_col_for), wu2_for = w_u2_reg[f](ind, weights_col_for);

						if ((wu1_here == 0.f) && (wu2_here == 0.f) && (wu1_for == 0.f) && (wu2_for == 0.f))
							continue;

						J_nX_here(0,1) = wu1_here*u2_der_ref(2,s1_a+s_reg*s2_a) - wu2_here*u1_der_ref(2,s1_a+s_reg*s2_a);
						J_nX_here(0,2) = wu2_here*u1_der_ref(1,s1_a+s_reg*s2_a) - wu1_here*u2_der_ref(1,s1_a+s_reg*s2_a);
						J_nX_here(1,2) = wu1_here*u2_der_ref(0,s1_a+s_reg*s2_a) - wu2_here*u1_der_ref(0,s1_a+s_reg*s2_a);
						J_nX_here(1,0) = -J_nX_here(0,1);
						J_nX_here(2,0) = -J_nX_here(0,2);
						J_nX_here(2,1) = -J_nX_here(1,2);	

						J_nX_forward(0,1) = wu1_for*u2_der_ref(2,s1_b+s_reg*s2_b) - wu2_for*u1_der_ref(2,s1_b+s_reg*s2_b);
						J_nX_forward(0,2) = wu2_for*u1_der_ref(1,s1_b+s_reg*s2_b) - wu1_for*u2_der_ref(1,s1_b+s_reg*s2_b);
						J_nX_forward(1,2) = wu1_for*u2_der_ref(0,s1_b+s_reg*s2_b) - wu2_for*u1_der_ref(0,s1_b+s_reg*s2_b);
						J_nX_forward(1,0) = -J_nX_forward(0,1);
						J_nX_forward(2,0) = -J_nX_forward(0,2);
						J_nX_forward(2,1) = -J_nX_forward(1,2);

						//First part of the Jacobian
						const Matrix3f J_reg_1 = Kr_sqrt*(J_nu[s1_b + s_reg*s2_b]*J_nX_forward - J_nu[s1_a + s_reg*s2_a]*J_nX_here)*inv_dist; 

						//Second part of the Jacobian: (n(s1for) - n(s1))*dif(norm(vec_s1for_s1))/dX
						const float J_dif_coords = w_contverts_reg[f](ind, weights_col_for) - w_contverts_reg[f](ind, weights_col);
						const Matrix3f J_reg_2 = -Kr_sqrt*J_dif_coords*(prev_coef*J_norm_dist); 

						//Whole Jacobian
						const Matrix3f J_reg = J_reg_1 + J_reg_2;

						//Update triplets
						j_elem.push_back(Tri(J_row, unk_per_vertex*cp, J_reg(0,0)));
						j_elem.push_back(Tri(J_row, unk_per_vertex*cp+1, J_reg(0,1)));
						j_elem.push_back(Tri(J_row, unk_per_vertex*cp+2, J_reg(0,2)));
						j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp, J_reg(1,0)));
						j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp+1, J_reg(1,1)));
						j_elem.push_back(Tri(J_row+1, unk_per_vertex*cp+2, J_reg(1,2)));
						j_elem.push_back(Tri(J_row+2, unk_per_vertex*cp, J_reg(2,0)));
						j_elem.push_back(Tri(J_row+2, unk_per_vertex*cp+1, J_reg(2,1)));
						j_elem.push_back(Tri(J_row+2, unk_per_vertex*cp+2, J_reg(2,2)));
					}	

					//Fill the residuals
					R.middleRows(J_row,3) = Kr_sqrt*inv_dist*(n_reg_f.col(s1_b+s_reg*s2_b) - n_reg_f.col(s1_a+s_reg*s2_a));
					//R(J_row) = Kr_sqrt*(nx_reg_f(s1_b,s2_b) - nx_reg_f(s1_a,s2_a))*inv_dist;
					//R(J_row+1) = Kr_sqrt*(ny_reg_f(s1_b,s2_b) - ny_reg_f(s1_a,s2_a))*inv_dist;
					//R(J_row+2) = Kr_sqrt*(nz_reg_f(s1_b,s2_b) - nz_reg_f(s1_a,s2_a))*inv_dist;
					J_row += 3;			
				}
			}
		}
}

void Mod3DfromRGBD::fill_J_RegEdges(unsigned int &J_row)
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

void Mod3DfromRGBD::fill_J_RegCTF(unsigned int &J_row)
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

void Mod3DfromRGBD::fill_J_RegAtraction(unsigned int &J_row)
{
	K_atrac = K_atrac_total; //*float(num_faces);
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

void Mod3DfromRGBD::fill_J_RegVertColor(unsigned int &J_row)
{
	const float K_color_sqrt = sqrtf(K_color_reg);
	
	for (unsigned int f=0; f<num_faces; f++)
		for (unsigned int e=0; e<4; e++)
		{						
			const unsigned int ind_e0 = e;
			const unsigned int ind_e1 = (e+1)%4;

			const unsigned int vert_e0 = face_verts(ind_e0,f);
			const unsigned int vert_e1 = face_verts(ind_e1,f);

			const float color_dif = vert_colors(vert_e1) - vert_colors(vert_e0);

			j_elem.push_back(Tri(J_row, 4*vert_e0+3, -K_color_sqrt));
			j_elem.push_back(Tri(J_row, 4*vert_e1+3, K_color_sqrt));


			//Fill the residual
			R(J_row) = K_color_sqrt*color_dif;
			J_row++;
		}	
}

void Mod3DfromRGBD::optimizeUBackground_LM()
{
	unsigned int cont = 0;
	unsigned int inner_cont;
	float aver_lambda = 0.f;
	float aver_num_iters = 0.f;
	float aver_first_uincr = 0.f;

	//Iterative solver
	float lambda, energy_ratio, energy_old, energy;
	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	const float limit_uincr = 0.05f*sqrtf(num_faces);
	const float lambda_limit = 100000.f;
	const float u_incr_min = 0.001f;
	const float res_threshold = float(max(rows,cols))/10.f;	//Pay attention to this
	float norm_uincr;
	const float lambda_mult = 3.f;


	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix3f rot_inv = cam_trans_inv[i].block<3,3>(0,0);
		const Vector3f tvec_inv = cam_trans_inv[i].block<3,1>(0,3);

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (!is_object[i](v,u) && valid[i](v,u))
				{
					energy = res_pixels[i].col(v+rows*u).squaredNorm();

					//Don't use this pixel if it is very far from the current model
					if (sqrtf(energy) > res_threshold)
						continue;

					//Refs to accelerate access
					float &u1_ref = u1[i](v,u);
					float &u2_ref = u2[i](v,u);
					int &uface_ref = uface[i](v,u);
					float &u1_old_ref = u1_old[i](v,u);
					float &u2_old_ref = u2_old[i](v,u);
					int &uface_old_ref = uface_old[i](v,u);
					float &u1_incr_ref = u1_incr[i](v,u);
					float &u2_incr_ref = u2_incr[i](v,u);

					energy_ratio = 2.f;
					norm_uincr = 1.f;
					lambda = 10.f;
					inner_cont = 0;

					while (energy_ratio > 1.0002f)
					{				
						//Old equal to new for the next iteration
						u1_old_ref = u1_ref;
						u2_old_ref = u2_ref;
						uface_old_ref = uface_ref;
						energy_old = energy;

						//Fill the Jacobian with the gradients with respect to the internal points
						if (surf_t[i](0,v+rows*u) <= 0.f)
							printf("\n Warning!! The model is behind the camera. Problems with the projection");

						Matrix<float, 2, 3> J_pi;
						const float inv_z = 1.f / surf_t[i](0,v+rows*u);

						J_pi << fx*surf_t[i](1,v+rows*u)*square(inv_z), -fx*inv_z, 0.f,
								fy*surf_t[i](2,v+rows*u)*square(inv_z), 0.f, -fy*inv_z;
	
						Matrix2f J;
						Vector3f u_der_vec = u1_der[i].col(v+rows*u);
						J.col(0) = J_pi*rot_inv*u_der_vec;
						u_der_vec = u2_der[i].col(v+rows*u);
						J.col(1) = J_pi*rot_inv*u_der_vec;

						const Vector2f R = -res_pixels[i].col(v+rows*u);
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

							u1_incr_ref = sol(0);
							u2_incr_ref = sol(1);
							norm_uincr = sqrt(square(sol(0)) + square(sol(1)));

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
							const float u1_new = u1_old_ref + u1_incr_ref;
							const float u2_new = u2_old_ref + u2_incr_ref;
							bool alarm = false;
							if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
								alarm = updateInternalPointCrossingEdges(i, v, u);
							else
							{
								u1_ref = u1_new;
								u2_ref = u2_new;
								uface_ref = uface_old_ref;
							}

							//Re-evaluate the mesh with the new parametric coordinates
							evaluateSubDivSurfacePixel(i, v, u);

							//Compute the residuals
							surf_t[i].col(v+rows*u) = rot_inv*surf[i].col(v+rows*u) + tvec_inv;
							//surf_t[i](0,v+rows*u) = mytrans_inv(0,0)*mx[i](v,u) + mytrans_inv(0,1)*my[i](v,u) + mytrans_inv(0,2)*mz[i](v,u) + mytrans_inv(0,3);
							//surf_t[i](1,v+rows*u) = mytrans_inv(1,0)*mx[i](v,u) + mytrans_inv(1,1)*my[i](v,u) + mytrans_inv(1,2)*mz[i](v,u) + mytrans_inv(1,3);
							//surf_t[i](2,v+rows*u) = mytrans_inv(2,0)*mx[i](v,u) + mytrans_inv(2,1)*my[i](v,u) + mytrans_inv(2,2)*mz[i](v,u) + mytrans_inv(2,3);
							if (surf_t[i](0,v+rows*u) <= 0.f)
								printf("\n Depth coordinate of the internal correspondence is equal or inferior to zero after the transformation!!!");

							res_pixels[i](0,v+rows*u) = float(u) - (fx*(surf_t[i](1,v+rows*u) / surf_t[i](0,v+rows*u)) + disp_u);
							res_pixels[i](1,v+rows*u) = float(v) - (fy*(surf_t[i](2,v+rows*u) / surf_t[i](0,v+rows*u)) + disp_v);

							//Compute the energy associated to this pixel
							energy = res_pixels[i].col(v+rows*u).squaredNorm();
							//if (alarm)
							//{
							//	printf("\n Projection error (%d, %d) = %f", v, u, sqrt(energy));
							//	drawRayAndCorrespondence(i, v, u);
							//}


							if (energy > energy_old)
							{
								lambda *= lambda_mult;
								//printf("\n Energy is higher than before");
								//cout << endl << "Lambda updated: " << lambda;
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
								u1_ref = u1_old_ref;
								u2_ref = u2_old_ref;
								uface_ref = uface_old_ref;
								energy_increasing = false;
								energy = energy_old;
							}
						}

						energy_ratio = energy_old / energy;
					}

					//printf("\n Iters = %d, lambda = %.3f, ratio = %.5f, u_incr = %.4f, res = %.3f", inner_cont, lambda, energy_ratio, norm_uincr, sqrtf(energy));

					//Statistics
					aver_lambda += lambda;
					aver_num_iters += float(inner_cont);
					cont++;

				}
	}

	//printf("\n **** Average lambda = %f, aver_iter_per_pixel = %f, aver_first_uincr = %f ****", aver_lambda/cont, aver_num_iters/cont, aver_first_uincr/cont);
}

void Mod3DfromRGBD::optimizeUDataterm_LM()
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
	for (unsigned int i = 0; i < num_images; i++)
	{
		const Matrix3f rot_inv = cam_trans_inv[i].block<3,3>(0,0);
		const Vector3f tvec_inv = cam_trans_inv[i].block<3,1>(0,3);

		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			if (is_object[i](v, u))
			{
				energy = Kp*res_pos[i].col(v+rows*u).squaredNorm() + Kn*res_normals[i].col(v+rows*u).squaredNorm();
				energy_ratio = 2.f;
				norm_uincr = 1.f;
				lambda = 0.1f;
				inner_cont = 0;

				while (energy_ratio > 1.0005f)
				{
					u1_old[i](v, u) = u1[i](v, u);
					u2_old[i](v, u) = u2[i](v, u);
					uface_old[i](v, u) = uface[i](v, u);
					energy_old = energy;

					//Re-evaluate the normal derivatives
					computeNormalDerivatives_Analyt(i,v,u);

					//Fast access to normals
					const Vector3f n_vec = normals[i].col(v+rows*u);

					//Fill the Jacobian with the gradients with respect to the internal points
					Matrix<float, 6, 2> J;
					Matrix<float, 3, 2> u_der; 
					u_der << u1_der[i](0,v+rows*u), u2_der[i](0,v+rows*u), u1_der[i](1,v+rows*u), u2_der[i](1,v+rows*u), u1_der[i](2,v+rows*u), u2_der[i](2,v+rows*u);
					J.topRows(3) = -Kp_sqrt*rot_inv*u_der;

					const float inv_norm = 1.f/n_vec.norm();
					Matrix<float, 3, 2> n_der_u;
					n_der_u << n_der_u1[i](0,v+rows*u), n_der_u2[i](0,v+rows*u), n_der_u1[i](1,v+rows*u), n_der_u2[i](1,v+rows*u), n_der_u1[i](2,v+rows*u), n_der_u2[i](2,v+rows*u);
					Matrix3f J_nu;
					J_nu.row(0) << square(n_vec(1)) + square(n_vec(2)), -n_vec(0)*n_vec(1), -n_vec(0)*n_vec(2);
					J_nu.row(1) << -n_vec(0)*n_vec(1), square(n_vec(0)) + square(n_vec(2)), -n_vec(1)*n_vec(2);
					J_nu.row(2) << -n_vec(0)*n_vec(2), -n_vec(1)*n_vec(2), square(n_vec(0)) + square(n_vec(1));
					J_nu *= inv_norm*square(inv_norm);
					J.bottomRows(3) = -Kn_sqrt*rot_inv*J_nu*n_der_u;


					Matrix2f JtJ; JtJ.multiply_AtA(J);
					Matrix<float, 6, 1> R; R << Kp_sqrt*res_pos[i].col(v+rows*u),  Kn_sqrt*res_normals[i].col(v+rows*u);//******************************************************* 
																														//Kn_sqrt*res_nx[i](v,u), Kn_sqrt*res_ny[i](v,u), Kn_sqrt*res_nz[i](v,u);
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

						u1_incr[i](v, u) = sol(0);
						u2_incr[i](v, u) = sol(1);
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
						const float u1_new = u1_old[i](v, u) + u1_incr[i](v, u);
						const float u2_new = u2_old[i](v, u) + u2_incr[i](v, u);
						if ((u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f))
							updateInternalPointCrossingEdges(i, v, u);
						else
						{
							u1[i](v, u) = u1_new;
							u2[i](v, u) = u2_new;
							uface[i](v, u) = uface_old[i](v, u);
						}

						//Re-evaluate the mesh with the new parametric coordinates
						evaluateSubDivSurfacePixel(i, v, u);

						//Compute residuals
						surf_t[i].col(v+rows*u) = rot_inv*surf[i].col(v+rows*u) + tvec_inv;
						res_pos[i].col(v+rows*u) = xyz_image[i].col(v+rows*u) - surf_t[i].col(v+rows*u);
						//res_pos[i](0,v+rows*u) = depth[i](v,u) - surf_t[i](0,v+rows*u);
						//res_pos[i](1,v+rows*u) = x_image[i](v,u) - surf_t[i](1,v+rows*u);
						//res_pos[i](2,v+rows*u) = y_image[i](v,u) - surf_t[i](2,v+rows*u);

						const Vector3f n_vec_new = normals[i].col(v+rows*u);
						normals_t[i].col(v+rows*u) = rot_inv*n_vec_new;

						const float inv_norm = 1.f/n_vec_new.norm();
						res_normals[i].col(v+rows*u) = normals_image[i].col(v+rows*u) - inv_norm*normals_t[i].col(v+rows*u);
						//res_nx[i](v,u) = normals_image[i](0, v+rows*u) - inv_norm*normals_t[i](0,v+rows*u);
						//res_ny[i](v,u) = normals_image[i](1, v+rows*u) - inv_norm*normals_t[i](1,v+rows*u);
						//res_nz[i](v,u) = normals_image[i](2, v+rows*u) - inv_norm*normals_t[i](2,v+rows*u);

						//Compute the energy associated to this pixel
						energy = Kp*res_pos[i].col(v+rows*u).squaredNorm() + Kn*res_normals[i].col(v+rows*u).squaredNorm();

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
							u1[i](v, u) = u1_old[i](v, u);
							u2[i](v, u) = u2_old[i](v, u);
							uface[i](v, u) = uface_old[i](v, u);
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
	}

	//printf("\n **** Average lambda = %f, aver_iter_per_pixel = %f, aver_first_uincr = %f ****", aver_lambda/cont, aver_num_iters/cont, aver_first_uincr/cont);
}

void Mod3DfromRGBD::solveDT2_LM()	
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForDTBackground();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for Levenberg-Marquardt
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
					J_rows += 6;

	J_rows += num_images*nsamples;

	if (with_reg_normals)		J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_edges)			J_rows += 8*num_faces;
	if (with_reg_ctf)			J_rows += 3*num_verts;
	if (with_reg_atraction)		J_rows += 4*num_faces;
	if (with_reg_edges_iniShape) J_rows += 8*num_faces;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();
	computeTransCoordAndResiduals();
	optimizeUDataterm_LM();
	new_energy = computeEnergyDT2();
	utils::CTicTac clock; 

	if (paper_visualization)
		takePictureLimitSurface(false);

	printf("\n It enters the loop");

	//									Iterative solver
	//====================================================================================
	for (unsigned int iter = 0; iter < max_iter; iter++)
	{
		
		clock.Tic();
		unsigned int cont = 0;
		R.fill(0.f);

		//Occasional search for the correspondences
		if (((iter+1) % 5 == 0)&&(ctf_level > 2))
		{
			searchBetterUDataterm();
			evaluateSubDivSurface();
			computeTransCoordAndResiduals();
			printf("\n Global search. Energy after it = %f", new_energy = computeEnergyDT2());
		}

		//Update old variables
		last_energy = new_energy;
		vert_coords_old = vert_coords;
		cam_mfold_old = cam_mfold;

		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n It starts to compute the Jacobians"); clock.Tic();

		//							Compute the Jacobians
		//------------------------------------------------------------------------------------
		for (unsigned int i = 0; i < num_images; i++)
		{
			//Keep the last solution for u
			u1_old_outer[i] = u1[i];
			u2_old_outer[i] = u2[i];
			uface_old_outer[i] = uface[i];

			//Foreground
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (is_object[i](v,u))
					{
						//Warning
						if (surf_t[i](0,v+rows*u) <= 0.f)
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");

						//Data alignment
						fill_J_EpPixel(i, v, u, cont);

						//Normal alignment
						fill_J_EnPixel(i, v, u, cont);
					}

			//Background term with DT^2
			fill_J_BackDT2(i, cont);
		}

		//printf("\n It finishes with Jacobians (without regularization). Time = %f", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals)			fill_J_RegNormals(cont);
		if (with_reg_normals_good)		fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)		fill_J_RegNormalsCurvature(cont);
		if (with_reg_edges) 			fill_J_RegEdges(cont);
		if (with_reg_ctf)				fill_J_RegCTF(cont);
		if (with_reg_atraction)			fill_J_RegAtraction(cont);
		if (with_reg_edges_iniShape)	fill_J_RegEdgesIniShape(cont);

		//printf("\n It finishes with Jacobians (with regularization). Time = %f", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver
		J.setFromTriplets(j_elem.begin(), j_elem.end()); j_elem.clear(); 	//printf("\n It creates the sparse matrix. Time = %f", clock.Tac()); clock.Tic();
		SparseMatrix<float> JtJ_sparse = J.transpose()*J;					//printf("\n It computes JtJ. Time = %f", clock.Tac()); clock.Tic();
		MatrixXf JtJ = MatrixXf(JtJ_sparse);								//printf("\n It transforms JtJ to dense. Time = %f", clock.Tac()); clock.Tic();

		VectorXf b = -J.transpose()*R;										//printf("\n It computes b. Time = %f", clock.Tac()); clock.Tic();
		MatrixXf JtJ_lm;


		energy_increasing = true;
		unsigned int cont_inner = 0;

		//printf("\n It enters the loop solver-energy-check. Time = %f", clock.Tac()); clock.Tic();

		//			Update the control vertices and the camera poses and adapt step sizes
		//-----------------------------------------------------------------------------------------
		while (energy_increasing)
		{
			//Solve
			JtJ_lm = JtJ;
			for (unsigned int j=0; j<J_cols; j++)
				JtJ_lm(j,j) = (1.f + adap_mult)*JtJ_lm(j,j);


			//Solve the system
			increments = JtJ_lm.ldlt().solve(b);

			//printf("\n It solves with LM. Time = %f", clock.Tac()); clock.Tic();
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//printf("\n It updates variables. Time = %f", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			for (unsigned int i = 0; i < num_images; i++)
			{
				u1[i] = u1_old_outer[i];
				u2[i] = u2_old_outer[i];
				uface[i] = uface_old_outer[i];
			}
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	//printf("\n It creates topology, evaluates the surface and computes residuals. Time = %f", clock.Tac()); clock.Tic();
			optimizeUDataterm_LM();				//printf("\n It solves closest correspondence foreground. Time = %f", clock.Tac()); clock.Tic();
			sampleSurfaceForDTBackground();
			new_energy = computeEnergyDT2();


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
			if (cont_inner > 15) 
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
				evaluateSubDivSurface();			
				if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
					evaluateSubDivSurfaceRegularization();
				computeTransCoordAndResiduals();
				optimizeUDataterm_LM();
				sampleSurfaceForDTBackground();
				new_energy = computeEnergyDT2();

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

		//Show the update
		if (!paper_visualization) {	showMesh(); showCamPoses(); showSubSurface(); showRenderedModel(); }
		else						takePictureLimitSurface(false);

		
		//printf("\n Time to finish everything else = %f", clock.Tac()); clock.Tic();
		printf("\n New_energy = %f, last_energy = %f, iter time(s) = %.3f", new_energy, last_energy, runtime);
		if ((energy_increasing)||(new_energy > last_energy - 0.0001f))
		{
			printf("\n Optimization finished because energy does not decrease anymore");
			printf("\n Average runtime = %f", aver_runtime / (iter+1));
			break;
		}
		else if (iter == max_iter - 1)
			printf("\n Average runtime = %f", aver_runtime / max_iter);
	}
}


float Mod3DfromRGBD::computeEnergyDT2()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = res_pos[i].col(v+rows*u).norm();
					energy_d += Kp*square(min(res, truncated_res));

					if (fit_normals_old)
					{
						const float resn = res_normals[i].col(v+rows*u).norm(); 
						energy_d += Kn*n_weights[i](v+rows*u)*square(min(resn, truncated_resn));	
					}
				}

		for (unsigned int s = 0; s < nsamples; s++)
		{
			const int v = int(pixel_DT_v[i](s)), u = int(pixel_DT_u[i](s));
			energy_b += alpha*square(DT[i](v,u));
		}
	}

	//Regularization
	if (with_reg_normals)			energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)		energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)		energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)				energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)				energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)			energy_r += computeEnergyRegAtraction();
	if (with_reg_edges_iniShape)	energy_r += computeEnergyRegEdgesIniShape();
	if (with_reg_arap)				energy_r += computeEnergyRegArap();
	if (with_reg_rot_arap)			energy_r += computeEnergyRegRotArap();

	const float energy_o = energy_d + energy_r + energy_b;
	//printf("\n Energies: overall = %.4f, dataterm = %.4f, reg = %.4f, backg = %.4f", energy_o, energy_d, energy_r, energy_b);

	return energy_o;
}

float Mod3DfromRGBD::computeEnergyBS()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = res_pos[i].col(v+rows*u).norm();
					energy_d += Kp*square(min(res, truncated_res));

					const float resn = res_normals[i].col(v+rows*u).norm(); 
					energy_d += Kn*n_weights[i](v+rows*u)*square(min(resn, truncated_resn));	
				}

		for (unsigned int s = 0; s < nsamples; s++)
		{
			//**************** I should interpolate ************************
			//const float v = pixel_DT_v[i](s), u = pixel_DT_u[i](s);
			//const int v_up = ceilf(pixel_DT_v[i](s)), v_down = floorf(pixel_DT_v[i](s));
			//const int u_up = ceilf(pixel_DT_u[i](s)), u_down = floorf(pixel_DT_u[i](s));
			//const float DT_here = (v_up-v)*(u_up-u)*DT[i](v_down, u_down) + 

			const int v = roundf(pixel_DT_v[i](s)), u = roundf(pixel_DT_u[i](s)); 
			energy_b += alpha*square(DT[i](v,u)*norm_n_DT(s));
		}
	}

	//Regularization
	if (with_reg_normals)			energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)		energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)		energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)				energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)				energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)			energy_r += computeEnergyRegAtraction();
	if (with_reg_edges_iniShape)	energy_r += computeEnergyRegEdgesIniShape();
	if (with_reg_arap)				energy_r += computeEnergyRegArap();
	if (with_reg_rot_arap)			energy_r += computeEnergyRegRotArap();

	const float energy_o = energy_d + energy_r + energy_b;
	//printf("\n Energies: overall = %.4f, dataterm = %.4f, reg = %.4f, backg = %.4f", energy_o, energy_d, energy_r, energy_b);

	return energy_o;
}

float Mod3DfromRGBD::computeEnergyBG()
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = res_pos[i].col(v+rows*u).norm();
					energy_d += Kp*square(min(res, truncated_res));

					if (fit_normals_old)
					{
						const float resn = res_normals[i].col(v+rows*u).norm(); 
						energy_d += Kn*n_weights[i](v+rows*u)*square(min(resn, truncated_resn));
					}

					if (with_color)
						energy_d += Kc*square(res_color[i](v+rows*u));
				}

		for (unsigned int s = 0; s < nsamples; s++)
		{
			//**************** I should interpolate ************************
			//const float v = pixel_DT_v[i](s), u = pixel_DT_u[i](s);
			//const int v_up = ceilf(pixel_DT_v[i](s)), v_down = floorf(pixel_DT_v[i](s));
			//const int u_up = ceilf(pixel_DT_u[i](s)), u_down = floorf(pixel_DT_u[i](s));
			//const float DT_here = (v_up-v)*(u_up-u)*DT[i](v_down, u_down) + 

			const int v = roundf(pixel_DT_v[i](s)), u = roundf(pixel_DT_u[i](s)); 
			energy_b += alpha*square(DT[i](v,u))*(square(u1_dernorm_DT(s)) + square(u2_dernorm_DT(s)));
		}
	}

	//Regularization
	if (with_reg_normals)			energy_r += computeEnergyRegNormals();
	if (with_reg_normals_good)		energy_r += computeEnergyRegNormalsGood();
	if (with_reg_normals_4dir)		energy_r += computeEnergyRegNormals4dir();
	if (with_reg_edges)				energy_r += computeEnergyRegEdges();
	if (with_reg_ctf)				energy_r += computeEnergyRegCTF();
	if (with_reg_atraction)			energy_r += computeEnergyRegAtraction();
	if (with_reg_edges_iniShape)	energy_r += computeEnergyRegEdgesIniShape();
	if (with_reg_arap)				energy_r += computeEnergyRegArap();
	if (with_reg_rot_arap)			energy_r += computeEnergyRegRotArap();
	if (with_color)					energy_r += computeEnergyRegVertColor();

	const float energy_o = energy_d + energy_r + energy_b;
	//printf("\n Energies: overall = %.4f, dataterm = %.4f, reg = %.4f, backg = %.4f", energy_o, energy_d, energy_r, energy_b);

	return energy_o;
}




