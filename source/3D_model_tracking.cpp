// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"


void Mod3DfromRGBD::rotMatricesFromTwist()
{
	Matrix3f kai_mat;
	
	for (unsigned int v=0; v<num_verts; v++)
	{
		kai_mat << 0.f, -rot_mfold[v](2), rot_mfold[v](1),
				rot_mfold[v](2), 0.f, -rot_mfold[v](0),
				-rot_mfold[v](1), rot_mfold[v](0), 0.f;

		rot_arap[v] = kai_mat.exp();
	}
}


float Mod3DfromRGBD::computeEnergyRegArap()
{
	float energy = 0.f;

	for (unsigned int v=0; v<num_verts; v++)
		for (unsigned int e=0; e<valence(v); e++)
		{						
			const unsigned int other_v = neighbourhood(e,v);

			const Vector3f edge_new = (vert_coords.col(other_v) - vert_coords.col(v)).matrix();
			const Vector3f edge_old = (vert_coords_reg.col(other_v) - vert_coords_reg.col(v)).matrix();
			const Vector3f dif = edge_new - rot_arap[v]*edge_old;

			energy += K_arap*dif.squareNorm();
		}

	return energy;	
}

float Mod3DfromRGBD::computeEnergyRegRotArap()
{
	float energy = 0.f;
	
	for (unsigned int v=0; v<num_verts; v++)
		for (unsigned int e=0; e<valence(v); e++)
		{
			const unsigned int v2 = neighbourhood(e,v);
			energy += K_rot_arap*(rot_mfold[v] - rot_mfold[v2]).squaredNorm(); ///dist;
		}		

	return energy;
}

void Mod3DfromRGBD::findOppositeVerticesFace()
{
	opposite_verts.resize(16, num_faces);
	for (unsigned int f=0; f<num_faces; f++)
	{
		unsigned int row_insert = 0;	
		for (unsigned int k=0; k<4; k++)
		{
			const unsigned int fadj = face_adj(k,f);

			//First: vertices in f which are not in fadj
			for (unsigned int v=0; v<4; v++)
			{
				const int vert_here = face_verts(v,f);
				bool v_repeated = false;
				for (unsigned int v_fadj=0; v_fadj<4; v_fadj++)
					if (face_verts(v_fadj,fadj) == vert_here)
						v_repeated = true;

				if (!v_repeated)
				{
					opposite_verts(row_insert,f) = vert_here;
					row_insert++;
				}
			}

			//Second: vertices in fadj which are not in f
			for (unsigned int v=0; v<4; v++)
			{
				const int vert_here = face_verts(v,fadj);
				bool v_repeated = false;
				for (unsigned int v_f=0; v_f<4; v_f++)
					if (face_verts(v_f,f) == vert_here)
						v_repeated = true;

				if (!v_repeated)
				{
					opposite_verts(row_insert,f) = vert_here;
					row_insert++;
				}
			}
		}
	}
}

void Mod3DfromRGBD::solveNB_Arap()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images + 3*num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += fit_normals_old ? 6 : 3;
					J_cols += 2;
				}

	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_arap)			J_rows += num_eq_arap;
	if (with_reg_rot_arap)		J_rows += num_eq_arap;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();

	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();

	computeTransCoordAndResiduals();
	new_energy = computeEnergyNB();

	//if (paper_visualization) takePictureLimitSurface(true);

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
		if (((iter+1) % 5 == 0)&&(num_faces > 30))
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
		rot_mfold_old = rot_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n Start to compute the Jacobian"); clock.Tic();

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
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							
						//Data alignment
						fill_J_EpArap(i, v, u, cont);

						//Normal alignment
						if (fit_normals_old)
							fill_J_EnArap(i, v, u, cont);										
					}
		}

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_arap)			fill_J_RegArap(cont);
		if (with_reg_rot_arap)		fill_J_RegRotArap(cont);

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

			for (unsigned int j=0; j<J_cols; j++)
			{
				const float lm_correction = (1.f + adap_mult)*JtJ_lm.coeffRef(j,j);
				if (lm_correction != 0.f)
					JtJ_lm.coeffRef(j,j) = lm_correction;
				else
				{
					JtJ_lm.insert(j,j) = 1.f;
					printf("\n Null value in the diagonal (unknown %d)", j);
					printf("\n 3*Num_verts = %d, +3*num_faces = %d", 3*num_verts, 3*num_verts+3*num_faces);
				}
			}

			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)		printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 		printf("Solving failed");

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

			//Rotations of ARAP
			for (unsigned int v=0; v<num_verts; v++)
				for (unsigned int k=0; k<3; k++)
					rot_mfold[v](k) = rot_mfold_old[v](k) + increments(cont++);
			rotMatricesFromTwist();

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
							//if (high_uincr > 10.f)		printf("\n warning high incr = %f.  image = %d, v = %d, u = %d", high_uincr, i, v, u);	

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

			//printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();

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
			if ((cont_inner > 5) || ((new_energy/last_energy > convergence_ratio)&&(energy_increasing == false)))
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
					rot_mfold = rot_mfold_old;
					rotMatricesFromTwist();
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
		if (paper_visualization)	;//takePictureLimitSurface(true);
		else						{showMesh(); showCamPoses(); showSubSurface(); showRenderedModel();}
		//saveSceneAsImage();
		//takePictureDataArch();
		


		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
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

void Mod3DfromRGBD::solveSK_Arap()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	//Variables for the LM solver
	unsigned int J_rows = 0, J_cols = 3*num_verts + optimize_cameras*6*num_images + 3*num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += fit_normals_old ? 6 : 3;
					J_cols += 2;
				}
				else if (valid[i](v,u))
					J_rows++;

	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_arap)			J_rows += num_eq_arap;
	if (with_reg_rot_arap)		J_rows += num_eq_arap;

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
		if (((iter+1) % 8 == 0)&&(num_faces > 30))
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
		rot_mfold_old = rot_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		u1_old_outer = u1_old;
		u2_old_outer = u2_old;
		uface_old_outer = uface_old;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n Start to compute the Jacobian"); clock.Tic();

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
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							
						//Data alignment
						fill_J_EpArap(i, v, u, cont);

						//Normal alignment
						if (fit_normals_old)
							fill_J_EnArap(i, v, u, cont);										
					}

		//Background
		for (unsigned int i = 0; i < num_images; i++)
			for (unsigned int u = 0; u < cols; u++)
				for (unsigned int v = 0; v < rows; v++)
					if (valid[i](v,u) && !is_object[i](v,u))
						fill_J_BackSKPixel(i, v, u, cont);

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_arap)			fill_J_RegArap(cont);
		if (with_reg_rot_arap)		fill_J_RegRotArap(cont);

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
			}

			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)		printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 		printf("Solving failed");

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

			//Rotations of ARAP
			for (unsigned int v=0; v<num_verts; v++)
				for (unsigned int k=0; k<3; k++)
					rot_mfold[v](k) = rot_mfold_old[v](k) + increments(cont++);
			rotMatricesFromTwist();

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

			//printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue; }

			optimizeUBackground_LM();
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
			//Try one iteration with a very low lambda
			//if (cont_inner > 5) 
			//{
			//	adap_mult *= 0.001f;
			//}
			//Try a complete search
			if ((cont_inner > 5) || ((new_energy/last_energy > convergence_ratio)&&(energy_increasing == false)))
			{
				adap_mult *= 100.f;
				
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
					rot_mfold = rot_mfold_old;
					rotMatricesFromTwist();
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
		if (!paper_visualization)
		{
			showMesh();
			showCamPoses();
			showSubSurface();
			showRenderedModel();
		}
		//saveSceneAsImage();
		//takePictureDataArch();


		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
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

void Mod3DfromRGBD::solveDT2_Arap()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForDTBackground();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	unsigned int J_rows = num_images*nsamples, J_cols = 3*num_verts + optimize_cameras*6*num_images + 3*num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += fit_normals_old ? 6 : 3;
					J_cols += 2;
				}

	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_arap)			J_rows += num_eq_arap;
	if (with_reg_rot_arap)		J_rows += num_eq_arap;

	J.resize(J_rows, J_cols);
	R.resize(J_rows);
	increments.resize(J_cols);

	//Prev computations
	evaluateSubDivSurface();

	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		evaluateSubDivSurfaceRegularization();

	computeTransCoordAndResiduals();
	new_energy = computeEnergyDT2();

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
		if (((iter+1) % 5 == 0)&&(num_faces > 30))
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
		rot_mfold_old = rot_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n Start to compute the Jacobian"); clock.Tic();

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
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							
						//Data alignment
						fill_J_EpArap(i, v, u, cont);

						//Normal alignment
						if (fit_normals_old)
							fill_J_EnArap(i, v, u, cont);										
					}

			//Background term with DT^2
			fill_J_BackDT2(i, cont);
		}

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_arap)			fill_J_RegArap(cont);
		if (with_reg_rot_arap)		fill_J_RegRotArap(cont);

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

			for (unsigned int j=0; j<J_cols; j++)
			{
				const float lm_correction = (1.f + adap_mult)*JtJ_lm.coeffRef(j,j);
				if (lm_correction != 0.f)
					JtJ_lm.coeffRef(j,j) = lm_correction;
				else
				{
					JtJ_lm.insert(j,j) = 1.f;
					printf("\n Null value in the diagonal (unknown %d)", j);
					printf("\n 3*Num_verts = %d, +3*num_faces = %d", 3*num_verts, 3*num_verts+3*num_faces);
				}
			}

			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)		printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 		printf("Solving failed");

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

			//Rotations of ARAP
			for (unsigned int v=0; v<num_verts; v++)
				for (unsigned int k=0; k<3; k++)
					rot_mfold[v](k) = rot_mfold_old[v](k) + increments(cont++);
			rotMatricesFromTwist();

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
							//if (high_uincr > 10.f)		printf("\n warning high incr = %f.  image = %d, v = %d, u = %d", high_uincr, i, v, u);	

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

			//printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue; }

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
			if ((cont_inner > 5) || ((new_energy/last_energy > convergence_ratio)&&(energy_increasing == false)))
			{
				//Last attempt to reduce the energy
				printf("\n Last attempt to reduce the energy");
				searchBetterUDataterm();
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
					rot_mfold = rot_mfold_old;
					rotMatricesFromTwist();
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
		if (!paper_visualization)
		{
			showMesh();
			showCamPoses();
			showSubSurface();
			showRenderedModel();
		}
		//saveSceneAsImage();
		//takePictureDataArch();


		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
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

void Mod3DfromRGBD::solveBS_Arap()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForBSTerm();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	unsigned int J_rows = num_images*nsamples, J_cols = 3*num_verts + optimize_cameras*6*num_images + 3*num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += 6;
					J_cols += 2;
				}

	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_arap)			J_rows += num_eq_arap;
	if (with_reg_rot_arap)		J_rows += num_eq_arap;

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
		if (((iter+1) % 5 == 0)&&(num_faces > 30))
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
		rot_mfold_old = rot_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n Start to compute the Jacobian"); clock.Tic();

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
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							
						//Data alignment
						fill_J_EpArap(i, v, u, cont);

						//Normal alignment
						fill_J_EnArap(i, v, u, cont);										
					}

			//Background term with DT^2
			fill_J_BackBS(i, cont);
		}

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_arap)			fill_J_RegArap(cont);
		if (with_reg_rot_arap)		fill_J_RegRotArap(cont);

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

			for (unsigned int j=0; j<J_cols; j++)
			{
				const float lm_correction = (1.f + adap_mult)*JtJ_lm.coeffRef(j,j);
				if (lm_correction != 0.f)
					JtJ_lm.coeffRef(j,j) = lm_correction;
				else
				{
					JtJ_lm.insert(j,j) = 1.f;
					printf("\n Null value in the diagonal (unknown %d)", j);
					printf("\n 3*Num_verts = %d, +3*num_faces = %d", 3*num_verts, 3*num_verts+3*num_faces);
				}
			}

			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)		printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 		printf("Solving failed");

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

			//Rotations of ARAP
			for (unsigned int v=0; v<num_verts; v++)
				for (unsigned int k=0; k<3; k++)
					rot_mfold[v](k) = rot_mfold_old[v](k) + increments(cont++);
			rotMatricesFromTwist();

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
							//if (high_uincr > 10.f)		printf("\n warning high incr = %f.  image = %d, v = %d, u = %d", high_uincr, i, v, u);	

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

			//printf("\n Update variables - %f sec", clock.Tac()); clock.Tic();

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue; }

			evaluateSurfaceForBSSamples();
			new_energy = computeEnergyBS();

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
			if ((cont_inner > 5) || ((new_energy/last_energy > convergence_ratio)&&(energy_increasing == false)))
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
					rot_mfold = rot_mfold_old;
					rotMatricesFromTwist();
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
		if (!paper_visualization)
		{
			showMesh();
			showCamPoses();
			showSubSurface();
			showRenderedModel();
		}
		//saveSceneAsImage();
		//takePictureDataArch();


		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
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

void Mod3DfromRGBD::solveBG_Arap()
{
	//								Initialize
	//======================================================================================
	float last_energy, new_energy, aver_runtime = 0.f;
	bool energy_increasing;

	sampleSurfaceForBSTerm();	//Must be here because it sets the number of samples. Correct it in the future!!!!!!!!!!!!!!!

	//Variables for the LM solver
	unsigned int J_rows = 2*num_images*nsamples, J_cols = unk_per_vertex*num_verts + optimize_cameras*6*num_images + 3*num_verts;
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))	
				{
					J_rows += fit_normals_old ? 3 + unk_per_vertex : unk_per_vertex;
					J_cols += 2;
				}

	if (with_reg_normals_good)	J_rows += 6*num_faces*square(s_reg);
	if (with_reg_normals_4dir)	J_rows += 6*num_faces*(square(s_reg) + square(s_reg-1));
	if (with_reg_arap)			J_rows += num_eq_arap;
	if (with_reg_rot_arap)		J_rows += num_eq_arap;
	if (with_color)				J_rows += 4*num_faces;

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
		if (((iter+1) % 5 == 0)&&(num_faces > 30))
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
		rot_mfold_old = rot_mfold;
		u1_old = u1;
		u2_old = u2;
		uface_old = uface;
		if (with_color) vert_colors_old = vert_colors;

		//Evaluate surface and compute residuals for the current solution
		evaluateSubDivSurface();
		computeTransCoordAndResiduals();

		//printf("\n Start to compute the Jacobian"); clock.Tic();

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
							printf("\n Warning!! A point of the model is behind the camera, which will surely be catastrophic");
							
						//Data alignment
						fill_J_EpArap(i, v, u, cont);

						//Normal alignment
						if (fit_normals_old)
							fill_J_EnArap(i, v, u, cont);										
					}

			//Background term with DT^2
			fill_J_BackBG(i, cont);
		}

		//printf("\n Fill J (dataterm) - %f sec", clock.Tac()); clock.Tic();

		//Include regularization
		if (with_reg_normals_good)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_normals_4dir)	fill_J_RegNormalsCurvature(cont);
		if (with_reg_arap)			fill_J_RegArap(cont);
		if (with_reg_rot_arap)		fill_J_RegRotArap(cont);
		if (with_color)				fill_J_RegVertColor(cont);
		//printf("\n Fill J (regularization) - %f sec", clock.Tac()); clock.Tic();

		//Prepare Levenberg solver - It seems that creating J within the method makes it faster
		const float max_j = 0.f; //2e-2f;
		vector<Tri> j_elem_trunc;
		for (unsigned int k=0; k<j_elem.size(); k++)
			if (abs(j_elem[k].value()) > max_j)
				j_elem_trunc.push_back(j_elem[k]);
		//printf("\n percentage of used elements (J) = %f", 100.f*float(j_elem_trunc.size())/float(j_elem.size()));


		J.setFromTriplets(j_elem_trunc.begin(), j_elem_trunc.end()-1); j_elem.clear();
		const SparseMatrix<float> JtJ_sparse = J.transpose()*J;
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

			for (unsigned int j=0; j<J_cols; j++)
			{
				const float lm_correction = (1.f + adap_mult)*JtJ_lm.coeffRef(j,j);
				if (lm_correction != 0.f)
					JtJ_lm.coeffRef(j,j) = lm_correction;
				else
				{
					JtJ_lm.insert(j,j) = 0.001f;
					printf("\n Null value in the diagonal (unknown %d)", j);
					//printf("\n 3*Num_verts = %d, +3*num_faces = %d", 3*num_verts, 3*num_verts+3*num_faces);
				}
			}

			//Solve the system
			SimplicialLDLT<SparseMatrix<float>> solver;
			solver.compute(JtJ_lm);
			if(solver.info()!=Success)		printf("Decomposition failed");
			increments = solver.solve(b);
			if(solver.info()!=Success) 		printf("Solving failed");
			
			//Update variables
			cont = 0;
			
			//Control vertices
			for (unsigned int k = 0; k < num_verts; k++)
			{
				for (unsigned int c = 0; c < 3; c++)
					vert_coords(c, k) = vert_coords_old(c,k) + increments(cont++);

				if (with_color)
					vert_colors(k) = vert_colors_old(k) + increments(cont++);
			}


			//Camera poses
			if (optimize_cameras)
			{
				for (unsigned int i = 0; i < num_images; i++)
					for (unsigned int k = 0; k < 6; k++)
						cam_mfold[i](k) = cam_mfold_old[i](k) + increments(cont++);
				computeCameraTransfandPosesFromTwist();
			}

			//Rotations of ARAP
			for (unsigned int v=0; v<num_verts; v++)
				for (unsigned int k=0; k<3; k++)
					rot_mfold[v](k) = rot_mfold_old[v](k) + increments(cont++);
			rotMatricesFromTwist();

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
							//if (high_uincr > 10.f)		printf("\n warning high incr = %f.  image = %d, v = %d, u = %d", high_uincr, i, v, u);	

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

			//Check whether the energy is increasing or decreasing
			createTopologyRefiner();		
			evaluateSubDivSurface();			
			if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
				evaluateSubDivSurfaceRegularization();
			computeTransCoordAndResiduals();	

			//Don't use this solution if the model goes behind the cameras
			if (behind_cameras) {adap_mult *= 4.f; behind_cameras = false; continue; }

			evaluateSurfaceForBGSamples();
			new_energy = computeEnergyBG();


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
			if ((cont_inner > 5) || ((new_energy/last_energy > convergence_ratio)&&(energy_increasing == false)))
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
					rot_mfold = rot_mfold_old;
					rotMatricesFromTwist();
					u1 = u1_old;
					u2 = u2_old;
					uface = uface_old;
					if (with_color) vert_colors = vert_colors_old;
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
		if (!paper_visualization)
		{
			showMesh();
			showCamPoses(); 
			showSubSurface();
			showRenderedModel();
		}
		//saveSceneAsImage();
		//takePictureDataArch();


		//printf("\n Time to finish everything else - %f sec", clock.Tac()); clock.Tic();
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



void Mod3DfromRGBD::findNeighbourhoodsForArap()
{
	//Neighbourhood - Four adjacent vertices
	neighbourhood.resize(6,num_verts); //Assuming that valences are never higher than 6
	neighbourhood.fill(-1);
	valence.resize(num_verts);

	for (unsigned int v=0; v<num_verts; v++)
	{
		vector<unsigned int> neigh_repeated;
		
		for (unsigned int f=0; f<num_faces; f++)
		{
			for (int k=0; k<4; k++)
				if (v == face_verts(k,f))
				{
					const unsigned int prev = k-1 < 0 ? 3 : k-1;
					const unsigned int foll = (k+1)%4;
					const unsigned int v_prev = face_verts(prev,f);
					const unsigned int v_foll = face_verts(foll,f);

					neigh_repeated.push_back(v_prev);
					neigh_repeated.push_back(v_foll);
					break;
				}
		}

		sort(neigh_repeated.begin(), neigh_repeated.end());
		valence(v) = neigh_repeated.size()/2;

		for (unsigned int k=0; k<valence(v); k++)
			neighbourhood(k,v) = neigh_repeated[2*k];
	}

	num_eq_arap = 3*valence.matrix().sumAll();
}


void Mod3DfromRGBD::fill_J_EpArap(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const float Kp_sqrt = sqrtf(Kp);
	const float Kc_sqrt = sqrtf(Kc);
	const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);
	
	const float res = res_pos[i].col(v+rows*u).norm();
	if (res < truncated_res)		//****************************** warning for the color *******************
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
				
				//Geometry
				for (unsigned int k=0; k<3; k++)
					for (unsigned int l=0; l<3; l++)
						if (prod(l,k) != 0.f)
							j_elem.push_back(Tri(J_row+l, unk_per_vertex*cp+k, prod(l,k)*v_weight));

				//Color
				if ((with_color) && (v_weight != 0.f))
					j_elem.push_back(Tri(J_row+3, unk_per_vertex*cp+3, -Kc_sqrt*v_weight));
			}
		}


		//Camera poses
		if (optimize_cameras)
		{
			//Vector4f t_point; t_point << mx_t[i](v, u), my_t[i](v, u), mz_t[i](v, u), 1.f;
			const Vector3f t_point = surf_t[i].col(v+rows*u); 
			const unsigned int cam_shift = unk_per_vertex*num_verts + 6*i;

			//translations
			j_elem.push_back(Tri(J_row, cam_shift, -Kp_sqrt));
			j_elem.push_back(Tri(J_row+1, cam_shift + 1, -Kp_sqrt));
			j_elem.push_back(Tri(J_row+2, cam_shift + 2, -Kp_sqrt));

			//rotations
			j_elem.push_back(Tri(J_row+1, cam_shift + 3, Kp_sqrt*t_point(2)));
			j_elem.push_back(Tri(J_row+2, cam_shift + 3, -Kp_sqrt*t_point(1)));

			j_elem.push_back(Tri(J_row, cam_shift + 4, -Kp_sqrt*t_point(2)));
			j_elem.push_back(Tri(J_row+2, cam_shift + 4, Kp_sqrt*t_point(0)));

			j_elem.push_back(Tri(J_row, cam_shift + 5, Kp_sqrt*t_point(1)));
			j_elem.push_back(Tri(J_row+1, cam_shift + 5, -Kp_sqrt*t_point(0)));

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
		const unsigned int ind_bias = unk_per_vertex*num_verts + optimize_cameras*6*num_images + 3*num_verts + 2*(J_row/(unk_per_vertex + 3*fit_normals_old)); 

		//Geometry
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + k, J_u(l,k)));

		//Color
		if (with_color)
		{
			j_elem.push_back(Tri(J_row+3, ind_bias, -Kc_sqrt*u1_der_color[i](v+rows*u)));
			j_elem.push_back(Tri(J_row+3, ind_bias+1, -Kc_sqrt*u2_der_color[i](v+rows*u)));
		}

		//Fill the residuals
		R.middleRows(J_row,3) = Kp_sqrt*res_pos[i].col(v+rows*u);
		//R(J_row) = Kp_sqrt*res_x[i](v,u);
		//R(J_row+1) = Kp_sqrt*res_y[i](v,u);
		//R(J_row+2) = Kp_sqrt*res_z[i](v,u);
		if (with_color) R(J_row+3) = Kc_sqrt*res_color[i](v+rows*u);
	}
	//Simplest (and probably bad) solution to the problem of underdetermined unknowns for the solver
	else
	{
		const unsigned int ind_bias = unk_per_vertex*num_verts + optimize_cameras*6*num_images + 3*num_verts + 2*(J_row/(unk_per_vertex+3*fit_normals_old)); 
		for (unsigned int k=0; k<2; k++)
			for (unsigned int l=0; l<3; l++)
				j_elem.push_back(Tri(J_row+l, ind_bias + k, 1.f));	
	}

	J_row += unk_per_vertex;	
}

void Mod3DfromRGBD::fill_J_EnArap(unsigned int i, unsigned int v, unsigned int u, unsigned int &J_row)
{
	const float Kn_sqrt = sqrtf(Kn);
	const float wn_sqrt = sqrtf(n_weights[i](v +rows*u));	
	const float resn = res_normals[i].col(v+rows*u).norm();
	const int index = v+rows*u;

	if (resn < truncated_resn)
	{
		const Matrix3f T_inv = cam_trans_inv[i].block<3,3>(0,0);
		
		const float inv_norm = 1.f/normals[i].col(index).norm();
		//const float inv_norm = 1.f/sqrtf(square(nx[i](v,u)) + square(ny[i](v,u)) + square(nz[i](v,u)));

		Matrix3f J_nu, J_nX;
		const float J_nu_xy = -normals[i](0,index)*normals[i](1,index);
		const float J_nu_xz = -normals[i](0,index)*normals[i](2,index);
		const float J_nu_yz = -normals[i](1,index)*normals[i](2,index);
		J_nu.row(0) << square(normals[i](1,index)) + square(normals[i](2,index)), J_nu_xy, J_nu_xz;
		J_nu.row(1) << J_nu_xy, square(normals[i](0,index)) + square(normals[i](2,index)), J_nu_yz;
		J_nu.row(2) << J_nu_xz, J_nu_yz, square(normals[i](0,index)) + square(normals[i](1,index));
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
						j_elem.push_back(Tri(J_row+l, unk_per_vertex*cp+k, J_norm_fit(l,k)));
			}
		}

		//Camera pose	
		if (optimize_cameras)
		{
			const Vector3f normal = normals[i].col(index);
			const Vector3f n_t = Kn_sqrt*wn_sqrt*T_inv*inv_norm*normal;
			const unsigned int cam_shift = unk_per_vertex*num_verts + 6*i;

			//rotations only for the normals
			j_elem.push_back(Tri(J_row+1, cam_shift + 3, n_t(2)));
			j_elem.push_back(Tri(J_row+2, cam_shift + 3, -n_t(1)));

			j_elem.push_back(Tri(J_row, cam_shift + 4, -n_t(2)));
			j_elem.push_back(Tri(J_row+2, cam_shift + 4, n_t(0)));

			j_elem.push_back(Tri(J_row, cam_shift + 5, n_t(1)));
			j_elem.push_back(Tri(J_row+1, cam_shift + 5, -n_t(0)));

			////General expression (inefficient)
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
		const unsigned int ind_bias = unk_per_vertex*num_verts + optimize_cameras*6*num_images + 3*num_verts + 2*(J_row/(unk_per_vertex+3)); 
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


void Mod3DfromRGBD::fill_J_RegEdgesIniShape(unsigned int &J_row)
{
	K_ini = K_ini_total*float(num_faces);
	const float K_ini_sqrt = sqrtf(K_ini);
	const float sqrt_2 = sqrtf(2.f);
	
	for (unsigned int f=0; f<num_faces; f++)
	{
		//Sides of the faces
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
			//const float rL = length_0 - length_1;

			const Vector3f J_vert_e0 = K_ini_sqrt*(vert_coords.col(vert_e0) - vert_coords.col(vert_e1))/length;
			const Vector3f J_vert_e1 = -J_vert_e0;

			j_elem.push_back(Tri(J_row, 3*vert_e0, J_vert_e0(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e1, J_vert_e1(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+1, J_vert_e0(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+1, J_vert_e1(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+2, J_vert_e0(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+2, J_vert_e1(2)));

			//Fill the residual
			R(J_row) = K_ini_sqrt*(length - length_ini);
			J_row++;
		}

		//opposite vertices of adjacent faces
		for (unsigned int k=0; k<4; k++)
		{							
			const unsigned int fadj = face_adj(k,f);
			const unsigned int vert_e0 = opposite_verts(0+4*k,f);
			const unsigned int vert_e1 = opposite_verts(1+4*k,f);
			const unsigned int vert_e2 = opposite_verts(2+4*k,f);
			const unsigned int vert_e3 = opposite_verts(3+4*k,f);

			const Vector3f edge = (0.5f*(vert_coords.col(vert_e2) + vert_coords.col(vert_e3))
								 - 0.5f*(vert_coords.col(vert_e0) + vert_coords.col(vert_e1))).square().matrix();
			const float length = sqrtf(edge.sumAll());
			const Vector3f edge_ini = (0.5f*(vert_coords_reg.col(vert_e2) + vert_coords_reg.col(vert_e3))
									 - 0.5f*(vert_coords_reg.col(vert_e0) + vert_coords_reg.col(vert_e1))).square().matrix();
			const float length_ini = sqrtf(edge_ini.sumAll());

			const Vector3f J_vert_e01 = 0.5f*K_ini_sqrt*(vert_coords.col(vert_e0) + vert_coords.col(vert_e1)
													   - vert_coords.col(vert_e2) + vert_coords.col(vert_e3))/length;
			const Vector3f J_vert_e23 = -J_vert_e01;

			j_elem.push_back(Tri(J_row, 3*vert_e0, J_vert_e01(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e1, J_vert_e01(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e2, J_vert_e23(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e3, J_vert_e23(0)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+1, J_vert_e01(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+1, J_vert_e01(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+1, J_vert_e23(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e3+1, J_vert_e23(1)));
			j_elem.push_back(Tri(J_row, 3*vert_e0+2, J_vert_e01(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e1+2, J_vert_e01(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e2+2, J_vert_e23(2)));
			j_elem.push_back(Tri(J_row, 3*vert_e3+2, J_vert_e23(2)));

			//Fill the residual
			R(J_row) = K_ini_sqrt*(length - length_ini);
			J_row++;
		}
	}
}

void Mod3DfromRGBD::fill_J_RegArap(unsigned int &J_row)
{
	const float K_arap_sqrt = sqrtf(K_arap);
	const unsigned int j_col_bias = unk_per_vertex*num_verts + optimize_cameras*6*num_images;
	
	for (unsigned int v=0; v<num_verts; v++)
	{
		for (unsigned int e=0; e<valence(v); e++)
		{						
			const unsigned int other_v = neighbourhood(e,v);

			const Vector3f edge_new = (vert_coords.col(other_v) - vert_coords.col(v)).matrix();
			const Vector3f edge_old = (vert_coords_reg.col(other_v) - vert_coords_reg.col(v)).matrix();
			const Vector3f edge_old_t = rot_arap[v]*edge_old;
			const Vector3f edge_dif = edge_new - edge_old_t;

			//WRT the control vertices
			j_elem.push_back(Tri(J_row, unk_per_vertex*v, -K_arap_sqrt));
			j_elem.push_back(Tri(J_row, unk_per_vertex*other_v, K_arap_sqrt));
			j_elem.push_back(Tri(J_row+1, unk_per_vertex*v+1, -K_arap_sqrt));
			j_elem.push_back(Tri(J_row+1, unk_per_vertex*other_v+1, K_arap_sqrt));
			j_elem.push_back(Tri(J_row+2, unk_per_vertex*v+2, -K_arap_sqrt));
			j_elem.push_back(Tri(J_row+2, unk_per_vertex*other_v+2, K_arap_sqrt));

			//WRT the rotations
			const unsigned int j_col_r = j_col_bias + 3*v;
			j_elem.push_back(Tri(J_row+1, j_col_r, K_arap_sqrt*edge_old_t(2)));	
			j_elem.push_back(Tri(J_row+2, j_col_r, -K_arap_sqrt*edge_old_t(1)));

			j_elem.push_back(Tri(J_row, j_col_r+1, -K_arap_sqrt*edge_old_t(2)));	
			j_elem.push_back(Tri(J_row+2, j_col_r+1, K_arap_sqrt*edge_old_t(0)));

			j_elem.push_back(Tri(J_row, j_col_r+2, K_arap_sqrt*edge_old_t(1)));	
			j_elem.push_back(Tri(J_row+1, j_col_r+2, -K_arap_sqrt*edge_old_t(0)));

			//General expression (less efficient)
			//for (unsigned int k=0; k<3; k++)
			//{
			//	const Vector3f j_rot = -K_arap_sqrt*mat_der_xi[k+3].block<3,3>(0,0)*edge_old_t;
			//	const unsigned int j_col = 3*num_verts + optimize_cameras*6*num_images + 3*v + k;

			//	j_elem.push_back(Tri(J_row, j_col, j_rot(0)));
			//	j_elem.push_back(Tri(J_row+1, j_col, j_rot(1)));	
			//	j_elem.push_back(Tri(J_row+2, j_col, j_rot(2)));	
			//}

			//Fill the residual
			R(J_row) = K_arap_sqrt*(edge_dif(0));
			R(J_row+1) = K_arap_sqrt*(edge_dif(1));
			R(J_row+2) = K_arap_sqrt*(edge_dif(2));
			J_row += 3;
		}
	}
}

void Mod3DfromRGBD::fill_J_RegRotArap(unsigned int &J_row)
{
	const float K_sqrt = sqrtf(K_rot_arap);
	const unsigned int j_col_bias = unk_per_vertex*num_verts + optimize_cameras*6*num_images;
	
	for (unsigned int v=0; v<num_verts; v++)
		for (unsigned int e=0; e<valence(v); e++)
		{						
			const unsigned int v2 = neighbourhood(e,v);
			const Vector3f res = rot_mfold[v2] - rot_mfold[v];

			for (unsigned int l=0; l<3; l++)
			{
				j_elem.push_back(Tri(J_row, j_col_bias + 3*v2 + l, K_sqrt));
				j_elem.push_back(Tri(J_row, j_col_bias + 3*v + l, -K_sqrt));

				//Fill the residual
				R(J_row) = K_sqrt*res(l);
				J_row++;
			}
		}	
}




