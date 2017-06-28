// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "face_transition_tests_3D.h"


void TestTransitions3D::loadInitialMesh()
{
	//Initial mesh - A cube
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

	cout << endl << "Face vertices: " << endl << face_verts;

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
	const float size = 0.2f;
	float min_x = -size, min_y = -size, min_z = -size;
	float max_x = size, max_y = size, max_z = size;

	vert_coords.resize(3, num_verts);
	vert_coords.col(0) << min_x, min_y, max_z;
	vert_coords.col(1) << max_x, min_y, max_z;
	vert_coords.col(2) << min_x, max_y, max_z;
	vert_coords.col(3) << max_x, max_y, max_z;
	vert_coords.col(4) << min_x, max_y, min_z;
	vert_coords.col(5) << max_x, max_y, min_z;
	vert_coords.col(6) << min_x, min_y, min_z;
	vert_coords.col(7) << max_x, min_y, min_z;

	//Show the mesh on the 3D Scene
	showMesh();
}

void TestTransitions3D::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1000, 900);
	window.setPos(900, 0);
	window.setCameraZoom(3);
	window.setCameraAzimuthDeg(0);
	window.setCameraElevationDeg(45);
	window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	scene = window.get3DSceneAndLock();

	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	scene->insert(control_mesh);

	//Vertex numbers
	for (unsigned int v = 0; v < 8; v++)
	{
		opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
		vert_nums->setString(std::to_string(v));
		vert_nums->setScale(0.02f);
		vert_nums->setColor(0.5, 0, 0);
		scene->insert(vert_nums);
	}

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	reference->setScale(0.2f);
	scene->insert(reference);

	//Internal model (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(4);
	points->enablePointSmooth(true);
	scene->insert(points);

	window.unlockAccess3DScene();
	window.repaint();
}

void TestTransitions3D::showMesh()
{
	scene = window.get3DSceneAndLock();

	//Control mesh
	opengl::CMesh3DPtr control_mesh = scene->getByClass<CMesh3D>(0);
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);

	//Show vertex numbers
	for (unsigned int v = 0; v < num_verts; v++)
	{
		opengl::CText3DPtr vert_nums = scene->getByClass<CText3D>(v);
		vert_nums->setLocation(vert_coords(0, v), vert_coords(1, v), vert_coords(2, v));
	}

	window.unlockAccess3DScene();
	window.repaint();
}


void TestTransitions3D::createTopologyRefiner()
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

void TestTransitions3D::testSingleParticle()
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	// Create a Far::PtexIndices to help find indices of ptex faces.
	Far::PtexIndices ptexIndices(*refiner);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	float pcoord[3], u1d[3], u2d[2];
	unsigned int face = 2;
	float u1 = 0.8f;
	float u2 = 0.4f;
	float u1_incr = 0.03f;
	float u2_incr = 0.01f;
	for (unsigned int i = 0; i < 150; ++i)
	{
		// Locate the patch corresponding to the face ptex idx and (s,t)
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u1, u2); assert(handle);

		// Evaluate the patch weights, identify the CVs and compute the limit frame:
		patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);
		pcoord[0] = eval.point[0]; pcoord[1] = eval.point[1]; pcoord[2] = eval.point[2];


		//Check crossing border and perform transition if necessary
		//-------------------------------------------------------------------------------------------------------
		float u1_new = u1 + u1_incr;
		float u2_new = u2 + u2_incr;
		bool crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);
		while (crossing)
		{	
			//Find the new face	and the coordinates of the crossing point within the old face and the new face
			unsigned int face_new; 
			float aux, dif, u1_cross, u2_cross;
			bool face_found = false;
			
			if (u1_new < 0.f)
			{ 
				dif = u1;
				const float u2t = u2 - u2_incr*dif / u1_incr;
				if ((u2t >= 0.f) && (u2t <= 1.f))
				{ 
					face_new = face_adj(3, face); aux = u2t; face_found = true; 
					u1_cross = 0.f; u2_cross = u2t;
				}
			}
			if ((u1_new > 1.f) && (!face_found))
			{ 
				dif = 1.f - u1; 
				const float u2t = u2 + u2_incr*dif / u1_incr;
				if ((u2t >= 0.f) && (u2t <= 1.f))
				{
					face_new = face_adj(1, face); aux = 1.f - u2t; face_found = true;
					u1_cross = 1.f; u2_cross = u2t;
				}
			}
			if ((u2_new < 0.f) && (!face_found))
			{ 
				dif = u2; 
				const float u1t = u1 - u1_incr*dif / u2_incr;
				if ((u1t >= 0.f) && (u1t <= 1.f))
				{ 
					face_new = face_adj(0, face); aux = 1.f - u1t; face_found = true;
					u1_cross = u1t; u2_cross = 0.f;
				}
			}
			if ((u2_new > 1.f) && (!face_found))
			{ 
				dif = 1.f - u2;
				const float u1t = u1 + u1_incr*dif / u2_incr;
				if ((u1t >= 0.f) && (u1t <= 1.f))
				{ 
					face_new = face_adj(2, face); aux = u1t; face_found = true;
					u1_cross = u1t; u2_cross = 1.f;
				}
			}

			//Evaluate the subdivision surface at the edge (with respect to the original face)
			Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(face, u1_cross, u2_cross); assert(handle);
			patchTable->EvaluateBasis(*handle1, u1_cross, u2_cross, pWeights, dsWeights, dtWeights);
			Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);
			eval.Clear();
			for (int cv = 0; cv < cvs.size(); ++cv)
				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

			u1d[0] = eval.deriv1[0]; u1d[1] = eval.deriv1[1]; u1d[2] = eval.deriv1[2];
			u2d[0] = eval.deriv2[0]; u2d[1] = eval.deriv2[1]; u2d[2] = eval.deriv2[2];
			Matrix<float, 3, 2> J_Sa; J_Sa << u1d[0], u2d[0], u1d[1], u2d[1], u1d[2], u2d[2];

			//Find the coordinates of the crossing point as part of the new face
			unsigned int conf;
			for (unsigned int f = 0; f < 4; f++)
				if (face_adj(f, face_new) == face) { conf = f; }

			switch (conf)
			{
			case 0: u1 = aux; u2 = 0.f; break;
			case 1: u1 = 1.f; u2 = aux; break;
			case 2:	u1 = 1-aux; u2 = 1.f; break;
			case 3:	u1 = 0.f; u2 = 1-aux; break;
			}

			//Evaluate the subdivision surface at the edge (with respect to the original face)
			Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(face_new, u1, u2); assert(handle);
			patchTable->EvaluateBasis(*handle2, u1, u2, pWeights, dsWeights, dtWeights);
			cvs = patchTable->GetPatchVertices(*handle2);
			eval.Clear();
			for (int cv = 0; cv < cvs.size(); ++cv)
				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

			//Derivatives
			u1d[0] = eval.deriv1[0]; u1d[1] = eval.deriv1[1]; u1d[2] = eval.deriv1[2];
			u2d[0] = eval.deriv2[0]; u2d[1] = eval.deriv2[1]; u2d[2] = eval.deriv2[2];
			Matrix<float, 3, 2> J_Sb; J_Sb << u1d[0], u2d[0], u1d[1], u2d[1], u1d[2], u2d[2];

			//Compute the new u increments
			Vector2f du_prev; du_prev << u1_new - u1_cross, u2_new - u2_cross;
			MatrixXf prod = J_Sa*du_prev;
			MatrixXf AtA, AtB;
			AtA.multiply_AtA(J_Sb);
			AtB.multiply_AtB(J_Sb, prod);
			Vector2f du_new = AtA.inverse()*AtB;

			//printf("\n New face = %d, u1 = %.4f, u2 = %.4f, u1_new = %.4f, n2_new = %.4f", face_new, u1, u2, u1_new, u2_new);
			//printf("\n Original remaining incr = (%f, %f), New remaining incr = (%f, %f) \n", du_prev(0), du_prev(1), du_new(0), du_new(1));

			u1_new = u1 + du_new(0);
			u2_new = u2 + du_new(1);
			face = face_new;

			//Keep the same speed
			du_prev << u1_incr, u2_incr;
			prod = J_Sa*du_prev;
			AtB.multiply_AtB(J_Sb, prod);
			du_new = AtA.inverse()*AtB;

			u1_incr = du_new(0);
			u2_incr = du_new(1);

			crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);
			if (crossing == true)
				printf("\n Crossing several face borders in a row!!");
		}

		u1 = u1_new;
		u2 = u2_new;
		//---------------------------------------------------------------------------------------------------------------------------------

		const float norm_incr = sqrtf(square(u1_incr) + square(u2_incr));
		printf("\n face = %d, u1 = %.3f, u2 = %.3f, u1_incr = %.3f, u2_incr = %.3f, norm = %f", face, u1, u2, u1_incr, u2_incr, norm_incr);

		//Show
		scene = window.get3DSceneAndLock();
		CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
		points->push_back(pcoord[0], pcoord[1], pcoord[2], 0.f, 0.f, 0.f);
		window.unlockAccess3DScene();
		window.repaint();

		system::sleep(50);
	}
}

void TestTransitions3D::testMultipleParticles()
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	// Create a Far::PtexIndices to help find indices of ptex faces.
	Far::PtexIndices ptexIndices(*refiner);

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int cont = 0;

	float pcoord[3], u1d[3], u2d[2];
	unsigned int face = 2;
	const unsigned int size = 20;
	ArrayXf u1(size), u2(size);
	u1 = 0.5f*VectorXf::Random(size); u1 += 0.5f;
	u2 = 0.5f*VectorXf::Random(size); u2 += 0.5f;

	for (unsigned int k = 0; k < size; k++)
	{
		float u1_incr = 0.03f;
		float u2_incr = 0.01f;

		float r, g, b;
		utils::colormap(mrpt::utils::cmJET, float(k) / float(size), r, g, b);

		for (unsigned int i = 0; i < 150; ++i)
		{
			// Locate the patch corresponding to the face ptex idx and (s,t)
			Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u1[k], u2[k]); assert(handle);

			// Evaluate the patch weights, identify the CVs and compute the limit frame:
			patchTable->EvaluateBasis(*handle, u1[k], u2[k], pWeights, dsWeights, dtWeights);

			Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

			LimitFrame eval; eval.Clear();
			for (int cv = 0; cv < cvs.size(); ++cv)
				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

			pcoord[0] = eval.point[0]; pcoord[1] = eval.point[1]; pcoord[2] = eval.point[2];


			//Check crossing border and perform transition if necessary
			//-------------------------------------------------------------------------------------------------------
			float u1_new = u1[k] + u1_incr;
			float u2_new = u2[k] + u2_incr;
			bool crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

			while (crossing)
			{
				//Find the new face	and the coordinates of the crossing point within the old face and the new face
				unsigned int face_new;
				float aux, dif, u1_cross, u2_cross;
				bool face_found = false;

				if (u1_new < 0.f)
				{
					dif = u1[k];
					const float u2t = u2[k] - u2_incr*dif / u1_incr;
					if ((u2t >= 0.f) && (u2t <= 1.f))
					{
						face_new = face_adj(3, face); aux = u2t; face_found = true;
						u1_cross = 0.f; u2_cross = u2t;
					}
				}
				if ((u1_new > 1.f) && (!face_found))
				{
					dif = 1.f - u1[k];
					const float u2t = u2[k] + u2_incr*dif / u1_incr;
					if ((u2t >= 0.f) && (u2t <= 1.f))
					{
						face_new = face_adj(1, face); aux = 1.f - u2t; face_found = true;
						u1_cross = 1.f; u2_cross = u2t;
					}
				}
				if ((u2_new < 0.f) && (!face_found))
				{
					dif = u2[k];
					const float u1t = u1[k] - u1_incr*dif / u2_incr;
					if ((u1t >= 0.f) && (u1t <= 1.f))
					{
						face_new = face_adj(0, face); aux = 1.f - u1t; face_found = true;
						u1_cross = u1t; u2_cross = 0.f;
					}
				}
				if ((u2_new > 1.f) && (!face_found))
				{
					dif = 1.f - u2[k];
					const float u1t = u1[k] + u1_incr*dif / u2_incr;
					if ((u1t >= 0.f) && (u1t <= 1.f))
					{
						face_new = face_adj(2, face); aux = u1t; face_found = true;
						u1_cross = u1t; u2_cross = 1.f;
					}
				}

				//Evaluate the subdivision surface at the edge (with respect to the original face)
				Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(face, u1_cross, u2_cross); assert(handle);
				patchTable->EvaluateBasis(*handle1, u1_cross, u2_cross, pWeights, dsWeights, dtWeights);
				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);
				eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				u1d[0] = eval.deriv1[0]; u1d[1] = eval.deriv1[1]; u1d[2] = eval.deriv1[2];
				u2d[0] = eval.deriv2[0]; u2d[1] = eval.deriv2[1]; u2d[2] = eval.deriv2[2];
				Matrix<float, 3, 2> J_Sa; J_Sa << u1d[0], u2d[0], u1d[1], u2d[1], u1d[2], u2d[2];


				//Find the coordinates of the crossing point as part of the new face
				unsigned int conf;
				for (unsigned int f = 0; f < 4; f++)
					if (face_adj(f, face_new) == face) { conf = f; }

				switch (conf)
				{
				case 0: u1[k] = aux; u2[k] = 0.f; break;
				case 1: u1[k] = 1.f; u2[k] = aux; break;
				case 2:	u1[k] = 1 - aux; u2[k] = 1.f; break;
				case 3:	u1[k] = 0.f; u2[k] = 1 - aux; break;
				}

				//Evaluate the subdivision surface at the edge (with respect to the original face)
				Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(face_new, u1[k], u2[k]); assert(handle);
				patchTable->EvaluateBasis(*handle2, u1[k], u2[k], pWeights, dsWeights, dtWeights);
				cvs = patchTable->GetPatchVertices(*handle2);
				eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Derivatives
				u1d[0] = eval.deriv1[0]; u1d[1] = eval.deriv1[1]; u1d[2] = eval.deriv1[2];
				u2d[0] = eval.deriv2[0]; u2d[1] = eval.deriv2[1]; u2d[2] = eval.deriv2[2];
				Matrix<float, 3, 2> J_Sb; J_Sb << u1d[0], u2d[0], u1d[1], u2d[1], u1d[2], u2d[2];

				//Compute the new u increments
				Vector2f du_prev; du_prev << u1_new - u1_cross, u2_new - u2_cross;
				MatrixXf prod = J_Sa*du_prev;
				MatrixXf AtA, AtB;
				AtA.multiply_AtA(J_Sb);
				AtB.multiply_AtB(J_Sb, prod);
				Vector2f du_new = AtA.inverse()*AtB;

				//printf("\n New face = %d, u1 = %.4f, u2 = %.4f, u1_new = %.4f, n2_new = %.4f", face_new, u1[k], u2[k], u1_new, u2_new);
				//printf("\n Original remaining incr = (%f, %f), New remaining incr = (%f, %f) \n", du_prev(0), du_prev(1), du_new(0), du_new(1));

				u1_new = u1[k] + du_new(0);
				u2_new = u2[k] + du_new(1);
				face = face_new;

				//Keep the same speed
				du_prev << u1_incr, u2_incr;
				prod = J_Sa*du_prev;
				AtB.multiply_AtB(J_Sb, prod);
				du_new = AtA.inverse()*AtB;

				u1_incr = du_new(0);
				u2_incr = du_new(1);

				crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);
				if (crossing == true)
					printf("\n Crossing several face borders in a row!!");
			}

			u1[k] = u1_new;
			u2[k] = u2_new;
			//---------------------------------------------------------------------------------------------------------------------------------

			const float norm_incr = sqrtf(square(u1_incr) + square(u2_incr));
			//printf("\n face = %d, u1 = %.3f, u2 = %.3f, u1_incr = %.3f, u2_incr = %.3f, norm = %f", face, u1[k], u2[k], u1_incr, u2_incr, norm_incr);

			//Show
			scene = window.get3DSceneAndLock();
			CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
			points->push_back(pcoord[0], pcoord[1], pcoord[2], r, g, b);
			window.unlockAccess3DScene();
			window.repaint();

			system::sleep(3);
		}
	}
}


void TestTransitions3D::testGradientDescentGravity()
{
	//The target
	const float target[3] = {0.3f, 0.1f, 0.f};

	scene = window.get3DSceneAndLock();

	CSpherePtr sphere = opengl::CSphere::Create(0.02f);
	sphere->setLocation(target[0], target[1], target[2]);
	scene->insert(sphere);

	CVectorField3DPtr gradients = opengl::CVectorField3D::Create();
	gradients->setPointSize(3.f);
	gradients->setPointColor(0.f, 0.f, 0.f);
	scene->insert(gradients);

	window.unlockAccess3DScene();
	window.repaint();

	
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);

	// Create a Far::PtexIndices to help find indices of ptex faces.
	Far::PtexIndices ptexIndices(*refiner);

	const unsigned int size = 40;
	ArrayXXf x(size, size), y(size, size), z(size, size);
	ArrayXXf u1(size,size), u2(size,size);
	ArrayXXf u1_inc(size, size), u2_inc(size, size);
	u1 = 0.5f*ArrayXXf::Random(size,size); u1 += 0.5f;
	u2 = 0.5f*ArrayXXf::Random(size,size); u2 += 0.5f;
	ArrayXXi face(size, size); face.fill(0);
	ArrayXXf resx(size, size), resy(size, size), resz(size, size);
	MatrixXf gradx(size, size), grady(size, size), gradz(size, size);
	float pWeights[20], dsWeights[20], dtWeights[20];

	//Optimize for these points
	//-----------------------------------------------------------------------
	const unsigned int max_iter = 600;
	const float sz_u = 0.01f; //Step size
	for (unsigned int i = 0; i < max_iter; i++)
	{

		for (unsigned int u = 0; u < size; u++)
			for (unsigned int v = 0; v < size; v++)
			{
				// Evaluate the subdivision surface
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face(v, u), u1(v, u), u2(v, u)); assert(handle);
				patchTable->EvaluateBasis(*handle, u1(v, u), u2(v, u), pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				x(v, u) = eval.point[0];
				y(v, u) = eval.point[1];
				z(v, u) = eval.point[2];

				//Compute the residuals
				resx(v, u) = target[0] - eval.point[0];
				resy(v, u) = target[1] - eval.point[1];
				resz(v, u) = target[2] - eval.point[2];

				//Compute increments for the internal points
				u1_inc(v, u) = 2.f * sz_u*(resx(v, u)*eval.deriv1[0] + resy(v, u)*eval.deriv1[1] + resz(v, u)*eval.deriv1[2]);
				u2_inc(v, u) = 2.f * sz_u*(resx(v, u)*eval.deriv2[0] + resy(v, u)*eval.deriv2[1] + resz(v, u)*eval.deriv2[2]);

				const float m = 12.f;
				gradx(v, u) = m*(eval.deriv1[0] * u1_inc(v, u) + eval.deriv2[0] * u2_inc(v, u));
				grady(v, u) = m*(eval.deriv1[1] * u1_inc(v, u) + eval.deriv2[1] * u2_inc(v, u));
				gradz(v, u) = m*(eval.deriv1[2] * u1_inc(v, u) + eval.deriv2[2] * u2_inc(v, u));

				//Check crossing border and perform transition if necessary
				//-------------------------------------------------------------------------------------------------------
				float u1_new = u1(v, u) + u1_inc(v, u);
				float u2_new = u2(v, u) + u2_inc(v, u);
				bool crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

				while (crossing)
				{
					//Find the new face	and the coordinates of the crossing point within the old face and the new face
					unsigned int face_new;
					float aux, dif, u1_cross, u2_cross;
					bool face_found = false;

					if (u1_new < 0.f)
					{
						dif = u1(v, u);
						const float u2t = u2(v, u) - u2_inc(v, u)*dif / u1_inc(v, u);
						if ((u2t >= 0.f) && (u2t <= 1.f))
						{
							face_new = face_adj(3, face(v, u)); aux = u2t; face_found = true;
							u1_cross = 0.f; u2_cross = u2t;
						}
					}
					if ((u1_new > 1.f) && (!face_found))
					{
						dif = 1.f - u1(v, u);
						const float u2t = u2(v, u) + u2_inc(v, u)*dif / u1_inc(v, u);
						if ((u2t >= 0.f) && (u2t <= 1.f))
						{
							face_new = face_adj(1, face(v, u)); aux = 1.f - u2t; face_found = true;
							u1_cross = 1.f; u2_cross = u2t;
						}
					}
					if ((u2_new < 0.f) && (!face_found))
					{
						dif = u2(v, u);
						const float u1t = u1(v, u) - u1_inc(v, u)*dif / u2_inc(v, u);
						if ((u1t >= 0.f) && (u1t <= 1.f))
						{
							face_new = face_adj(0, face(v, u)); aux = 1.f - u1t; face_found = true;
							u1_cross = u1t; u2_cross = 0.f;
						}
					}
					if ((u2_new > 1.f) && (!face_found))
					{
						dif = 1.f - u2(v, u);
						const float u1t = u1(v, u) + u1_inc(v, u)*dif / u2_inc(v, u);
						if ((u1t >= 0.f) && (u1t <= 1.f))
						{
							face_new = face_adj(2, face(v, u)); aux = u1t; face_found = true;
							u1_cross = u1t; u2_cross = 1.f;
						}
					}

					//Evaluate the subdivision surface at the edge (with respect to the original face)
					Far::PatchTable::PatchHandle const * handle1 = patchmap.FindPatch(face(v, u), u1_cross, u2_cross); assert(handle);
					patchTable->EvaluateBasis(*handle1, u1_cross, u2_cross, pWeights, dsWeights, dtWeights);
					Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle1);
					eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

					Matrix<float, 3, 2> J_Sa; J_Sa << eval.deriv1[0], eval.deriv2[0], eval.deriv1[1], eval.deriv2[1], eval.deriv1[2], eval.deriv2[2];

					//Find the coordinates of the crossing point as part of the new face
					unsigned int conf;
					for (unsigned int f = 0; f < 4; f++)
						if (face_adj(f, face_new) == face(v, u)) { conf = f; }

					switch (conf)
					{
					case 0: u1(v, u) = aux; u2(v, u) = 0.f; break;
					case 1: u1(v, u) = 1.f; u2(v, u) = aux; break;
					case 2:	u1(v, u) = 1 - aux; u2(v, u) = 1.f; break;
					case 3:	u1(v, u) = 0.f; u2(v, u) = 1 - aux; break;
					}

					//Evaluate the subdivision surface at the edge (with respect to the original face)
					Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(face_new, u1(v, u), u2(v, u)); assert(handle);
					patchTable->EvaluateBasis(*handle2, u1(v, u), u2(v, u), pWeights, dsWeights, dtWeights);
					cvs = patchTable->GetPatchVertices(*handle2);
					eval.Clear();
					for (int cv = 0; cv < cvs.size(); ++cv)
						eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

					Matrix<float, 3, 2> J_Sb; J_Sb << eval.deriv1[0], eval.deriv2[0], eval.deriv1[1], eval.deriv2[1], eval.deriv1[2], eval.deriv2[2];

					//Compute the new u increments
					Vector2f du_prev; du_prev << u1_new - u1_cross, u2_new - u2_cross;
					MatrixXf prod = J_Sa*du_prev;
					MatrixXf AtA, AtB;
					AtA.multiply_AtA(J_Sb);
					AtB.multiply_AtB(J_Sb, prod);
					Vector2f du_new = AtA.inverse()*AtB;

					u1_new = u1(v, u) + du_new(0);
					u2_new = u2(v, u) + du_new(1);
					face(v, u) = face_new;
					u1_inc(v, u) = du_new(0);
					u2_inc(v, u) = du_new(1);

					//printf("\n New face = %d, u1 = %.4f, u2 = %.4f, u1_new = %.4f, n2_new = %.4f", face_new, u1(v, u), u2(v, u), u1_new, u2_new);
					//printf("\n Original remaining incr = (%f, %f), New remaining incr = (%f, %f) \n", du_prev(0), du_prev(1), du_new(0), du_new(1));

					crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);
				}

				u1(v, u) = u1_new;
				u2(v, u) = u2_new;
				//-------------------------------------------------------------------------------------------------------
			}


			//Show
			scene = window.get3DSceneAndLock();
			
			MatrixXf xx = x.matrix(), yy = y.matrix(), zz = z.matrix();
			gradients->setPointCoordinates(xx, yy, zz);
			gradients->setVectorField(gradx, grady, gradz);

			window.unlockAccess3DScene();
			window.repaint();
			system::sleep(40);

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



