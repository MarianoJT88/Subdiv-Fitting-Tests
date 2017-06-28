// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "face_transition_tests_2D.h"


void TestTransitions3D::loadInitialMesh1()
{
	//Load a plane with 9 faces
	num_verts = 16;
	num_faces = 9;

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1, 5, 4;
	face_verts.col(1) << 1, 2, 6, 5;
	face_verts.col(2) << 2, 3, 7, 6;
	face_verts.col(3) << 4, 5, 9, 8;
	face_verts.col(4) << 5, 6, 10, 9;
	face_verts.col(5) << 6, 7, 11, 10;
	face_verts.col(6) << 8, 9, 13, 12;
	face_verts.col(7) << 9, 10, 14, 13;
	face_verts.col(8) << 10, 11, 15, 14;

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
	//Uniform template
	const float min_x = 0.f, min_y = 0.f, min_z = 0.f;
	float max_x = 1.f, max_y = 1.f, max_z = 0.f;

	vert_coords.resize(3, num_verts);
	for (unsigned int u = 0; u < 4; u++)
		for (unsigned int v = 0; v < 4; v++)
		{
			vert_coords(0, v + 4 * u) = min_x + float(v)*(max_x - min_x)/3.f;
			vert_coords(1, v + 4 * u) = min_y + float(u)*(max_x - min_x)/3.f;
			vert_coords(2, v + 4 * u) = 0.f;
		}

	//Perturb some points
	vert_coords(0, 9) += 0.1f;
	vert_coords(1, 9) -= 0.2f;
	vert_coords(0, 10) += 0.3f;
	vert_coords(1, 10) += 0.2f;
}

void TestTransitions3D::loadInitialMesh2()
{
	//Load a plane with 9 faces
	num_verts = 16;
	num_faces = 9;

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1, 5, 4;
	face_verts.col(1) << 1, 2, 6, 5;
	face_verts.col(2) << 2, 3, 7, 6;
	face_verts.col(3) << 4, 5, 9, 8;
	face_verts.col(4) << 5, 6, 10, 9;
	face_verts.col(5) << 6, 7, 11, 10;
	face_verts.col(6) << 8, 9, 13, 12;
	face_verts.col(7) << 9, 10, 14, 13;
	face_verts.col(8) << 10, 11, 15, 14;

	cout << endl << "Face vertices: " << endl << face_verts;

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
	//Uniform template
	const float min_x = 0.f, min_y = 0.f, min_z = 0.f;
	float max_x = 1.f, max_y = 1.f, max_z = 0.f;

	vert_coords.resize(3, num_verts);
	for (unsigned int u = 0; u < 4; u++)
		for (unsigned int v = 0; v < 4; v++)
		{
			vert_coords(0, v + 4 * u) = min_x + float(v)*(max_x - min_x) / 3.f;
			vert_coords(1, v + 4 * u) = min_y + float(u)*(max_x - min_x) / 3.f;
			vert_coords(2, v + 4 * u) = 0.f;
		}

	//strech the middle row
	for (unsigned int v = 0; v < 4; v++)
	{
		vert_coords(1, v + 4) = 0.07f;
		vert_coords(1, v + 8) = 0.93f;
	}
}

void TestTransitions3D::loadInitialMeshVertexVal3()
{
	//Load a plane with 9 faces
	num_verts = 21;
	num_faces = 13;

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1, 5, 4;
	face_verts.col(1) << 1, 2, 6, 5;
	face_verts.col(2) << 2, 3, 7, 6;
	face_verts.col(3) << 4, 5, 9, 8;
	face_verts.col(4) << 5, 16, 19, 9;
	face_verts.col(5) << 5, 6, 17, 16;
	face_verts.col(6) << 16, 17, 10, 19;
	face_verts.col(7) << 6, 7, 18, 17;
	face_verts.col(8) << 17, 18, 11, 10;
	face_verts.col(9) << 8, 9, 13, 12;
	face_verts.col(10) << 9, 19, 20, 13;
	face_verts.col(11) << 19, 10, 14, 20;
	face_verts.col(12) << 10, 11, 15, 14;
	cout << endl << "Face vertices: " << endl << face_verts;

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
	//Uniform template
	const float min_x = 0.f, min_y = 0.f, min_z = 0.f;
	float max_x = 1.f, max_y = 1.f, max_z = 0.f;

	vert_coords.resize(3, num_verts);
	vert_coords.row(2).fill(0.f);

	for (unsigned int u = 0; u < 4; u++)
		for (unsigned int v = 0; v < 4; v++)
		{
			vert_coords(0, v + 4 * u) = min_x + float(v)*(max_x - min_x) / 3.f;
			vert_coords(1, v + 4 * u) = min_y + float(u)*(max_x - min_x) / 3.f;
		}


	vert_coords.col(16) = 0.5f*(vert_coords.col(0) + vert_coords.col(15));
	vert_coords.col(17) = 0.5f*(vert_coords.col(6) + vert_coords.col(10));
	vert_coords.col(18) = 0.5f*(vert_coords.col(7) + vert_coords.col(11));
	vert_coords.col(19) = 0.5f*(vert_coords.col(9) + vert_coords.col(10));
	vert_coords.col(20) = 0.5f*(vert_coords.col(13) + vert_coords.col(14));
}

void TestTransitions3D::loadInitialMeshVertexVal8()
{
	//Load a plane with 9 faces
	num_verts = 49;
	num_faces = 32;

	//Fill the type of poligons (triangles or quads)
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);

	//Fill the vertices per face
	face_verts.resize(4, num_faces);	//The first number does not do anything, you can write there what you want, it will keep its original definition
	face_verts.col(0) << 0, 1, 8, 7;
	face_verts.col(1) = face_verts.col(0) + 1;
	face_verts.col(2) = face_verts.col(1) + 1;
	face_verts.col(3) = face_verts.col(2) + 1;
	face_verts.col(4) = face_verts.col(3) + 1;
	face_verts.col(5) = face_verts.col(4) + 1;

	face_verts.col(6) = face_verts.col(0) + 7;
	face_verts.col(7) = face_verts.col(6) + 1;
	face_verts.col(8) << 9, 10, 24, 16;
	face_verts.col(9) << 10, 11, 18, 24;
	face_verts.col(10) = face_verts.col(7) + 3;
	face_verts.col(11) = face_verts.col(10) + 1;

	face_verts.col(12) = face_verts.col(6) + 7;
	face_verts.col(13) << 15, 16, 24, 22;
	face_verts.col(14) << 18, 19, 26, 24;
	face_verts.col(15) = face_verts.col(11) + 7;

	face_verts.col(16) = face_verts.col(12) + 7;
	face_verts.col(17) << 22, 24, 30, 29;
	face_verts.col(18) << 24, 26, 33, 32;
	face_verts.col(19) = face_verts.col(15) + 7;

	face_verts.col(20) = face_verts.col(16) + 7;
	face_verts.col(21) = face_verts.col(20) + 1;
	face_verts.col(22) << 30, 24, 38, 37;
	face_verts.col(23) << 24, 32, 39, 38;
	face_verts.col(24) = face_verts.col(10) + 21;
	face_verts.col(25) = face_verts.col(24) + 1;

	face_verts.col(26) = face_verts.col(20) + 7;
	face_verts.col(27) = face_verts.col(26) + 1;
	face_verts.col(28) = face_verts.col(27) + 1;
	face_verts.col(29) = face_verts.col(28) + 1;
	face_verts.col(30) = face_verts.col(29) + 1;
	face_verts.col(31) = face_verts.col(30) + 1;

	cout << endl << "Face vertices: " << endl << face_verts;

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
	//Uniform template
	const float min_x = 0.f, min_y = 0.f, min_z = 0.f;
	float max_x = 1.f, max_y = 1.f, max_z = 0.f;

	vert_coords.resize(3, num_verts);
	vert_coords.row(2).fill(0.f);

	for (unsigned int u = 0; u < 7; u++)
		for (unsigned int v = 0; v < 7; v++)
		{
			vert_coords(0, v + 7 * u) = min_x + float(v)*(max_x - min_x) / 6.f;
			vert_coords(1, v + 7 * u) = min_y + float(u)*(max_x - min_x) / 6.f;
		}
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
	control_mesh->setLineWidth(4.f);
	control_mesh->setEdgeColor(0.f, 0.f, 0.f);
	scene->insert(control_mesh);

	////Vertex numbers
	//for (unsigned int v = 0; v < num_verts; v++)
	//{
	//	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	//	vert_nums->setString(std::to_string(v));
	//	vert_nums->setScale(0.01f);
	//	vert_nums->setColor(0.5, 0, 0);
	//	scene->insert(vert_nums);
	//}

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	reference->setScale(0.2f);
	scene->insert(reference);

	//Internal model (subdivision surface)
	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	points->setPointSize(1.5f);
	points->enablePointSmooth(true);
	scene->insert(points);

	//Particles
	opengl::CPointCloudColouredPtr particles = opengl::CPointCloudColoured::Create();
	particles->setPointSize(3.f);
	particles->enablePointSmooth(true);
	scene->insert(particles);

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

void TestTransitions3D::showSubSurface()
{
	scene = window.get3DSceneAndLock();

	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
	points->clear();

	for (unsigned int k = 0; k < mx.size(); k++)
		points->push_back(mx(k), my(k), mz(k), 1.f, 1.f, 1.f);

	window.unlockAccess3DScene();
	window.repaint();
}

void TestTransitions3D::showSubSurface2()
{
	scene = window.get3DSceneAndLock();

	CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
	points->clear();

	for (unsigned int k = 0; k < mx.size(); k++)
	{
		if ((u1(k) == 0.f) || (u1(k) == 1.f) || (u2(k) == 0.f) || (u2(k) == 1.f))
			points->push_back(mx(k), my(k), mz(k), 0.f, 0.f, 0.f);

		else
			points->push_back(mx(k), my(k), mz(k), 1.f, 1.f, 1.f);
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
	options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

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

	const int maxIsolation = 0;
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

void TestTransitions3D::evaluateSubDivSurface()
{
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	//Far::PtexIndices ptexIndices(*refiner); 	// Create a Far::PtexIndices to help find indices of ptex faces.

	float pWeights[20], dsWeights[20], dtWeights[20];
	unsigned int count = 0;
	const unsigned int samples_u = 60, samples_v = 5;
	mx.resize(2*num_faces*samples_u*samples_v,1);
	my.resize(2*num_faces*samples_u*samples_v,1);
	mz.resize(2*num_faces*samples_u*samples_v,1);
	u1.resize(2*num_faces*samples_u*samples_v,1);
	u2.resize(2*num_faces*samples_u*samples_v,1);


	//Evaluate the surface with parametric coordinates
	for (unsigned int f = 0; f < num_faces; f++)
	{
		for (unsigned int u = 0; u < samples_u; u++)
			for (unsigned int v = 0; v < samples_v; v++)
			{
				const float u1s = float(u) / float(samples_u - 1);
				const float u2s = float(v) / float(samples_v - 1);

				// Locate the patch corresponding to the face ptex idx and (s,t)
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1s, u2s); assert(handle);
				if (!handle) continue;

				// Evaluate the patch weights, identify the CVs and compute the limit frame:
				patchTable->EvaluateBasis(*handle, u1s, u2s, pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Save the 3D coordinates
				mx(count) = eval.point[0];
				my(count) = eval.point[1];
				mz(count) = eval.point[2];

				u1(count) = u1s;
				u2(count) = u2s;

				count++;
			}

		for (unsigned int u = 0; u < samples_v; u++)
			for (unsigned int v = 0; v < samples_u; v++)
			{
				const float u1s = float(u) / float(samples_v - 1);
				const float u2s = float(v) / float(samples_u - 1);

				// Locate the patch corresponding to the face ptex idx and (s,t)
				Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u1s, u2s); assert(handle);
				if (!handle) continue;

				// Evaluate the patch weights, identify the CVs and compute the limit frame:
				patchTable->EvaluateBasis(*handle, u1s, u2s, pWeights, dsWeights, dtWeights);

				Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

				LimitFrame eval; eval.Clear();
				for (int cv = 0; cv < cvs.size(); ++cv)
					eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

				//Save the 3D coordinates
				mx(count) = eval.point[0];
				my(count) = eval.point[1];
				mz(count) = eval.point[2];

				u1(count) = u1s;
				u2(count) = u2s;

				count++;
			}
	}
}

void TestTransitions3D::evaluateTransitions()
{
	//Create a particle
	unsigned int face_ini[3] = {4, 3, 0};
	float u1_ini[3] = {0.7f, 0.8f, 0.5f};
	float u2_ini[3] = {0.2f, 0.9f, 0.5f + 10e-8f};
	float u1_incr_ini[3] = {0.008, 0.01f, 0.006f};
	float u2_incr_ini[3] = {0.001f, 0.005f, 0.006f};
	const unsigned int iter = 200.f;

	//Define variables
	// Create a Far::PatchMap to help locating patches in the table
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];
	float pcoord[3], u1d[3], u2d[2];

	//--------------------------------------------------------------------------------------------------------
	//Move the particle throught the subdiv surface imposing tangent continuity
	//--------------------------------------------------------------------------------------------------------
	for (unsigned int k = 0; k < 3; k++)
	{
		unsigned int face = face_ini[k];
		float u1 = u1_ini[k], u2 = u2_ini[k];
		float u1_incr = u1_incr_ini[k], u2_incr = u2_incr_ini[k];

		for (unsigned int i = 0; i < iter; i++)
		{
			// Locate the patch corresponding to the face ptex idx and (s,t)
			Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u1, u2);

			// Evaluate the patch weights, identify the CVs and compute the limit frame:
			patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);

			Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

			LimitFrame eval; eval.Clear();
			for (int cv = 0; cv < cvs.size(); ++cv)
				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

			pcoord[0] = eval.point[0]; pcoord[1] = eval.point[1]; pcoord[2] = eval.point[2];

			//Check if crossing borders
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
				case 2:	u1 = 1 - aux; u2 = 1.f; break;
				case 3:	u1 = 0.f; u2 = 1 - aux; break;
				}

				//Evaluate the subdivision surface at the edge (with respect to the original face)
				Far::PatchTable::PatchHandle const * handle2 = patchmap.FindPatch(face_new, u1, u2);
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
			}

			u1 = u1_new;
			u2 = u2_new;
			

			//Show
			scene = window.get3DSceneAndLock();
			if (i == 0)
			{
				CSpherePtr ini = opengl::CSphere::Create(0.01f);
				ini->setLocation(pcoord[0], pcoord[1], pcoord[2]);
				scene->insert(ini);
			}
			else
			{
				CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(1);
				points->push_back(pcoord[0], pcoord[1], pcoord[2], 0.f, 0.f, 1.f);			
			}
			window.unlockAccess3DScene();
			window.repaint();

		}

		//-------------------------------------------------------------------------------------------
		//Move the particle throught the subdiv surface with direct transition between faces
		//-------------------------------------------------------------------------------------------
		face = face_ini[k];
		u1 = u1_ini[k]; u2 = u2_ini[k];
		u1_incr = u1_incr_ini[k]; u2_incr = u2_incr_ini[k];
		int extra_rot;

		for (unsigned int i = 0; i < iter; i++)
		{
			// Locate the patch corresponding to the face ptex idx and (s,t)
			Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(face, u1, u2);

			// Evaluate the patch weights, identify the CVs and compute the limit frame:
			patchTable->EvaluateBasis(*handle, u1, u2, pWeights, dsWeights, dtWeights);

			Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

			LimitFrame eval; eval.Clear();
			for (int cv = 0; cv < cvs.size(); ++cv)
				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

			pcoord[0] = eval.point[0]; pcoord[1] = eval.point[1]; pcoord[2] = eval.point[2];

			//Check if crossing borders
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
						extra_rot = -1;
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
						extra_rot = 1;
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
						extra_rot = 2;
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
						extra_rot = 0;
					}
				}

				//Find the coordinates of the crossing point as part of the new face
				unsigned int conf;
				for (unsigned int f = 0; f < 4; f++)
					if (face_adj(f, face_new) == face) { conf = f; }

				switch (conf)
				{
				case 0: u1 = aux; u2 = 0.f; break;
				case 1: u1 = 1.f; u2 = aux; break;
				case 2:	u1 = 1 - aux; u2 = 1.f; break;
				case 3:	u1 = 0.f; u2 = 1 - aux; break;
				}

				//Compute the new u increments
				const float angle = 0.5f*M_PIf*(conf + extra_rot);
				Vector2f du_new, du_prev; du_prev << u1_new - u1_cross, u2_new - u2_cross;
				Matrix2f rot_matrix; rot_matrix << cosf(angle), -sinf(angle), sinf(angle), cosf(angle);
				du_new = rot_matrix*du_prev;

				//Adapt the increments
				du_prev << u1_incr, u2_incr;
				du_new = rot_matrix*du_prev;
				u1_incr = du_new(0);
				u2_incr = du_new(1);

				u1_new = u1 + du_new(0);
				u2_new = u2 + du_new(1);
				face = face_new;

				crossing = (u1_new < 0.f) || (u1_new > 1.f) || (u2_new < 0.f) || (u2_new > 1.f);

				//printf("\n New: face = %d, u1 = %.4f, u2 = %.4f, xc = %.4f, yc = %.4f, zc = %.4f", face, u1, u2, eval.point[0], eval.point[1], eval.point[2]);
				//printf("\n Original increments = (%f, %f), New increments = (%f, %f) \n", du_prev(0), du_prev(1), u1_incr, u2_incr);
			}

			u1 = u1_new;
			u2 = u2_new;

			//Show
			scene = window.get3DSceneAndLock();
			CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(1);
			points->push_back(pcoord[0], pcoord[1], pcoord[2], 0.f, 0.6f, 0.f);
			window.unlockAccess3DScene();
			window.repaint();

		}
	}
}




