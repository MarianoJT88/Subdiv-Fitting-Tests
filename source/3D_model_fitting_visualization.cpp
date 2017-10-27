// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"


void Mod3DfromRGBD::initializeScene()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1000, 1000); //window.resize(1000, 900);
	window.setPos(200, 0); //window.setPos(900, 0);
	window.setCameraZoom(4.5f); //window.setCameraZoom(3); 
	window.setCameraAzimuthDeg(180);//window.setCameraAzimuthDeg(0); 
	window.setCameraElevationDeg(5);//window.setCameraElevationDeg(45);	
	window.setCameraPointingToPoint(0.4f, 0.f, 0.f);//window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	//Saved configurations for different experiments
	//Arch - window.setCameraZoom(2.f) 2.5f; window.setCameraAzimuthDeg(155); window.setCameraElevationDeg(25); window.setCameraPointingToPoint(1.2f, 0.f, 0.f);
	//Tracking - window.setCameraZoom(4.2f); window.setCameraAzimuthDeg(180); window.setCameraElevationDeg(5); window.setCameraPointingToPoint(0.4f, 0.f, 0.f);
	//Exp2 (teddy seg?) - window.setCameraZoom(1.8,1.8,2); window.setCameraAzimuthDeg(205); window.setCameraElevationDeg(8); window.setCameraPointingToPoint(0.f, 0.f, 0.f)?;
	//Exp3 (teddy null?) - window.setCameraZoom(1.4); window.setCameraAzimuthDeg(225); window.setCameraElevationDeg(25); window.setCameraPointingToPoint(0.f, 0.f, 0.f)?;

	if (paper_visualization)
	{
		window.getDefaultViewport()->setCustomBackgroundColor(utils::TColorf(1.f, 1.f, 1.f));
		window.captureImagesStart();
	}

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


	//Extra viewport for the distance transform
	if ((solve_DT)&&(!paper_visualization))
	{
		COpenGLViewportPtr gl_view_aux = scene->createViewport("DT");
		gl_view_aux->setViewportPosition(10, 10, 320, 240);
		utils::CImage viewport_image;
		viewport_image.setFromMatrix(DT[0].matrix(), false);
		viewport_image.flipVertical();
		viewport_image.normalize();
		gl_view_aux->setImageView(viewport_image);

		COpenGLViewportPtr gl_depth_im = scene->createViewport("depth");
		gl_depth_im->setViewportPosition(10, 700, 320, 240);
		MatrixXf mat_aux = (is_object[0].cast<float>() + valid[0].cast<float>()).matrix();
		viewport_image.setFromMatrix(mat_aux, false);
		viewport_image.flipVertical();
		viewport_image.normalize();
		gl_depth_im->setImageView(viewport_image);
	}

	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	control_mesh->setLineWidth(2.f); //2.f
	control_mesh->setPointSize(4.f); //8.f
	if (paper_visualization)
	{
		control_mesh->setEdgeColor(0.9f, 0.f, 0.f);
		control_mesh->setVertColor(0.6f, 0.f, 0.f);
	}
	scene->insert(control_mesh);

	//Vertex numbers
	//for (unsigned int v = 0; v < 2000; v++)
	//{
	//	opengl::CText3DPtr vert_nums = opengl::CText3D::Create();
	//	vert_nums->setString(std::to_string(v));
	//	vert_nums->setScale(0.003f);
	//	vert_nums->setColor(0.5, 0, 0);
	//	scene->insert(vert_nums);
	//}

	if (!paper_visualization)
	{
		//Reference
		opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
		reference->setScale(0.05f);
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
			points->setPointSize(5.f); //4.f
			points->enablePointSmooth(true);
			scene->insert(points);

			//Insert points
			float r, g, b;

			if (vis_errors)	{r = 0.f; g = 0.7f; b = 0.f;}
			else			{utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);}
			
			for (unsigned int v = 0; v < rows; v++)
				for (unsigned int u = 0; u < cols; u++)
					if  (is_object[i](v,u) && (depth[i](v,u) < 2.5f))
					{
						if (with_color)
						{
							r = intensity[i](v,u);
							g = r;
							b = r;
						}
						points->push_back(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), r, g, b);
					}
						

					//else if (valid[i](v,u))
					//	points->push_back(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), 0.f, 0.f, 0.f);
		}


		//Internal model (subdivision surface)
		opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
		points->setPointSize(2.f);
		points->enablePointSmooth(true);
		scene->insert(points);

	//	//Whole subdivision surface
	//	opengl::CPointCloudPtr subsurface = opengl::CPointCloud::Create();
	//	subsurface->setPointSize(4.f);
	//	subsurface->setColor(0.8f, 0.f, 0.f);
	//	subsurface->enablePointSmooth(true);
	//	//subsurface->setPose(CPose3D(0.f, 0.8f, 0.f, 0.f, 0.f, 0.f));
	//	scene->insert(subsurface);

		//Surface normals
		const float fact = 0.01f;
		for (unsigned int i = 0; i < num_images; i++)
		{	
			opengl::CSetOfLinesPtr normals = opengl::CSetOfLines::Create();
			normals->setColor(0, 0, 0.8f);
			normals->setLineWidth(1.f);
			scene->insert(normals);

			//float r, g, b;
			//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
			//normals->setColor(r, g, b);
			//for (unsigned int v = 0; v < rows; v++)
			//	for (unsigned int u = 0; u < cols; u++)
			//	if (is_object[i](v, u))
			//		normals->appendLine(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 
			//		depth[i](v, u) + fact*nx_image[i](v, u), x_image[i](v, u) + fact*ny_image[i](v, u), y_image[i](v, u) + fact*nz_image[i](v, u));
		}
	}


	//3D Model
	opengl::CMesh3DPtr model = opengl::CMesh3D::Create();
	if (!paper_visualization)	model->setPose(CPose3D(0.f, 1.5f, 0.f, 0.f, 0.f, 0.f));
	model->enableShowVertices(false);
	model->enableShowEdges(false);
	model->enableShowFaces(true);
	model->enableFaceNormals(true);
	//model->enableTransparency(true);
	model->setFaceColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setVertColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setEdgeColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setPointSize(1.f);
	model->setLineWidth(1.f);
	scene->insert(model);

	//Mesh to study the normals
	//opengl::CMeshPtr normals = opengl::CMesh::Create(false, 0.f, 0.4f, 0.f, 0.6f);
	//normals->setPose(CPose3D(0.f, 1.f, 0.f, 0.f, 0.f, 0.f));
	//const MatrixXf red = nx_image[0].abs().matrix();
	//const MatrixXf green = ny_image[0].abs().matrix();
	//const MatrixXf blue = nz_image[0].abs().matrix();
	//utils::CImage normal_im; normal_im.setFromRGBMatrices(red, green, blue, true);
	//normals->assignImage(normal_im);
	//scene->insert(normals);
	
	////Regularization samples
	//opengl::CPointCloudColouredPtr reg = opengl::CPointCloudColoured::Create();
	//reg->setPointSize(3.f);
	//reg->enablePointSmooth(true);
	//reg->setPose(CPose3D(0.f, -0.8f, 0.f, 0.f, 0.f, 0.f));
	//scene->insert(reg);

	window.unlockAccess3DScene();
	window.repaint();
}


void Mod3DfromRGBD::initializeSceneDataArch()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	window.resize(1000, 1000); //window.resize(1000, 900);
	window.setPos(200, 0); //window.setPos(900, 0);
	window.setCameraZoom(4.f);
	window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(35);
	window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	window.setCameraZoom(2.f); window.setCameraAzimuthDeg(155); window.setCameraElevationDeg(25); window.setCameraPointingToPoint(1.2f, 0.f, 0.f);

	//Saved configurations for different experiments
	//Arch - window.setCameraZoom(2.f) 2.5f/1.8; window.setCameraAzimuthDeg(155); window.setCameraElevationDeg(25); window.setCameraPointingToPoint(1.2f, 0.f, 0.f);
	//Tracking Manu - window.setCameraZoom(4.2f); window.setCameraAzimuthDeg(180); window.setCameraElevationDeg(5); window.setCameraPointingToPoint(0.4f, 0.f, 0.f);
	//Tracking me - window.setCameraZoom(4.f); window.setCameraAzimuthDeg(180);	window.setCameraElevationDeg(35); window.setCameraPointingToPoint(0.f, 0.f, 0.f);
	//Exp2 (teddy seg?) - window.setCameraZoom(1.8,1.8,2); window.setCameraAzimuthDeg(205); window.setCameraElevationDeg(8); window.setCameraPointingToPoint(0.f, 0.f, 0.f)?;
	//Exp3 (teddy null?) - window.setCameraZoom(1.4); window.setCameraAzimuthDeg(225); window.setCameraElevationDeg(25); window.setCameraPointingToPoint(0.f, 0.f, 0.f)?;

	if (paper_visualization)
	{
		window.getDefaultViewport()->setCustomBackgroundColor(utils::TColorf(1.f, 1.f, 1.f));
		window.captureImagesStart();
	}

	scene = window.get3DSceneAndLock();

	// Lights:
	scene->getViewport()->setNumberOfLights(2);
	mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	light0.light_ID = 0;
	light0.setPosition(-0.5f, 0.f, 0.5f, 0.f);

	mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	light1.light_ID = 1;


	//Control mesh
	opengl::CMesh3DPtr control_mesh = opengl::CMesh3D::Create();
	control_mesh->enableShowEdges(true);
	control_mesh->enableShowFaces(false);
	control_mesh->enableShowVertices(true);
	control_mesh->setLineWidth(2.f);
	control_mesh->setPointSize(8.f);
	if (paper_visualization)
	{
		control_mesh->setEdgeColor(0.9f, 0.f, 0.f);
		control_mesh->setVertColor(0.6f, 0.f, 0.f);
	}
	scene->insert(control_mesh);


	const unsigned int num_sizes = 10;
	for (unsigned int s = 0; s < num_sizes; s++)
	{
		//Points
		opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
		points->setPointSize(5.f + 0.75*s);
		points->enablePointSmooth(true);
		scene->insert(points);
			
		//for (unsigned int v = 0; v < rows; v++)
		//	for (unsigned int u = 0; u < cols; u++)
		//		if  (is_object[0](v,u) && (depth[0](v,u) < 2.5f))
		//			points->push_back(depth[0](v,u), x_image[0](v,u), y_image[0](v,u), 0.f, 1.f, 0.f);

				//else if (valid[i](v,u))
				//	points->push_back(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), 0.f, 0.f, 0.f);
	}

	//Lines
	opengl::CSetOfLinesPtr normals = opengl::CSetOfLines::Create();
	normals->setColor(0, 0, 0.8f);
	normals->setLineWidth(1.f);
	scene->insert(normals);

	const float fact = 0.01f;
	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//normals->setColor(r, g, b);
	//for (unsigned int v = 0; v < rows; v++)
	//	for (unsigned int u = 0; u < cols; u++)
	//	if (is_object[i](v, u))
	//		normals->appendLine(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 
	//		depth[i](v, u) + fact*nx_image[i](v, u), x_image[i](v, u) + fact*ny_image[i](v, u), y_image[i](v, u) + fact*nz_image[i](v, u));


	////Internal model (subdivision surface)
	//opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//points->setPointSize(2.f);
	//points->enablePointSmooth(true);
	//scene->insert(points);


	//3D Model
	opengl::CMesh3DPtr model = opengl::CMesh3D::Create();
	model->enableShowVertices(false); //false
	model->enableShowEdges(true); //false
	model->enableShowFaces(false); //true
	model->enableFaceNormals(false); //true
	//model->enableTransparency(true);
	model->setFaceColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setVertColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setEdgeColor(0.7f, 0.7f, 0.8f, 0.3f);
	model->setPointSize(1.f);
	model->setLineWidth(1.f);
	scene->insert(model);

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::takePictureLimitSurface(bool last)
{
	//takePictureDataArch();
	
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;

	//8 for the surface, 5 for the fine mesh
	const unsigned int num_ref = 8;
	
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

	//Coordinates conversion
	const Matrix4f &mytrans_inv = cam_trans_inv[3]; //Only for the teddy!!!! ***********

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	//for (unsigned int cv = 0; cv < num_verts_now; cv++)
	//{
	//	const float vx = mytrans_inv(0, 0)*vert_coords_now(0, cv) + mytrans_inv(0, 1)*vert_coords_now(1, cv) + mytrans_inv(0, 2)*vert_coords_now(2, cv) + mytrans_inv(0, 3);
	//	const float vy = mytrans_inv(1, 0)*vert_coords_now(0, cv) + mytrans_inv(1, 1)*vert_coords_now(1, cv) + mytrans_inv(1, 2)*vert_coords_now(2, cv) + mytrans_inv(1, 3);
	//	const float vz = mytrans_inv(2, 0)*vert_coords_now(0, cv) + mytrans_inv(2, 1)*vert_coords_now(1, cv) + mytrans_inv(2, 2)*vert_coords_now(2, cv) + mytrans_inv(2, 3);

	//	vert_coords_now(0,cv) = vx;
	//	vert_coords_now(1,cv) = vy;
	//	vert_coords_now(2,cv) = vz;
	//}
	is_quad.resize(num_faces_now, 1); is_quad.fill(true);
	model->loadMesh(num_verts_now, num_faces_now, is_quad, face_verts_now, vert_coords_now);


	opengl::CMesh3DPtr mesh = scene->getByClass<CMesh3D>(0);

	vert_coords_now.resize(3, num_verts);
	vert_coords_now = vert_coords;
	//for (unsigned int cv = 0; cv < num_verts; cv++)
	//{
	//	vert_coords_now(0,cv) = mytrans_inv(0, 0)*vert_coords(0, cv) + mytrans_inv(0, 1)*vert_coords(1, cv) + mytrans_inv(0, 2)*vert_coords(2, cv) + mytrans_inv(0, 3);
	//	vert_coords_now(1,cv) = mytrans_inv(1, 0)*vert_coords(0, cv) + mytrans_inv(1, 1)*vert_coords(1, cv) + mytrans_inv(1, 2)*vert_coords(2, cv) + mytrans_inv(1, 3);
	//	vert_coords_now(2,cv) = mytrans_inv(2, 0)*vert_coords(0, cv) + mytrans_inv(2, 1)*vert_coords(1, cv) + mytrans_inv(2, 2)*vert_coords(2, cv) + mytrans_inv(2, 3);
	//}

	is_quad.resize(num_faces, 1); is_quad.fill(true);
	if (!last) 	mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords_now);
	else 		mesh->loadMesh(0, 0, is_quad, face_verts, vert_coords_now);


	window.unlockAccess3DScene();
	window.repaint();	

	system::sleep(2500);


	// Open file: find the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;
	string  name;

	while (!free_name)
	{
		nFile++;
		sprintf_s(aux, "image_%03u.png", nFile );
		name = f_folder + aux;
		free_name = !system::fileExists(name);
	}

	utils::CImage img;
	window.getLastWindowImage(img);
	img.saveToFile(name);
}

void Mod3DfromRGBD::takePictureDataArch()
{
	unsigned int num_verts_now = num_verts;
	unsigned int num_faces_now = num_faces;
	Array<int, 4, Dynamic> face_verts_now = face_verts;
	Array<float, 3, Dynamic> vert_coords_now = vert_coords;
	Far::TopologyRefiner *refiner_now;
	std::vector<Vertex> verts_now = verts;

	const unsigned int num_ref = 5;
	
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


	//opengl::CMesh3DPtr mesh = scene->getByClass<CMesh3D>(0);
	//vert_coords_now.resize(3, num_verts);
	//vert_coords_now = vert_coords;
	//is_quad.resize(num_faces, 1); is_quad.fill(true);
	//if (!last)	mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords_now);
	//else		mesh->loadMesh(0, 0, is_quad, face_verts, vert_coords_now);


	//Show data and connecting lines (testing)
	const unsigned int num_sizes = 10;
	for (unsigned int s = 0; s < num_sizes; s++)
	{
		//Points (data)
		opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(s);
		points->clear();
		points->setPose(cam_poses[0]);

		
		const float max_res = 0.04f, max_red = 0.8f, max_green = 0.8f; //max_res = 0.02 arch, max_res = 0.04 tracking
		const float thresh_low = float(s)*max_res/float(num_sizes);
		const float thresh_high = s < num_sizes-1 ? float(s+1)*max_res/float(num_sizes) : 1000.f;
			
		for (unsigned int v = 0; v < rows; v++)
			for (unsigned int u = 0; u < cols; u++)
				if (is_object[0](v, u) && (depth[0](v,u) < 2.5f))
				{
					const float res = res_pos[0].col(v+rows*u).norm();
					if ((res <= thresh_high)&&(res > thresh_low))
					{
						const float red = min(max_red, res/max_res);
						const float green = max(0.f, max_green*(1.f-res/max_res));
						points->push_back(depth[0](v,u), x_image[0](v,u), y_image[0](v,u), red, green, 0.f);	
					}
				}	
	}

	//Connecting lines
	//const float fact_norm = 0.03f;
	//float r,g,b;
	//CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(i);
	//conect->clear();
	//conect->setPose(cam_poses[i]);
	//conect->setColor(0.5f, 0.5f, 0.5f);
	////utils::colormap(mrpt::utils::cmJET,float(i) / float(num_images),r,g,b);
	////conect->setColor(r,g,b);
	//for (unsigned int u = 0; u < cols; u++)
	//	for (unsigned int v = 0; v < rows; v++)
	//		if (is_object[i](v,u))
	//			conect->appendLine(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), mx_t[i](v,u), my_t[i](v,u), mz_t[i](v,u));


	window.unlockAccess3DScene();
	window.repaint();	

	//system::sleep(2500);


	// Open file: find the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;
	string  name;

	while (!free_name)
	{
		nFile++;
		sprintf_s(aux, "image_%03u.png", nFile );
		name = f_folder + aux;
		free_name = !system::fileExists(name);
	}

	utils::CImage img;
	window.getLastWindowImage(img);
	img.saveToFile(name);
}




void Mod3DfromRGBD::saveSceneAsImage()
{
	// Open file: find the first free file-name.
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;
	string  name;

	while (!free_name)
	{
		nFile++;
		sprintf_s(aux, "image_%03u.png", nFile );
		name = f_folder + aux;
		free_name = !system::fileExists(name);
	}

	utils::CImage img;
	window.getLastWindowImage(img);
	img.saveToFile(name);
}

void Mod3DfromRGBD::showDTAndDepth()
{
	if ((solve_DT)&&(!paper_visualization))
	{
		COpenGLViewportPtr gl_view_dt = scene->getViewport("DT");
		gl_view_dt->setViewportPosition(10, 10, 320, 240);
		utils::CImage viewport_image;
		viewport_image.setFromMatrix(DT[0].matrix(), false);
		viewport_image.flipVertical();
		viewport_image.normalize();
		gl_view_dt->setImageView(viewport_image);

		COpenGLViewportPtr gl_depth_im = scene->getViewport("depth");
		gl_depth_im->setViewportPosition(10, 700, 320, 240);
		MatrixXf mat_aux = (is_object[0].cast<float>() + valid[0].cast<float>()).matrix();
		viewport_image.setFromMatrix(mat_aux, false);
		viewport_image.flipVertical();
		viewport_image.normalize();
		gl_depth_im->setImageView(viewport_image);
	}
}

void Mod3DfromRGBD::showRenderedModel()
{
	scene = window.get3DSceneAndLock();

	opengl::CMesh3DPtr model = scene->getByClass<CMesh3D>(1);
	model->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);
	if (with_color)
	{
		Matrix3Xf mesh_colors(3, num_verts);
		mesh_colors.row(0) = vert_colors;
		mesh_colors.row(1) = vert_colors;
		mesh_colors.row(2) = vert_colors;
		model->loadVertColors(mesh_colors);
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showCamPoses()
{
	scene = window.get3DSceneAndLock();

	for (unsigned int i = 0; i < num_images; i++)
	{
		//Points (data)
		opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(i);
		points->setPose(cam_poses[i]);
		if ((vis_errors)&&(!with_color))
		{
			const float max_res = 0.04f;
			const float max_red = 0.8;
			const float max_green = 0.8f;
			unsigned int index = 0;
			
			for (unsigned int v = 0; v < rows; v++)
				for (unsigned int u = 0; u < cols; u++)
					if (is_object[i](v, u) && (depth[i](v,u) < 2.5f))
					{
						const float res = res_pos[i].col(v+rows*u).norm();
						const float red = min(max_red, res/max_res);
						const float green = max(0.f, max_green*(1.f-res/max_res));

						//const float red = 1.f - n_weights[i](v + rows*u);
						//const float green = n_weights[i](v + rows*u);

						points->setPointColor_fast(index, red, green, 0.f);
						index++;	
					}					
		}

		//Cameras
		opengl::CFrustumPtr frustum = scene->getByClass<CFrustum>(i);
		frustum->setPose(cam_poses[i]);

		//Normals
		//opengl::CSetOfLinesPtr normals = scene->getByClass<CSetOfLines>(i);
		//normals->setPose(cam_poses[i]);
		//normals->clear();

		//const float fact = 0.01f;
		//for (unsigned int v = 0; v < rows; v++)
		//	for (unsigned int u = 0; u < cols; u++)
		//		if (is_object[i](v, u))
		//		{
		//			const float res = sqrtf(square(res_x[i](v,u)) + square(res_y[i](v,u)) + square(res_z[i](v,u)));
		//			if (res > 0.03f)
		//				normals->appendLine(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 
		//				depth[i](v, u) + fact*nx_image[i](v, u), x_image[i](v, u) + fact*ny_image[i](v, u), y_image[i](v, u) + fact*nz_image[i](v, u));
		//		}
				
	}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::showNewData(bool new_pointcloud)
{
	scene = window.get3DSceneAndLock();

	for (unsigned int i = 0; i < num_images; i++)
	{
		//Points (data)
		if (new_pointcloud)
		{
			opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(i);
			points->clear();
			points->setPose(cam_poses[i]);

			const float max_res = 0.04f, max_red = 0.8, max_green = 0.8f;
			float r,g,b;

			if (vis_errors)	{r = 0.f; g = 0.7f; b = 0.f;}
			else			{utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);}
			
			for (unsigned int v = 0; v < rows; v++)
				for (unsigned int u = 0; u < cols; u++)
					if (is_object[i](v, u) && (depth[i](v,u) < 2.5f))
					{
						if (with_color)
						{
							r = intensity[i](v,u);
							g = r;
							b = r;						
						}				
						else if (vis_errors)
						{
							const float res = res_pos[i].col(v+rows*u).norm();
							r = min(max_red, res/max_res);
							g = max(0.f, max_green*(1.f-res/max_res));
						}

						points->push_back(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), r, g, b);	
					}	
		}

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
	control_mesh->loadMesh(num_verts, num_faces, is_quad, face_verts, vert_coords);

	//Show vertex numbers
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
	{
		const MatrixXf &surf_ref = surf[i];
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				const int index = v+rows*u;
				if (is_object[i](v,u))
					points->push_back(surf_ref(0,index), surf_ref(1,index), surf_ref(2,index), 0.7f, 0.7f, 0.7f);

				else if((solve_SK)&&(valid[i](v,u)))
					points->push_back(surf_ref(0,index), surf_ref(1,index), surf_ref(2,index), 0.f, 0.f, 0.f);
			}
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
	const float fact_norm = 0.03f;
	float r,g,b;
	for (unsigned int i = 0; i < num_images; i++)
	{
		CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(i);
		conect->clear();
		conect->setPose(cam_poses[i]);
		//utils::colormap(mrpt::utils::cmJET,float(i) / float(num_images),r,g,b);
		//conect->setColor(r,g,b);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v,u))
				{
					const int index = v+rows*u;
					conect->appendLine(depth[i](v,u), x_image[i](v,u), y_image[i](v,u), surf_t[i](0,index), surf_t[i](1,index), surf_t[i](2,index));
				}
	}

	//Normals of the correspondences
	//const float scale = 10.f;
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(i);
	//	conect->clear();
	//	conect->setColor(0,0,1);
	//	for (unsigned int u = 0; u < cols; u++)
	//		for (unsigned int v = 0; v < rows; v++)
	//			if (is_object[i](v,u))
	//				conect->appendLine(mx[i](v,u), my[i](v,u), mz[i](v,u),
	//									mx[i](v,u) + scale*nx[i](v,u), my[i](v,u) + scale*ny[i](v,u), mz[i](v,u) + scale*nz[i](v,u));
	//}

	////Normals of correspondences (reg)
	//const float scale = 0.05f;
	//CSetOfLinesPtr conect = scene->getByClass<CSetOfLines>(0);
	//conect->clear();
	//conect->setColor(0,0,1);
	//for (unsigned int v = 0; v<num_verts; v++)
	//	conect->appendLine(mx_reg[v](0), my_reg[v](0), mz_reg[v](0),
	//					   mx_reg[v](0) + scale*n_verts(0,v), my_reg[v](0) + scale*n_verts(1,v), mz_reg[v](0) + scale*n_verts(2,v));

	////Different evaluations of the surface
	//const unsigned int sampl = max(10, int(100.f/sqrtf(num_faces)));
	//const float fact = 1.f / float(sampl-1);
	//Far::PatchMap patchmap(*patchTable);
	//float pWeights[20], dsWeights[20], dtWeights[20];

	//CPointCloudPtr subsurface = scene->getByClass<CPointCloud>(0); subsurface->clear();
	//Far::PtexIndices ptexIndices(*refiner);
 //   int nfaces = ptexIndices.GetNumFaces();

	////Show the subdiv surface evaluated at the control vertices
	//for (unsigned int v = 0; v<num_verts; v++)
	//{
	//	// Locate the patch corresponding to the face ptex idx and (s,t)
	//	Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(vert_u(2,v), vert_u(0,v), vert_u(1,v)); assert(handle);

	//	// Evaluate the patch weights, identify the CVs and compute the limit frame:
	//	patchTable->EvaluateBasis(*handle, vert_u(0,v), vert_u(1,v), pWeights, dsWeights, dtWeights);

	//	Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

	//	LimitFrame eval; eval.Clear();
	//	for (int cv = 0; cv < cvs.size(); ++cv)
	//		eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

	//	//Insert the point
	//	subsurface->insertPoint(eval.point[0], eval.point[1], eval.point[2]);
	//}

	//Show the whole surface
	//for (unsigned int f = 0; f<nfaces; f++)
	//	for (unsigned int u = 0; u < sampl; u++)
	//		for (unsigned int v = 0; v < sampl; v++)
	//		{
	//			// Locate the patch corresponding to the face ptex idx and (s,t)
	//			Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(f, u*fact, v*fact); assert(handle);

	//			// Evaluate the patch weights, identify the CVs and compute the limit frame:
	//			patchTable->EvaluateBasis(*handle, u*fact, v*fact, pWeights, dsWeights, dtWeights);

	//			Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);

	//			LimitFrame eval; eval.Clear();
	//			for (int cv = 0; cv < cvs.size(); ++cv)
	//				eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

	//			//Insert the point
	//			subsurface->insertPoint(eval.point[0], eval.point[1], eval.point[2]);
	//		}

	//Show the regularization samples and the residual associated to them
	//if (with_reg_normals)
	//{
	//	CPointCloudColouredPtr reg = scene->getByClass<CPointCloudColoured>(num_images+1);
	//	reg->clear();
	//	Kr = Kr_total/float(square(s_reg));
	//	const float k_energy = 60.f;
	//	for (unsigned int f = 0; f<num_faces; f++)
	//		for (unsigned int s1 = 0; s1 < s_reg-1; s1++)
	//			for (unsigned int s2 = 0; s2 < s_reg-1; s2++)
	//			{
	//				const float dist_s1 = square(mx_reg[f](s1+1,s2) - mx_reg[f](s1,s2)) + square(my_reg[f](s1+1,s2) - my_reg[f](s1,s2)) + square(mz_reg[f](s1+1,s2) - mz_reg[f](s1,s2));
	//				const float dist_s2 = square(mx_reg[f](s1,s2+1) - mx_reg[f](s1,s2)) + square(my_reg[f](s1,s2+1) - my_reg[f](s1,s2)) + square(mz_reg[f](s1,s2+1) - mz_reg[f](s1,s2));

	//				const float energy = Kr*((square(nx_reg[f](s1+1,s2) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1+1,s2) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1+1,s2) - nz_reg[f](s1,s2)))/dist_s1
	//								+(square(nx_reg[f](s1,s2+1) - nx_reg[f](s1,s2)) + square(ny_reg[f](s1,s2+1) - ny_reg[f](s1,s2))	+ square(nz_reg[f](s1,s2+1) - nz_reg[f](s1,s2)))/dist_s2);
	//			
	//				utils::colormap(mrpt::utils::cmJET, sqrtf(k_energy*energy), r,g,b);			
	//				reg->push_back(mx_reg[f](s1,s2), my_reg[f](s1,s2), mz_reg[f](s1,s2), r, g, b);
	//			}
	//}

	window.unlockAccess3DScene();
	window.repaint();
}

void Mod3DfromRGBD::drawRayAndCorrespondence(unsigned int i, unsigned int v, unsigned int u)
{
	showMesh(); 
	showCamPoses(); 
	showSubSurface();
	
	scene = window.get3DSceneAndLock();

	//Ray
	CSetOfLinesPtr ray = opengl::CSetOfLines::Create();
	ray->setColor(0.f, 0.f, 0.7f);
	ray->setPose(cam_poses[i]);
	scene->insert( ray );

	const float disp_u = 0.5f*float(cols - 1);
	const float disp_v = 0.5f*float(rows - 1);
	const float x = 4.f;
	const float y = x*(u - disp_u)/fx;
	const float z = x*(v - disp_v)/fy;
	ray->appendLine(0.f, 0.f, 0.f, x, y, z);

	//Point
	CPointCloudPtr point = opengl::CPointCloud::Create();
	point->setColor(0.f, 0.f, 0.7f);
	point->enablePointSmooth();
	point->setPointSize(10.f);
	scene->insert( point );

	point->insertPoint(surf[i](v+rows*u), surf[i](1,v+rows*u), surf[i](2,v+rows*u));
	printf("\n coordinates: %f %f %d", u1[i](v,u), u2[i](v,u), uface[i](v,u));


	CPointCloudPtr point2 = opengl::CPointCloud::Create();
	point2->setColor(0.f, 0.f, 0.7f);
	point2->enablePointSmooth();
	point2->setPointSize(10.f);
	point2->setPose(cam_poses[i]);
	scene->insert( point2 );

	point2->insertPoint(surf_t[i](v+rows*u), surf_t[i](1,v+rows*u), surf_t[i](2,v+rows*u));

	window.unlockAccess3DScene();
	window.repaint();

	system::os::getch();

	//Delete objects
	scene = window.get3DSceneAndLock();
	scene->removeObject(ray);
	scene->removeObject(point);
	scene->removeObject(point2);
	window.unlockAccess3DScene();
	window.repaint();
}

