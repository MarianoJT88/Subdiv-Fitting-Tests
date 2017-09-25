// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"

void Mod3DfromRGBD::saveMeshToFile()
{
	//Open file
	string dir = "C:/Users/jaimez/Desktop/mesh.obj";

	std::ofstream	f_mesh;
	f_mesh.open(dir.c_str());
		
	//Save vertices
	for (unsigned int v=0; v<num_verts; v++)
		f_mesh << "v " << vert_coords(0,v) << " " << vert_coords(1,v) << " " << vert_coords(2,v) << endl;

	//Save faces
	for (unsigned int f=0; f<num_faces; f++)
	{
		f_mesh << "f ";
		for (unsigned int k=0; k<4; k++)
			f_mesh << face_verts(k,f) << "//" << f << " ";		
		f_mesh << endl;
	}

	f_mesh.close();
}

void Mod3DfromRGBD::createImagesFromCube()
{
	//Render one image - unitary cube
	const float dist = 1.f;

	for (unsigned int i = 0; i<num_images; i++)
		for (unsigned int u = 0; u<cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				depth[i](v,u) = dist;
				x_image[i](v,u) = (u - 0.5f*(cols-1))/(cols-1);
				y_image[i](v,u) = (v - 0.5f*(rows-1))/(rows-1);
				xyz_image[i].col(v+rows*u) << dist, x_image[i](v,u), y_image[i](v,u);
				is_object[i](v,u) = true;
				valid[i](v,u) = true;
			}
}

void Mod3DfromRGBD::loadInitialMesh()
{
	//if (image_set == 5)
	//{
	//	loadInitialMeshUShape();	
	//}
	//else
	//if ((image_set == 6)||(image_set == 7))
	//{
	//	loadMeshFromKinectFusion();
	//}
	//else
	{
		//Initial mesh - A cube/parallelepiped
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
		float min_x, max_x, min_y, max_y, min_z, max_z;

		//Find the center of gravity and create a cube around it
		if (seg_from_background)
		{
			float mean_x = 0.f, mean_y = 0.f, mean_z = 0.f;
			unsigned int cont = 0;
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

							mean_x += x_t; mean_y += y_t; mean_z += z_t;
							cont++;
						}
			}

			mean_x /= float(cont); mean_y /= float(cont); mean_z /= float(cont);

			if (image_set == 3)
			{
				max_x = mean_x + ini_size; min_x = mean_x - ini_size;
				max_y = mean_y + ini_size; min_y = mean_y - ini_size;
				max_z = mean_z + ini_size; min_z = mean_z - ini_size;
			}
		}

		//If the segmentation is perfect -> get the bounding box of the 3D point cloud
		else
		{
			min_x = 10.f; min_y = 10.f; min_z = 10.f;
			max_x = -10.f; max_y = -10.f; max_z = -10.f;

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
		}


		//Small initialization for the teddy (image_set 2)
		//max_x += 0.05f; min_x += 0.05f;
		//min_z += 0.02f; min_z += 0.02f;
		//min_y -= 0.06f; max_y -= 0.06f;
		//const float x_margin = -0.3f*(max_x - min_x);
		//const float y_margin = -0.3f*(max_y - min_y);
		//const float z_margin = -0.1f*(max_z - min_z);

		//Small initialization for the Arch - Starting from one column
		//max_x -= 0.05f; min_x -= 0.05f;
		//min_z += 0.02f; min_z += 0.02f;
		//min_y -= 0.25f; max_y -= 0.25f;
		//const float x_margin = -0.3f*(max_x - min_x); //-0.3f
		//const float y_margin = -0.3f*(max_y - min_y); //-0.3f
		//const float z_margin = -0.3f*(max_z - min_z); //-0.1f

		//Small initialization for the Arch - Starting from the upper part
		//max_x -= 0.05f; min_x -= 0.05f;
		//min_z += 0.2f; min_z += 0.2f;
		//min_y -= 0.25f; max_y += 0.25f;
		//const float x_margin = -0.3f*(max_x - min_x); //-0.3f
		//const float y_margin = -0.3f*(max_y - min_y); //-0.3f
		//const float z_margin = -0.3f*(max_z - min_z); //-0.1f

		//Standard
		const float factor = small_initialization ? -0.3f : 0.f;
		const float x_margin = factor*(max_x - min_x);
		const float y_margin = factor*(max_y - min_y);
		const float z_margin = factor*(max_z - min_z);


		vert_coords.resize(3, num_verts); vert_coords_old.resize(3, num_verts);
		vert_coords_reg.resize(3, num_verts); 
		vert_coords.col(0) << min_x - x_margin, min_y - y_margin, max_z + z_margin;
		vert_coords.col(1) << max_x + x_margin, min_y - y_margin, max_z + z_margin;
		vert_coords.col(2) << min_x - x_margin, max_y + y_margin, max_z + z_margin;
		vert_coords.col(3) << max_x + x_margin, max_y + y_margin, max_z + z_margin;
		vert_coords.col(4) << min_x - x_margin, max_y + y_margin, min_z - z_margin;
		vert_coords.col(5) << max_x + x_margin, max_y + y_margin, min_z - z_margin;
		vert_coords.col(6) << min_x - x_margin, min_y - y_margin, min_z - z_margin;
		vert_coords.col(7) << max_x + x_margin, min_y - y_margin, min_z - z_margin;
	}


	//Resize variables for the optimization and copy original mesh (tracking)
	vert_coords_reg = vert_coords; //For ctf_regularization, edge_ini and arap
	vert_coords_old = vert_coords;
	rot_arap.resize(num_faces);
	rot_mfold.resize(num_faces);
	rot_mfold_old.resize(num_faces);
	for (unsigned int f=0; f<num_faces; f++)
	{
		rot_arap[f].setIdentity();
		rot_mfold[f].fill(0.f);
		rot_mfold_old[f].fill(0.f);
	}

	//Regularization (normals)
	if (with_reg_normals || with_reg_normals_good || with_reg_normals_4dir)
		initializeRegForFitting();
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
}


void Mod3DfromRGBD::readAssocFileKinectFusion()
{
    //Open Assoc file
	std::ifstream	f_assoc;
	string filename = im_set_dir;
	filename.append("rgbd_assoc.txt");

	cout << endl << "Filename (assoc): " << filename;
	f_assoc.open(filename.c_str());

	if (f_assoc.fail())
		throw std::runtime_error("\nError finding the trajectory file.");


    // first load all groundtruth timestamps and poses
    std::string line;
    while (std::getline(f_assoc, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestampDepth, timestampColor;
        std::string fileDepth, fileColor;
        if (!(iss >> timestampColor >> fileColor >> timestampDepth >> fileDepth))
            break;

        timestamps.push_back(timestampDepth);
        files_depth.push_back(fileDepth);
        files_color.push_back(fileColor);
    }

    f_assoc.close();
}

void Mod3DfromRGBD::loadImagesFromKinectFusion()
{
	float depth_scale = 1.f / 5000.f;
	float depth_offset = 0.f;

	//char aux[30];
	const float norm_factor = 1.f/255.f;
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	//Images
	for (unsigned int i = 0; i < num_images; i++)
	{
		//Depth
		//float ind = i*(trajectory.size()-2)/(num_images) + 1;
		//sprintf_s(aux, "depth/%.6f.png", ind);
		//string name = im_set_dir + aux;

		int ind = i*(trajectory.size()-2)/(num_images) + 1;
		string name = im_set_dir + files_depth.at(ind);
		cout << endl << "Filename (image): " << name;

		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		const float disp_v = 0.5f*(rows - 1);

		cv::Mat im_d = cv::imread(name, -1);
		cv::Mat depth_float;

		im_d.convertTo(depth_float, CV_32FC1, depth_scale);
		for (unsigned int u = 0; u<cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample) + depth_offset;
				depth[i](v, u) = d;
				x_image[i](v, u) = (u - disp_u)*d*inv_fd;
				y_image[i](v, u) = (v - disp_v)*d*inv_fd;
				xyz_image[i].col(v+rows*u) << d, x_image[i](v, u), y_image[i](v, u);
			}

		//Load their poses into the right variables
		cam_ini[i] = trajectory.at(ind).block<4,4>(0,0);
		CMatrixDouble44 mat = cam_ini[i];
		cam_poses[i] = CPose3D(mat);
	}

}

void Mod3DfromRGBD::loadPosesFromKinectFusion()
{
	//Open Trajectory file
	std::ifstream	f_trajectory;
	string filename = im_set_dir;
	filename.append("traj.txt");


	cout << endl << "Filename: " << filename;
	f_trajectory.open(filename.c_str());

	if (f_trajectory.fail())
		throw std::runtime_error("\nError finding the trajectory file.");

	// first load all groundtruth timestamps and poses
    std::string line;
    while (std::getline(f_trajectory, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestamp;
        float tx, ty, tz;
        float qx, qy, qz, qw;
        if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
            break;

        timestamps.push_back(timestamp);

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        Eigen::Vector3f tVec(tx, ty, tz);
        pose.topRightCorner(3,1) = tVec;
        //Eigen::Quaternionf quat(qw, qx, qy, qz);
		Eigen::Quaternionf quat(qw, qx, qy, qz);
        pose.topLeftCorner(3,3) = quat.toRotationMatrix();

		CPose3D pose_corr = CPose3D(0,0,0,0,-M_PI/2, M_PI/2);
		CMatrixDouble44 mat_aux; pose_corr.getHomogeneousMatrix(mat_aux);
		Matrix4f mat_corr = mat_aux.cast<float>();


		MatrixXf aux_pose = pose*mat_corr;
        trajectory.push_back(aux_pose);
    }

    // align all poses so that the initial pose is the identity matrix
	bool firstPoseIsIdentity = false;
    if (firstPoseIsIdentity && !trajectory.empty())
    {
        Eigen::Matrix4f initPose = trajectory[0];

		//Correction
        for (unsigned int i = 0; i < trajectory.size(); ++i)
            trajectory[i] = initPose.inverse() * trajectory[i];
    }

	f_trajectory.close();
}

void Mod3DfromRGBD::loadMeshFromKinectFusion()
{
	//Open file
	string dir = im_set_dir;

	std::ifstream	f_mesh;
	dir.append("quad_mesh.obj");

	cout << endl << "Filename: " << dir;
	f_mesh.open(dir.c_str());
		
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


	//Load the vertices
	vert_coords.resize(3, num_verts);

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
			iss >> aux_v >> vert_coords(0,cont) >> vert_coords(1,cont) >> vert_coords(2,cont);	
			cont++;
		}
		else
			is_vert = false;
	}

	//Load the faces
	face_verts.resize(4, num_faces);
	cont = 0;
	printf("\nReading faces...");

	while (line.at(0) == 'v')
		std::getline(f_mesh, line);

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

	face_verts -= 1;
         

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

void Mod3DfromRGBD::loadMeshFromFile()
{
	//Open file
	string dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting-Private/data/meshes/";
				//"C:/Users/Mariano/Programas/GitHub/OpenSubdiv-Model-Fitting-Private/data/meshes/";

	std::ifstream	f_mesh;
	dir.append("good_mesh_person_no_thumbs.obj");

	cout << endl << "Filename: " << dir;
	f_mesh.open(dir.c_str());
		
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


	//Load the vertices
	vert_coords.resize(3, num_verts);

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
			//iss >> aux_v >> vert_coords(0,cont) >> vert_coords(1,cont) >> vert_coords(2,cont);	
			iss >> aux_v >> vert_coords(1,cont) >> vert_coords(2,cont) >> vert_coords(0,cont);	
			cont++;
		}
		else
			is_vert = false;
	}
	vert_coords *= 0.075f; //0.082f
	vert_coords.row(0) = -vert_coords.row(0);

	//Load the faces
	is_quad.resize(num_faces, 1);
	is_quad.fill(true);
	face_verts.resize(4, num_faces);
	cont = 0;
	printf("\nReading faces...");

	while (line.at(0) == 'v')
		std::getline(f_mesh, line);

	do
	{
		std::istringstream iss(line);
		char aux_c1; int aux_num;
		iss >> aux_c1;
		for (unsigned int k=0; k<4; k++)
		{
			if (iss.eof())
			{
				is_quad(cont) = false;
				face_verts(3,cont) = 0;

				//I don't include triangles
				//cont--; 
				//num_faces--;
			}
			else
				iss >> face_verts(3-k,cont) >> aux_c1 >> aux_num >> aux_c1 >> aux_num; //To have normals defined properly			
		}
		//printf("\n Read: %d %d %d %d", face_verts(0,cont), face_verts(1,cont), face_verts(2,cont), face_verts(3,cont));
		
		cont++;

	} while (std::getline(f_mesh, line));
	face_verts -= 1;


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

	//Necessary for some of the regularization terms
	findOppositeVerticesFace();
	findNeighbourhoodsForArap();


	//Resize variables for the optimization and copy original mesh (tracking)
	vert_coords_reg = vert_coords; //For ctf_regularization, edge_ini and arap
	vert_coords_old = vert_coords;
	rot_arap.resize(num_verts);
	rot_mfold.resize(num_verts);
	rot_mfold_old.resize(num_verts);
	for (unsigned int v=0; v<num_verts; v++)
	{
		rot_arap[v].setIdentity();
		rot_mfold[v].fill(0.f);
		rot_mfold_old[v].fill(0.f);
	}

	//Regularization (normals)
	initializeRegForFitting();

	//Initialize colors (if used)
	if (with_color)
	{
		vert_colors.resize(1, num_verts);
		vert_colors.fill(0.5f);
	}


	////Perturb the vertices
	//for (unsigned int v=0; v<num_verts; v++)
	//	for (unsigned int k=0; k<3; k++)
	//		vert_coords(k,v) += 0.00005f*float(rand()%1000);
}


void Mod3DfromRGBD::loadInputs()
{
	string name;
	switch (image_set) {

	case 1:
		createImagesFromCube();
		break;

	case 2:
		im_set_dir = im_dir + "images Teddy/";
		seg_from_background = false;
		num_images = min(int(num_images), 5);
		loadImagesStandard();
		max_depth_segmentation = 1.f;
		max_radius_segmentation = 0.3f;
		plane_res_segmentation = 0.05f;
		computeDepthSegmentationFromPlane();
		break;

	case 3:
		im_set_dir = im_dir + "teddy remove background/";
		seg_from_background = true;
		num_images = min(int(num_images), 4);
		loadImagesStandard();
		computeDepthSegmentationFromBackground();
		break;

	case 4:
		im_set_dir = im_dir + "person front/";
		seg_from_background = false;
		num_images = 1;
		loadImagesStandard();
		computeDepthSegmentationDepthRange(0.4f, 2.f);
		break;

	case 5:
		im_set_dir = im_dir + "u shape/";
		seg_from_background = false;
		num_images = min(int(num_images), 1);
		loadImagesStandard();
		computeDepthSegmentationDepthRange(0.f, 1.32f);
		//computeDepthSegmentationUShape();
		break;

	case 6:
		im_dir = "D:/RGBD sequences subdiv fitting/";
		//im_dir = "C:/Users/Mariano/Desktop/datasets subdiv fitting/";
		im_set_dir = im_dir + "dataset_robert/";
		seg_from_background = false;
		num_images = min(int(num_images), 20);
		readAssocFileKinectFusion();
		loadPosesFromKinectFusion();
		loadImagesFromKinectFusion();
		computeDepthSegmentationFromBoundingBox();
		break;

	case 7:
		im_dir = "D:/RGBD sequences subdiv fitting/";
		//im_dir = "C:/Users/Mariano/Desktop/datasets subdiv fitting/";
		im_set_dir = im_dir + "dataset_teddy2/";
		seg_from_background = false;
		num_images = min(int(num_images), 20);
		readAssocFileKinectFusion();
		loadPosesFromKinectFusion();
		loadImagesFromKinectFusion();
		max_depth_segmentation = 0.7f;
		max_radius_segmentation = 0.3f;
		plane_res_segmentation = 0.05f;
		computeDepthSegmentationFromPlane();
		//computeDepthSegmentationFromBoundingBox();
		break;

	case 8:
		im_set_dir = im_dir + "sculpture1/";
		num_images = min(int(num_images), 8);
		seg_from_background = false;
		loadImagesStandard();
		max_depth_segmentation = 0.35f;
		max_radius_segmentation = 0.3f;
		plane_res_segmentation = 0.05f;
		computeDepthSegmentationFromPlane();
		break;

	case 9:
		im_set_dir = im_dir + "sculpture2/";
		num_images = min(int(num_images), 8);
		seg_from_background = false;
		loadImagesStandard();
		max_depth_segmentation = 0.45f;
		max_radius_segmentation = 0.22f;
		plane_res_segmentation = 0.05f;
		computeDepthSegmentationFromPlane();
		break;

	case 10:
		im_set_dir = im_dir + "motorbike1/";
		num_images = min(int(num_images), 8);
		seg_from_background = false;
		loadImagesStandard();
		max_depth_segmentation = 0.55f;
		max_radius_segmentation = 0.3f;
		plane_res_segmentation = 0.04f;
		computeDepthSegmentationFromPlane();
		break;

	case 11:
		im_set_dir = im_dir + "moped1/";
		num_images = min(int(num_images), 8);
		seg_from_background = false;
		loadImagesStandard();
		max_depth_segmentation = 0.55f;
		max_radius_segmentation = 0.3f;
		plane_res_segmentation = 0.04f;
		computeDepthSegmentationFromPlane();
		break;
	}
}

void Mod3DfromRGBD::loadImageFromSequence(int im_num, bool first_image, int seq_ID)
{
	//Read Image
	//---------------------------------------------------------
	float depth_scale = 1.f / 5000.f;
	char aux[30];
	const float norm_factor = 1.f/255.f;
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	////Intensity
	//sprintf_s(aux, "i%d.png", i);
	//name = dir + aux;
	//cv::Mat im_i = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	//for (unsigned int u = 0; u < cols; u++)
	//	for (unsigned int v = 0; v < rows; v++)
	//		intensity[i - 1](v,u) = norm_factor*im_i.at<unsigned char>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);

	//Depth
	sprintf_s(aux, "d%d.png", im_num);
	string name = im_set_dir + aux;
	const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
	const float disp_u = 0.5f*(cols - 1);
	const float disp_v = 0.5f*(rows - 1);
	cv::Mat im_d = cv::imread(name, -1), depth_float;

	im_d.convertTo(depth_float, CV_32FC1, depth_scale);
	for (unsigned int u = 0; u < cols; u++)
		for (unsigned int v = 0; v<rows; v++)
		{
			const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);
			depth[0](v,u) = d;
			x_image[0](v,u) = (u - disp_u)*d*inv_fd;
			y_image[0](v,u) = (v - disp_v)*d*inv_fd;
			xyz_image[0].col(v+rows*u) << d, x_image[0](v,u), y_image[0](v,u);
		}

	//Color (if used)
	if (with_color)
	{
		sprintf_s(aux, "i%d.png", im_num);
		string name = im_set_dir + aux;
		//cv::Mat im_d = cv::imread(name, -1);
		cv::Mat im_int = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		const float norm_factor = 1.f/255.f;

		for (unsigned int u = 0; u<cols; u++)
			for (unsigned int v = 0; v<rows; v++)
				intensity[0](v,u) = norm_factor*im_int.at<unsigned char>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);		
	}


	//Segment the image
	//--------------------------------------------
	//computeDepthSegmentationGrowRegFromCenter(); //Me tracking 1
	if (seq_ID == 1) computeDepthSegmentationDepthRange(0.5, 2.5);  //Manu tracking 1
	else if (seq_ID == 2) computeDepthSegmentationDepthRange(0.5, 1.9f); //Mariano tracking 1


	//Set initial cam pose
	//------------------------------------------------------
	if (first_image)
	{
		cam_poses.resize(1);
		//cam_poses[0].setFromValues(-1.65f, -0.15f, -0.28f, utils::DEG2RAD(5.7f), utils::DEG2RAD(-20.f), utils::DEG2RAD(0.f)); //Me tracking 1
		//cam_poses[0].setFromValues(-1.98f, -0.25f, 0.05f, utils::DEG2RAD(5.7f), utils::DEG2RAD(-5.f), utils::DEG2RAD(0.f)); //Manu tracking 2
		if (seq_ID == 1) cam_poses[0].setFromValues(-2.10f, -0.35f, 0.1f, utils::DEG2RAD(5.7f), utils::DEG2RAD(-8.f), utils::DEG2RAD(0.f)); //Manu tracking 2
		else if (seq_ID == 2) cam_poses[0].setFromValues(1.55f, -0.27f, 0.22f, utils::DEG2RAD(169.f), utils::DEG2RAD(-5.f), utils::DEG2RAD(0.f)); //Mariano tracking 1

		//Get and store the initial transformation matrices
		CMatrixDouble44 aux_mat;
		cam_poses[0].getHomogeneousMatrix(aux_mat);
		cam_trans[0] = aux_mat.cast<float>();

		const Matrix3f rot_mat = cam_trans[0].block<3, 3>(0, 0).transpose();
		const Vector3f tra_vec = cam_trans[0].block<3, 1>(0, 3);

		cam_trans_inv[0].topLeftCorner<3, 3>() = rot_mat;
		cam_trans_inv[0].block<3, 1>(0, 3) = -rot_mat*tra_vec;
		cam_trans_inv[0].row(3) << 0.f, 0.f, 0.f, 1.f;
		cam_ini[0] = cam_trans_inv[0].topLeftCorner<4, 4>();
		cam_mfold[0].assign(0.f);
	}
}
	
void Mod3DfromRGBD::loadImagesStandard()
{	
	float depth_scale = 1.f / 5000.f;
	float depth_offset = 0.f;
	if (image_set == 5)
	{
		depth_scale = 1.f/200000.f;
		depth_offset = 1.f;
	}

	string name;

	char aux[30];
	const float norm_factor = 1.f/255.f;
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	//Load background depth
	if (seg_from_background)
	{
		sprintf_s(aux, "d_background.png");
		name = im_set_dir + aux;
		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		const float disp_v = 0.5f*(rows - 1);

		cv::Mat im_d = cv::imread(name, -1);
		cv::Mat depth_float;

		im_d.convertTo(depth_float, CV_32FC1, depth_scale);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);
				depth_background(v,u) = d;
			}
	}

	for (unsigned int i = 1; i <= num_images; i++)
	{
		////Intensity
		//sprintf_s(aux, "i%d.png", i);
		//name = dir + aux;

		//cv::Mat im_i = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		//for (unsigned int u = 0; u < cols; u++)
		//	for (unsigned int v = 0; v < rows; v++)
		//		intensity[i - 1](v,u) = norm_factor*im_i.at<unsigned char>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample);

		//Depth
		sprintf_s(aux, "d%d.png", i);
		name = im_set_dir + aux;
		const float inv_fd = 2.f*tan(0.5f*fovh_d) / float(cols);
		const float disp_u = 0.5f*(cols - 1);
		const float disp_v = 0.5f*(rows - 1);

		cv::Mat im_d = cv::imread(name, -1), depth_float;
		im_d.convertTo(depth_float, CV_32FC1, depth_scale);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v<rows; v++)
			{
				const float d = depth_float.at<float>(im_rows - 1 - v*downsample, im_cols - 1 - u*downsample) + depth_offset;
				depth[i-1](v, u) = d;
				x_image[i-1](v, u) = (u - disp_u)*d*inv_fd;
				y_image[i-1](v, u) = (v - disp_v)*d*inv_fd;
				xyz_image[i-1].col(v+rows*u) << d, x_image[i-1](v, u), y_image[i-1](v, u);
			}
	}
}

void Mod3DfromRGBD::saveSilhouetteToFile()
{
	//Load image with original resolution
	//-------------------------------------------------------
	string name;
	char aux[30];
	const unsigned int im_rows = rows*downsample, im_cols = cols*downsample;

	sprintf_s(aux, "d%d.png", 1);
	name = im_set_dir + aux;

	const float depth_scale = 1.f / 5000.f;
	cv::Mat im_d = cv::imread(name, -1), depth_float;
	im_d.convertTo(depth_float, CV_32FC1, depth_scale);
	ArrayXXf depth_big(im_rows,im_cols);
	Array<bool, Dynamic, Dynamic> valid_big(im_rows,im_cols), is_object_big(im_rows,im_cols);

	for (unsigned int u = 0; u<im_cols; u++)
		for (unsigned int v = 0; v<im_rows; v++)
			depth_big(v,u) = depth_float.at<float>(im_rows-1-v, im_cols-1-u);

	//Segment image
	//-------------------------------------------------------------
	valid_big.fill(true);
	is_object_big.fill(false);
		
	for (unsigned int u = 0; u < im_cols; u++)
		for (unsigned int v = 0; v < im_rows; v++)
		{
			if ((depth_big(v,u) > 0.f)&&(depth_big(v,u) < 2.f))
				is_object_big(v,u) = true;

			else if (depth_big(v,u) == 0.f)
				valid_big(v,u) = false;
		}

	//Project 3D model and superpose it with the data silhouette
	//------------------------------------------------------------------------
	const float disp_u = 0.5f*float(im_cols - 1);
	const float disp_v = 0.5f*float(im_rows - 1);

	const unsigned int nsamples_per_edge = 20;
	nsamples = square(nsamples_per_edge)*num_faces;

	//Create aux variables
	ArrayXf mx_samples(nsamples), my_samples(nsamples), mz_samples(nsamples);
	ArrayXf u1_samples(nsamples), u2_samples(nsamples);
	ArrayXi uface_samples(nsamples);

	const float fact = 1.f / float(nsamples_per_edge-1); //**********************
	for (unsigned int f = 0; f < num_faces; f++) 
		for (unsigned int u1 = 0; u1 < nsamples_per_edge; u1++)
			for (unsigned int u2 = 0; u2 < nsamples_per_edge; u2++)
			{
				const unsigned int ind = f*square(nsamples_per_edge) + u1*nsamples_per_edge + u2;
				u1_samples(ind) = float(u1)*fact;
				u2_samples(ind) = float(u2)*fact;
				uface_samples(ind) = f;
			}

	//Evaluate the surface
	Far::PatchMap patchmap(*patchTable);
	float pWeights[20], dsWeights[20], dtWeights[20];

	for (unsigned int s = 0; s < nsamples; s++)
	{
		Far::PatchTable::PatchHandle const * handle = patchmap.FindPatch(uface_samples(s), u1_samples(s), u2_samples(s)); assert(handle);
		patchTable->EvaluateBasis(*handle, u1_samples(s), u2_samples(s), pWeights, dsWeights, dtWeights);

		Far::ConstIndexArray cvs = patchTable->GetPatchVertices(*handle);
		LimitFrame eval; eval.Clear();
		for (int cv = 0; cv < cvs.size(); ++cv)
			eval.AddWithWeight(verts[cvs[cv]], pWeights[cv], dsWeights[cv], dtWeights[cv]);

		mx_samples(s) = eval.point[0];
		my_samples(s) = eval.point[1];
		mz_samples(s) = eval.point[2];
	}

	//Project the samples
	const Matrix4f &T_inv = cam_trans_inv[0];
	Array<bool, Dynamic, Dynamic> sil_mod(im_rows, im_cols); sil_mod.fill(false);
	const float fx_big = float(im_cols) / (2.f*tan(0.5f*fovh_d));
	const float fy_big = float(im_rows) / (2.f*tan(0.5f*fovv_d));

	for (unsigned int s = 0; s < nsamples; s++)
	{
		Vector4f m_sample = {mx_samples(s), my_samples(s), mz_samples(s), 1.f};
		Vector3f m_t_sample = T_inv.topRows(3)*m_sample;
		const float u_proj = fx_big*m_t_sample(1) / m_t_sample(0) + disp_u;
		const float v_proj = fy_big*m_t_sample(2) / m_t_sample(0) + disp_v;

		if ((u_proj > im_cols-1) || (u_proj < 0) || (v_proj > im_rows-1) || (v_proj < 0))
			continue;
		else
		{
			unsigned int u_up = ceilf(u_proj), u_down = floorf(u_proj);
			unsigned int v_up = ceilf(v_proj), v_down = floorf(v_proj);

			sil_mod(v_up, u_up) = true;
			sil_mod(v_up, u_down) = true;
			sil_mod(v_down, u_up) = true;
			sil_mod(v_down, u_down) = true;
		}
	}

	//Create color image with the silhouette
	//--------------------------------------------------
	cv::Mat im_sil(im_rows, im_cols, CV_8UC3);
    float r[3], g[3], b[3];
    for (unsigned int l=0; l<3; l++)
    {
        const float indx = float(l)/float(3-1);
        mrpt::utils::colormap(mrpt::utils::cmJET, indx, r[l], g[l], b[l]);
		r[l] *= 255;
		g[l] *= 255;
		b[l] *= 255;
    }

	const int scale = 30;
    for (unsigned int u=0; u<im_cols; u++)
        for (unsigned int v=0; v<im_rows; v++)
		{
			if (is_object_big(v,u))
				im_sil.at<cv::Vec3b>(im_rows-v,u) = cv::Vec3b(r[0]+scale*sil_mod(v,u), g[0]+scale*sil_mod(v,u), b[0]+scale*sil_mod(v,u));
			else if (valid_big(v,u))
				im_sil.at<cv::Vec3b>(im_rows-v,u) = cv::Vec3b(r[1]+scale*sil_mod(v,u), g[1]+scale*sil_mod(v,u), b[1]+scale*sil_mod(v,u));
			else
				im_sil.at<cv::Vec3b>(im_rows-v,u) = cv::Vec3b(r[2]+scale*sil_mod(v,u), g[2]+scale*sil_mod(v,u), b[2]+scale*sil_mod(v,u));
		}





	//Save image
	string name_im = "C:/Users/Mariano/Desktop/im_sil.png";
	cv::imwrite(name_im.c_str(), im_sil);

}

void Mod3DfromRGBD::createFileToSaveEnergy()
{
	//Open file
	string dir = f_folder + "energy_file.txt";
	f_energy.open(dir.c_str());
}

void Mod3DfromRGBD::saveCurrentEnergyInFile(bool insert_newline)
{
	float energy_d = 0.f, energy_b = 0.f, energy_r = 0.f;
	const float const_concave_back = 1.f - eps_rel; //(1.f - eps/tau);

	for (unsigned int i = 0; i < num_images; i++)
	{
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (is_object[i](v, u))
				{
					const float res = res_pos[i].col(v+rows*u).norm();
					energy_d += Kp*square(min(res, truncated_res));

					const float resn = res_normals[i].col(v+rows*u).norm();
					energy_d += Kn*n_weights[i](v + rows*u)*square(min(resn, truncated_resn));	
				}
				else if ((!solve_DT)&&(valid[i](v,u)))
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

		if (solve_DT)
		{
			for (unsigned int s = 0; s < nsamples; s++)
			{
				const int v = int(pixel_DT_v[i](s)), u = int(pixel_DT_u[i](s));
				energy_b += alpha*square(DT[i](v,u));
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

	//Save
	f_energy << energy_d << " " << energy_b << " " << energy_r;
	if (insert_newline)
		f_energy << endl;
}

void Mod3DfromRGBD::saveResultsCamExperiment()
{
	//Open file, find the first free file-name
	//----------------------------------------
	char	aux[100];
	int     nFile = 0;
	bool    free_name = false;
	string	dir;

	while (!free_name)
	{
		nFile++;
		sprintf(aux, "bt_cameras_%03u.txt", nFile );
		dir = f_folder + aux;
		free_name = !system::fileExists(dir);
	}
	
	//string dir = f_folder + "energy_file.txt";
	f_energy.open(dir.c_str());
	
	//Save results
	//---------------------------------------
	for (unsigned int k=0; k<6; k++)
	{
		for (unsigned int i=0; i<num_images; i++)
			f_energy << cam_pert[i](k) << " ";
	
		f_energy << endl;
	}

	for (unsigned int k=0; k<6; k++)
	{
		for (unsigned int i=0; i<num_images; i++)
			f_energy << cam_mfold[i](k) << " ";
	
		f_energy << endl;
	}

	saveCurrentEnergyInFile(false);

	for (unsigned int k=3; k<num_images; k++) 
		f_energy << " " << 0.f;

	f_energy.close();
}




