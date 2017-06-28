// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************

#include "3D_model_fitting.h"


void Mod3DfromRGBD::computeDepthSegmentationDepthRange(float min_d, float max_d)
{
	
	//Simple case: consider points below a certain depth threshold 
	for (unsigned int i = 0; i < num_images; i++)
	{
		valid[i].fill(true);
		is_object[i].fill(false);
		
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
			{
				if ((depth[i](v,u) > min_d)&&(depth[i](v,u) < max_d))
					is_object[i](v,u) = true;

				if (depth[i](v,u) == 0.f)
					valid[i](v,u) = false;
			}
	}

	//Introduce outliers
	//for (unsigned int i = 0; i < num_images; i++)
	//	for (unsigned int u = 0; u < cols; u++)
	//		for (unsigned int v = 0; v < rows; v++)
	//			if (depth[i](v,u) > 2.f)
	//			{
	//				const float rand_num = rand()%2000;
	//				if (rand_num == 0)
	//					is_object[i](v,u) = true;
	//			}
}

void Mod3DfromRGBD::computeDepthSegmentationClosestObject()
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

void Mod3DfromRGBD::computeDepthSegmentationGrowRegFromCenter()
{
	vector<Array2i> buffer;
	Array<bool,Dynamic,Dynamic> checked(rows,cols);
	const float dist_thres = square(0.005f*downsample); //It is actually a threshold for the distance square(^2)

	//Grow segmentation from the central pixel
	for (unsigned int i = 0; i < num_images; i++)
	{
		checked.fill(false);
		valid[i].fill(true);
		
		//Find the closest point
		Array2i min_coords; min_coords.fill(0);
		for (unsigned int u = 0; u < cols; u++)
			for (unsigned int v = 0; v < rows; v++)
				if (depth[i](v, u) == 0.f)
					valid[i](v, u) = false;

		//Expand from the closest point
		min_coords(0) = rows/2;
		min_coords(1) = cols/2;
		buffer.push_back(min_coords);
		while (!buffer.empty())
		{
			const int v = buffer.back()(0);
			const int u = buffer.back()(1);
			buffer.pop_back();
			checked(v,u) = true;

			for (int k = -1; k < 2; k++)
				for (int l = -1; l < 2; l++)
				{
					Array2i ind; ind(0) = v + l; ind(1) = u + k;

					if ((ind(0) < 0) || (ind(1) < 0) || (ind(0) >= rows) || (ind(1) >= cols))
						continue;

					else if (checked(ind(0), ind(1)))
						continue;

					const float dist = square(depth[i](v,u) - depth[i](ind(0), ind(1)))
									+ square(x_image[i](v,u) - x_image[i](ind(0), ind(1)))
									+ square(y_image[i](v,u) - y_image[i](ind(0), ind(1)));

					if (dist < dist_thres)
						buffer.push_back(ind);
				}
		}

		//Copy result to the segmentation
		is_object[i].swap(checked);
	}
}


void Mod3DfromRGBD::computeDepthSegmentationFromPlane()
{
	vector<Vector3f> alpha; alpha.resize(num_images);

	//RANSAC
	//E(a) = sum((a1*x + a2*y + a3*z - 1)^2)
	
	for (unsigned int i = 0; i < num_images; i++)
	{
		std::vector<float> depth_sorted;
		
		//First find the number of pixels used
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) < max_depth_segmentation))
					depth_sorted.push_back(depth[i](v,u));

		std::sort(depth_sorted.begin(), depth_sorted.end());

		const float percent_furthest = 0.1f;
		const float depth_threshold = depth_sorted[ceil((1.f - percent_furthest)*depth_sorted.size())];

		vector<Vector3f> inliers;
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if ((depth[i](v,u) > depth_threshold)&&(depth[i](v,u) < max_depth_segmentation))
				{
					Vector3f point; point <<  depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					inliers.push_back(point);
				}

		const unsigned int num_iter_ransac = 5;

		for (unsigned int r=0; r<num_iter_ransac; r++)
		{
			//Create matrices to solve the system
			unsigned int num_points = inliers.size();
			MatrixXf A(num_points, 3);
			VectorXf b(num_points); b.fill(1.f);
			VectorXf residuals(num_points); residuals.fill(1.f);

			for (unsigned int k=0; k<num_points; k++)
				A.row(k) = inliers[k].transpose();

			MatrixXf AtA, AtB;
			AtA.multiply_AtA(A);
			AtB.multiply_AtB(A,b);
			alpha[i] = AtA.ldlt().solve(AtB);
			residuals = A*alpha[i]-b;

			//Find new inliers
			inliers.clear();
			for (unsigned int u=0; u<cols; u++)
				for (unsigned int v=0; v<rows; v++)
					if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) < max_depth_segmentation))
					{
						Vector3f point; point <<  depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
						const float res = abs((alpha[i].transpose()*point).value() - 1.f);
						if (res < plane_res_segmentation)
							inliers.push_back(point);
					}
		}


		//Check which points belong to the plane
		is_object[i].fill(true);
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if ((depth[i](v,u) > 0.f)&&(depth[i](v,u) <= max_depth_segmentation))
				{
					Matrix<float, 1, 3> r; r<< depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					if (abs((r*alpha[i]).value() - 1.f) < plane_res_segmentation)
						is_object[i](v,u) = false;
				}
				else if ((depth[i](v,u) == 0.f)||(depth[i](v,u) > max_depth_segmentation))
					is_object[i](v,u) = false;
			}

		//Filter the mask to reject false positives
		//First, reject those points far from the center of gravity of the object
		Array3f cent_of_grav; cent_of_grav.fill(0.f);
		unsigned int num_points = 0;
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
				{
					Array3f point; point << depth[i](v,u), x_image[i](v,u), y_image[i](v,u);
					cent_of_grav += point;
					num_points++;
				}
		cent_of_grav /= num_points;

		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (is_object[i](v,u))
					if (sqrtf(square(depth[i](v,u) - cent_of_grav(0)) + square(x_image[i](v,u) - cent_of_grav(1)) +  square(y_image[i](v,u) - cent_of_grav(2))) > max_radius_segmentation)
						is_object[i](v,u) = false;

		//Build invalid mask
		valid[i].fill(true);
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
				if (depth[i](v,u) == 0.f)
					valid[i](v,u) = false;
	}

	////I check whether it works or not...
	//global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	//window.resize(1000, 900);
	//window.setPos(900, 0);
	//window.setCameraZoom(3);
	//window.setCameraAzimuthDeg(0);
	//window.setCameraElevationDeg(45);
	//window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	//scene = window.get3DSceneAndLock();

	//// Lights:
	//scene->getViewport()->setNumberOfLights(2);
	//mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	//light0.light_ID = 0;
	////light0.setPosition(2.5f,0,0.f,1.f);
	//light0.setDirection(0.f, 0.f, -1.f);

	//mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	//light1.light_ID = 1;
	//light1.setPosition(0.0f, 0, 0.f, 1.f);
	//light1.setDirection(0.f, 1.f, 0.f);

	////Reference
	//opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	//reference->setScale(0.2f);
	//scene->insert(reference);

	////Points
	//CPose3D pose_aux = CPose3D(0,0,0,0,0,0);
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//	points->setPointSize(3.f);
	//	points->enablePointSmooth(true);
	//	points->setPose(pose_aux);
	//	scene->insert(points);

	//	//Insert points
	//	//float r, g, b;
	//	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//	for (unsigned int v = 0; v < rows; v++)
	//		for (unsigned int u = 0; u < cols; u++)
	//		{
	//			if (is_object[i](v, u))
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 1.f, 0.f, 0.f);
	//			else
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 0.1f, 0.1f, 0.1f);
	//		}

	//	//Sample the plane
	//	opengl::CPointCloudColouredPtr plane = opengl::CPointCloudColoured::Create();
	//	plane->setPointSize(3.f);
	//	plane->enablePointSmooth(true);
	//	plane->setPose(pose_aux);
	//	scene->insert(plane);
	//	for (unsigned int s = 0; s < 20; s++)
	//		for (unsigned int t = 0; t < 20; t++)
	//		{
	//			const float x = 0.f + s*0.1f;
	//			const float y = -0.5f + t*0.05f;
	//			const float z = (1.f - alpha[i](0)*x - alpha[i](1)*y)/alpha[i](2);
	//			plane->push_back(x, y, z, 0.f , 1.f, 0.f);
	//		}

	//	pose_aux.y_incr(1.5f);
	//}

	//window.unlockAccess3DScene();
	//window.repaint();
	//system::os::getch();
}

void Mod3DfromRGBD::computeDepthSegmentationFromBackground()
{
	const float max_depth_dif = 0.03f;

	//For every image
	for (unsigned int i = 0; i < num_images; i++)
	{
		is_object[i].fill(false);
		valid[i].fill(true);

		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (abs(depth[i](v,u) - depth_background(v,u)) > max_depth_dif*square(depth[i](v,u)))
					is_object[i](v,u) = true;

				if (depth[i](v,u) == 0.f)
				{
					valid[i](v,u) = false;
					is_object[i](v,u) = false;
				}
			}
	}

	////I check whether it works or not...
	//global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 50000000;
	//window.resize(1000, 900);
	//window.setPos(900, 0);
	//window.setCameraZoom(3);
	//window.setCameraAzimuthDeg(0);
	//window.setCameraElevationDeg(45);
	//window.setCameraPointingToPoint(0.f, 0.f, 0.f);

	//scene = window.get3DSceneAndLock();

	//// Lights:
	//scene->getViewport()->setNumberOfLights(2);
	//mrpt::opengl::CLight & light0 = scene->getViewport()->getLight(0);
	//light0.light_ID = 0;
	////light0.setPosition(2.5f,0,0.f,1.f);
	//light0.setDirection(0.f, 0.f, -1.f);

	//mrpt::opengl::CLight & light1 = scene->getViewport()->getLight(1);
	//light1.light_ID = 1;
	//light1.setPosition(0.0f, 0, 0.f, 1.f);
	//light1.setDirection(0.f, 1.f, 0.f);

	////Reference
	//opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	//reference->setScale(0.2f);
	//scene->insert(reference);

	////Points
	//CPose3D pose_aux = CPose3D(0,0,0,0,0,0);
	//for (unsigned int i = 0; i < num_images; i++)
	//{
	//	opengl::CPointCloudColouredPtr points = opengl::CPointCloudColoured::Create();
	//	points->setPointSize(3.f);
	//	points->enablePointSmooth(true);
	//	points->setPose(pose_aux);
	//	scene->insert(points);

	//	//Insert points
	//	//float r, g, b;
	//	//utils::colormap(mrpt::utils::cmJET, float(i) / float(num_images), r, g, b);
	//	for (unsigned int v = 0; v < rows; v++)
	//		for (unsigned int u = 0; u < cols; u++)
	//		{
	//			if (is_object[i](v, u))
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 1.f, 0.f, 0.f);
	//			else
	//				points->push_back(depth[i](v, u), x_image[i](v, u), y_image[i](v, u), 0.1f, 0.1f, 0.1f);
	//		}

	//	pose_aux.y_incr(1.5f);
	//}

	//window.unlockAccess3DScene();
	//window.repaint();
	//system::os::getch();
}

void Mod3DfromRGBD::computeDepthSegmentationFromBoundingBox()
{
	//Open volume file
	std::ifstream	f_volume;
	string filename = im_set_dir;
	filename.append("volume.txt");

	f_volume.open(filename.c_str());

	if (f_volume.fail())
		throw std::runtime_error("\nError finding the volume file.");

	//Get the edge of the bounding box
	float cube_edge;
	f_volume >> cube_edge;

	f_volume.close();
	
	for (unsigned int i = 0; i < num_images; i++)
		for (unsigned int u=0; u<cols; u++)
			for (unsigned int v=0; v<rows; v++)
			{
				if (depth[i](v,u) < 0.01f)
				{
					valid[i](v,u) = false;
					is_object[i](v,u) = false;
				}
				else
				{
					valid[i](v,u) = true;

					//Transform the points to the global coordinate system
					const float x_t = trajectory[i](0,0)*depth[i](v,u) + trajectory[i](0,1)*x_image[i](v,u) + trajectory[i](0,2)*y_image[i](v,u) + trajectory[i](0,3);
					const float y_t = trajectory[i](1,0)*depth[i](v,u) + trajectory[i](1,1)*x_image[i](v,u) + trajectory[i](1,2)*y_image[i](v,u) + trajectory[i](1,3);
					const float z_t = trajectory[i](2,0)*depth[i](v,u) + trajectory[i](2,1)*x_image[i](v,u) + trajectory[i](2,2)*y_image[i](v,u) + trajectory[i](2,3);

					if ((abs(x_t) < 0.5f*cube_edge)&&(abs(y_t) < 0.5f*cube_edge)&&(abs(z_t) < 0.5f*cube_edge))
						is_object[i](v,u) = true;
					else
						is_object[i](v,u) = false;			
				}					
			}
}

void Mod3DfromRGBD::saveSegmentationToFile(unsigned int image)
{
	string dir = "C:/Users/jaimez/programs/GitHub/OpenSubdiv-Model-Fitting/data/";
	string name;
	char aux[30];
	
	//Save segmentation
	sprintf_s(aux, "seg_image.png");
	name = dir + aux;
	cv::Mat seg_cv(rows, cols, CV_16U);
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
			if (is_object[image](v,u) == true)
				seg_cv.at<unsigned short>(rows-1-v,u) = 0;
			else if (valid[image](v,u) == true)
				seg_cv.at<unsigned short>(rows-1-v,u) = 10000;
			else
				seg_cv.at<unsigned short>(rows-1-v,u) = 20000;
		}

	cv::imwrite(name.c_str(), seg_cv);
}



