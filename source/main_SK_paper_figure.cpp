/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2016, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include <mrpt/gui.h>
#include <mrpt/opengl.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/system/threads.h>
#include <mrpt/math.h>
#include <Eigen/dense>

using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::maps;
using namespace std;


int main ( int argc, char** argv )
{

	//										Create scene
	//========================================================================================
	gui::CDisplayWindow3D window;
	opengl::COpenGLScenePtr	scene;
	mrpt::global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 1000000;
	window.setWindowTitle("RGB-D camera frame");
	window.resize(800,600);
	window.setPos(500,50);
	window.setCameraZoom(5);
	window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(5);
	window.getDefaultViewport()->setCustomBackgroundColor(utils::TColorf(1.f, 1.f, 1.f, 1.f));
	scene = window.get3DSceneAndLock();

	const float max_r = 1.2f, eps = 0.05f, tau = 1.f;
	const int ns = 201;

	opengl::CMeshPtr mesh = opengl::CMesh::Create();
	Eigen::MatrixXf sk(ns,ns), r(ns,ns), g(ns,ns), b(ns,ns); 
	r.fill(0.5f); g.fill(0.5f); b.fill(0.5f);
	utils::CImage image;

	for (int u=0; u<ns; u++)
		for (int v=0; v<ns; v++)
		{
			const float r1 = 2.f*max_r*float(u-(ns-1)/2)/float(ns-1);
			const float r2 = 2.f*max_r*float(v-(ns-1)/2)/float(ns-1);
			const float norm_r = sqrt(r1*r1 + r2*r2);

			if (norm_r < eps)
			{
				sk(v,u) = (1-eps/tau)*(1.f - square(norm_r)/(eps*tau));
				//color(v,u,1) = sk(v,u);
				//color(v,u,2) = 0;
				//color(v,u,3) = 0;
			}
			else if (norm_r < tau)
			{
				sk(v,u) = square(1 - norm_r/tau);
				//color(v,u,1) = 0.5 + sk(v,u);
				//color(v,u,2) = 0.5;
				//color(v,u,3) = 0.5;
			}
			else
			{
				sk(v,u) = 0.f;
				//color(v,u,1) = 0.5 + sk(v,u);
				//color(v,u,2) = 0.5;
				//color(v,u,3) = 0.5;
			}
		}

	mesh->setGridLimits(-max_r, max_r, -max_r, max_r);
	image.setFromRGBMatrices(r, g, b);
	mesh->assignImageAndZ(image, sk);
	scene->insert(mesh);


	math::CMatrixFloat22 cov;
	cov << square(tau), 0.f, 0.f, square(tau);
	opengl::CEllipsoidPtr disc1 = opengl::CEllipsoid::Create();
	disc1->setColor(0.f, 0.f, 0.f);
	disc1->setQuantiles(1.f);
	disc1->set2DsegmentsCount(200);
	disc1->setLineWidth(4.f);
	disc1->setCovMatrix(cov);
	disc1->setLocation(0.f, 0.f, 0.005f);
	scene->insert( disc1 );


	window.unlockAccess3DScene();
	window.repaint();

	system::os::getch();
	return 0;
}



