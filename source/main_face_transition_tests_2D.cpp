// *************************************************
// Author: Mariano Jaimez Tarifa 
// Developed in MLP Microsoft Research Cambridge
// October 2015
//**************************************************


#include "face_transition_tests_2D.h"

int main()
{
	TestTransitions3D face_trans;

	//Create initial mesh
	face_trans.loadInitialMeshVertexVal3();

	//Create the 3D scene
	face_trans.initializeScene();

	//Show the mesh
	//face_trans.showMesh();

	//Evaluate the surface
	face_trans.createTopologyRefiner();
	face_trans.evaluateSubDivSurface();
	face_trans.showSubSurface2();
	face_trans.evaluateTransitions();

	system::os::getch();
	return 0;
}


//
//	//==============================================================================
//	//									Main operation
//	//==============================================================================
//
//	int pushed_key = 0;
//	bool anything_new = 0;
//    bool clean_sf = 0;
//    bool realtime = 0;
//	int stop = 0;
//    unsigned int up = odo.cols/2, vp = odo.rows/2;
//    CTicTac	main_clock, aux_clock; float pyrtime;
//    CImage image;
//
//	//Initialize variables
//    odo.createImagePyramidExperiments();
//	
//	while (!stop)
//	{	
//
//		if (odo.m_window.keyHit())
//			pushed_key = odo.m_window.getPushedKey();
//		else
//			pushed_key = 0;
//
//		switch (pushed_key) {
//			
//        //Capture a new frame
//		case  'n':
//            //odo.loadFrame();
//            main_clock.Tic();
//            odo.createImagePyramidExperiments();
//            pyrtime = main_clock.Tac();
//            printf("\n Runtime ima_pyr_exp = %f", pyrtime);
//
//            odo.computeOcclusionWeightsFast();
//            if (odo.kmeans_lowres)
//                odo.kMeansLowResolution();
//            else
//                odo.kMeansWithoutColor();
//
//            odo.createOptLabelImage();
//
//            anything_new = 1;
//            clean_sf = 1;
//			break;
//
//        //Compute the solution (CPU)
//        case 'a':
//            odo.mainIteration();
//            odo.createOptLabelImage();
//            anything_new = 1;
//            break;
//
//        //Turn on/off real-time estimation
//        case 's':
//            realtime = !realtime;
//            break;
//
////        //Increase the number of iterations
////        case '+':
////            odo.iter_irls += 1;
////            printf("\n Iterations irls: %d", odo.iter_irls);
////            break;
//
////        //Decrease the number of iterations
////        case '-':
////            if (odo.iter_irls > 0)
////                odo.iter_irls -= 1;
////            printf("\n Iterations irls: %d", odo.iter_irls);
////            break;
//
//        //Search a given point and plot information about it
//        case 'd':
//        {
//            int key_inner = 'h';
//            while (key_inner != '\r')
//            {
//                if (odo.m_window.keyHit())
//                    key_inner = odo.m_window.getPushedKey();
//
//                else
//                    key_inner = 0;
//
//                if ((key_inner == 314)&&(up < odo.cols-1)) //Left (':')
//                    up++;
//
//                else if ((key_inner == 316)&&(up > 0)) //Right ('<')
//                    up--;
//
//                else if ((key_inner == 315)&&(vp < odo.rows-1)) //Up (';')
//                    vp++;
//
//                else if ((key_inner == 317)&&(vp > 0)) //Down ('=')
//                    vp--;
//
//                //Represent it
//                odo.m_scene = odo.m_window.get3DSceneAndLock();
//                sel_point->clear();
//                sel_point->insertPoint(odo.depth_old[repr_level](vp,up), odo.xx_old[repr_level](vp,up), odo.yy_old[repr_level](vp,up));
//                odo.m_window.unlockAccess3DScene();
//                odo.m_window.repaint();
//                mrpt::system::sleep(1);
//            }
//
//            //Printf info
//            printf("\n depth_old = %f, depth = %f, depth_occ = %f", odo.depth_old[repr_level](vp,up), odo.depth[repr_level](vp,up), odo.depth_occ(vp,up));
//            fflush(stdout);
//            break;
//        }
//			
//		//Close the program
//		case 'p':
//			stop = 1;
//			break;
//		}
//	
//		if (anything_new)
//		{
//			odo.m_scene = odo.m_window.get3DSceneAndLock();
//
//			kin_points->clear();
//			for (unsigned int y=1; y<odo.cols-1; y++)
//				for (unsigned int z=1; z<odo.rows-1; z++)
//                    if (odo.depth[repr_level](z,y) > 0)
//                        kin_points->push_back(odo.depth[repr_level](z,y), odo.xx[repr_level](z,y), odo.yy[repr_level](z,y), 0.f, 1.f, 1.f);
//
//            image.setFromRGBMatrices(odo.olabels_image[0], odo.olabels_image[1], odo.olabels_image[2], true);
//            image.flipVertical();
//            vp_labels->setImageView(image);
////            depth_obj->assignImage(image);
//
//            //image.setFromRGBMatrices(odo.olabels_mod[0], odo.olabels_mod[1], odo.olabels_mod[2], true);
////            image.setFromMatrix(odo.color_wf);
////            color_gl->assignImage(image);
//			
//			odo.m_window.unlockAccess3DScene();
//			odo.m_window.repaint();
//
//			anything_new = 0;
//		}
//	}
//
//	return 0;
//}

