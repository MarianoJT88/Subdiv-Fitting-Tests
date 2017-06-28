/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2016, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#include <mrpt/utils.h>
#include <Eigen/dense>
#include <Eigen/sparse>
#include <stdio.h>


using namespace Eigen;
using namespace std;
using namespace mrpt;
typedef Triplet<float> Tri;

int main ( int argc, char** argv )
{
	utils::CTicTac clock;
	
	//Build a matrix
	const unsigned int rows = 50000, cols = 500;
	vector<Tri> j_elem;
	
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
		{
			float elem = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			if (elem > 0.8f)
				j_elem.push_back(Tri(v, u, elem));
		}
			

	//Fill triplets in col-major order to build col-major and row-major sparse matrices and compare times
	//---------------------------------------------------------------------------------------------------
	{
		SparseMatrix<float,ColMajor> smat_1; smat_1.resize(rows,cols);
		clock.Tic();
		smat_1.setFromTriplets(j_elem.begin(), j_elem.end());
		const float t1 = clock.Tac();

		SparseMatrix<float,RowMajor> smat_2; smat_2.resize(rows,cols);
		clock.Tic();
		smat_2.setFromTriplets(j_elem.begin(), j_elem.end());
		const float t2 = clock.Tac();

		printf("\n Time col-major = %f, time row-major = %f", t1, t2);
		//Surprisingly, if triplets are stored in col-major order then a row-major matrix can be built faster from them, and viceversa
	}

	//Check if the matrix format changes after transposition
	//--------------------------------------------------------------------------------------------------
	{
		SparseMatrix<float,ColMajor> smat_1; smat_1.resize(rows,cols);
		smat_1.setFromTriplets(j_elem.begin(), j_elem.end());
		printf("\n Is transposed matrix row-major = %d, is smat_1 row-major = %d", smat_1.transpose().IsRowMajor, smat_1.IsRowMajor);
		//Yes, it changes the matrix format
	}

	//Check if matrix products (+transposition) are slower when the matrix formats are different
	//---------------------------------------------------------------------------------------------------------------
	{
		SparseMatrix<float,ColMajor> smat_1; smat_1.resize(rows,cols);
		smat_1.setFromTriplets(j_elem.begin(), j_elem.end());
		SparseMatrix<float,RowMajor> smat_2; smat_2.resize(rows,cols);
		smat_2.setFromTriplets(j_elem.begin(), j_elem.end());
		printf("\n is smat1 compressed = %d, is smat2 compressed = %d", smat_1.isCompressed(), smat_2.isCompressed());

		clock.Tic();
		SparseMatrix<float> prod1 = smat_1.transpose()*smat_1;
		const float t1 = clock.Tac();

		clock.Tic();
		SparseMatrix<float> prod2 = smat_2.transpose()*smat_2;
		const float t2 = clock.Tac();

		printf("\n t1 = %f, t2 = %f", t1, t2);
		//t1 is higher than t2 but there isn't a huge difference...
	}
	

	









	//Checking the format of the transposed matrix
	
	//Test for the product JtJ - Comparing col-major and row-major


	return 0;
}



