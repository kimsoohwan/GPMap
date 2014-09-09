#ifndef _GPMAP_TRAINING_DATA_HPP_
#define _GPMAP_TRAINING_DATA_HPP_

// PCL
#include <pcl/point_types.h>		// pcl::PointXYZ, pcl::Normal, pcl::PointNormal
#include <pcl/point_cloud.h>		// pcl::PointCloud

// GPMap
#include "util/data_types.hpp"			// PointXYZVList, Matrix, MatrixPtr, Vector, VectorPtr
#include "features/surface_normal.hpp"	// norm, pcl::isFinite<pcl::PointNormal>

namespace GPMap {

/** @brief	Count the number of finite points */
void numFinitePoints(const pcl::PointCloud<pcl::PointNormal>	&pointNormalCloud,
							const Indices										&indices,
							size_t												&Nf,	// number of function observation
							size_t												&Nd)	// number of derivative observation
{
	// reset
	Nf = 0;
	Nd = 0;

	for(Indices::const_iterator iter = indices.begin(); iter != indices.end(); ++iter)
	{
		// index
		assert(*iter >= 0 && *iter < static_cast<int>(pointNormalCloud.points.size()));

		// point normal
		const pcl::PointNormal &pointNormal = pointNormalCloud.points[*iter];

		// check finite
		if(!pcl::isFinite<pcl::PointNormal>(pointNormal)) continue;
		
		// count
		if(pointNormal.curvature < 0)		Nf++;	// function observation
		else										Nd++;	// derivative observation
	}
}

/** @brief	Generate training data from a surface normal cloud
  * @param[in] pPointNormalCloud		Function/derivative/all observations in pcl::PointCloud<pcl::PointNormal>
  *											Note that function observations are alse represented as point normals.
  *											Their x/y/z are hit points.
  *											Their normal_x/y/z are unit ray back vectors from hit points to sensor positions.
  *											Their curvature = -1, which can be used to check whether it is a function observation or a derivative one.
  * @Todo	Optimization or using getMatrixXfMap()
  */
void generateTrainingData(const PointNormalCloudConstPtr		&pPointNormalCloud,
								  const Indices							&indices,
								  const float								gap,
								  MatrixPtr &pX, MatrixPtr &pXd, VectorPtr &pYYd)
{
	// K: NN by NN, NN = N + Nd*D
	// 
	// for example, when D = 3
	//                  | f(x) | df(xd)/dx_1, df(xd)/dx_2, df(xd)/dx_3
	//                  |  N   |     Nd            Nd           Nd
	// ---------------------------------------------------------------
	// f(x)        : N  |  FF  |     FD1,         FD2,         FD3
	// df(xd)/dx_1 : Nd |   -  |    D1D1,        D1D2,        D1D3  
	// df(xd)/dx_2 : Nd |   -  |      - ,        D2D2,        D2D3  
	// df(xd)/dx_3 : Nd |   -  |      - ,          - ,        D3D3

	assert(gap >= 0);

	// some constants
	//const bool fUseFunctionObservations(gap > 0.f);	// function/derivative observations
	const size_t D = 3;	// number of dimensions
	size_t Nf;				// number of function observations
	size_t Nd;				// number of derivative observation
	numFinitePoints(*pPointNormalCloud, indices, Nf, Nd);
	const size_t N = 2*Nf + Nd;	// hit/empty points from function observations and virtual hit points from derivative observations

	// memory allocation
	pX.reset(new Matrix(N, 3));			// hit/empty points from function observations and virtual hit points from derivative observations
	pXd.reset(new Matrix(Nd, 3));			// surface normals from derivative observations
	pYYd.reset(new Vector(N + Nd*D));	// all
		
	// assignment
	int i = 0;
	for(Indices::const_iterator iter = indices.begin(); iter != indices.end(); ++iter)
	{
		// index
		assert(*iter >= 0 && *iter < static_cast<int>(pPointNormalCloud->points.size()));

		// point normal
		const pcl::PointNormal &pointNormal(pPointNormalCloud->points[*iter]);

		// check finite
		if(!pcl::isFinite<pcl::PointNormal>(pointNormal)) continue;

		// function observation
		if(pointNormal.curvature < 0)
		{
			// hit point: Nf
			(*pX)(i, 0)		= pointNormal.x;
			(*pX)(i, 1)		= pointNormal.y;
			(*pX)(i, 2)		= pointNormal.z;
			(*pYYd)(i)		= 0.f;

			// empty point: Nf
			(*pX)(Nf + i, 0)	= pointNormal.x + gap * pointNormal.normal_x;
			(*pX)(Nf + i, 1)	= pointNormal.y + gap * pointNormal.normal_y;
			(*pX)(Nf + i, 2)	= pointNormal.z + gap * pointNormal.normal_z;
			(*pYYd)(Nf+i)	= -gap; // outside: negative distance
			//(*pYYd)(Nf+i)		= gap; // ???
		}

		// derivative observation
		else
		{
			// virtual hit point: Nd
			(*pX)(2*Nf + i, 0)		= pointNormal.x;
			(*pX)(2*Nf + i, 1)		= pointNormal.y;
			(*pX)(2*Nf + i, 2)		= pointNormal.z;
			(*pYYd)(2*Nf+i)			= 0.f;

			// surface normal
			(*pXd)(i, 0)			= pointNormal.x;
			(*pXd)(i, 1)			= pointNormal.y;
			(*pXd)(i, 2)			= pointNormal.z;
			(*pYYd)(2*Nf+Nd + i)			= pointNormal.normal_x;
			(*pYYd)(2*Nf+Nd + Nd+i)		= pointNormal.normal_y;
			(*pYYd)(2*Nf+Nd + 2*Nd+i)	= pointNormal.normal_z;
		}

		// next index
		i++;
	}
}


}

#endif