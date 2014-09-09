#ifndef _SURFACE_NORMAL_VECTOR_HPP_
#define _SURFACE_NORMAL_VECTOR_HPP_

// STL
#include <limits>		// std::numeric_limits<T>::epsilon()
#include <cmath>
#include <vector>

// PCL
#include <pcl/point_types.h>					// pcl::PointXYZ, pcl::Normal, pcl::PointNormal
#include <pcl/point_cloud.h>					// pcl::PointCloud
#include <pcl/point_cloud.h>					// pcl::PointCloud
#include <pcl/kdtree/kdtree_flann.h>		// pcl::search::KdTree
#ifdef _OPENMP
#include <pcl/features/normal_3d_omp.h>	// pcl::NormalEstimationOMP
#include <pcl/surface/mls_omp.h>				// pcl::MovingLeastSquaresOMP
#else
#include <pcl/features/normal_3d.h>			// pcl::NormalEstimation
#include <pcl/surface/mls.h>					// pcl::MovingLeastSquares
#endif

// GPMap
#include "util/data_types.hpp"				// PointXYZVList
#include "remove_NAN.hpp"

namespace GPMap {

template <typename NormalT>
inline float norm(const NormalT &normal)
{
	return sqrt(normal.normal_x * normal.normal_x
				 + normal.normal_y * normal.normal_y
				 + normal.normal_z * normal.normal_z);
}

}

namespace pcl {

/** @brief		Extention of pcl::isFinite 
  * @details	Refer to 
  *				template <typename PointT> inline bool pcl::isFinite (const PointT &pt)
  *				template <> inline bool isFinite<pcl::Normal> (const pcl::Normal &n)
  */
template <>
inline bool
isFinite<pcl::PointNormal>(const pcl::PointNormal &pt_n)
{
   return (pcl_isfinite(pt_n.x) && 
			  pcl_isfinite(pt_n.y) && 
			  pcl_isfinite(pt_n.z) &&
			  pcl_isfinite(pt_n.normal_x) && 
			  pcl_isfinite(pt_n.normal_y) && 
			  pcl_isfinite(pt_n.normal_z) &&
			  GPMap::norm(pt_n) > std::numeric_limits<float>::epsilon());
			  //pt_n.curvature > 0
}

}

namespace GPMap {

template <typename NormalT>
inline bool normalizeNormalVector(NormalT &normal)
{
	if(pcl::isFinite<NormalT>(normal))
	{
		// length
		const float length = norm(normal);
		assert(std::numeric_limits<float>::epsilon());

		// normalization
		normal.normal_x /= length;
		normal.normal_y /= length;
		normal.normal_z /= length;

		return true;
	}
	return false;
}

template <typename NormalT>
void normalizeNormalVectorCloud(typename pcl::PointCloud<NormalT>::Ptr &pNormalVectorCloud)
{
	if(!pNormalVectorCloud) return;

	// new normal vector cloud
	pcl::PointCloud<NormalT>::Ptr pTempNormalVectorCloud(new pcl::PointCloud<NormalT>());
	pTempNormalVectorCloud->points.resize(pNormalVectorCloud->points.size());

	// for each normal vector
	size_t j = 0;
	for(size_t i = 0; i < pNormalVectorCloud->points.size(); i++)
	{
		if(normalizeNormalVector(pNormalVectorCloud->points[i]))
		{
			pTempNormalVectorCloud->points[j++] = pNormalVectorCloud->points[i];
		}
	}

	// Resize to the correct size
	if (j != pNormalVectorCloud->points.size())
	{
		pTempNormalVectorCloud->points.resize(j);
	}

	pTempNormalVectorCloud->height = 1;
	pTempNormalVectorCloud->width  = static_cast<uint32_t>(j);

	// swap
	pNormalVectorCloud = pTempNormalVectorCloud;
}

// check if n is consistently oriented towards the viewpoint and flip otherwise
// angle between Psensor - Phit and Normal should be less than 90 degrees
// dot(Psensor - Phit, Normal) > 0
void flipSurfaceNormals(const pcl::PointXYZ							&sensorPosition,
								pcl::PointCloud<pcl::PointNormal>		&pointNormals)
{
	pcl::PointCloud<pcl::PointNormal>::iterator iter;
	for(iter	= pointNormals.begin(); iter != pointNormals.end(); iter++)
	{
		if((sensorPosition.x - iter->x) * iter->normal_x + 
		   (sensorPosition.y - iter->y) * iter->normal_y + 
		   (sensorPosition.z - iter->z) * iter->normal_z < 0)
		{
			iter->normal_x *= -1.f;
			iter->normal_y *= -1.f;
			iter->normal_z *= -1.f;
		}
	}
}

// pcl::Normal: float normal[3], curvature
pcl::PointCloud<pcl::Normal>::Ptr 
estimateSurfaceNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr		&pPointCloud,
							  const pcl::PointXYZ									&sensorPosition,
							  const bool												bSearchNearestNeighbor,
							  const float												param)
{
	// surface normal vectors
	// Create the normal estimation class, and pass the input dataset to it
#ifdef _OPENMP
	// Warning
	// PCL 1.6.0\3rdParty\Boost\include\boost/bind/bind.hpp(586): 
	// warning C4244: 'argument' : conversion from 'double' to 'int', possible loss of data
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
#else
	// Warning
	// PCL 1.6.0\3rdParty\Boost\include\boost/bind/bind.hpp(586): 
	// warning C4244: 'argument' : conversion from 'double' to 'int', possible loss of data
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
#endif

	// Set the input points
	ne.setInputCloud(pPointCloud);

	// Set the view point
	ne.setViewPoint(sensorPosition.x, sensorPosition.y, sensorPosition.z);

	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	// Use a FLANN-based KdTree to perform neighborhood searches
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	// Use all neighbors in a sphere of radius
	// Specify the size of the local neighborhood to use when computing the surface normals
	if(bSearchNearestNeighbor)		ne.setKSearch(param);
	else									ne.setRadiusSearch(param);

	// Set the search surface (i.e., the points that will be used when search for the input points?neighbors)
	ne.setSearchSurface(pPointCloud);

	// Compute the surface normals
	pcl::PointCloud<pcl::Normal>::Ptr pNormals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*pNormals);

	return pNormals;
}

// pcl::PointNormal: float x, y, znormal[3], curvature
pcl::PointCloud<pcl::PointNormal>::Ptr 
estimateSurfaceNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr		&pPointCloud, 
							  const pcl::PointXYZ									&sensorPosition,
							  const bool												bSearchNearestNeighbor,
							  const float												param,
							  std::vector<int>										&index)
{
	// estimate surface normals
	pcl::PointCloud<pcl::Normal>::Ptr pNormals = estimateSurfaceNormals(pPointCloud, sensorPosition, bSearchNearestNeighbor, param);

	// concatenate
	pcl::PointCloud<pcl::PointNormal>::Ptr pPointNormals(new pcl::PointCloud<pcl::PointNormal>());
	pcl::concatenateFields<pcl::PointXYZ, pcl::Normal, pcl::PointNormal>(*pPointCloud, *pNormals, *pPointNormals);

	// extract NaN
	removeNaNNormalsFromPointCloud<pcl::PointNormal>(*pPointNormals, *pPointNormals, index);

	return pPointNormals;
}

// pcl::PointNormal: float x, y, znormal[3], curvature
pcl::PointCloud<pcl::PointNormal>::Ptr 
smoothAndNormalEstimation(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr	&cloud,
								  const double													radius)
{
	// Smoothing and normal estimation based on polynomial reconstruction
	// Moving Least Squares (MLS) surface reconstruction method can be used to smooth and resample noisy data

	// Create a KD-Tree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());

	// Init object (second point type is for the normals, even if unused)
#ifdef _OPENMP
	pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointNormal> mls;
#else
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
#endif

	// Set parameters
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(kdtree);
	mls.setSearchRadius(radius);
	// void 	setPointDensity (int desired_num_points_in_radius);
	// void 	setDilationVoxelSize (float voxel_size)
	// void 	setDilationIterations (int iterations);
	// void 	setSqrGaussParam (double sqr_gauss_param)
	// void 	setPolynomialOrder (int order)
	// void 	setPolynomialFit (bool polynomial_fit)

	// Reconstruct
	// PCL v1.6
#if 1
	mls.setComputeNormals(true);

	// Output has the pcl::PointNormal type in order to store the normals calculated by MLS
	pcl::PointCloud<pcl::PointNormal>::Ptr mls_points(new pcl::PointCloud<pcl::PointNormal>());
	mls.process(*mls_points);
	return mls_points;
#else
	mls.reconstruct(*pPointCloud);

	// Output has the pcl::PointNormal type in order to store the normals calculated by MLS
	pPointNormals = mls.getOutputNormals();
	//mls.setOutputNormals(mls_points);
#endif
}


class ByNearestNeighbors		{};
class ByMovingLeastSquares		{};

template <typename EstimateMethod>
void estimateSurfaceNormals(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>	&pointClouds, 
									 const PointXYZVList													&sensorPositionList,
									 const bool																fSearchRadius, // SearchRadius or SearchK
									 const float															param,			// radius or k
									 std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>		&pointNormalCloudPtrList);

template <>
void estimateSurfaceNormals<ByNearestNeighbors>(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>		&pointClouds,
																const PointXYZVList												&sensorPositionList,
																const bool															fSearchNearestK, // SearchRadius or SearchNearestK
																const float															param,			// radius or k
																std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>			&pointNormalCloudPtrList)
{
	// resize
	pointNormalCloudPtrList.resize(pointClouds.size());

	// for each point cloud
	for(size_t i = 0; i < pointClouds.size(); i++)
	{
		std::cout << "Estimate surface normals: " << i << " ... ";
		pointNormalCloudPtrList[i] = estimateSurfaceNormals(pointClouds[i], 
																		 sensorPositionList[i], 
																		 fSearchNearestK, 
																		 param, 
																		 std::vector<int>());
		std::cout << pointNormalCloudPtrList[i]->size() << " normals." << std::endl;
	}
}

template <>
void estimateSurfaceNormals<ByMovingLeastSquares>(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>	&pointCloudPtrList,
																  const PointXYZVList												&sensorPositionList,
																  const bool															fSearchNearestK, // SearchRadius or SearchNearestK
																  const float															param,			// radius or k
																  std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>		&pointNormalCloudPtrList)
{
	// resize
	pointNormalCloudPtrList.resize(pointCloudPtrList.size());

	// for each point cloud
	for(size_t i = 0; i < pointCloudPtrList.size(); i++)
	{
		std::cout << "Estimate surface normals: " << i << " ... ";

		// estimate the surface normals
		pointNormalCloudPtrList[i] = smoothAndNormalEstimation(pointCloudPtrList[i], param);

		// normalize
		normalizeNormalVectorCloud<pcl::PointNormal>(pointNormalCloudPtrList[i]);

		// flip toward the sensor position
		flipSurfaceNormals(sensorPositionList[i], *(pointNormalCloudPtrList[i]));

		std::cout << pointNormalCloudPtrList[i]->size() << " normals." << std::endl;
	}
}


/** @brief Compute a unit vector from the hit point to the sensor position, P_O */
template <typename PointT>
inline void unitBackRayVector(const PointT				&hitPoint, 
										const pcl::PointXYZ		&sensorPosition,
										pcl::PointNormal			&P_O)
{
	// check finite
	assert(pcl::isFinite<PointT>(hitPoint) && pcl::isFinite<pcl::PointXYZ>(sensorPosition));

	// hit point
	P_O.x = hitPoint.x;
	P_O.y = hitPoint.y;
	P_O.z = hitPoint.z;

	// vector from a hit point to the origin (sensorPosition)
	P_O.normal_x = sensorPosition.x - hitPoint.x;
	P_O.normal_y = sensorPosition.y - hitPoint.y;
	P_O.normal_z = sensorPosition.z - hitPoint.z;
	const float length = sqrt(P_O.normal_x*P_O.normal_x + P_O.normal_y*P_O.normal_y + P_O.normal_z*P_O.normal_z);

	// check length
	assert(length > std::numeric_limits<float>::epsilon());

	// normalize
	P_O.normal_x /= length;
	P_O.normal_y /= length;
	P_O.normal_z /= length;

	// curvature
	P_O.curvature = -1.f;
}

/** @brief Compute unit vectors from the hit points to the sensor position, P_O */
pcl::PointCloud<pcl::PointNormal>::Ptr unitRayBackVectors(const pcl::PointCloud<pcl::PointXYZ>		&pointCloud,
																			 const pcl::PointXYZ								&sensorPosition)
{
	// memory allocation
	pcl::PointCloud<pcl::PointNormal>::Ptr pPointNormalCloud(new pcl::PointCloud<pcl::PointNormal>());
	pPointNormalCloud->resize(pointCloud.size());

	// unit vectors
	for(size_t i = 0; i < pointCloud.points.size(); i++)
		unitBackRayVector(pointCloud.points[i], sensorPosition, pPointNormalCloud->points[i]);

	return pPointNormalCloud;
}

/** @brief Compute unit vectors from the hit points to the sensor position, P_O */
void unitRayBackVectors(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>	&pointCloudPtrList,
								const PointXYZVList													&sensorPositionList,
								std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>		&pointNormalCloudPtrList)
{
	// resize
	pointNormalCloudPtrList.resize(pointCloudPtrList.size());

	// for each point cloud
	for(size_t i = 0; i < pointCloudPtrList.size(); i++)
	{
		std::cout << "Calculate unit ray back vectors: " << i << " ... ";

		// calculate unit ray back vectors
		pointNormalCloudPtrList[i] = unitRayBackVectors(*pointCloudPtrList[i], sensorPositionList[i]);

		std::cout << pointNormalCloudPtrList[i]->size() << " normals." << std::endl;
	}
}

}
#endif