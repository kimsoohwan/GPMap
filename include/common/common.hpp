#ifndef _COMMON_HPP_
#define _COMMON_HPP_

// STL
#include <algorithm>		// min, max

// PCL
#include <pcl/point_types.h>		// pcl::PointXYZ, pcl::Normal, pcl::PointNormal
#include <pcl/point_cloud.h>		// pcl::PointCloud
#include <pcl/common/common.h>	// pcl::getMinMax3D

namespace GPMap {

/** @brief		Find the minimum of two pcl::PointXYZ */
template <typename PointT1, typename PointT2>
inline pcl::PointXYZ minPointXYZ(const PointT1 &p1, const PointT2 &p2)
{
	return pcl::PointXYZ(min<float>(p1.x, p2.x), min<float>(p1.y, p2.y), min<float>(p1.z, p2.z));
}

/** @brief		Find the maximum of two pcl::PointXYZ */
template <typename PointT1, typename PointT2>
inline pcl::PointXYZ maxPointXYZ(const PointT1 &p1, const PointT2 &p2)
{
	return pcl::PointXYZ(max<float>(p1.x, p2.x), max<float>(p1.y, p2.y), max<float>(p1.z, p2.z));
}

template <typename PointT>
void getMinMaxPointXYZ(const pcl::PointCloud<PointT>	&pointCloud,
							  pcl::PointXYZ						&min_pt, 
							  pcl::PointXYZ						&max_pt)
{
	// min/max points with the same point type
	PointT min_pt_temp, max_pt_temp;
	
	// get min max
	pcl::getMinMax3D(pointCloud, min_pt_temp, max_pt_temp);

	// set
	min_pt.x = min_pt_temp.x;
	min_pt.y = min_pt_temp.y;
	min_pt.z = min_pt_temp.z;

	max_pt.x = max_pt_temp.x;
	max_pt.y = max_pt_temp.y;
	max_pt.z = max_pt_temp.z;
}

template <typename PointT>
void getMinMaxPointXYZ(const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&pPointClouds,
							  pcl::PointXYZ &min_pt, pcl::PointXYZ &max_pt)
{
	pcl::PointXYZ min_pt_temp, max_pt_temp;

	// for each point cloud
	for(size_t i = 0; i < pPointClouds.size(); i++)
	{
		// get min max
		getMinMaxPointXYZ<PointT>(*pPointClouds[i], min_pt_temp, max_pt_temp);

		// compare
		if(i == 0)
		{
			min_pt = min_pt_temp;
			max_pt = max_pt_temp;
		}
		else
		{
			min_pt = minPointXYZ(min_pt, min_pt_temp);
			max_pt = maxPointXYZ(max_pt, max_pt_temp);
		}
	}
}

/** @brief Combine two point cloud list */
template <typename PointT>
void combinePointCloud(const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&pointCloudList1,
							  const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&pointCloudList2,
							  std::vector<typename pcl::PointCloud<PointT>::Ptr>				&pointCloudList)
{
	// size check
	assert(pointCloudList1.size() == pointCloudList2.size());
	pointCloudList.resize(pointCloudList1.size());

	// for each hit/empty point cloud, add them to the output list
	for(size_t i = 0; i < pointCloudList1.size(); i++)
	{
		pointCloudList[i].reset(new pcl::PointCloud<PointT>());
		*pointCloudList[i] += *pointCloudList1[i];
		*pointCloudList[i] += *pointCloudList2[i];
	}
}

}

#endif