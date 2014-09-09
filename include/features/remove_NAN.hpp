#ifndef _REMOVE_NAN_FROM_POINT_CLOUD_HPP_
#define _REMOVE_NAN_FROM_POINT_CLOUD_HPP_

// STL
#include <vector>

// PCL
#include <pcl/point_cloud.h>					// pcl::PointCloud

namespace GPMap {

// PCL 1.7
template <typename PointT> 
void removeNaNNormalsFromPointCloud(const pcl::PointCloud<PointT>		&cloud_in, 
											   pcl::PointCloud<PointT>				&cloud_out,
												std::vector<int>						&index)
{
	// If the clouds are not the same, prepare the output
	if (&cloud_in != &cloud_out)
	{
		cloud_out.header = cloud_in.header;
		cloud_out.points.resize(cloud_in.points.size());
	}

	// Reserve enough space for the indices
	index.resize(cloud_in.points.size());
	size_t j = 0;
	for (size_t i = 0; i < cloud_in.points.size(); ++i)
	{
		if (!pcl_isfinite(cloud_in.points[i].normal_x) || 
			 !pcl_isfinite(cloud_in.points[i].normal_y) || 
			 !pcl_isfinite(cloud_in.points[i].normal_z))
			 continue;
		cloud_out.points[j] = cloud_in.points[i];
		index[j] = static_cast<int>(i);
		j++;
	}

	// Resize to the correct size
	if (j != cloud_in.points.size())
	{
		cloud_out.points.resize(j);
		index.resize(j);
	}

	cloud_out.height = 1;
	cloud_out.width  = static_cast<uint32_t>(j);
}

}
#endif