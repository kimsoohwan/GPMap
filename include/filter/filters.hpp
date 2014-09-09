#ifndef FILTERS_FOR_POINT_CLOUDS_HPP
#define FILTERS_FOR_POINT_CLOUDS_HPP

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/normal_refinement.h>
//#include <pcl/filters/impl/normal_refinement.hpp>

namespace GPMap {

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr 
statisticalOutlierRemoval(typename pcl::PointCloud<PointT>::ConstPtr cloud,
							const int K,
							const double stdMultiplier,
							const bool bReturnOutliers = false,
							typename pcl::PointCloud<PointT>::Ptr outliers = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>()))
{	
	// calculate average distances for all points
	// estimate the mean and the variance of the average distances
	// threshold = mean + std_mul_ * stddev;

	// build the filter
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud (cloud);
	sor.setMeanK(K);						// number of neighbors
	sor.setStddevMulThresh(stdMultiplier);	// standard deviation multiplier

	// apply filter
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
	sor.filter (*cloud_filtered);

	// filtered point cloud
	if(bReturnOutliers)
	{
		sor.setNegative (true);
		sor.filter (*outliers);
	}

	return cloud_filtered;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr 
radiusOutlierRemoval(typename pcl::PointCloud<PointT>::ConstPtr cloud,
						const double radius,
						const int minNeighborsInRadius,
						const bool bReturnOutliers = false,
						typename pcl::PointCloud<PointT>::Ptr outliers = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>()))
{
	// build the filter
	pcl::RadiusOutlierRemoval<PointT> outrem;
	outrem.setInputCloud(cloud);
	outrem.setRadiusSearch(radius);
	outrem.setMinNeighborsInRadius(minNeighborsInRadius);

	// apply filter
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
	outrem.filter (*cloud_filtered);

	// filtered point cloud
	if(bReturnOutliers)
	{
		outrem.setNegative (true);
		outrem.filter (*outliers);
	}

	return cloud_filtered;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
cropBox(const typename pcl::PointCloud<PointT>::ConstPtr cloud,
		   const Eigen::Vector4f &minPoint, const Eigen::Vector4f &maxPoint)
{
	// build the filter
	pcl::CropBox<PointT> cropFilter;
	cropFilter.setInputCloud (cloud); 
	cropFilter.setMin(minPoint); 
	cropFilter.setMax(maxPoint); 
	//cropFilter.setTranslation(boxTranslatation); 
	//cropFilter.setRotation(boxRotation); 

	// apply filter
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
	cropFilter.filter (*cloud_filtered);

	return cloud_filtered;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr 
rangeRemoval(typename pcl::PointCloud<PointT>::ConstPtr cloud,
				const PointT &minPoint, const PointT &maxPoint,
				const bool bRemoveFar = true)
{
	if(bRemoveFar)
	{
		// build the condition
		pcl::ConditionAnd<PointT>::Ptr range_cond (new pcl::ConditionAnd<PointT> ());

		// condition which a given point must satisfy for it to remain in the point cloud
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, minPoint.x)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, minPoint.y)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, minPoint.z)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, maxPoint.x)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, maxPoint.y)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, maxPoint.z)));

		// build the filter
		pcl::ConditionalRemoval<PointT> condrem (range_cond);
		condrem.setInputCloud (cloud);
		//condrem.setKeepOrganized(true);
		
		// apply filter
		pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
		condrem.filter (*cloud_filtered);

		return cloud_filtered;

		//pcl::PassThrough<pcl::PointXYZ> pass;
		//pass.setInputCloud (cloud);
		//pass.setFilterFieldName ("z");
		//pass.setFilterLimits (0.0, 1.0);
		////pass.setFilterLimitsNegative (true);
		//pass.filter (*cloud_filtered);
		//return cloud_filtered;
	}
	else
	{
		// build the condition
		pcl::ConditionOr<PointT>::Ptr range_cond (new pcl::ConditionOr<PointT> ());

		// condition which a given point must satisfy for it to remain in the point cloud
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, minPoint.x)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, minPoint.y)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, minPoint.z)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, maxPoint.x)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, maxPoint.y)));
		range_cond->addComparison (pcl::FieldComparison<PointT>::ConstPtr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, maxPoint.z)));

		// build the filter
		pcl::ConditionalRemoval<PointT> condrem (range_cond);
		condrem.setInputCloud (cloud);
		//condrem.setKeepOrganized(true);
		
		// apply filter
		pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
		condrem.filter (*cloud_filtered);

		return cloud_filtered;
	}
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
rangeRemoval(typename pcl::PointCloud<PointT>::ConstPtr cloud,
				const PointT &origin,
				const float range,
				const bool bRemoveClose = true)
{
	// build the condition which a given point must satisfy for it to remain in the point cloud
	// (p - q)'(p - q) > r^2
	// => p'p - 2q'p + q'q - r^2 > 0
	// => p'Ap + 2v'p + c [OP] 0
	//    : A = eye(3), v = -q, c = q'q - r^2, OP = >
	pcl::ConditionAnd<PointT>::Ptr quad_cond(new pcl::ConditionAnd<PointT> ());
	quad_cond->addComparison(pcl::TfQuadraticXYZComparison<PointT>::ConstPtr(
								new pcl::TfQuadraticXYZComparison<PointT>(bRemoveClose ? pcl::ComparisonOps::GT : pcl::ComparisonOps::LT,
																			Eigen::Matrix3f::Identity(), 
																			Eigen::Vector3f(-origin.x, -origin.y, -origin.z), 
																			(origin.x)*(origin.x) + (origin.y)*(origin.y) + (origin.z)*(origin.z) - range*range)));
		
	// build the filter
	pcl::ConditionalRemoval<PointT> condrem (quad_cond);
	condrem.setInputCloud (cloud);
	//condrem.setKeepOrganized(true);
		
	// apply filter
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
	condrem.filter (*cloud_filtered);

	return cloud_filtered;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
downSampling(const typename pcl::PointCloud<PointT>::ConstPtr		&pCloud,
			  const float														leafSize)
{
	// build the filter
	pcl::VoxelGrid<PointT> vg;
	vg.setInputCloud(pCloud);
	vg.setLeafSize(leafSize, leafSize, leafSize);

	// apply filter
	pcl::PointCloud<PointT>::Ptr pFilteredCloud(new pcl::PointCloud<PointT>());
	vg.filter(*pFilteredCloud);

	return pFilteredCloud;
}

template <typename PointT>
void downSampling(const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&cloudPtrList,
					 const float																	leafSize,
					 std::vector<typename pcl::PointCloud<PointT>::Ptr>				&filteredCloudPtrList)
{
	// reset
	filteredCloudPtrList.resize(cloudPtrList.size());

	// for each point cloud
	for(size_t i = 0; i < cloudPtrList.size(); i++)
	{
		filteredCloudPtrList[i] = downSampling<PointT>(cloudPtrList[i], leafSize);
	}
}

template <typename NormalT>
typename pcl::PointCloud<NormalT>::Ptr
normalRefinement(typename pcl::PointCloud<NormalT>::Ptr normals, const int k)
{
	// nearest neighbor search
	std::vector<std::vector<int> >		k_indices;
	std::vector<std::vector<float> >	k_sqr_distances;
      
	pcl::search::KdTree<NormalT> kdtree;
	kdtree.setInputCloud(normals);
	kdtree.nearestKSearchT(*normals, std::vector<int>(), k, k_indices, k_sqr_distances);

	// build the filter
	pcl::NormalRefinement<NormalT> nr(k_indices, k_sqr_distances);
	nr.setInputCloud (normals);

	// apply filter
	pcl::PointCloud<NormalT>::Ptr normals_refined(new pcl::PointCloud<NormalT>());
	nr.filter(*normals_refined);

	return normals_refined;
}

}

#endif