#if 1
// Eigen
#include "serialization/eigen_serialization.hpp" // Eigen
// includes followings inside of it
//		- #define EIGEN_NO_DEBUG		// to speed up
//		- #define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
//		- #include <Eigen/Core>

// PCL
#include <pcl/point_types.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h>

// OpenGP
#include "gp.h"

// GPMap
#include "common/common.hpp"					// combinePointCloud
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "visualization/cloud_viewer.hpp"	// show
#include "features/surface_normal.hpp"		// estimateSurfaceNormals
#include "filter/filters.hpp"					// downSampling
#include "octree/data_partitioning.hpp"	// randomSampling
#include "octree/octree_viewer.hpp"			// OctreeViewer

#include "octree/octree_gpmap.hpp"			// OctreeGPMap
#include "octree/octree_container.hpp"		// OctreeGPMapContainer
#include "bcm/gaussian.hpp"					// GaussianDistribution

using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] setting - directory
	const std::string strDataFolder				("E:/Documents/GitHub/Data/");
	const std::string strDataName					("bunny");
	const std::string strInputFolder				(strDataFolder + strDataName + "/input/");
	const std::string strIntermediateFolder	(strDataFolder + strDataName + "/intermediate/");

	// [0] setting - observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]		= {"bun000", "bun090", "bun180", "bun270"};
	const std::string strFileNameAll				=  "bunny_all";
	StringList strObsFileNames(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [0] setting - input data folder
	std::cout << "[Input Data]" << std::endl;
	int fOctreeDownSampling;
	std::cout << "No sampling(-1), Random sampling(0) or octree-based down sampling(1)?";
	std::cin >> fOctreeDownSampling;

	float param;
	std::string strIntermediateSampleFolder;
	// no sampling
	if(fOctreeDownSampling < 0)
	{
		strIntermediateSampleFolder	= strIntermediateFolder;
	}
	// octree-based down sampling
	else if(fOctreeDownSampling > 0)
	{
		// leaf size
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)

		// sub-folder
		std::stringstream ss;
		ss << strIntermediateFolder << "down_sampling_" << param << "/";
		strIntermediateSampleFolder = ss.str();
	}
	else
	{
		// sampling ratio
		std::cout << "Random sampling ratio: ";
		std::cin >> param;	// 0.5, 0.3, 0.2, 0.1

		// sub-folder
		std::stringstream ss;
		ss << strIntermediateFolder << "random_sampling_" << param << "/";
		strIntermediateSampleFolder = ss.str();
	}

	// [1-1] Hit Points - Sequential - Down Sampling
	PointXYZCloudPtrList sampledHitPointCloudPtrList;
	loadPointCloud<pcl::PointXYZ>(sampledHitPointCloudPtrList, strObsFileNames, strIntermediateSampleFolder, ".pcd");
	//show<pcl::PointXYZ>("Sequential Down Sampled Hit Points", sampledHitPointCloudPtrList);


	// [1-2] Hit Points - All - Down Sampling
	PointXYZCloudPtr pAllSampledHitPointCloud(new PointXYZCloud());
	loadPointCloud<pcl::PointXYZ>(pAllSampledHitPointCloud, strFileNameAll, strIntermediateSampleFolder, ".pcd");
	//show<pcl::PointXYZ>("All Down Sampled Hit Points", pAllSampledHitPointCloud);

	// GPMap
	const double BLOCK_SIZE = 0.2;
	const double NUM_CELLS_PER_AXIS
	typedef OctreeGPMapContainer<GaussianDistribution>	LeafT;
	typedef OctreeGPMap<GP::MeanZero, GP::CovSEiso, GP::LikGauss, GP::InfExactGeneral, LeafT> OctreeGPMapT;
	OctreeGPMapT gpmap(BLOCK_SIZE, 
							 NUM_CELLS_PER_AXIS, 
							 MIN_NUM_POINTS_TO_PREDICT, 
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS,
							 FLAG_RAMDOMLY_SAMPLE_POINTS);

	// octree
	//pcl::octree::OctreePointCloud<pcl::PointXYZ> octreePointCloud(0.02);
	//octreePointCloud.setInputCloud(pAllSampledHitPointCloud);
	//octreePointCloud.addPointsFromInputCloud();

	// octree viewer
	//OctreeViewer<pcl::PointXYZ, pcl:octree::OctreePointCloud<pcl::PointXYZ> > octreeViewer(octreePointCloud);

	for(size_t i = 0; i < sampledHitPointCloudPtrList.size(); i++)
	{
		std::cout << "Number of " << i << "-th Hit Points: " << sampledHitPointCloudPtrList[i]->points.size() << std::endl;
		for(size_t j = 0; j < sampledHitPointCloudPtrList[i]->points.size(); j++)
		{
			const pcl::PointXYZ point(sampledHitPointCloudPtrList[i]->points[j]);
			sampledHitPointCloudPtrList[i]->points[j].x = -point.y;
			sampledHitPointCloudPtrList[i]->points[j].y = -point.x;
			sampledHitPointCloudPtrList[i]->points[j].z = -point.z;
		}
	}
	show<pcl::PointXYZ>("Hit Points", sampledHitPointCloudPtrList, 0, 0, false, false, false, true);
		
	for(size_t j = 0; j < pAllSampledHitPointCloud->points.size(); j++)
	{
		const pcl::PointXYZ point(pAllSampledHitPointCloud->points[j]);
		pAllSampledHitPointCloud->points[j].x = -point.y;
		pAllSampledHitPointCloud->points[j].y = -point.x;
		pAllSampledHitPointCloud->points[j].z = -point.z;
	}
	show<pcl::PointXYZ>("All Hit Points", pAllSampledHitPointCloud, 0, 0, false, false, false);

	system("pause");
}

#endif