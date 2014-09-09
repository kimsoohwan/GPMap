#ifndef _TEST_OCTREE_GPMAP_HPP_
#define _TEST_OCTREE_GPMAP_HPP_

// Google Test
#include "gtest/gtest.h"

// PCL
#include <pcl/io/pcd_io.h>			// pcl::io::loadPCDFile, savePCDFile
#include <pcl/common/common.h>	// pcl::getMinMax3D

// GPMap
// LINK ERROR: cannot open file 'libboost_filesystem-vc100-mt-sgd-1_49.lib'
// LINK ERROR: cannot open file 'libboost_filesystem-vc100-mt-s-1_49.lib'
// #include "io/io.hpp"							// loadPointClouds, savePointClouds, loadSensorPositionList, 
#include "common/common.hpp"					// getMinMaxPointXYZ
#include "filter/filters.hpp"					// cropBox
#include "visualization/cloud_viewer.hpp"	// show
#include "octree/octree_gpmap.hpp"
using namespace GPMap;

class TestOctreeGPMap : public ::testing::Test,
								public OctreeGPMap<pcl::PointNormal>
{
public:
	TestOctreeGPMap()
		: BLOCK_SIZE(0.01),
		  NUM_CELLS_PER_AXIS(10),
		  INDEPENDENT_BCM(true), //INDEPENDENT_BCM(false),
		  POINT_DUPLICATION(false),
		  OctreeGPMap<pcl::PointNormal>(BLOCK_SIZE, NUM_CELLS_PER_AXIS, INDEPENDENT_BCM, POINT_DUPLICATION),
		  sensorPosition(7.2201273e-01, 2.5926464e-02, 1.6074278e-01),
		  pPointNormalCloud(new pcl::PointCloud<pcl::PointNormal>())
	{
		// some constants
		const size_t NUM_DATA = 4; 
		//const std::string strInputDataFolder ("../../data/input/bunny/");
		//const std::string strOutputDataFolder("../../data/output/bunny/");
		//const std::string strFilenames_[] = {"bun000", "bun090", "bun180", "bun270"};
		//StringList strFileNames(strFilenames_, strFilenames_ + NUM_DATA); 

		// [1] load/save hit points
		//loadPointClouds<pcl::PointXYZ>(hitPointCloudList, strFileNames, strInputDataFolder, "_hit_points.pcd");		// original pcd files which are transformed in global coordinates

		// [2] load sensor positions
		//loadSensorPositionList(sensorPositionList, strFileNames, strInputDataFolder, "_camera_position.txt");
		pcl::PointXYZ sensorPosition(7.2201273e-01, 2.5926464e-02, 1.6074278e-01);

		// [3] load/save surface normals
		//loadPointClouds<pcl::PointNormal>(pPointNormalClouds, strFileNames, strInputDataFolder, "_normals.pcd");		// original pcd files which are transformed in global coordinates
		pcl::io::loadPCDFile<pcl::PointNormal>("../../data/input/bunny/bun000_normals.pcd", *pPointNormalCloud);
		
		// [4] bounding box
		pcl::PointNormal min_pt, max_pt;
		pcl::getMinMax3D(*pPointNormalCloud, min_pt, max_pt);
		defineBoundingBox(min_pt, max_pt);

		// [5] crop point cloud into four parts
		pPointNormalCloudList.resize(NUM_DATA);
		pcl::PointXYZ mid_pt((min_pt.x+max_pt.x)/2.f, 
									(min_pt.y+max_pt.y)/2.f,
									(min_pt.z+max_pt.z)/2.f);

		Eigen::Vector4f min_pt1, max_pt1;
		Eigen::Vector4f min_pt2, max_pt2;
		Eigen::Vector4f min_pt3, max_pt3;
		Eigen::Vector4f min_pt4, max_pt4;

		min_pt1 << min_pt.x, mid_pt.y, mid_pt.z;
		max_pt1 << max_pt.x, max_pt.y, max_pt.z;

		min_pt2 << min_pt.x, min_pt.y, mid_pt.z;
		max_pt2 << max_pt.x, mid_pt.y, max_pt.z;

		min_pt3 << min_pt.x, min_pt.y, min_pt.z;
		max_pt3 << max_pt.x, mid_pt.y, mid_pt.z;

		min_pt4 << min_pt.x, mid_pt.y, min_pt.z;
		max_pt4 << max_pt.x, max_pt.y, mid_pt.z;

		// crop
		pPointNormalCloudList[0] = cropBox<pcl::PointNormal>(pPointNormalCloud, min_pt1, max_pt1);
		pPointNormalCloudList[1] = cropBox<pcl::PointNormal>(pPointNormalCloud, min_pt2, max_pt2);
		pPointNormalCloudList[2] = cropBox<pcl::PointNormal>(pPointNormalCloud, min_pt3, max_pt3);
		pPointNormalCloudList[3] = cropBox<pcl::PointNormal>(pPointNormalCloud, min_pt4, max_pt4);
		assert(pPointNormalCloud->size() == (pPointNormalCloudList[0]->size()
													  + pPointNormalCloudList[1]->size()
													  + pPointNormalCloudList[2]->size()
													  + pPointNormalCloudList[3]->size()));

		// show
		show<pcl::PointNormal>("Cropped Bunny000", pPointNormalCloudList, 0.01);
	}

protected:
		const double		BLOCK_SIZE;
		const size_t		NUM_CELLS_PER_AXIS;
		const bool			INDEPENDENT_BCM;
		const bool			POINT_DUPLICATION;

		//PointXYZCloudPtrList			hitPointCloudList;
		//PointXYZVList					sensorPositionList;
		//PointNormalCloudPtrList		pPointNormalClouds;

		// In order not to use boost::filesystem, because of its compatability with Google Test
		pcl::PointXYZ					sensorPosition;
		PointNormalCloudPtr			pPointNormalCloud;
		PointNormalCloudPtrList		pPointNormalCloudList;
};

/** @brief Update by mean vectors and covariance matrices */
TEST_F(TestOctreeGPMap, BoundingBoxTest)
{
	double minX, minY, minZ, maxX, maxY, maxZ;
	getBoundingBox(minX, minY, minZ, maxX, maxY, maxZ);
	std::cout << minX << ", " << minY << ", " << minZ << std::endl;
	EXPECT_EQ(minX, 0.0);
	EXPECT_EQ(minY, 0.0);
	EXPECT_EQ(minZ, 0.0);
}

#endif