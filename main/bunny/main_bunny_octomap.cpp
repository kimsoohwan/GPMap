#if 1

// STL
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

// GPMap
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "visualization/cloud_viewer.hpp"	// show
#include "octomap/octomap.hpp"				// OctoMap
using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] setting - directory
	const std::string strOriginalInputDataFolder	("E:/Documents/GitHub/ScanDataAlignment/data/original/bunny/");
	const std::string strInputDataFolder			("../../data/bunny/input/");
	const std::string strIntermediateDataFolder	("../../data/bunny/intermediate/");
	const std::string strOctomapOutputDataFolder	("../../data/bunny/output/octomap/");
	create_directory(strOctomapOutputDataFolder);

	// [0] setting - observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]	= {"bun000", "bun090", "bun180", "bun270"};
	StringList strObsFileNameList(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [0] octomap Scene Graph

	// robot poses
	octomath::Pose6D poses[] = {octomath::Pose6D(octomath::Vector3(0, 0, 0),											octomath::Quaternion(0, 0, 0, 1)),
										 octomath::Pose6D(octomath::Vector3(2.20761e-05, -3.34606e-05, -7.20881e-05), octomath::Quaternion(0.000335889, -0.708202, 0.000602459, 0.706009)),
										 octomath::Pose6D(octomath::Vector3(0.000116991, 2.47732e-05, -4.6283e-05),	octomath::Quaternion(-0.00215148, 0.999996, -0.0015001, 0.000892527)),
										 octomath::Pose6D(octomath::Vector3(0.000130273, 1.58623e-05, 0.000406764),	octomath::Quaternion(0.000462632, 0.707006, -0.00333301, 0.7072))};

	// hit points
	PointXYZCloudPtrList originalHitPointCloudPtrList;
	loadPointCloud<pcl::PointXYZ>(originalHitPointCloudPtrList, strObsFileNameList, strOriginalInputDataFolder, ".ply");

	// scene graph
	octomap::ScanGraph scanGraph;
	for(size_t i = 0; i < NUM_OBSERVATIONS; i++)
	{
		// point cloud
		octomap::Pointcloud pc;
		pcd2pc<pcl::PointXYZ>(*(originalHitPointCloudPtrList[i]), pc);
		std::cout << "pc.size() = " << pc.size() << std::endl;

		// add node
		scanGraph.addNode(&pc, poses[i]);
	}

	// save
	//scanGraph.writeBinary("bunny.graph");
	scanGraph.writeBinary(strOctomapOutputDataFolder + "bunny.graph");
	system("pause");
	exit(0);

	// [1] load data
	std::cout << "[Input Data]" << std::endl;
	int fOctreeDownSampling;
	std::cout << "No sampling(-1), Random sampling(0) or octree-based down sampling(1)?";
	std::cin >> fOctreeDownSampling;

	float param;
	std::string strDownSampledIntermediateDataFolder;
	std::string strDownSampledOctomapOutputDataFolder;
	if(fOctreeDownSampling > 0)
	{
		// leaf size
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)

		// sub folder
		std::stringstream ss;
		ss << "down_sampling_" << param << "/";
		strDownSampledIntermediateDataFolder	= strIntermediateDataFolder	+ ss.str();
		strDownSampledOctomapOutputDataFolder	= strOctomapOutputDataFolder	+ ss.str();
	}
	else if(fOctreeDownSampling < 0)
	{
		strDownSampledIntermediateDataFolder	= strIntermediateDataFolder	+ "original/";
		strDownSampledOctomapOutputDataFolder	= strOctomapOutputDataFolder	+ "original/";
	}
	else
	{
		// sampling ratio
		std::cout << "Random sampling ratio: ";
		std::cin >> param;	// 0.5, 0.3, 0.2, 0.1

		// sub folder
		std::stringstream ss;
		ss << "random_sampling_" << param << "/";
		strDownSampledIntermediateDataFolder	= strIntermediateDataFolder	+ ss.str();
		strDownSampledOctomapOutputDataFolder	= strOctomapOutputDataFolder	+ ss.str();
	}
	create_directory(strDownSampledOctomapOutputDataFolder);

	// [1-1] load original/down-sampled hit points
	const std::string strOriginalIntermediateDataFolder = strIntermediateDataFolder + "original/";
	PointXYZCloudPtrList hitPointCloudPtrList;
	loadPointCloud<pcl::PointXYZ>(hitPointCloudPtrList, strObsFileNameList, strOriginalIntermediateDataFolder, ".pcd");

	PointXYZCloudPtrList sampledHitPointCloudPtrList;
	loadPointCloud<pcl::PointXYZ>(sampledHitPointCloudPtrList, strObsFileNameList, strDownSampledIntermediateDataFolder, ".pcd");

	// [1-2] load sensor positions
	PointXYZVList sensorPositionList;
	loadSensorPositionList(sensorPositionList, strObsFileNameList, strInputDataFolder, "_camera_position.txt");
	assert(NUM_OBSERVATIONS == hitPointCloudPtrList.size() && NUM_OBSERVATIONS == sensorPositionList.size());

	// [2] Octomaps

	// resolution
	double	RESOLUTION = 0.001;
	std::cout << "Resolution: ";
	std::cin >> RESOLUTION;

	// sub folders
	std::stringstream ss;
	ss << "resolution_" << RESOLUTION << "/";
	const std::string strMyOctomapOutputDataFolder	(strDownSampledOctomapOutputDataFolder + ss.str());
	const std::string strLogFolder						(strMyOctomapOutputDataFolder + "log/");
	create_directory(strMyOctomapOutputDataFolder);
	create_directory(strLogFolder);

	// log file
	LogFile logFile(strLogFolder + "octomap.log");
		
	// setting
	OctoMap<NO_COLOR> octomap(RESOLUTION);


	// update
	CPU_Times octomap_elapsed, octomap_total_elapsed;
	octomap_total_elapsed.clear();
	for(size_t i = 0; i < sampledHitPointCloudPtrList.size(); i++)
	{
		logFile << "==== Updating the OctoMap with the point cloud #" << i << " ====" << std::endl;

		// update
		octomap_elapsed = octomap.update<pcl::PointXYZ, pcl::PointXYZ>(*(sampledHitPointCloudPtrList[i]), sensorPositionList[i]);

		// save
		std::stringstream ss;
		ss << strMyOctomapOutputDataFolder << "octomap_bunny_upto_" << i;
		octomap.save(ss.str());

		// accumulate cpu times
		octomap_total_elapsed += octomap_elapsed;
		logFile << octomap_elapsed << std::endl << std::endl;

		// save point cloud
		octomap::Pointcloud pc;
		for(size_t j = 0; j < sampledHitPointCloudPtrList[i]->size(); j++)
			pc.push_back(sampledHitPointCloudPtrList[i]->points[j].x, 
							 sampledHitPointCloudPtrList[i]->points[j].y, 
							 sampledHitPointCloudPtrList[i]->points[j].z);

		std::stringstream ss2;
		ss2 << strMyOctomapOutputDataFolder << "point_cloud_" << i << ".dat";
		std::ofstream ofs(ss2.str());
		pc.write(ofs);
	}

	// total time
	logFile << "============= Total Time =============" << std::endl;
	logFile << octomap_total_elapsed << std::endl << std::endl;

	// [4] evaluation
	logFile << "============= Evaluation =============" << std::endl;
	unsigned int num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown;
	octomap.evaluate<pcl::PointXYZ, pcl::PointXYZ>(hitPointCloudPtrList, sensorPositionList,
																	num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown);
	logFile << "Number of hit points: " << num_points << std::endl;
	logFile << "Number of correct voxels: " << num_voxels_correct << std::endl;
	logFile << "Number of wrong voxels: " << num_voxels_wrong << std::endl;
	logFile << "Number of unknown voxels: " << num_voxels_unknown << std::endl;
	logFile << "Correct rate (correct/(correct+wrong)): " << static_cast<float>(num_voxels_correct)/static_cast<float>(num_voxels_correct + num_voxels_wrong) << std::endl;

	system("pause");
}

#endif