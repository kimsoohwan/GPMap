#if 1
// Eigen
#include "serialization/eigen_serialization.hpp" // Eigen
// includes followings inside of it
//		- #define EIGEN_NO_DEBUG		// to speed up
//		- #define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
//		- #include <Eigen/Core>

// combinations
// -------------------------------------------------------------------------
// |                   Observations                  |       Process       |
// |=======================================================================|
// |   (FuncObs)  |    (DerObs)         |  (AllObs)  |       | Incremental |
// | hit points,  | virtual hit points, |  FuncObs,  | Batch |-------------|
// | empty points | surface normals     |  DerObs    |       | BCM  | iBCM |
// -------------------------------------------------------------------------

// GPMap
#include "common/common.hpp"					// combinePointCloud
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "visualization/cloud_viewer.hpp"	// show
#include "features/surface_normal.hpp"		// estimateSurfaceNormals
#include "filter/filters.hpp"					// downSampling
#include "octree/data_partitioning.hpp"	// randomSampling
using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] setting - directory
	const std::string strDataFolder				("E:/Documents/GitHub/Data/");
	const std::string strDataName					("bunny");
	const std::string strInputFolder				(strDataFolder + strDataName + "/input/");
	const std::string strIntermediateFolder	(strDataFolder + strDataName + "/intermediate/");
	create_directory(strIntermediateFolder);

	// [0] setting - observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]	= {"bun000", "bun090", "bun180", "bun270"};
	const std::string strFileNameAll			=  "bunny_all";
	StringList strObsFileNames(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [0] setting - parameters for estimating surface normals
	const bool FLAG_SERACH_BY_RADIUS = false; // SearchNearestK or SearchRadius
	const float searchRadius = 0.01;

	// show?
	bool fShow;
	std::cout << "Do you wish to see the results? (0/1)"; 
	std::cin >> fShow;

	//////////////////////////////////////////////////////////////////////////////////////////
	//                               Original Observations                                  //
	//////////////////////////////////////////////////////////////////////////////////////////

	// done before?
	bool fRunOriginalObservations;
	std::cout << "Do you wish to pass(0) or run(1) the orignial observations? "; 
	std::cin >> fRunOriginalObservations;

	// [1-1] Hit Points - Sequential
	PointXYZCloudPtrList hitPointCloudPtrList;
	if(fRunOriginalObservations)
	{
		loadPointCloud<pcl::PointXYZ>(hitPointCloudPtrList, strObsFileNames, strInputFolder, ".ply");				// original ply files which are transformed in global coordinates
		savePointCloud<pcl::PointXYZ>(hitPointCloudPtrList, strObsFileNames, strIntermediateFolder, ".pcd");
		if(fShow) show<pcl::PointXYZ>("Sequential Hit Points", hitPointCloudPtrList);
	}
	else
	{
		// it will be used for random/down sampling later
		loadPointCloud<pcl::PointXYZ>(hitPointCloudPtrList, strObsFileNames, strIntermediateFolder, ".pcd");
	}

	// [1-2] Hit Points - All
	PointXYZCloudPtr pAllHitPointCloud(new PointXYZCloud());
	if(fRunOriginalObservations)
	{
		for(size_t i = 0; i < hitPointCloudPtrList.size(); i++)	(*pAllHitPointCloud) += (*(hitPointCloudPtrList[i]));
		savePointCloud<pcl::PointXYZ>(pAllHitPointCloud, strFileNameAll, strIntermediateFolder, ".pcd");
		if(fShow) show<pcl::PointXYZ>("All Hit Points", pAllHitPointCloud);
	}
	else
	{
		// it will be used for random/down sampling later
		loadPointCloud<pcl::PointXYZ>(pAllHitPointCloud, strFileNameAll, strIntermediateFolder, ".pcd");
	}

	// [2] Sensor Positions
	PointXYZVList sensorPositionList;
	loadSensorPositionList(sensorPositionList, strObsFileNames, strInputFolder, "_camera_position.txt");
	assert(NUM_OBSERVATIONS == hitPointCloudPtrList.size() && NUM_OBSERVATIONS == sensorPositionList.size());

	if(fRunOriginalObservations)
	{
		// [3-1] Function Observations (Hit Points + Unit Ray Back Vectors) - Sequential
		PointNormalCloudPtrList funcObsCloudPtrList;
		unitRayBackVectors(hitPointCloudPtrList, sensorPositionList, funcObsCloudPtrList);
		savePointCloud<pcl::PointNormal>(funcObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_func_obs.pcd");
		//loadPointCloud<pcl::PointNormal>(funcObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_func_obs.pcd");
		if(fShow) show<pcl::PointNormal>("Sequential Unit Ray Back Vectors", funcObsCloudPtrList, 0.005);

		// [3-2] Function Observations (Hit Points + Unit Ray Back Vectors) - All
		PointNormalCloudPtr pAllFuncObs(new PointNormalCloud());
		for(size_t i = 0; i < funcObsCloudPtrList.size(); i++)	*pAllFuncObs += *funcObsCloudPtrList[i];
		savePointCloud<pcl::PointNormal>(pAllFuncObs, strFileNameAll, strIntermediateFolder, "_func_obs.pcd");
		//loadPointCloud<pcl::PointNormal>(pAllFuncObs, strFileNameAll, strIntermediateFolder, "_func_obs.pcd");
		if(fShow) show<pcl::PointNormal>("All Unit Ray Back Vectors", pAllFuncObs, 0.005);

		// [4-1] Derivative Observations (Virtual Hit Points + Surface Normal Vectors) - Sequential
		PointNormalCloudPtrList derObsCloudPtrList;
		//estimateSurfaceNormals<ByNearestNeighbors>(hitPointCloudPtrList, sensorPositionList, FLAG_SERACH_BY_RADIUS, searchRadius, derObsCloudPtrList);
		estimateSurfaceNormals<ByMovingLeastSquares>(hitPointCloudPtrList, sensorPositionList, FLAG_SERACH_BY_RADIUS, searchRadius, derObsCloudPtrList);
		savePointCloud<pcl::PointNormal>(derObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_der_obs.pcd");
		//loadPointCloud<pcl::PointNormal>(derObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_der_obs.pcd");
		if(fShow) show<pcl::PointNormal>("Sequential Surface Normals", derObsCloudPtrList, 0.005);

		// [4-2] Derivative Observations (Virtual Hit Points + Surface Normal Vectors) - All
		PointNormalCloudPtr pAllDerObs(new PointNormalCloud());
		for(size_t i = 0; i < derObsCloudPtrList.size(); i++)	(*pAllDerObs) += (*(derObsCloudPtrList[i]));
		savePointCloud<pcl::PointNormal>(pAllDerObs, strFileNameAll, strIntermediateFolder, "_der_obs.pcd");
		//loadPointCloud<pcl::PointNormal>(pAllDerObs, strFileNameAll, strIntermediateFolder, "_der_obs.pcd");
		if(fShow) show<pcl::PointNormal>("All Surface Normals", pAllDerObs, 0.005);
	}

	//////////////////////////////////////////////////////////////////////////////////////////
	//                           Down Sampled Observations                                  //
	//////////////////////////////////////////////////////////////////////////////////////////

	bool fOctreeDownSampling;
	std::cout << "Random sampling(0) or octree-based down sampling(1)?";
	std::cin >> fOctreeDownSampling;

	// sub-folder for samples
	std::string strIntermediateSampleFolder;
	float param;
	if(fOctreeDownSampling)
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

		// sub folder
		std::stringstream ss;
		ss << strIntermediateFolder << "random_sampling_" << param << "/";
		strIntermediateSampleFolder = ss.str();
	}

	// [1-1] Hit Points - Sequential - Down Sampling
	PointXYZCloudPtrList sampledHitPointCloudPtrList;
	if(fOctreeDownSampling)	downSampling  <pcl::PointXYZ>(hitPointCloudPtrList, param, sampledHitPointCloudPtrList);
	else							randomSampling<pcl::PointXYZ>(hitPointCloudPtrList, param, sampledHitPointCloudPtrList);
	savePointCloud<pcl::PointXYZ>(sampledHitPointCloudPtrList, strObsFileNames, strIntermediateSampleFolder, ".pcd");
	//loadPointCloud<pcl::PointXYZ>(sampledHitPointCloudPtrList, strObsFileNames, strIntermediateSampleFolder, ".pcd");
	if(fShow) show<pcl::PointXYZ>("Sequential Down Sampled Hit Points", sampledHitPointCloudPtrList);

	// sampling rate
	std::cout << "Hit Points - Sequential - Down Sampling" << std::endl;
	int totalNumPoints(0), totalNumSampledPoints(0);
	for(size_t i = 0; i < sampledHitPointCloudPtrList.size(); i++)
	{
		totalNumPoints				+= hitPointCloudPtrList[i]->points.size();
		totalNumSampledPoints	+= sampledHitPointCloudPtrList[i]->points.size();
		std::cout << "[" << i << "] sampling rate: " 
					 << sampledHitPointCloudPtrList[i]->points.size() << " / "
					 << hitPointCloudPtrList[i]->points.size() << " = " 
					 << 100.f * static_cast<float>(sampledHitPointCloudPtrList[i]->points.size())/static_cast<float>(hitPointCloudPtrList[i]->points.size()) << "%" << std::endl;
	}
	std::cout << "Total sampling rate: " 
				 << totalNumSampledPoints << " / "
				 << totalNumPoints << " = " 
				 << 100.f * static_cast<float>(totalNumSampledPoints)/static_cast<float>(totalNumPoints) << "%" << std::endl;

	// [1-2] Hit Points - All - Down Sampling
	PointXYZCloudPtr pAllSampledHitPointCloud(new PointXYZCloud());
	for(size_t i = 0; i < sampledHitPointCloudPtrList.size(); i++)	(*pAllSampledHitPointCloud) += (*(sampledHitPointCloudPtrList[i]));
	savePointCloud<pcl::PointXYZ>(pAllSampledHitPointCloud, strFileNameAll, strIntermediateSampleFolder, ".pcd");
	//loadPointCloud<pcl::PointXYZ>(pAllSampledHitPointCloud, strFileNameAll, strIntermediateSampleFolder, ".pcd");
	if(fShow) show<pcl::PointXYZ>("All Down Sampled Hit Points", pAllSampledHitPointCloud);

	// [3-1] Function Observations (Hit Points + Unit Ray Back Vectors) - Sequential - Down Sampling
	PointNormalCloudPtrList sampledFuncObsCloudPtrList;
	unitRayBackVectors(sampledHitPointCloudPtrList, sensorPositionList, sampledFuncObsCloudPtrList);
	savePointCloud<pcl::PointNormal>(sampledFuncObsCloudPtrList, strObsFileNames, strIntermediateSampleFolder, "_func_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(sampledFuncObsCloudPtrList, strObsFileNames, strIntermediateSampleFolder, "_func_obs.pcd");
	if(fShow) show<pcl::PointNormal>("Sequential Down Sampled Unit Ray Back Vectors", sampledFuncObsCloudPtrList, 0.005);

	// [3-2] Function Observations (Hit Points + Unit Ray Back Vectors) - All - Down Sampling
	PointNormalCloudPtr pAllSampledFuncObs(new PointNormalCloud());
	for(size_t i = 0; i < sampledFuncObsCloudPtrList.size(); i++)	*pAllSampledFuncObs += *sampledFuncObsCloudPtrList[i];
	savePointCloud<pcl::PointNormal>(pAllSampledFuncObs, strFileNameAll, strIntermediateSampleFolder, "_func_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(pAllSampledFuncObs, strFileNameAll, strIntermediateSampleFolder, "_func_obs.pcd");
	if(fShow) show<pcl::PointNormal>("All Down Sampled Unit Ray Back Vectors", pAllSampledFuncObs, 0.005);

	// [4-1] Derivative Observations (Virtual Hit Points + Surface Normal Vectors) - Sequential - Down Sampling
	PointNormalCloudPtrList sampledDerObsCloudPtrList;
	//estimateSurfaceNormals<ByNearestNeighbors>(sampledHitPointCloudPtrList, sensorPositionList, FLAG_SERACH_BY_RADIUS, searchRadius, sampledDerObsCloudPtrList);
	estimateSurfaceNormals<ByMovingLeastSquares>(sampledHitPointCloudPtrList, sensorPositionList, FLAG_SERACH_BY_RADIUS, searchRadius, sampledDerObsCloudPtrList);
	savePointCloud<pcl::PointNormal>(sampledDerObsCloudPtrList, strObsFileNames, strIntermediateSampleFolder, "_der_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(sampledDerObsCloudPtrList, strObsFileNames, strIntermediateSampleFolder, "_der_obs.pcd");
	if(fShow) show<pcl::PointNormal>("Sequential Down Sampled Surface Normals", sampledDerObsCloudPtrList, 0.005);

	// [4-2] Derivative Observations (Virtual Hit Points + Surface Normal Vectors) - All - Down Sampling
	PointNormalCloudPtr pAllSampledDerObs(new PointNormalCloud());
	for(size_t i = 0; i < sampledDerObsCloudPtrList.size(); i++)	(*pAllSampledDerObs) += (*(sampledDerObsCloudPtrList[i]));
	savePointCloud<pcl::PointNormal>(pAllSampledDerObs, strFileNameAll, strIntermediateSampleFolder, "_der_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(pAllSampledDerObs, strFileNameAll, strIntermediateSampleFolder, "_der_obs.pcd");
	if(fShow) show<pcl::PointNormal>("All Down Sampled Surface Normals", pAllSampledDerObs, 0.005);

	// [5] All (Func + Der) Observations
	//PointNormalCloudPtrList allObsCloudPtrList;
	//combinePointCloud<pcl::PointNormal>(funcObsCloudPtrList, derObsCloudPtrList, allObsCloudPtrList);
	//savePointCloud<pcl::PointNormal>(allObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_all_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(allObsCloudPtrList, strObsFileNames, strIntermediateFolder, "_all_obs.pcd");
	//if(fShow) show<pcl::PointNormal>("All Observation List", allObsCloudPtrList, 0.005);

	//PointNormalCloudPtr pAllObs(new PointNormalCloud());
	//*pAllObs += *pAllFuncObs;
	//*pAllObs += *pAllDerObs;
	//savePointCloud<pcl::PointNormal>(pAllObs, strFileNameAll, strIntermediateFolder, "_all_obs.pcd");
	//loadPointCloud<pcl::PointNormal>(pAllObs, strFileNameAll, strIntermediateFolder, "_all_obs.pcd");
	//if(fShow) show<pcl::PointNormal>("All Observations", pAllDerObs, 0.005, 0.001);

	system("pause");
}

#endif