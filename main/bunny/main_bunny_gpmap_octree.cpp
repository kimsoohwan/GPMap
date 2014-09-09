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
	const std::string strInputDataFolder						("../../data/bunny/input/");
	const std::string strOriginalIntermediateDataFolder	("../../data/bunny/intermediate/original/");
	const std::string strGPMapDataFolder						("../../data/bunny/output/gpmap/random_sampling_0.1/meta_data/");
	const std::string strOutputDataFolder0						("../../data/bunny/output/gpmap/random_sampling_0.1/octree/");
	const std::string strOutputLogFolder						(strOutputDataFolder0			 + "log/");
	create_directory(strOutputDataFolder0);
	create_directory(strOutputLogFolder);

	// [0] setting - observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]	= {"bun000", "bun090", "bun180", "bun270"};
	StringList strObsFileNames(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [0] setting - GPMaps
	const size_t NUM_GPMAPS = 10; 
	const std::string strGPMapFileNames_[]	= {
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(all)_iBCM",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_iBCM_upto_0",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_iBCM_upto_1",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_iBCM_upto_2",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_iBCM_upto_3",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(all)_BCM",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_BCM_upto_0",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_BCM_upto_1",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_BCM_upto_2",
															"block_0.02_cell_0.002_m_2_n_100_gap_0.001_func_obs(seq)_BCM_upto_3",
															};
	StringList strGPMapFileNames(strGPMapFileNames_, strGPMapFileNames_ + NUM_GPMAPS); 

	// [1] load/save hit points
	PointXYZCloudPtrList hitPointCloudPtrList;
	loadPointCloud<pcl::PointXYZ>(hitPointCloudPtrList, strObsFileNames, strOriginalIntermediateDataFolder, ".pcd");		// original pcd files which are transformed in global coordinates

	// [2] load sensor positions
	PointXYZVList sensorPositionList;
	loadSensorPositionList(sensorPositionList, strObsFileNames, strInputDataFolder, "_camera_position.txt");
	assert(NUM_OBSERVATIONS == hitPointCloudPtrList.size() && NUM_OBSERVATIONS == sensorPositionList.size());

	// [3] Setting
	double	RESOLUTION = 0.002;
	std::cout << "Resolution: ";
	std::cin >> RESOLUTION;

	// log file
	LogFile logFile;
	std::string strLogFilePath;
	strLogFilePath = strOutputLogFolder + "gpmap.log";
	logFile.open(strLogFilePath);

	// user input
	bool fRunTraining;
	logFile << "Do you wish to train PLSC hyperparameters? (0/1)";
	std::cin >> fRunTraining;
	logFile << fRunTraining << std::endl;

	// max iterations
	int maxIter; // 100
	if(fRunTraining)
	{
		logFile << "Max Iterations? (0 for no training): ";
		std::cin >> maxIter;	logFile << maxIter << std::endl;;
	}

	// PLSC hyperparameters
	bool fUsePredefinedHyperparameters;
	logFile << "Use Predefined PLSC Hyperparameters? (0/1) ";
	std::cin >> fUsePredefinedHyperparameters;
	logFile << fUsePredefinedHyperparameters << std::endl;;
	if(fUsePredefinedHyperparameters)
	{
		// Train By Minimizing Sum of Negative Log Prediction Probability
		//PLSC::alpha	= 2.64143f;		//0.05f;
		//PLSC::beta	= 1.f;			//0.0001f;

		// Train By Minimizing Minimizing Negative Evaluation Accuracy
		PLSC::alpha	= 1.f;
		PLSC::beta	= 0.f;		// a bit less? make it larger for detailed structure?
	}
	else
	{
		std::cout << "PLSC::alpha: ";			std::cin >> PLSC::alpha;		// 0.1f
		std::cout << "PLSC::beta: ";			std::cin >> PLSC::beta;		// 0.1f
	}
	logFile << "PLSC::alpha: "	<< PLSC::alpha	<< std::endl;
	logFile << "PLSC::beta: "	<< PLSC::beta	<< std::endl;

	// training
	if(fRunTraining && maxIter > 0)
	{
		// log file
		strLogFilePath = strOutputLogFolder + strGPMapFileNames[0] + "_training.log";
		logFile.open(strLogFilePath);

		// train criteria
		bool fTrainCriteria;
		logFile << "Train By Minimizing Sum of Negative Log Prediction Probability (0)"
					<< "\nor By Minimizing Negative Evaluation Accuracy (1)? ";
		std::cin >> fTrainCriteria;
		logFile << fTrainCriteria << std::endl;

		// GPMap
		pcl::PointCloud<pcl::PointNormal>::Ptr pPointCloudGPMap;
		loadPointCloud<pcl::PointNormal>(pPointCloudGPMap, strGPMapDataFolder + strGPMapFileNames[0] + ".pcd");

		// convert GPMap to OctoMap
		OctoMap<NO_COLOR> octomap_train(RESOLUTION, *pPointCloudGPMap);

		// Train By Minimizing Sum of Negative Log Prediction Probability
		if(!fTrainCriteria)
		{
			// consider both occupied and empty cells
			bool fConsiderBothOccupiedAndEmpty;
			logFile << "Consider both occupied and empty cells or not? (1/0) ";
			std::cin >> fConsiderBothOccupiedAndEmpty;
			logFile << fConsiderBothOccupiedAndEmpty << std::endl;

			// train
			const float sumNegLogPredProb = octomap_train.trainByMinimizingSumNegLogPredProb(pPointCloudGPMap, 
																														PLSC::alpha, 
																														PLSC::beta, 
																														fConsiderBothOccupiedAndEmpty, 
																														maxIter);
			logFile << "PLSC::alpha: "	<< PLSC::alpha	<< std::endl;
			logFile << "PLSC::beta: "	<< PLSC::beta	<< std::endl;
			logFile << "sumNegLogPredProb = "  << sumNegLogPredProb  << std::endl;
		}

		// Train By Minimizing Minimizing Negative Evaluation Accuracy
		else
		{
			// train
			const float negEvalAccu = octomap_train.trainByMinimizingNegEvalAccu(pPointCloudGPMap,
																										&hitPointCloudPtrList,
																										&sensorPositionList,
																										PLSC::alpha,
																										PLSC::beta, 
																										maxIter);
			logFile << "PLSC::alpha: "	<< PLSC::alpha	<< std::endl;
			logFile << "PLSC::beta: "	<< PLSC::beta	<< std::endl;
			logFile << "negEvalAccu = "  << negEvalAccu  << std::endl;
		}
	}

	// convert GPMap to Octree
	bool fConvertGPMap;
	logFile << "Do you wish to convert GPMap to Octree? (0/1)";
	std::cin >> fConvertGPMap;
	logFile << fConvertGPMap << std::endl;

	if(fConvertGPMap)
	{
		std::stringstream ss;
		ss << strOutputDataFolder0 << "PLSC_alpha_" << PLSC::alpha << "_PLSC_beta_" << PLSC::beta << "/";
		const std::string strOutputDataFolder			(ss.str());
		const std::string strColorOutputDataFolder	(ss.str() + "color/");
		create_directory(strOutputDataFolder);
		create_directory(strColorOutputDataFolder);

		// for each GPMap
		for(size_t i = 0; i < strGPMapFileNames.size(); i++)
		{
			std::cout << "================[ " << strGPMapFileNames[i] << " ]================" << std::endl;

			// log file
			strLogFilePath = strOutputLogFolder + strGPMapFileNames[i] + "_octree.log";
			logFile.open(strLogFilePath);

			// [4-1] GPMap
			pcl::PointCloud<pcl::PointNormal>::Ptr pPointCloudGPMap;
			loadPointCloud<pcl::PointNormal>(pPointCloudGPMap, strGPMapDataFolder + strGPMapFileNames[i] + ".pcd");

			// [4-2] convert GPMap to Octree
			OctoMap<NO_COLOR> octree(RESOLUTION, *pPointCloudGPMap);
			octree.save(strOutputDataFolder + strGPMapFileNames[i]);

			// [4-3] Convert GPMap to ColorOcTree
			// min, max of means and variances
			float minMean, maxMean, minVar, maxVar;
			getMinMaxMeanVarOfOccupiedCells(*pPointCloudGPMap, minMean, maxMean, minVar, maxVar);

			// file name
			std::stringstream ss;
			ss.precision(std::numeric_limits<float>::digits10);
			ss << strColorOutputDataFolder << strGPMapFileNames[i] << "_color_"
				<< "_min_var_" << std::scientific << minVar
				<< "_max_var_" << std::scientific << maxVar;

			// convert GPMap to ColorOcTree
			OctoMap<COLOR> color_octree(RESOLUTION, *pPointCloudGPMap);
			color_octree.save(ss.str());

			// [4-4] Evaluation
			unsigned int num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown;
			octree.evaluate<pcl::PointXYZ, pcl::PointXYZ>(hitPointCloudPtrList, sensorPositionList,
																			num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown);
			logFile << "Number of hit points: "			<< num_points				<< std::endl;
			logFile << "Number of correct voxels: "	<< num_voxels_correct	<< std::endl;
			logFile << "Number of wrong voxels: "		<< num_voxels_wrong		<< std::endl;
			logFile << "Number of unknown voxels: "	<< num_voxels_unknown	<< std::endl;
			logFile << "Correct rate (correct/(correct+wrong)): " << static_cast<float>(num_voxels_correct)/static_cast<float>(num_voxels_correct+num_voxels_wrong) << std::endl;	
			logFile << std::endl << std::endl;
		}
	}

	// [5] Octree with cells having less that the limit max variances
	while(true)
	{
		// thresholds for color
		float maxVarThld; std::cout << "max var thld: ";				std::cin >> maxVarThld;
		float minVarRange; std::cout << "min var color range: ";		std::cin >> minVarRange;

		std::stringstream ss;
		ss << strOutputDataFolder0 << "PLSC_alpha_" << PLSC::alpha << "_PLSC_beta_" << PLSC::beta << "/";
		const std::string strOutputDataFolder1			(ss.str());
		ss << "min_var_" << minVarRange << "_max_var_" << maxVarThld << "/";
		const std::string strOutputDataFolder			(ss.str());
		const std::string strColorOutputDataFolder	(ss.str() + "color/");
		create_directory(strOutputDataFolder1);
		create_directory(strOutputDataFolder);
		create_directory(strColorOutputDataFolder);

		// for each GPMap
		for(size_t i = 0; i < strGPMapFileNames.size(); i++)
		{
			std::cout << "================[ " << strGPMapFileNames[i] << " ]================" << std::endl;

			// log file
			strLogFilePath = strOutputDataFolder + strGPMapFileNames[i] + "_color_octree.log";
			logFile.open(strLogFilePath);

			// [5-1] loading GPMap
			pcl::PointCloud<pcl::PointNormal>::Ptr pPointCloudGPMap;
			loadPointCloud<pcl::PointNormal>(pPointCloudGPMap, strGPMapDataFolder + strGPMapFileNames[i] + ".pcd");

			// [5-2] convert GPMap to Octree
			OctoMap<NO_COLOR> octree(RESOLUTION, *pPointCloudGPMap, maxVarThld);
			octree.save(strOutputDataFolder + strGPMapFileNames[i]);

			// [4-3] Convert GPMap to ColorOcTree
			// min, max of means and variances
			//float minMean, maxMean, minVar, maxVar;
			//getMinMaxMeanVarOfOccupiedCells(*pPointCloudGPMap, minMean, maxMean, minVar, maxVar);

			// file name
			//std::stringstream ss;
			//ss.precision(std::numeric_limits<float>::digits10);
			//ss << strColorOutputDataFolder << strGPMapFileNames[i] << "_min_var_" << std::scientific << minVar;

			// Convert GPMap to ColorOcTree
			OctoMap<COLOR> color_octree(RESOLUTION, *pPointCloudGPMap, minVarRange, maxVarThld, maxVarThld); 
			color_octree.save(strColorOutputDataFolder + strGPMapFileNames[i]);

			logFile << std::endl << std::endl;
		}
	}

	system("pause");
}

#endif