#ifndef _MACRO_GPMAP_HPP_
#define _MACRO_GPMAP_HPP_

// STL
#include <string>
#include <vector>

// GP
#include "gp.h"						// LogFile
using GP::LogFile;

// GPMap
#include "util/data_types.hpp"				// PointNormalCloudPtrList
#include "util/timer.hpp"						// CPU_Times
#include "common/common.hpp"					// getMinMaxPointXYZ
#include "octree/octree_gpmap.hpp"			// OctreeGPMap
#include "octree/octree_container.hpp"		// OctreeGPMapContainer
#include "bcm/bcm.hpp"							// BCM
#include "bcm/bcm_serializable.hpp"			// BCM_Serializable
#include "bcm/gaussian.hpp"					// GaussianDistribution
#include "visualization/cloud_viewer.hpp"	// show

const int RUN_ALL_WITH_TRAINING_ONCE = false;
//const int RUN_ALL_WITH_TRAINING_ONCE = true;

namespace GPMap {

/** @brief Train hyperparameters with all-in-one observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void gpmap_training(const double		BLOCK_SIZE, 
						  const size_t		NUM_CELLS_PER_AXIS,
						  const size_t		MIN_NUM_POINTS_TO_PREDICT,
						  const size_t		MAX_NUM_POINTS_TO_PREDICT,
						  typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	&logHyp,				// hyperparameters
						  const pcl::PointCloud<pcl::PointNormal>::ConstPtr	&pAllPointNormalCloud,	// observations
						  const float														gap,							// gap
						  const int															maxIter,						// number of iterations for training before update, 100
						  const int															numRandomBlocks)			// number of randomly selected blocks (<=0 for all), 100
{
	// log file
	LogFile logFile;

	// gpmap with Gaussian Distribution leaf nodes
	pcl::PointXYZ min_pt, max_pt;
	getMinMaxPointXYZ<pcl::PointNormal>(*pAllPointNormalCloud, min_pt, max_pt);

	// gpmap
	// the contrainter type is meaningless for training
	//typedef OctreeGPMapContainer<GaussianDistribution>
	typedef OctreeGPMapContainer<BCM>	LeafT;
	typedef OctreeGPMap<MeanFunc, CovFunc, LikFunc, InfMethod, LeafT> OctreeGPMapT;
	const bool	FLAG_INDEPENDENT_TEST_POSITIONS		= true; // meaningless for training, this is for predicting test positions
	const bool	FLAG_DO_NOT_RAMDOMLY_SAMPLE_POINTS	= false;
	const bool	FLAG_DO_NOT_DUPLICATE_POINTS			= false;
	OctreeGPMapT gpmap(BLOCK_SIZE, 
							 NUM_CELLS_PER_AXIS, 
							 MIN_NUM_POINTS_TO_PREDICT, 
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS,
							 FLAG_DO_NOT_RAMDOMLY_SAMPLE_POINTS,
							 FLAG_DO_NOT_DUPLICATE_POINTS);

	// set bounding box
	logFile << "[0] Set bounding box" << std::endl << std::endl;
	gpmap.defineBoundingBox(min_pt, max_pt);

	// set input cloud
	logFile << "[1] Set input cloud" << std::endl << std::endl;
	gpmap.setInputCloud(pAllPointNormalCloud, gap);

	// add points from the input cloud
	logFile << "[2] Add points from the input cloud" << std::endl;
	logFile << gpmap.addPointsFromInputCloud() << std::endl << std::endl;

	// train
	logFile << "[4] Learning hyperparameters"		<< std::endl;
	logFile << "- Train - Max Iterations: "		<< maxIter				<< std::endl;
	logFile << "- Train - Num Random Blocks: "	<< numRandomBlocks	<< std::endl;
	CPU_Timer timer;
	GP::DlibScalar nlZ = gpmap.train(logHyp, maxIter, numRandomBlocks);
	CPU_Times t_training = timer.elapsed();
	logFile << "- nlZ: " << nlZ << std::endl << std::endl;
	logFile << t_training << std::endl << std::endl;

	// trained hyperparameters
	int j = 0;
	for(int i = 0; i < logHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(logHyp.mean(i)) << std::endl;
	for(int i = 0; i < logHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(logHyp.cov(i))  << std::endl;
	for(int i = 0; i < logHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(logHyp.lik(i))  << std::endl;
}

/** @brief	Building a GPMap with all-in-one observations
  * @note	Note that there exist no max num points limit for prediction.
  *			So, sampling the observations would be a good idea for memory issue. */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void gpmap_batch(const double		BLOCK_SIZE, 
					  const size_t		NUM_CELLS_PER_AXIS,
					  const size_t		MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
					  size_t				MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
					  const size_t		FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
					  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	&logHyp,	// hyperparameters
					  const pcl::PointCloud<pcl::PointNormal>::ConstPtr		&pAllPointNormalCloud,							// observations
					  const float															gap,													// gap
					  const int																maxIterBeforeUpdate,					// number of iterations for training before update
					  const std::string													&strPCDFilePathWithoutExtension)	// save file path
{
	// log file
	LogFile logFile;

	// times
	CPU_Times	t_update_training;
	CPU_Times	t_update_predict;
	CPU_Times	t_update_combine;

	// get bounding box
	pcl::PointXYZ min_pt, max_pt;
	getMinMaxPointXYZ<pcl::PointNormal>(*pAllPointNormalCloud, min_pt, max_pt);

	// gpmap with Gaussian Distribution leaf nodes
	if(!FLAG_RAMDOMLY_SAMPLE_POINTS)	MAX_NUM_POINTS_TO_PREDICT = 0;	// no data partitioning
	const float	FLAG_INDEPENDENT_TEST_POSITIONS = true;					// predict means and variances
	typedef OctreeGPMapContainer<GaussianDistribution>	LeafT;
	typedef OctreeGPMap<MeanFunc, CovFunc, LikFunc, InfMethod, LeafT> OctreeGPMapT;
	OctreeGPMapT gpmap(BLOCK_SIZE, 
							 NUM_CELLS_PER_AXIS, 
							 MIN_NUM_POINTS_TO_PREDICT, 
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS,
							 FLAG_RAMDOMLY_SAMPLE_POINTS);

	// set bounding box
	logFile << "[0] Set bounding box" << std::endl << std::endl;
	gpmap.defineBoundingBox(min_pt, max_pt);

	// set input cloud
	logFile << "[1] Set input cloud" << std::endl << std::endl;
	gpmap.setInputCloud(pAllPointNormalCloud, gap);

	// add points from the input cloud
	logFile << "[2] Add points from the input cloud" << std::endl;
	logFile << gpmap.addPointsFromInputCloud() << std::endl << std::endl;

	// update using GPR
	logFile << "[3] Update using GPR" << std::endl;
	gpmap.update(logHyp, maxIterBeforeUpdate, t_update_training, t_update_predict, t_update_combine);
	logFile << "- Training hyp: " << t_update_training << std::endl << std::endl;
	logFile << "- Predict GPR:  " << t_update_predict  << std::endl << std::endl;
	logFile << "- Update BCM:   " << t_update_combine  << std::endl << std::endl;

	// save
	logFile << "[4] Save" << std::endl << std::endl;
	gpmap.saveAsPointCloud(strPCDFilePathWithoutExtension);
}

/** @brief	Building a GPMap with all-in-one observations with possible incremental update
  * @note	Note that if the number of points exceed the limit, incremental update is activated */
template<typename BCM_T,
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void gpmap_incremental(const double		BLOCK_SIZE, 
							  const size_t		NUM_CELLS_PER_AXIS,
							  const size_t		MIN_NUM_POINTS_TO_PREDICT,
							  const size_t		MAX_NUM_POINTS_TO_PREDICT,			// if 0, pure batch (memory issue). Otherwise, incrementally updated with subsets
							  const bool		FLAG_INDEPENDENT_TEST_POSITIONS, // iBCM or BCM
							  const size_t		FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
							  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	&logHyp,	// hyperparameters
							  const pcl::PointCloud<pcl::PointNormal>::ConstPtr		&pAllPointNormalCloud,							// observations
							  const float															gap,													// gap
							  const int																maxIterBeforeUpdate,					// number of iterations for training before update
							  const std::string													&strPCDFilePathWithoutExtension)	// save file path
{
	// log file
	LogFile logFile;

	// times
	CPU_Times	t_update_training;
	CPU_Times	t_update_predict;
	CPU_Times	t_update_combine;

	// get bounding box
	pcl::PointXYZ min_pt, max_pt;
	getMinMaxPointXYZ<pcl::PointNormal>(*pAllPointNormalCloud, min_pt, max_pt);

	// gpmap with BCM leaf nodes
	typedef OctreeGPMapContainer<BCM_T>	LeafT;
	typedef OctreeGPMap<MeanFunc, CovFunc, LikFunc, InfMethod, LeafT> OctreeGPMapT;
	OctreeGPMapT gpmap(BLOCK_SIZE, 
							 NUM_CELLS_PER_AXIS, 
							 MIN_NUM_POINTS_TO_PREDICT, 
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS,
							 FLAG_RAMDOMLY_SAMPLE_POINTS);

	// set bounding box
	logFile << "[0] Set bounding box" << std::endl << std::endl;
	gpmap.defineBoundingBox(min_pt, max_pt);

	// set input cloud
	logFile << "[1] Set input cloud" << std::endl << std::endl;
	gpmap.setInputCloud(pAllPointNormalCloud, gap);

	// add points from the input cloud
	logFile << "[2] Add points from the input cloud" << std::endl;
	logFile << gpmap.addPointsFromInputCloud() << std::endl << std::endl;

	// update using GPR
	logFile << "[3] Update using GPR" << std::endl;
	gpmap.update(logHyp, maxIterBeforeUpdate, t_update_training, t_update_predict, t_update_combine);
	logFile << "- Training hyp: " << t_update_training << std::endl << std::endl;
	logFile << "- Predict GPR:  " << t_update_predict  << std::endl << std::endl;
	logFile << "- Update BCM:   " << t_update_combine  << std::endl << std::endl;

	// save
	logFile << "[4] Save" << std::endl << std::endl;
	gpmap.saveAsPointCloud(strPCDFilePathWithoutExtension);
}

/** @brief Building a GPMap with sequential observations
  *			incrementally if the number of points exceed the limit */
template<typename BCM_T,
			template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void gpmap_incremental(const double		BLOCK_SIZE,							// block size
							  const size_t		NUM_CELLS_PER_AXIS,				// number of cells per each axie
							  const size_t		MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
							  const size_t		MAX_NUM_POINTS_TO_PREDICT,		// if 0, pure incremental (memory issue). Otherwise, incrementally updated with subsets
							  const bool		FLAG_INDEPENDENT_TEST_POSITIONS,	// independent BCM or BCM
							  const size_t		FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
							  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	&logHyp,				// hyperparameters
							  const std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>	&pointNormalCloudList,	// observations
							  const float																gap,							// gap
							  const int																	maxIterBeforeUpdate,					// number of iterations for training before update
							  const std::string														&strPCDFilePathWithoutExtension)	// save file path
{
	// log file
	LogFile logFile;

	// times
	CPU_Times	t_elaped;
	CPU_Times	t_update_training;
	CPU_Times	t_update_predict;
	CPU_Times	t_update_combine;

	CPU_Times	t_add_point_total;
	CPU_Times	t_update_training_total;
	CPU_Times	t_update_predict_total;
	CPU_Times	t_update_combine_total;
	
	t_add_point_total.clear();
	t_update_training_total.clear();
	t_update_predict_total.clear();
	t_update_combine_total.clear();

	// get bounding box
	pcl::PointXYZ min_pt, max_pt;
	getMinMaxPointXYZ<pcl::PointNormal>(pointNormalCloudList, min_pt, max_pt);

	// gpmap with BCM leaf nodes
	typedef OctreeGPMapContainer<BCM_T>	LeafT;
	typedef OctreeGPMap<MeanFunc, CovFunc, LikFunc, InfMethod, LeafT> OctreeGPMapT;
	OctreeGPMapT gpmap(BLOCK_SIZE,
							 NUM_CELLS_PER_AXIS,
							 MIN_NUM_POINTS_TO_PREDICT,
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS,
							 FLAG_RAMDOMLY_SAMPLE_POINTS);

	// set bounding box
	logFile << "[0] Set bounding box" << std::endl << std::endl;
	gpmap.defineBoundingBox(min_pt, max_pt);

	// for each observation
	for(size_t i = 0; i < pointNormalCloudList.size(); i++)
	{
		logFile << "==== Updating the GPMap with the point cloud #" << i << " ====" << std::endl;

		// set input cloud
		logFile << "[1] Set input cloud" << std::endl << std::endl;
		gpmap.setInputCloud(pointNormalCloudList[i], gap);

		// add points from the input cloud
		logFile << "[2] Add points from the input cloud" << std::endl;
		t_elaped = gpmap.addPointsFromInputCloud();
		logFile << t_elaped << std::endl << std::endl;
		t_add_point_total += t_elaped;

		// update using GPR
		logFile << "[3] Update using GPR" << std::endl;
		gpmap.update(logHyp, maxIterBeforeUpdate, t_update_training, t_update_predict, t_update_combine);
		logFile << "- Training hyp: " << t_update_training << std::endl << std::endl;
		logFile << "- Predict GPR:  " << t_update_predict  << std::endl << std::endl;
		logFile << "- Update BCM:   " << t_update_combine   << std::endl << std::endl;
		t_update_training_total		+= t_update_training;
		t_update_predict_total		+= t_update_predict;
		t_update_combine_total		+= t_update_combine;

		// save
		logFile << "[4] Save" << std::endl << std::endl;
		std::stringstream ss;
		ss << strPCDFilePathWithoutExtension << "_upto_" << i;
		gpmap.saveAsPointCloud(ss.str());

		// last
		//if(i == pointNormalCloudList.size() - 1) gpmap.saveAsPointCloud(strPCDFilePathWithoutExtension);
	}

	// total time
	logFile << "============= Total Time =============" << std::endl;
	logFile << "- Total"					<< std::endl << t_add_point_total + t_update_training_total + t_update_predict_total + t_update_combine_total << std::endl << std::endl;
	logFile << "- Total: Add point"	<< std::endl << t_add_point_total << std::endl << std::endl;
	logFile << "- Total: Update"		<< std::endl << t_update_training_total + t_update_predict_total + t_update_combine_total << std::endl << std::endl;
	logFile << "- Total: Update - Training hyp: " << t_update_training_total << std::endl << std::endl;
	logFile << "- Total: Update - Predict GPR:  " << t_update_predict_total  << std::endl << std::endl;
	logFile << "- Total: Update - Update BCM:   " << t_update_combine_total  << std::endl << std::endl;
}

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_with_all_in_one_observations(const double				BLOCK_SIZE,								// block size
																		  const size_t				NUM_CELLS_PER_AXIS,					// number of cells per each axie
																		  const size_t				MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																		  const size_t				MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																		  const std::string		&strAllInOneObsFileName,			// all-in-one observations input file name
																		  const std::string		&strAllInOneObsFileNamePrefix,	// input file name prefix
																		  const std::string		&strAllInOneObsFileNameSuffix,	// input file name suffix
																		  const float				gap,										// gap
																		  const std::string		&strOutputFileName,					// output data file prefix
																		  const std::string		&strLogFolder,							// log folder
																		  typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	&logHyp)	// hyperparameters
{
	// log file
	const std::string strFileName = strOutputFileName + "(all_in_one)_training";
	LogFile logFile(strLogFolder + strFileName + ".log");	

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strAllInOneObsFileName, strAllInOneObsFileNamePrefix, strAllInOneObsFileNameSuffix);
	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations For Training", pAllInOneObs, 0.005);

	// training
	while(true)
	{
		// continue?
		bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
		if(!fContinue)
		{
			std::cout << "Do you wish to train hyperparameters? (0/1) "; 
			std::cin  >> fContinue;
			if(!fContinue) break;
		}

		// max iterations
		int maxIter;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			maxIter = 50;
		}
		else
		{
			logFile  << "\tMax iterations? (0 for no training): ";
			std::cin >> maxIter;
		}
		logFile  << maxIter << std::endl;;
		if(maxIter <= 0) break;

		// number of random blocks
		size_t numRandomBlocks;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			numRandomBlocks = 0;
		}
		else
		{
			logFile  << "\tNumber of random blocks? (0 for all) ";	
			std::cin >> numRandomBlocks;
		}
		logFile  << numRandomBlocks << std::endl;;

		// hyperparameters
		bool fUsePredefinedHyperparameters;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			fUsePredefinedHyperparameters = true;
		}
		else
		{
			logFile  << "\tUse Predefined Hyperparameters? (0/1) ";
			std::cin >> fUsePredefinedHyperparameters;
		}
		logFile  << fUsePredefinedHyperparameters << std::endl;;
		if(!fUsePredefinedHyperparameters)
		{
			float hyp;
			for(int i = 0; i < logHyp.mean.size(); i++) { std::cout << "\t\thyp.mean(" << i << ") = "; std::cin >> hyp; logHyp.mean(i) = log(hyp); }
			for(int i = 0; i < logHyp.cov.size();  i++) { std::cout << "\t\thyp.cov(" << i << ") = ";  std::cin >> hyp; logHyp.cov(i)  = log(hyp); }
			for(int i = 0; i < logHyp.lik.size();  i++) { std::cout << "\t\thyp.lik(" << i << ") = ";  std::cin >> hyp; logHyp.lik(i)  = log(hyp); }
		}

		// log hyperparameters before training
		logFile << "Hyperparameters before training" << std::endl;
		for(int i = 0; i < logHyp.mean.size(); i++) { logFile  << "\thyp.mean(" << i << ") = " << exp(logHyp.mean(i)) << std::endl; }
		for(int i = 0; i < logHyp.cov.size(); i++)  { logFile  << "\thyp.cov("  << i << ") = " << exp(logHyp.cov(i))  << std::endl; }
		for(int i = 0; i < logHyp.lik.size(); i++)  { logFile  << "\thyp.lik("  << i << ") = " << exp(logHyp.lik(i))  << std::endl; }

		// training
		gpmap_training<MeanFunc, 
							CovFunc, 
							LikFunc, 
							InfMethod>(BLOCK_SIZE,						// block size
										  NUM_CELLS_PER_AXIS,			// number of cells per each axie
										  MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
										  MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
										  logHyp,							// hyperparameters
										  pAllInOneObs,					// observations
										  gap,								// gap
										  maxIter,							// number of iterations for training before update
										  numRandomBlocks);				// number of randomly selected blocks

		// log hyperparameters after training
		logFile << "Hyperparameters after training" << std::endl;
		for(int i = 0; i < logHyp.mean.size(); i++) { logFile  << "\thyp.mean(" << i << ") = " << exp(logHyp.mean(i)) << std::endl; }
		for(int i = 0; i < logHyp.cov.size(); i++)  { logFile  << "\thyp.cov("  << i << ") = " << exp(logHyp.cov(i))  << std::endl; }
		for(int i = 0; i < logHyp.lik.size(); i++)  { logFile  << "\thyp.lik("  << i << ") = " << exp(logHyp.lik(i))  << std::endl; }

		// next
		if(RUN_ALL_WITH_TRAINING_ONCE) break;
	}
}

/** @brief	Build GPMaps with Sampled All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_with_sampled_all_in_one_observations(const double				BLOCK_SIZE,										// block size
																		 const size_t				NUM_CELLS_PER_AXIS,							// number of cells per each axie
																		 const size_t				MIN_NUM_POINTS_TO_PREDICT,					// min number of points to predict
																		 const size_t				MAX_NUM_POINTS_TO_PREDICT,					// max number of points to predict
																		 const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																										&logHyp,											// hyperparameters
																		 const std::string		&strAllInOneObsFileName,					// all-in-one observations input file name
																		 const std::string		&strAllInOneObsFileNamePrefix,			// input file name prefix
																		 const std::string		&strAllInOneObsFileNameSuffix,			// input file name suffix
																		 const float				gap,												// gap
																		 const int					maxIterBeforeUpdate,							// number of iterations for training before update
																		 const std::string		&strOutputFolder,								// output data folder
																		 const std::string		&strOutputFileName,							// output data file prefix
																		 const std::string		&strLogFolder)									// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		std::cout << "Do you wish to build GPMaps with randomly sampled all-in-one observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strFileName;
	LogFile logFile;

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strAllInOneObsFileName, strAllInOneObsFileNamePrefix, strAllInOneObsFileNameSuffix);
	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations for Prediction", pAllInOneObs, 0.005);

	// GPMap - Sampled All-in-One Observations - Batch
	strFileName = strOutputFileName + "(all_samples)_Batch";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_RAMDOMLY_SAMPLE_POINTS = true;
	gpmap_batch<MeanFunc, 
					CovFunc, 
					LikFunc, 
					InfMethod>(BLOCK_SIZE,								// block size
									NUM_CELLS_PER_AXIS,					// number of cells per each axie
									MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
									MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
									FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
									logHyp,									// hyperparameters
									pAllInOneObs,							// observations
									gap,										// gap
									maxIterBeforeUpdate,					// number of iterations for training before update
									strOutputFolder + strFileName);	// save file path
}

/** @brief	Build GPMaps with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_with_all_in_one_observations(const double				BLOCK_SIZE,										// block size
															  const size_t				NUM_CELLS_PER_AXIS,							// number of cells per each axie
															  const size_t				MIN_NUM_POINTS_TO_PREDICT,					// min number of points to predict
															  const size_t				MAX_NUM_POINTS_TO_PREDICT,					// max number of points to predict
															  const size_t				FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
															  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																							&logHyp,											// hyperparameters
															  const std::string		&strAllInOneObsFileName,					// all-in-one observations input file name
															  const std::string		&strAllInOneObsFileNamePrefix,			// input file name prefix
															  const std::string		&strAllInOneObsFileNameSuffix,			// input file name suffix
															  const float				gap,												// gap
															  const int					maxIterBeforeUpdate,							// number of iterations for training before update
															  const std::string		&strOutputFolder,								// output data folder
															  const std::string		&strOutputFileName,							// output data file prefix
															  const std::string		&strLogFolder)									// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		if(FLAG_RAMDOMLY_SAMPLE_POINTS)	std::cout << "Do you wish to build GPMaps with randomly sampled all-in-one observations? (0/1) ";
		else										std::cout << "Do you wish to build GPMaps with all-in-one observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strFileName;
	LogFile logFile;

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strAllInOneObsFileName, strAllInOneObsFileNamePrefix, strAllInOneObsFileNameSuffix);
	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations for Prediction", pAllInOneObs, 0.005);

	// [1] GPMap - All-in-One Observations - Incremental Update (iBCM)
	if(FLAG_RAMDOMLY_SAMPLE_POINTS)	strFileName = strOutputFileName + "(all_samples)_iBCM";
	else										strFileName = strOutputFileName + "(all)_iBCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_INDEPENDENT_TEST_POSITIONS = true;
	gpmap_incremental<BCM,
							MeanFunc, 
							CovFunc, 
							LikFunc, 
							InfMethod>(BLOCK_SIZE,								// block size
										  NUM_CELLS_PER_AXIS,					// number of cells per each axie
										  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
										  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
										  FLAG_INDEPENDENT_TEST_POSITIONS,	// predict means and variances
										  FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
										  logHyp,									// hyperparameters
										  pAllInOneObs,							// observations
										  gap,										// gap for free points
										  maxIterBeforeUpdate,					// number of iterations for training before update
										  strOutputFolder + strFileName);	// save file path

	// [2] GPMap - All-in-One Observations - Incremental Update (BCM)
	//if(FLAG_RAMDOMLY_SAMPLE_POINTS)	strFileName = strOutputFileName + "(all_samples)_BCM";
	//else										strFileName = strOutputFileName + "(all)_BCM";
	//logFile.open(strLogFolder + strFileName + ".log");	
	//const bool FLAG_DEPENDENT_TEST_POSITIONS = false;
	//gpmap_incremental<BCM_Serializable,
	//						MeanFunc, 
	//						CovFunc, 
	//						LikFunc, 
	//						InfMethod>(BLOCK_SIZE,								// block size
	//									  NUM_CELLS_PER_AXIS,					// number of cells per each axie
	//									  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
	//									  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
	//									  FLAG_DEPENDENT_TEST_POSITIONS,		// predict means and covariances
	//									  FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
	//									  logHyp,									// hyperparameters
	//									  pAllInOneObs,							// observations
	//									  gap,										// gap for free points
	//									  maxIterBeforeUpdate,					// number of iterations for training before update
	//									  strOutputFolder + strFileName);	// save file path
}

/** @brief	Build GPMaps with Sequential [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_with_sequential_observations(const double				BLOCK_SIZE,										// block size
															  const size_t				NUM_CELLS_PER_AXIS,							// number of cells per each axie
															  const size_t				MIN_NUM_POINTS_TO_PREDICT,					// min number of points to predict
															  const size_t				MAX_NUM_POINTS_TO_PREDICT,					// max number of points to predict
															  const size_t				FLAG_RAMDOMLY_SAMPLE_POINTS,				// randomly sample points in each leaf node
															  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																												&logHyp,										// hyperparameters
															  const std::vector<std::string>			&strSequentialObsFileNameList,		// sequential observations input file name
															  const std::string							&strSequentialObsFileNamePrefix,		// input file name prefix
															  const std::string							&strSequentialObsFileNameSuffix,		// input file name suffix
															  const float									gap,											// gap
															  const int										maxIterBeforeUpdate,						// number of iterations for training before update
															  const std::string							&strOutputFolder,							// output data folder
															  const std::string							&strOutputFileName,						// output data file prefix
															  const std::string							&strLogFolder)								// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		if(FLAG_RAMDOMLY_SAMPLE_POINTS)	std::cout << "Do you wish to build GPMaps with randomly sampled sequential observations? (0/1) ";
		else										std::cout << "Do you wish to build GPMaps with sequential observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strFileName;
	LogFile logFile;

	// loading sequential data
	PointNormalCloudPtrList ObsCloudPtrList;
	loadPointCloud<pcl::PointNormal>(ObsCloudPtrList, strSequentialObsFileNameList, strSequentialObsFileNamePrefix, strSequentialObsFileNameSuffix);
	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("Sequential Observations for Prediction", ObsCloudPtrList, 0.005);

	// [1] GPMap - Sequential Observations - Incremental Update (iBCM)
	if(FLAG_RAMDOMLY_SAMPLE_POINTS)	strFileName = strOutputFileName + "(seq_samples)_iBCM";
	else										strFileName = strOutputFileName + "(seq)_iBCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_INDEPENDENT_TEST_POSITIONS = true;
	gpmap_incremental<BCM,
							MeanFunc, 
							CovFunc, 
							LikFunc, 
							InfMethod>(BLOCK_SIZE,								// block size
										  NUM_CELLS_PER_AXIS,					// number of cells per each axie
										  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
										  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
										  FLAG_INDEPENDENT_TEST_POSITIONS,	// predict means and variances
										  FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
										  logHyp,									// hyperparameters
										  ObsCloudPtrList,						// observations
										  gap,										// gap for free points
										  maxIterBeforeUpdate,					// number of iterations for training before update
										  strOutputFolder + strFileName);	// save file path

	// [2] GPMap - Sequential Observations - Incremental Update (BCM)
	if(FLAG_RAMDOMLY_SAMPLE_POINTS)	strFileName = strOutputFileName + "(seq_samples)_BCM";
	else										strFileName = strOutputFileName + "(seq)_BCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_DEPENDENT_TEST_POSITIONS = false;
	gpmap_incremental<BCM_Serializable,
							MeanFunc, 
							CovFunc, 
							LikFunc, 
							InfMethod>(BLOCK_SIZE,								// block size
										  NUM_CELLS_PER_AXIS,					// number of cells per each axie
										  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
										  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
										  FLAG_DEPENDENT_TEST_POSITIONS,		// predict means and covariances
										  FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
										  logHyp,									// hyperparameters
										  ObsCloudPtrList,						// observations
										  gap,										// gap for free points
										  maxIterBeforeUpdate,					// number of iterations for training before update
										  strOutputFolder + strFileName);	// save file path
}

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps with
  *				- Sampled All-in-One	[Function/Derivative/All] Observations
  *				- All-in-One			[Function/Derivative/All] Observations
  *				- Sequential			[Function/Derivative/All] Observations
  */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_and_build_gpmaps(const double				BLOCK_SIZE,												// block size
														  const size_t				NUM_CELLS_PER_AXIS,									// number of cells per each axie
														  const size_t				MIN_NUM_POINTS_TO_PREDICT,							// min number of points to predict
														  const size_t				MAX_NUM_POINTS_TO_PREDICT,							// max number of points to predict
														  typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																										&logHyp,									// hyperparameters
														  const std::vector<std::string>		&strSequentialObsFileNameList,	// sequential observations input file name
														  const std::string						&strAllInOneObsFileName,			// all-in-one observations input file name
														  const std::string						&strObsFileNamePrefix,				// input file name prefix
														  const std::string						&strObsFileNameSuffix,				// input file name suffix
														  const float								gap,										// gap
														  const int									maxIterBeforeUpdate,					// number of iterations for training before update
														  const std::string						&strOutputFolder,						// output data folder
														  const std::string						&strOutputFileName,					// output data file prefix
														  const std::string						&strLogFolder)							// log folder
{
	// [1-0] Training
	train_hyperparameters_with_all_in_one_observations<MeanFunc,
																		CovFunc, 
																		LikFunc, 
																		InfMethod>(BLOCK_SIZE,							// block size
																					  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																					  strAllInOneObsFileName,			// **all-in-one observation input file name**
																					  strObsFileNamePrefix,				// input file name prefix
																					  strObsFileNameSuffix,				// input file name suffix
																					  gap,									// gap
																					  strOutputFileName,					// output file name prefix
																					  strLogFolder,						// log folder
																					  logHyp);								// hyperparameters

	//bool FLAG_RAMDOMLY_SAMPLE_POINTS = true;
	bool FLAG_RAMDOMLY_SAMPLE_POINTS = false;
	//for(size_t i = 0; i < 2; i++)
	//{
		// [1-1] Sampled All-in-One Observations
		if(FLAG_RAMDOMLY_SAMPLE_POINTS)
		{
			build_gpmaps_with_sampled_all_in_one_observations<MeanFunc,
																			  CovFunc, 
																			  LikFunc, 
																			  InfMethod>(BLOCK_SIZE,							// block size
																							 NUM_CELLS_PER_AXIS,				// number of cells per each axie
																							 MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																							 MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																							 logHyp,								// hyperparameters
																							 strAllInOneObsFileName,			// **all-in-one observation file name**
																							 strObsFileNamePrefix,				// file name prefix
																							 strObsFileNameSuffix,				// file name suffix
																							 gap,									// gap
																							 maxIterBeforeUpdate,				// number of iterations for training before update
																							 strOutputFolder,					// output data folder
																							 strOutputFileName,					// output file name prefix
																							 strLogFolder);						// log folder
		}

		//// [1-2] Sequential Observations
		//build_gpmaps_with_sequential_observations<MeanFunc,
		//														CovFunc, 
		//														LikFunc, 
		//														InfMethod>(BLOCK_SIZE,							// block size
		//																	  NUM_CELLS_PER_AXIS,				// number of cells per each axie
		//																	  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
		//																	  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
		//																	  FLAG_RAMDOMLY_SAMPLE_POINTS,		// randomly sample points in each leaf node
		//																	  logHyp,								// hyperparameters
		//																	  strSequentialObsFileNameList,	// **sequential observation file names**
		//																	  strObsFileNamePrefix,				// file name prefix
		//																	  strObsFileNameSuffix,				// file name suffix
		//																	  gap,									// gap
		//																	  maxIterBeforeUpdate,				// number of iterations for training before update
		//																	  strOutputFolder,					// output data folder
		//																	  strOutputFileName,					// output file name prefix
		//																	  strLogFolder);						// log folder	

		// [1-3] All-in-One Observations
		build_gpmaps_with_all_in_one_observations<MeanFunc,
																CovFunc, 
																LikFunc, 
																InfMethod>(BLOCK_SIZE,							// block size
																			  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																			  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																			  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																			  FLAG_RAMDOMLY_SAMPLE_POINTS,	// randomly sample points in each leaf node
																			  logHyp,								// hyperparameters
																			  strAllInOneObsFileName,			// **all-in-one observation file name**
																			  strObsFileNamePrefix,				// file name prefix
																			  strObsFileNameSuffix,				// file name suffix
																			  gap,									// gap
																			  maxIterBeforeUpdate,				// number of iterations for training before update
																			  strOutputFolder,					// output data folder
																			  strOutputFileName,					// output file name prefix
																			  strLogFolder);						// log folder

		// next
		//FLAG_RAMDOMLY_SAMPLE_POINTS = !FLAG_RAMDOMLY_SAMPLE_POINTS;
	//}
}

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps with
  *				- Sampled All-in-One	[Function/Derivative/All] Observations
  *				- All-in-One			[Function/Derivative/All] Observations
  *				- Sequential			[Function/Derivative/All] Observations
  */
template<template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_and_build_gpmaps_using_MeanGlobalGP(const double				BLOCK_SIZE,													// block size
																					const size_t				NUM_CELLS_PER_AXIS,										// number of cells per each axie
																					const size_t				MIN_NUM_POINTS_TO_PREDICT,								// min number of points to predict
																					const size_t				MAX_NUM_POINTS_TO_PREDICT,								// max number of points to predict
																					typename GP::MeanGlobalGP<float>::GlobalHyp	
																																	&logGlobalHyp,							// global hyperparameters
																					typename GP::GaussianProcess<float, GP::MeanGlobalGP, CovFunc, LikFunc, InfMethod>::Hyp	
																																	&logLocalHyp,							// local hyperparameters
																					const std::vector<std::string>		&strSequentialObsFileNameList,	// sequential observations input file name
																					const std::string							&strAllInOneObsFileName,			// all-in-one observations input file name
																					const std::string							&strObsFileNamePrefix,				// input file name prefix
																					const std::string							&strGlobalObsFileNameSuffix,		// global input file name suffix
																					const std::string							&strObsFileNameSuffix,				// input file name suffix
																					const float									gap,										// gap
																					const int									maxIterBeforeUpdate,					// number of iterations for training before update
																					const std::string							&strOutputFolder,						// output data folder
																					const std::string							&strOutputFileName,					// output data file prefix
																					const std::string							&strLogFolder)							// log folder
{
	// [0] Training Global GP
	// log file
	const std::string strFileName = strOutputFileName + "_MeanGlobalGP";
	LogFile logFile(strLogFolder + strFileName + ".log");	

	// loading all-in-one data
	PointNormalCloudPtr pGlobalAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pGlobalAllInOneObs, strAllInOneObsFileName, strObsFileNamePrefix, strGlobalObsFileNameSuffix);
	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("Global All-in-one Observations for Training and Prediction", pGlobalAllInOneObs, 0.005);

	// indices
	const int N = pGlobalAllInOneObs->points.size();
	std::vector<int> indices(N);
	std::generate(indices.begin(), indices.end(), UniqueNonZeroInteger());

	// training data
	MatrixPtr pX, pXd; VectorPtr pYYd;
	generateTrainingData(pGlobalAllInOneObs, indices, gap, pX, pXd, pYYd);	
	GP::DerivativeTrainingData<float> globalTrainingData;
	globalTrainingData.set(pX, pXd, pYYd);
	//GP::TrainingData<float> globalTrainingData;
	//globalTrainingData.set(pX, pYYd);

	// training and set
	while(true)
	{
		// continue?
		bool fTrainGlobalHyp = RUN_ALL_WITH_TRAINING_ONCE;
		if(!fTrainGlobalHyp)
		{
			std::cout << "Would you like to train the global hyperparameters? (0/1) ";
			std::cin >> fTrainGlobalHyp;
		}

		// max iterations
		int maxIter = 0;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			maxIter = 50;
		}
		else if(fTrainGlobalHyp)
		{
			std::cout << "\tMax iterations? (0 for no training): ";
			std::cin >> maxIter;
		}

		// hyperparameters
		bool fUsePredefinedHyperparameters;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			fUsePredefinedHyperparameters = true;
		}
		else
		{
			logFile  << "\tUse Predefined Hyperparameters? (0/1) ";
			std::cin >> fUsePredefinedHyperparameters;
		}
		if(!fUsePredefinedHyperparameters)
		{
			float hyp;
			for(int i = 0; i < logLocalHyp.mean.size(); i++) { std::cout << "\t\thyp.mean(" << i << ") = "; std::cin >> hyp; logLocalHyp.mean(i) = log(hyp); }
			for(int i = 0; i < logLocalHyp.cov.size();  i++) { std::cout << "\t\thyp.cov(" << i << ") = ";  std::cin >> hyp; logLocalHyp.cov(i)  = log(hyp); }
			for(int i = 0; i < logLocalHyp.lik.size();  i++) { std::cout << "\t\thyp.lik(" << i << ") = ";  std::cin >> hyp; logLocalHyp.lik(i)  = log(hyp); }
		}

		// set
		GP::MeanGlobalGP<float>::set(logGlobalHyp, globalTrainingData, fTrainGlobalHyp, maxIter);
		if(!fTrainGlobalHyp || maxIter <= 0) break;
	}

	// do the same thing
	train_hyperparameters_and_build_gpmaps<GP::MeanGlobalGP, CovFunc, LikFunc, InfMethod>(BLOCK_SIZE,							// block size
																													  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																													  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																													  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																													  logLocalHyp,							// hyperparameters
																													  strSequentialObsFileNameList,	// sequential observations input file name
																													  strAllInOneObsFileName,			// all-in-one observations input file name
																													  strObsFileNamePrefix,				// input file name prefix
																													  strObsFileNameSuffix,				// input file name suffix
																													  gap,									// gap
																													  maxIterBeforeUpdate,				// number of iterations for training before update
																													  strOutputFolder,					// output data folder
																													  strOutputFileName,					// output data file prefix
																													  strLogFolder);						// log folder

}

}
#endif