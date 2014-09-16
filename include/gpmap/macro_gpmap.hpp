#ifndef _MACRO_GPMAP_HPP_
#define _MACRO_GPMAP_HPP_

// GPMap
#include "macro_gpmap_basics.hpp"
#include "gpmap/octree_viewer.hpp"			// OctreeViewer
#include "common/common.hpp"					// getMinMaxPointXYZ
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList

const int RUN_ALL_WITH_TRAINING_ONCE = false;
//const int RUN_ALL_WITH_TRAINING_ONCE = true;

namespace GPMap {

/** @brief	Show observations with an octree structure */
void show(const double				BLOCK_SIZE,						// block size
			 const std::string		&strFileName,					// all-in-one observations input file name
			 const std::string		&strFileNamePrefix,			// input file name prefix
			 const std::string		&strFileNameSuffix)			// input file name suffix
{
	// Function Observations (Hit Points + Unit Ray Back Vectors) - All - Down Sampling
	PointNormalCloudPtr pAllObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllObs, strFileName, strFileNamePrefix, strFileNameSuffix);

	// get bounding box
	pcl::PointXYZ min_pt, max_pt;
	getMinMaxPointXYZ<pcl::PointNormal>(*pAllObs, min_pt, max_pt);

	// GPMap
	typedef OctreeGPMapContainer<GaussianDistribution>	LeafT;
	typedef OctreeGPMap<GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs, LeafT> OctreeGPMapT;
	const GP_Coverage		GP_COVERAGE = LOCAL_GP;
	const size_t			NUM_CELLS_PER_AXIS = 0;							// number of cells per each axie
	const size_t			MIN_NUM_POINTS_TO_PREDICT = 0;				// min number of points to predict
	const size_t			MAX_NUM_POINTS_TO_PREDICT = 0;				// max number of points to predict
	const bool				FLAG_INDEPENDENT_TEST_POSITIONS = true;	// variance
	const float				GAP = 0.f;
	OctreeGPMapT gpmap(GP_COVERAGE, 
							 BLOCK_SIZE, 
							 NUM_CELLS_PER_AXIS, 
							 MIN_NUM_POINTS_TO_PREDICT, 
							 MAX_NUM_POINTS_TO_PREDICT, 
							 FLAG_INDEPENDENT_TEST_POSITIONS);

	// set bounding box
	gpmap.defineBoundingBox(min_pt, max_pt);

	// set input cloud
	gpmap.setInputCloud(pAllObs, GAP);

	// add points from the input cloud
	gpmap.addPointsFromInputCloud();

	// Octree Viewer
	OctreeViewer<pcl::PointNormal, OctreeGPMapT> octreeViewer(gpmap);
}
			 

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_with_all_in_one_observations(const GP_Coverage		GP_COVERAGE,					// GP coverage
																		  const double				BLOCK_SIZE,						// block size
																		  const size_t				NUM_CELLS_PER_AXIS,			// number of cells per each axie
																		  const size_t				MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																		  const size_t				MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																		  const std::string		&strFileName,					// all-in-one observations input file name
																		  const std::string		&strFileNamePrefix,			// input file name prefix
																		  const std::string		&strFileNameSuffix,			// input file name suffix
																		  const float				gap,								// gap
																		  const std::string		&strOutputFileName,			// output data file prefix
																		  const std::string		&strLogFolder,					// log folder
																		  typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp
																										&logHyp)							// hyperparameters
{
	// log file
	LogFile logFile;	

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strFileName, strFileNamePrefix, strFileNameSuffix);
	//if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations For Training", pAllInOneObs, 0.005);

	// training
	GP::DlibScalar nlZ = std::numeric_limits<GP::DlibScalar>::max();
	bool fFirstTry = true;
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
			std::cout  << "\tMax iterations? (0 for no training): ";
			std::cin >> maxIter;
			if(maxIter <= 0) break;
		}

		// do training
		if(fFirstTry)
		{
			// log file
			const std::string strFileName = strOutputFileName + "(all_in_one)_training";
			logFile.open(strLogFolder + strFileName + ".log");	
			fFirstTry = false;
		}

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

		// min value
		double minValue = 0;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			minValue = 1e-15;
		}
		else
		{
			logFile << "\tMin value? (default: 1e-10): ";
			std::cin >> minValue;
			if(minValue <= 0) minValue = 1e-10;
		}
		logFile  << minValue << std::endl;;

		// hyperparameters
		typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp tempLogHyp = logHyp;
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
			for(int i = 0; i < tempLogHyp.mean.size(); i++) { std::cout << "\t\thyp.mean(" << i << ") = "; std::cin >> hyp; tempLogHyp.mean(i) = log(hyp); }
			for(int i = 0; i < tempLogHyp.cov.size();  i++) { std::cout << "\t\thyp.cov(" << i << ") = ";  std::cin >> hyp; tempLogHyp.cov(i)  = log(hyp); }
			for(int i = 0; i < tempLogHyp.lik.size();  i++) { std::cout << "\t\thyp.lik(" << i << ") = ";  std::cin >> hyp; tempLogHyp.lik(i)  = log(hyp); }
		}

		// log hyperparameters before training
		logFile << "Hyperparameters before training" << std::endl;
		for(int i = 0; i < tempLogHyp.mean.size(); i++) { logFile  << "\thyp.mean(" << i << ") = " << exp(tempLogHyp.mean(i)) << std::endl; }
		for(int i = 0; i < tempLogHyp.cov.size(); i++)  { logFile  << "\thyp.cov("  << i << ") = " << exp(tempLogHyp.cov(i))  << std::endl; }
		for(int i = 0; i < tempLogHyp.lik.size(); i++)  { logFile  << "\thyp.lik("  << i << ") = " << exp(tempLogHyp.lik(i))  << std::endl; }

		// training
		GP::DlibScalar nlZ_temp = gpmap_training_with_all_in_one_observations<MeanFunc,
																									 CovFunc, 
																									 LikFunc, 
																									 InfMethod>(GP_COVERAGE,					// GP coverage
																													BLOCK_SIZE,						// block size
																													NUM_CELLS_PER_AXIS,			// number of cells per each axie
																													MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																													MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																													tempLogHyp,						// hyperparameters
																													pAllInOneObs,					// observations
																													gap,								// gap
																													maxIter,							// number of iterations for training before update
																													numRandomBlocks,				// number of randomly selected blocks
																													minValue);						// min value

		// log hyperparameters after training
		logFile << "Hyperparameters after training" << std::endl;
		for(int i = 0; i < tempLogHyp.mean.size(); i++) { logFile  << "\thyp.mean(" << i << ") = " << exp(tempLogHyp.mean(i)) << std::endl; }
		for(int i = 0; i < tempLogHyp.cov.size(); i++)  { logFile  << "\thyp.cov("  << i << ") = " << exp(tempLogHyp.cov(i))  << std::endl; }
		for(int i = 0; i < tempLogHyp.lik.size(); i++)  { logFile  << "\thyp.lik("  << i << ") = " << exp(tempLogHyp.lik(i))  << std::endl; }

		// save better hyperparameters
		if(nlZ_temp < nlZ)
		{
			nlZ = nlZ_temp;
			logHyp = tempLogHyp;
		}
		logFile << "Best hyperparameters, so far" << std::endl;
		logFile << "nlZ = " << nlZ << std::endl;
		for(int i = 0; i < logHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(logHyp.mean(i)) << std::endl;
		for(int i = 0; i < logHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(logHyp.cov(i))  << std::endl;
		for(int i = 0; i < logHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(logHyp.lik(i))  << std::endl;

		// next
		if(RUN_ALL_WITH_TRAINING_ONCE) break;
	}
}

/** @brief	Build GPMaps as Batch with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_batch_with_all_in_one_observations(const GP_Coverage		GP_COVERAGE,					// GP coverage
																	  const double				BLOCK_SIZE,						// block size
																	  const size_t				NUM_CELLS_PER_AXIS,			// number of cells per each axie
																	  const size_t				MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																	  const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																									&logHyp,							// hyperparameters
																	  const std::string		&strFileName,					// all-in-one observations input file name
																	  const std::string		&strFileNamePrefix,			// input file name prefix
																	  const std::string		&strFileNameSuffix,			// input file name suffix
																	  const float				gap,								// gap
																	  const int					maxIterBeforeUpdate,			// number of iterations for training before update
																	  const std::string		&strOutputFolder,				// output data folder
																	  const std::string		&strOutputFileName,			// output data file prefix
																	  const std::string		&strLogFolder)					// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		std::cout << "Do you wish to build GPMaps as batch with all-in-one observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strNewOutputFileName = strOutputFileName + "(all)_Batch";
	LogFile logFile(strLogFolder + strFileName + ".log");	

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strFileName, strFileNamePrefix, strFileNameSuffix);
	//if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations for Prediction", pAllInOneObs, 0.005);

	// GPMap - All-in-One Observations - Gaussian Distribution (Batch)
	gpmap_batch_with_all_in_one_observations<MeanFunc, 
														  CovFunc, 
														  LikFunc, 
														  InfMethod>(GP_COVERAGE,							// GP coverage
																		 BLOCK_SIZE,							// block size
																		 NUM_CELLS_PER_AXIS,					// number of cells per each axie
																		 MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																		 logHyp,									// hyperparameters
																		 pAllInOneObs,							// observations
																		 gap,										// gap for free points
																		 maxIterBeforeUpdate,				// number of iterations for training before update
																		 strOutputFolder + strNewOutputFileName);	// save file path
}

/** @brief	Build GPMaps incrementally with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_incrementally_with_all_in_one_observations(const GP_Coverage		GP_COVERAGE,					// GP coverage
																				 const double				BLOCK_SIZE,						// block size
																				 const size_t				NUM_CELLS_PER_AXIS,			// number of cells per each axie
																				 const size_t				MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																				 const size_t				MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																				 const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																												&logHyp,							// hyperparameters
																				 const std::string		&strFileName,					// all-in-one observations input file name
																				 const std::string		&strFileNamePrefix,			// input file name prefix
																				 const std::string		&strFileNameSuffix,			// input file name suffix
																				 const float				gap,								// gap
																				 const int					maxIterBeforeUpdate,			// number of iterations for training before update
																				 const std::string		&strOutputFolder,				// output data folder
																				 const std::string		&strOutputFileName,			// output data file prefix
																				 const std::string		&strLogFolder)					// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		std::cout << "Do you wish to build GPMaps incrementally with all-in-one observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strFileName;
	LogFile logFile;

	// loading all-in-one data
	PointNormalCloudPtr pAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pAllInOneObs, strFileName, strFileNamePrefix, strFileNameSuffix);
	//if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("All-in-one Observations for Prediction", pAllInOneObs, 0.005);

	// [1] GPMap - All-in-One Observations - Incremental Update (iBCM)
	strFileName = strOutputFileName + "(all)_iBCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_INDEPENDENT_TEST_POSITIONS = true;
	gpmap_batch_with_all_in_one_observations<BCM_Serializable, // BCM
														  MeanFunc, 
														  CovFunc, 
														  LikFunc, 
														  InfMethod>(GP_COVERAGE,								// GP coverage
																		 BLOCK_SIZE,								// block size
																		 NUM_CELLS_PER_AXIS,						// number of cells per each axie
																		 MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																		 MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																		 FLAG_INDEPENDENT_TEST_POSITIONS,	// predict means and variances
																		 logHyp,										// hyperparameters
																		 pAllInOneObs,								// observations
																		 gap,											// gap for free points
																		 maxIterBeforeUpdate,					// number of iterations for training before update
																		 strOutputFolder + strFileName);		// save file path

	// [2] GPMap - All-in-One Observations - Incremental Update (BCM)
	strFileName = strOutputFileName + "(all)_BCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_DEPENDENT_TEST_POSITIONS = false;
	gpmap_batch_with_all_in_one_observations<BCM_Serializable,
														  MeanFunc, 
														  CovFunc, 
														  LikFunc, 
														  InfMethod>(GP_COVERAGE,								// GP coverage
																		 BLOCK_SIZE,								// block size
																		 NUM_CELLS_PER_AXIS,						// number of cells per each axie
																		 MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																		 MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																		 FLAG_DEPENDENT_TEST_POSITIONS,		// predict means and covariances
																		 logHyp,										// hyperparameters
																		 pAllInOneObs,								// observations
																		 gap,											// gap for free points
																		 maxIterBeforeUpdate,					// number of iterations for training before update
																		 strOutputFolder + strFileName);		// save file path
}

/** @brief	Build GPMaps incrementally with Sequential [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void build_gpmaps_incrementally_with_sequential_observations(const GP_Coverage						GP_COVERAGE,					// GP coverage
																				 const double								BLOCK_SIZE,						// block size
																				 const size_t								NUM_CELLS_PER_AXIS,			// number of cells per each axie
																				 const size_t								MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																				 const size_t								MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																				 const typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																																&logHyp,							// hyperparameters
																				 const std::vector<std::string>		&strFileNameList,				// sequential observations input file name
																				 const std::string						&strFileNamePrefix,			// input file name prefix
																				 const std::string						&strFileNameSuffix,			// input file name suffix
																				 const float								gap,								// gap
																				 const int									maxIterBeforeUpdate,			// number of iterations for training before update
																				 const std::string						&strOutputFolder,				// output data folder
																				 const std::string						&strOutputFileName,			// output data file prefix
																				 const std::string						&strLogFolder)					// log folder
{
	// continue?	
	bool fContinue = RUN_ALL_WITH_TRAINING_ONCE;
	if(!fContinue)
	{
		std::cout << "Do you wish to build GPMaps incrementally with sequential observations? (0/1) ";
		std::cin >> fContinue;
		if(!fContinue) return;
	}

	// log file
	std::string strFileName;
	LogFile logFile;

	// loading sequential data
	PointNormalCloudPtrList cloudPtrList;
	loadPointCloud<pcl::PointNormal>(cloudPtrList, strFileNameList, strFileNamePrefix, strFileNameSuffix);
	//if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("Sequential Observations for Prediction", cloudPtrList, 0.005);

	// [1] GPMap - Sequential Observations - Incremental Update (iBCM)
	strFileName = strOutputFileName + "(seq)_iBCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_INDEPENDENT_TEST_POSITIONS = true;
	gpmap_incremental_with_sequential_observations<BCM_Serializable, // BCM
																  MeanFunc, 
																  CovFunc, 
																  LikFunc, 
																  InfMethod>(GP_COVERAGE,								// GP coverage
																				 BLOCK_SIZE,								// block size
																				 NUM_CELLS_PER_AXIS,						// number of cells per each axie
																				 MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																				 MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																				 FLAG_INDEPENDENT_TEST_POSITIONS,	// predict means and variances
																				 logHyp,										// hyperparameters
																				 cloudPtrList,								// observations
																				 gap,											// gap for free points
																				 maxIterBeforeUpdate,					// number of iterations for training before update
																				 strOutputFolder + strFileName);		// save file path

	// [2] GPMap - Sequential Observations - Incremental Update (BCM)
	strFileName = strOutputFileName + "(seq)_BCM";
	logFile.open(strLogFolder + strFileName + ".log");	
	const bool FLAG_DEPENDENT_TEST_POSITIONS = false;
	gpmap_incremental_with_sequential_observations<BCM_Serializable,
																  MeanFunc, 
																  CovFunc, 
																  LikFunc, 
																  InfMethod>(GP_COVERAGE,							// GP coverage
																				 BLOCK_SIZE,							// block size
																				 NUM_CELLS_PER_AXIS,					// number of cells per each axie
																				 MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																				 MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																				 FLAG_DEPENDENT_TEST_POSITIONS,	// predict means and covariances
																				 logHyp,									// hyperparameters
																				 cloudPtrList,							// observations
																				 gap,										// gap for free points
																				 maxIterBeforeUpdate,					// number of iterations for training before update
																				 strOutputFolder + strFileName);	// save file path
}

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps as Batch with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_and_build_gpmaps_batch_with_all_in_one_observations(const GP_Coverage		GP_COVERAGE,					// GP coverage
																										 const double				BLOCK_SIZE,						// block size
																										 const size_t				NUM_CELLS_PER_AXIS,			// number of cells per each axie
																										 const size_t				MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																										 const size_t				MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																										 typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																																		&logHyp,							// hyperparameters
																										 const std::string		&strFileName,					// all-in-one observations input file name
																										 const std::string		&strFileNamePrefix,			// input file name prefix
																										 const std::string		&strFileNameSuffix,			// input file name suffix
																										 const float				gap,								// gap
																										 const int					maxIterBeforeUpdate,			// number of iterations for training before update
																										 const std::string		&strOutputFolder,				// output data folder
																										 const std::string		&strOutputFileName,			// output data file prefix
																										 const std::string		&strLogFolder)					// log folder
{
	// training
	train_hyperparameters_with_all_in_one_observations<MeanFunc,
																		CovFunc, 
																		LikFunc, 
																		InfMethod>(GP_COVERAGE,						// GP coverage
																					  BLOCK_SIZE,						// block size
																					  NUM_CELLS_PER_AXIS,			// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																					  strFileName,						// all-in-one observations input file name
																					  strFileNamePrefix,				// input file name prefix
																					  strFileNameSuffix,				// input file name suffix
																					  gap,								// gap
																					  strOutputFileName,				// output data file prefix
																					  strLogFolder,					// log folder
																					  logHyp);							// hyperparameters


	// prediction
	build_gpmaps_batch_with_all_in_one_observations<MeanFunc,
																	CovFunc, 
																	LikFunc, 
																	InfMethod>(GP_COVERAGE,							// GP coverage
																				  BLOCK_SIZE,							// block size
																				  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																				  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																				  logHyp,								// hyperparameters
																				  strFileName,							// all-in-one observations input file name
																				  strFileNamePrefix,					// file name prefix
																				  strFileNameSuffix,					// file name suffix
																				  gap,									// gap
																				  maxIterBeforeUpdate,				// number of iterations for training before update
																				  strOutputFolder,					// output data folder
																				  strOutputFileName,					// output file name prefix
																				  strLogFolder);						// log folder
}

/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps incrementally with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_and_build_local_gpmaps_incrementally_with_all_in_one_observations(const double		BLOCK_SIZE,						// block size
																														 const size_t			NUM_CELLS_PER_AXIS,			// number of cells per each axie
																														 const size_t			MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																														 const size_t			MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																														 typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																																					&logHyp,							// hyperparameters
																														 const std::string	&strFileName,					// all-in-one observations input file name
																														 const std::string	&strFileNamePrefix,			// input file name prefix
																														 const std::string	&strFileNameSuffix,			// input file name suffix
																														 const float			gap,								// gap
																														 const int				maxIterBeforeUpdate,			// number of iterations for training before update
																														 const std::string	&strOutputFolder,				// output data folder
																														 const std::string	&strOutputFileName,			// output data file prefix
																														 const std::string	&strLogFolder)					// log folder
{
	// training
	train_hyperparameters_with_all_in_one_observations<MeanFunc,
																		CovFunc, 
																		LikFunc, 
																		InfMethod>(LOCAL_GP,									// GP coverage
																					  BLOCK_SIZE,								// block size
																					  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																					  strFileName,								// all-in-one observations input file name
																					  strFileNamePrefix,						// input file name prefix
																					  strFileNameSuffix,						// input file name suffix
																					  gap,										// gap
																					  strOutputFileName + "Local_GP",	// output data file prefix
																					  strLogFolder,							// log folder
																					  logHyp);									// hyperparameters

	// prediction
	build_gpmaps_incrementally_with_all_in_one_observations<MeanFunc,
																			CovFunc, 
																			LikFunc, 
																			InfMethod>(LOCAL_GP,									// GP coverage
																						  BLOCK_SIZE,								// block size
																						  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																						  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																						  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																						  logHyp,									// hyperparameters
																						  strFileName,								// all-in-one observations input file name
																						  strFileNamePrefix,						// file name prefix
																						  strFileNameSuffix,						// file name suffix
																						  gap,										// gap
																						  maxIterBeforeUpdate,					// number of iterations for training before update
																						  strOutputFolder,						// output data folder
																						  strOutputFileName + "Local_GP",	// output file name prefix
																						  strLogFolder);							// log folder
}


/** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps incrementally with Sequential [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_hyperparameters_and_build_local_gpmaps_incrementally_with_sequential_observations(const double							BLOCK_SIZE,						// block size
																															const size_t							NUM_CELLS_PER_AXIS,			// number of cells per each axie
																															const size_t							MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																															const size_t							MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																															typename GP::GaussianProcess<float, MeanFunc, CovFunc, LikFunc, InfMethod>::Hyp	
																																										&logHyp,							// hyperparameters
																															const std::string						&strFileName,					// all-in-one observations input file name
																															const std::vector<std::string>	&strFileNameList,				// sequential observations input file name
																															const std::string						&strFileNamePrefix,			// input file name prefix
																															const std::string						&strFileNameSuffix,			// input file name suffix
																															const float								gap,								// gap
																															const int								maxIterBeforeUpdate,			// number of iterations for training before update
																															const std::string						&strOutputFolder,				// output data folder
																															const std::string						&strOutputFileName,			// output data file prefix
																															const std::string						&strLogFolder)					// log folder
{
	// training
	train_hyperparameters_with_all_in_one_observations<MeanFunc,
																		CovFunc, 
																		LikFunc, 
																		InfMethod>(LOCAL_GP,									// GP coverage
																					  BLOCK_SIZE,								// block size
																					  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																					  strFileName,								// all-in-one observations input file name
																					  strFileNamePrefix,						// input file name prefix
																					  strFileNameSuffix,						// input file name suffix
																					  gap,										// gap
																					  strOutputFileName + "Local_GP",	// output data file prefix
																					  strLogFolder,							// log folder
																					  logHyp);									// hyperparameters

	// prediction
	build_gpmaps_incrementally_with_sequential_observations<MeanFunc,
																			  CovFunc, 
																			  LikFunc, 
																			  InfMethod>(LOCAL_GP,									// GP coverage
																							 BLOCK_SIZE,								// block size
																							 NUM_CELLS_PER_AXIS,						// number of cells per each axie
																							 MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																							 MAX_NUM_POINTS_TO_PREDICT,			// max number of points to predict
																							 logHyp,										// hyperparameters
																							 strFileNameList,							// sequential observations input file name
																							 strFileNamePrefix,						// file name prefix
																							 strFileNameSuffix,						// file name suffix
																							 gap,											// gap
																							 maxIterBeforeUpdate,					// number of iterations for training before update
																							 strOutputFolder,							// output data folder
																							 strOutputFileName + "Local_GP",		// output file name prefix
																							 strLogFolder);							// log folder
}

/** @brief	Train global hyperparameters with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class MeanFunc, 
			template<typename> class CovFunc, 
			template<typename> class LikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class InfMethod>
void train_global_hyperparameters_with_all_in_one_observations(const std::string		&strFileName,			// all-in-one observations input file name
																					const std::string		&strFileNamePrefix,	// input file name prefix
																					const std::string		&strFileNameSuffix,	// input file name suffix
																					const float				gap,						// gap
																					const std::string		&strOutputFileName,	// output data file prefix
																					const std::string		&strLogFolder,			// log folder
																					typename GP::MeanGlobalGP<float>::GlobalHyp 
																												&logGlobalHyp)			// global hyperparameters
{
	// [0] Training Global GP
	// log file
	const std::string strLogFileName = strOutputFileName + "_training";
	LogFile logFile(strLogFolder + strFileName + ".log");	

	// loading all-in-one data
	PointNormalCloudPtr pGlobalAllInOneObs(new PointNormalCloud());
	loadPointCloud<pcl::PointNormal>(pGlobalAllInOneObs, strFileName, strFileNamePrefix, strFileNameSuffix);
	//if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("Global All-in-one Observations for Training and Prediction", pGlobalAllInOneObs, 0.005);

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
	GP::DlibScalar nlZ = std::numeric_limits<GP::DlibScalar>::max();
	while(true)
	{
		// continue?
		bool fTrainGlobalHyp = RUN_ALL_WITH_TRAINING_ONCE;
		if(!fTrainGlobalHyp)
		{
			std::cout << "Would you like to train the global hyperparameters? (0/1) ";
			std::cin >> fTrainGlobalHyp;
			if(!fTrainGlobalHyp) break;
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
			if(maxIter <= 0) break;
		}

		// min value
		double minValue = 0;
		if(RUN_ALL_WITH_TRAINING_ONCE)
		{
			minValue = 1e-15;
		}
		else if(fTrainGlobalHyp)
		{
			std::cout << "\tMin value? (default: 1e-10): ";
			std::cin >> minValue;
			if(minValue <= 0) minValue = 1e-10;
		}

		// hyperparameters
		typename GP::MeanGlobalGP<float>::GlobalHyp tempLogGlobalHyp = logGlobalHyp;
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
			for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++) { std::cout << "\t\thyp.mean(" << i << ") = "; std::cin >> hyp; tempLogGlobalHyp.mean(i) = log(hyp); }
			for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++) { std::cout << "\t\thyp.cov(" << i << ") = ";  std::cin >> hyp; tempLogGlobalHyp.cov(i)  = log(hyp); }
			for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++) { std::cout << "\t\thyp.lik(" << i << ") = ";  std::cin >> hyp; tempLogGlobalHyp.lik(i)  = log(hyp); }
		}

		// log hyperparameters before training
		logFile << "Global hyperparameters before training" << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(tempLogGlobalHyp.mean(i)) << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(tempLogGlobalHyp.cov(i))  << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(tempLogGlobalHyp.lik(i))  << std::endl;

		// training
		typedef GP::GaussianProcess<float, GP::MeanZeroDerObs, GP::CovRQisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs> GPType;
		GP::DlibScalar nlZ_temp = GPType::train<GP::BOBYQA, GP::NoStopping>(tempLogGlobalHyp, globalTrainingData, maxIter, minValue);

		logFile << "Global hyperparameters after training" << std::endl;
		logFile << "nlZ = " << nlZ_temp << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(tempLogGlobalHyp.mean(i)) << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(tempLogGlobalHyp.cov(i))  << std::endl;
		for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(tempLogGlobalHyp.lik(i))  << std::endl;

		// save better hyperparameters
		if(nlZ_temp < nlZ)
		{
			nlZ = nlZ_temp;
			logGlobalHyp = tempLogGlobalHyp;
		}
		logFile << "Best global hyperparameters, so far" << std::endl;
		logFile << "nlZ = " << nlZ << std::endl;
		for(int i = 0; i < logGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(logGlobalHyp.mean(i)) << std::endl;
		for(int i = 0; i < logGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(logGlobalHyp.cov(i))  << std::endl;
		for(int i = 0; i < logGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(logGlobalHyp.lik(i))  << std::endl;
		
		if(!fTrainGlobalHyp || maxIter <= 0) break;
	}

	// set
	GP::MeanGlobalGP<float>::set(logGlobalHyp, globalTrainingData);
}

/** @brief	Train global and local hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps as Batch with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class LocalCovFunc, 
			template<typename> class LocalLikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class LocalInfMethod>
void train_hyperparameters_and_build_global_local_gpmaps_batch_with_all_in_one_observations(const double				BLOCK_SIZE,						// block size
																														  const size_t				NUM_CELLS_PER_AXIS,			// number of cells per each axie
																														  const size_t				MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																														  typename GP::MeanGlobalGP<float>::GlobalHyp	
																																						&logGlobalHyp,					// global hyperparameters
																														  typename GP::GaussianProcess<float, GP::MeanGlobalGP, LocalCovFunc, LocalLikFunc, LocalInfMethod>::Hyp	
																																						&logLocalHyp,					// glboal hyperparameters
																														  const std::string		&strFileName,					// all-in-one observations input file name
																														  const std::string		&strFileNamePrefix,			// input file name prefix
																														  const std::string		&strGlobalFileNameSuffix,	// input file name suffix
																														  const std::string		&strLocalFileNameSuffix,	// input file name suffix
																														  const float				gap,								// gap
																														  const int					maxIterBeforeUpdate,			// number of iterations for training before update
																														  const std::string		&strOutputFolder,				// output data folder
																														  const std::string		&strOutputFileName,			// output data file prefix
																														  const std::string		&strLogFolder)					// log folder
{
	// Global GP: training
	train_global_hyperparameters_with_all_in_one_observations<GP::MeanZeroDerObs,
																				 GP::CovRQisoDerObs, 
																				 GP::LikGaussDerObs, 
																				 GP::InfExactDerObs>(strFileName,								// all-in-one observations input file name
																											strFileNamePrefix,						// input file name prefix
																											strGlobalFileNameSuffix,				// input file name suffix
																											gap,											// gap
																											strOutputFileName + "_Global_GP",	// output data file prefix
																											strLogFolder,								// log folder
																											logGlobalHyp);								// global hyperparameters

	// Global GP: prediction
	build_gpmaps_batch_with_all_in_one_observations<GP::MeanZeroDerObs,
																	GP::CovRQisoDerObs, 
																	GP::LikGaussDerObs, 
																	GP::InfExactDerObs>(GLOBAL_GP,								// GP coverage
																							  BLOCK_SIZE,								// block size
																							  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																							  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																							  logGlobalHyp,							// global hyperparameters
																							  strFileName,								// all-in-one observations input file name
																							  strFileNamePrefix,						// file name prefix
																							  strGlobalFileNameSuffix,				// file name suffix
																							  gap,										// gap
																							  maxIterBeforeUpdate,					// number of iterations for training before update
																							  strOutputFolder,						// output data folder
																							  strOutputFileName + "_Global_GP",	// output file name prefix
																							  strLogFolder);							// log folder


	// Global GP + Local GP: train + prediction
	const size_t MAX_NUM_POINTS_TO_PREDICT = 0;
	train_hyperparameters_and_build_gpmaps_batch_with_all_in_one_observations<GP::MeanGlobalGP,
																									  LocalCovFunc, 
																									  LocalLikFunc, 
																									  LocalInfMethod>(GLOBAL_LOCAL_GP,									// GP coverage
																															BLOCK_SIZE,											// block size
																															NUM_CELLS_PER_AXIS,								// number of cells per each axie
																															MIN_NUM_POINTS_TO_PREDICT,						// min number of points to predict
																															MAX_NUM_POINTS_TO_PREDICT,						// max number of points to predict
																															logLocalHyp,										// local hyperparameters
																															strFileName,										// all-in-one observations input file name
																															strFileNamePrefix,								// input file name prefix
																															strLocalFileNameSuffix,							// input file name suffix
																															gap,													// gap
																															maxIterBeforeUpdate,								// number of iterations for training before update
																															strOutputFolder,									// output data folder
																															strOutputFileName + "_Global_Local_GP",	// output data file prefix
																															strLogFolder);										// log folder	

}

/** @brief	Train global and local hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps incrementally with All-in-One [Function/Derivative/All] Observations */
template<template<typename> class LocalCovFunc, 
			template<typename> class LocalLikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class LocalInfMethod>
void train_hyperparameters_and_build_global_local_gpmaps_incrementally_with_all_in_one_observations(const double				BLOCK_SIZE,							// block size
																																	 const size_t				NUM_CELLS_PER_AXIS,				// number of cells per each axie
																																	 const size_t				MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																																	 const size_t				MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																																	 typename GP::MeanGlobalGP<float>::GlobalHyp	
																																									&logGlobalHyp,						// global hyperparameters
																																	 typename GP::GaussianProcess<float, GP::MeanGlobalGP, LocalCovFunc, LocalLikFunc, LocalInfMethod>::Hyp	
																																									&logLocalHyp,						// local hyperparameters
																																	 const std::string		&strFileName,						// all-in-one observations input file name
																																	 const std::string		&strFileNamePrefix,				// input file name prefix
																																	 const std::string		&strGlobalFileNameSuffix,		// input file name suffix
																																	 const std::string		&strLocalFileNameSuffix,		// input file name suffix
																																	 const float				gap,									// gap
																																	 const int					maxIterBeforeUpdate,				// number of iterations for training before update
																																	 const std::string		&strOutputFolder,					// output data folder
																																	 const std::string		&strOutputFileName,				// output data file prefix
																																	 const std::string		&strLogFolder)						// log folder
{
	// Global GP: training
	train_global_hyperparameters_with_all_in_one_observations<GP::MeanZeroDerObs,
																				 GP::CovRQisoDerObs, 
																				 GP::LikGaussDerObs, 
																				 GP::InfExactDerObs>(strFileName,								// all-in-one observations input file name
																											strFileNamePrefix,						// input file name prefix
																											strGlobalFileNameSuffix,				// input file name suffix
																											gap,											// gap
																											strOutputFileName + "_Global_GP",	// output data file prefix
																											strLogFolder,								// log folder
																											logGlobalHyp);								// hyperparameters

	// Global GP: prediction
	build_gpmaps_batch_with_all_in_one_observations<GP::MeanZeroDerObs,
																	GP::CovRQisoDerObs, 
																	GP::LikGaussDerObs, 
																	GP::InfExactDerObs>(GLOBAL_GP,								// GP coverage
																							  BLOCK_SIZE,								// block size
																							  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																							  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																							  logGlobalHyp,							// global hyperparameters
																							  strFileName,								// all-in-one observations input file name
																							  strFileNamePrefix,						// file name prefix
																							  strGlobalFileNameSuffix,				// file name suffix
																							  gap,										// gap
																							  maxIterBeforeUpdate,					// number of iterations for training before update
																							  strOutputFolder,						// output data folder
																							  strOutputFileName + "_Global_GP",	// output file name prefix
																							  strLogFolder);							// log folder


	// Global GP + Local GP: train + prediction
	train_hyperparameters_and_build_gpmaps_incrementally_with_all_in_one_observations<GP::MeanGlobalGP,
																												 LocalCovFunc, 
																												 LocalLikFunc, 
																												 LocalInfMethod>(GLOBAL_LOCAL_GP,								// GP coverage
																																	  BLOCK_SIZE,										// block size
																																	  NUM_CELLS_PER_AXIS,							// number of cells per each axie
																																	  MIN_NUM_POINTS_TO_PREDICT,					// min number of points to predict
																																	  MAX_NUM_POINTS_TO_PREDICT,					// max number of points to predict
																																	  logLocalHyp,										// local hyperparameters
																																	  strFileName,										// all-in-one observations input file name
																																	  strFileNamePrefix,								// input file name prefix
																																	  strLocalFileNameSuffix,						// input file name suffix
																																	  gap,												// gap
																																	  maxIterBeforeUpdate,							// number of iterations for training before update
																																	  strOutputFolder,								// output data folder
																																	  strOutputFileName + "_Global_Local_GP",	// output data file prefix
																																	  strLogFolder);									// log folder

}

/** @brief	Train global and local hyperparameters with All-in-One [Function/Derivative/All] Observations and
  *			Build GPMaps incrementally with Sequential [Function/Derivative/All] Observations */
template<template<typename> class LocalCovFunc, 
			template<typename> class LocalLikFunc,
			template <typename, 
						 template<typename> class,
						 template<typename> class,
						 template<typename> class> class LocalInfMethod>
void train_hyperparameters_and_build_global_local_gpmaps_incrementally_with_sequential_observations(const double								BLOCK_SIZE,							// block size
																																	 const size_t								NUM_CELLS_PER_AXIS,				// number of cells per each axie
																																	 const size_t								MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																																	 const size_t								MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																																	 typename GP::MeanGlobalGP<float>::GlobalHyp	
																																													&logGlobalHyp,						// global hyperparameters
																																	 typename GP::GaussianProcess<float, GP::MeanGlobalGP, LocalCovFunc, LocalLikFunc, LocalInfMethod>::Hyp	
																																													&logLocalHyp,						// local hyperparameters
																																	 const std::string						&strFileName,						// global all-in-one observations input file name
																																	 const std::vector<std::string>		&strFileNameList,					// local sequential observations input file name
																																	 const std::string						&strFileNamePrefix,				// input file name prefix
																																	 const std::string						&strGlobalFileNameSuffix,		// global input file name suffix
																																	 const std::string						&strLocalFileNameSuffix,		// local input file name suffix
																																	 const float								gap,									// gap
																																	 const int									maxIterBeforeUpdate,				// number of iterations for training before update
																																	 const std::string						&strOutputFolder,					// output data folder
																																	 const std::string						&strOutputFileName,				// output data file prefix
																																	 const std::string						&strLogFolder)						// log folder
{
	// Global GP: training
	train_global_hyperparameters_with_all_in_one_observations<GP::MeanZeroDerObs,
																				 GP::CovRQisoDerObs, 
																				 GP::LikGaussDerObs, 
																				 GP::InfExactDerObs>(strFileName,								// all-in-one observations input file name
																											strFileNamePrefix,						// input file name prefix
																											strGlobalFileNameSuffix,				// input file name suffix
																											gap,											// gap
																											strOutputFileName + "_Global_GP",	// output data file prefix
																											strLogFolder,								// log folder
																											logGlobalHyp);								// hyperparameters

	// Global GP: prediction
	build_gpmaps_batch_with_all_in_one_observations<GP::MeanZeroDerObs,
																	GP::CovRQisoDerObs, 
																	GP::LikGaussDerObs, 
																	GP::InfExactDerObs>(GLOBAL_GP,								// GP coverage
																							  BLOCK_SIZE,								// block size
																							  NUM_CELLS_PER_AXIS,					// number of cells per each axie
																							  MIN_NUM_POINTS_TO_PREDICT,			// min number of points to predict
																							  logGlobalHyp,							// global hyperparameters
																							  strFileName,								// all-in-one observations input file name
																							  strFileNamePrefix,						// file name prefix
																							  strGlobalFileNameSuffix,				// file name suffix
																							  gap,										// gap
																							  maxIterBeforeUpdate,					// number of iterations for training before update
																							  strOutputFolder,						// output data folder
																							  strOutputFileName + "_Global_GP",	// output file name prefix
																							  strLogFolder);							// log folder

	// Global GP + Local GP: training
	train_hyperparameters_with_all_in_one_observations<GP::MeanGlobalGP,
																		LocalCovFunc, 
																		LocalLikFunc, 
																		LocalInfMethod>(GLOBAL_LOCAL_GP,									// GP coverage
																							 BLOCK_SIZE,										// block size
																							 NUM_CELLS_PER_AXIS,								// number of cells per each axie
																							 MIN_NUM_POINTS_TO_PREDICT,					// min number of points to predict
																							 MAX_NUM_POINTS_TO_PREDICT,					// max number of points to predict
																							 strFileName,										// all-in-one observations input file name
																							 strFileNamePrefix,								// input file name prefix
																							 strLocalFileNameSuffix,						// input file name suffix
																							 gap,													// gap
																							 strOutputFileName + "_Global_Local_GP",	// output data file prefix
																							 strLogFolder,										// log folder
																							 logLocalHyp);										// hyperparameters

	// Global GP + Local GP: prediction
	build_gpmaps_incrementally_with_sequential_observations<GP::MeanGlobalGP,
																			  LocalCovFunc, 
																			  LocalLikFunc, 
																			  LocalInfMethod>(GLOBAL_LOCAL_GP,									// GP coverage
																									BLOCK_SIZE,											// block size
																									NUM_CELLS_PER_AXIS,								// number of cells per each axie																							 MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																									MIN_NUM_POINTS_TO_PREDICT,						// min number of points to predict
																									MAX_NUM_POINTS_TO_PREDICT,						// max number of points to predict
																									logLocalHyp,										// hyperparameters
																									strFileNameList,									// sequential observations input file name
																									strFileNamePrefix,								// file name prefix
																									strLocalFileNameSuffix,							// file name suffix
																									gap,													// gap
																									maxIterBeforeUpdate,								// number of iterations for training before update
																									strOutputFolder,									// output data folder
																									strOutputFileName + "_Global_Local_GP",	// output file name prefix
																									strLogFolder);										// log folder
}

///** @brief	Train hyperparameters with All-in-One [Function/Derivative/All] Observations and
//  *			Build GPMaps with
//  *				- Sampled All-in-One	[Function/Derivative/All] Observations
//  *				- All-in-One			[Function/Derivative/All] Observations
//  *				- Sequential			[Function/Derivative/All] Observations
//  */
//template<template<typename> class CovFunc, 
//			template<typename> class LikFunc,
//			template <typename, 
//						 template<typename> class,
//						 template<typename> class,
//						 template<typename> class> class InfMethod>
//void train_hyperparameters_and_build_gpmaps_using_MeanGlobalGP(const double				BLOCK_SIZE,													// block size
//																					const size_t				NUM_CELLS_PER_AXIS,										// number of cells per each axie
//																					const size_t				MIN_NUM_POINTS_TO_PREDICT,								// min number of points to predict
//																					const size_t				MAX_NUM_POINTS_TO_PREDICT,								// max number of points to predict
//																					typename GP::MeanGlobalGP<float>::GlobalHyp	
//																																	&logGlobalHyp,							// global hyperparameters
//																					typename GP::GaussianProcess<float, GP::MeanGlobalGP, CovFunc, LikFunc, InfMethod>::Hyp	
//																																	&logLocalHyp,							// local hyperparameters
//																					const std::vector<std::string>		&strFileNameList,	// sequential observations input file name
//																					const std::string							&strFileName,			// all-in-one observations input file name
//																					const std::string							&strFileNamePrefix,				// input file name prefix
//																					const std::string							&strGlobalObsFileNameSuffix,		// global input file name suffix
//																					const std::string							&strFileNameSuffix,				// input file name suffix
//																					const float									gap,										// gap
//																					const int									maxIterBeforeUpdate,					// number of iterations for training before update
//																					const std::string							&strOutputFolder,						// output data folder
//																					const std::string							&strOutputFileName,					// output data file prefix
//																					const std::string							&strLogFolder)							// log folder
//{
//	// [0] Training Global GP
//	// log file
//	const std::string strFileName = strOutputFileName + "_MeanGlobalGP";
//	LogFile logFile(strLogFolder + strFileName + ".log");	
//
//	// loading all-in-one data
//	PointNormalCloudPtr pGlobalAllInOneObs(new PointNormalCloud());
//	loadPointCloud<pcl::PointNormal>(pGlobalAllInOneObs, strFileName, strFileNamePrefix, strGlobalObsFileNameSuffix);
//	if(!RUN_ALL_WITH_TRAINING_ONCE) show<pcl::PointNormal>("Global All-in-one Observations for Training and Prediction", pGlobalAllInOneObs, 0.005);
//
//	// indices
//	const int N = pGlobalAllInOneObs->points.size();
//	std::vector<int> indices(N);
//	std::generate(indices.begin(), indices.end(), UniqueNonZeroInteger());
//
//	// training data
//	MatrixPtr pX, pXd; VectorPtr pYYd;
//	generateTrainingData(pGlobalAllInOneObs, indices, gap, pX, pXd, pYYd);	
//	GP::DerivativeTrainingData<float> globalTrainingData;
//	globalTrainingData.set(pX, pXd, pYYd);
//	//GP::TrainingData<float> globalTrainingData;
//	//globalTrainingData.set(pX, pYYd);
//
//	// training and set
//	GP::DlibScalar nlZ = std::numeric_limits<GP::DlibScalar>::max();
//	while(true)
//	{
//		// continue?
//		bool fTrainGlobalHyp = RUN_ALL_WITH_TRAINING_ONCE;
//		if(!fTrainGlobalHyp)
//		{
//			std::cout << "Would you like to train the global hyperparameters? (0/1) ";
//			std::cin >> fTrainGlobalHyp;
//			if(!fTrainGlobalHyp) break;
//		}
//
//		// max iterations
//		int maxIter = 0;
//		if(RUN_ALL_WITH_TRAINING_ONCE)
//		{
//			maxIter = 50;
//		}
//		else if(fTrainGlobalHyp)
//		{
//			std::cout << "\tMax iterations? (0 for no training): ";
//			std::cin >> maxIter;
//			if(maxIter <= 0) break;
//		}
//
//		// hyperparameters
//		typename GP::MeanGlobalGP<float>::GlobalHyp tempLogGlobalHyp = logGlobalHyp;
//		bool fUsePredefinedHyperparameters;
//		if(RUN_ALL_WITH_TRAINING_ONCE)
//		{
//			fUsePredefinedHyperparameters = true;
//		}
//		else
//		{
//			logFile  << "\tUse Predefined Hyperparameters? (0/1) ";
//			std::cin >> fUsePredefinedHyperparameters;
//		}
//		if(!fUsePredefinedHyperparameters)
//		{
//			float hyp;
//			for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++) { std::cout << "\t\thyp.mean(" << i << ") = "; std::cin >> hyp; tempLogGlobalHyp.mean(i) = log(hyp); }
//			for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++) { std::cout << "\t\thyp.cov(" << i << ") = ";  std::cin >> hyp; tempLogGlobalHyp.cov(i)  = log(hyp); }
//			for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++) { std::cout << "\t\thyp.lik(" << i << ") = ";  std::cin >> hyp; tempLogGlobalHyp.lik(i)  = log(hyp); }
//		}
//
//		// log hyperparameters before training
//		logFile << "Global hyperparameters before training" << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(tempLogGlobalHyp.mean(i)) << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(tempLogGlobalHyp.cov(i))  << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(tempLogGlobalHyp.lik(i))  << std::endl;
//
//		// training
//		typedef GP::GaussianProcess<float, GP::MeanZeroDerObs, GP::CovRQisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs> GPType;
//		GP::DlibScalar nlZ_temp = GPType::train<GP::BOBYQA, GP::NoStopping>(tempLogGlobalHyp, globalTrainingData, maxIter);
//
//		logFile << "Global hyperparameters after training" << std::endl;
//		logFile << "nlZ = " << nlZ_temp << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(tempLogGlobalHyp.mean(i)) << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(tempLogGlobalHyp.cov(i))  << std::endl;
//		for(int i = 0; i < tempLogGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(tempLogGlobalHyp.lik(i))  << std::endl;
//
//		// save better hyperparameters
//		if(nlZ_temp < nlZ)
//		{
//			nlZ = nlZ_temp;
//			logGlobalHyp = tempLogGlobalHyp;
//		}
//		logFile << "Best global hyperparameters, so far" << std::endl;
//		logFile << "nlZ = " << nlZ << std::endl;
//		for(int i = 0; i < logGlobalHyp.mean.size(); i++)		logFile << "- mean[" << i << "] = " << expf(logGlobalHyp.mean(i)) << std::endl;
//		for(int i = 0; i < logGlobalHyp.cov.size();  i++)		logFile << "- cov["  << i << "] = " << expf(logGlobalHyp.cov(i))  << std::endl;
//		for(int i = 0; i < logGlobalHyp.lik.size();  i++)		logFile << "- lik["  << i << "] = " << expf(logGlobalHyp.lik(i))  << std::endl;
//		
//		if(!fTrainGlobalHyp || maxIter <= 0) break;
//	}
//
//	// set
//	GP::MeanGlobalGP<float>::set(logGlobalHyp, globalTrainingData);
//
//	// do the same thing
//	train_hyperparameters_and_build_gpmaps<GP::MeanGlobalGP, CovFunc, LikFunc, InfMethod>(BLOCK_SIZE,							// block size
//																													  NUM_CELLS_PER_AXIS,				// number of cells per each axie
//																													  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
//																													  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
//																													  logLocalHyp,							// hyperparameters
//																													  strFileNameList,	// sequential observations input file name
//																													  strFileName,			// all-in-one observations input file name
//																													  strFileNamePrefix,				// input file name prefix
//																													  strFileNameSuffix,				// input file name suffix
//																													  gap,									// gap
//																													  maxIterBeforeUpdate,				// number of iterations for training before update
//																													  strOutputFolder,					// output data folder
//																													  strOutputFileName,					// output data file prefix
//																													  strLogFolder);						// log folder
//
//}

}
#endif