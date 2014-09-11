#if 1
// Eigen
#include "serialization/eigen_serialization.hpp" // Eigen
// includes followings inside of it
//		- #define EIGEN_NO_DEBUG		// to speed up
//		- #define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
//		- #include <Eigen/Core>

// combinations
// ---------------------------------------------------------------------------------------------
// |                   Observations                   |         GPMap Building Process         |
// |===========================================================================================|
// |   (FuncObs)  |    (DerObs)         |  (AllObs)   |                   | Incremental Update |
// | hit points,  | virtual hit points, |  FuncObs,   | Single Prediction |--------------------|
// | empty points | surface normals     |  DerObs     |      (Batch)      |          |         |
// ---------------------------------------------------|                   |   iBCM   |   BCM   |
// |  Sequential  |  All-in-One  | Sampled *          |                   |          |         |
// ---------------------------------------------------------------------------------------------

// All-in-One	Sampled	[Function/Derivative/All] Observations + Batch
// Sequential	Sampled	[Function/Derivative/All] Observations + Incremental Update (iBCM/BCM)
// Sequential 	Origianl	[Function/Derivative/All] Observations + Incremental Update (iBCM/BCM)
// All-in-One	Sampled	[Function/Derivative/All] Observations + Incremental Update (iBCM/BCM)
// All-in-One	Origianl	[Function/Derivative/All] Observations + Incremental Update (iBCM/BCM)


// GPMap
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "octree/macro_gpmap.hpp"			// macro_gpmap
#include "octree/octree_viewer.hpp"			// OctreeViewer
#include "common/common.hpp"					// getMinMaxPointXYZ
using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] setting - GPMap constants
	// CELL_SIZE = 0.001
	//const double	BLOCK_SIZE	= 0.003;		const size_t	NUM_CELLS_PER_AXIS	= 3;		// NUM_CELLS_PER_BLOCK = 3*3*3 = 27						BCM/iBCM (too small BLOCK_SIZE to produce non-smooth signed distance function)
	//const double	BLOCK_SIZE	= 0.01;		const size_t	NUM_CELLS_PER_AXIS	= 10;		// NUM_CELLS_PER_BLOCK = 10*10*10 = 1,000				BCM/iBCM
	//const double	BLOCK_SIZE	= 0.009;		const size_t	NUM_CELLS_PER_AXIS	= 9;		// NUM_CELLS_PER_BLOCK = 9*9*9 = 729					iBCM (sparse_ell for Der Obs): good?
	//const double	BLOCK_SIZE	= 0.027;		const size_t	NUM_CELLS_PER_AXIS	= 27;		// NUM_CELLS_PER_BLOCK = 27*27*27 = 19,683			iBCM (sparse_ell for Der Obs): good?
	//const double	BLOCK_SIZE	= 0.06;		const size_t	NUM_CELLS_PER_AXIS	= 60;		// NUM_CELLS_PER_BLOCK = 60*60*60 = 216,000			iBCM (2*sparse_ell for Der Obs): very good?
	//const double	BLOCK_SIZE	= 0.15;		const size_t	NUM_CELLS_PER_AXIS	= 150;	// NUM_CELLS_PER_BLOCK = 150*150*150 = 3,375,000	iBCM (spase_ell for Func Obs)	

	const double	BLOCK_SIZE								= 0.02;
	const size_t	NUM_CELLS_PER_AXIS					= 10;		// NUM_CELLS_PER_BLOCK = 10*10*10 = 1,000
	const size_t	MIN_NUM_POINTS_TO_PREDICT			= 2;		// 10;
	const size_t	MAX_NUM_POINTS_TO_PREDICT			= 100;	// 200: too slow
	const bool		FLAG_DEPENDENT_TEST_POSITIONS		= false;
	const bool		FLAG_INDEPENDENT_TEST_POSITIONS	= true;
	const float		GAP										= 0.001f;
	const int		MAX_ITER_BEFORE_UPDATE				= 0;

	// [0] setting - directory
	const std::string strDataFolder				("E:/Documents/GitHub/Data/");
	const std::string strDataName					("bunny");
	const std::string strInputFolder				(strDataFolder + strDataName + "/input/");
	const std::string strIntermediateFolder	(strDataFolder + strDataName + "/intermediate/");
	const std::string strGPMapFolder				(strDataFolder + strDataName + "/output/gpmap/");
	const std::string strGPMapMetaDataFolder	(strGPMapFolder + "meta_data/");
	create_directory(strGPMapFolder);
	create_directory(strGPMapMetaDataFolder);

	// [0] setting - observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]		= {"bun000", "bun090", "bun180", "bun270"};
	const std::string strFileNameAll				=  "bunny_all";
	StringList strObsFileNames(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [0] setting - input data folder
	std::cout << "[Input Data]" << std::endl;
	int fOctreeDownSampling;
	std::cout << "No sampling(-1), Random sampling(0) or octree-based down sampling(1)? ";
	std::cin >> fOctreeDownSampling;

	float param;
	std::string strIntermediateSampleFolder;
	std::string strGPMapMetaDataSampleFolder;
	// no sampling
	if(fOctreeDownSampling < 0)
	{
		strIntermediateSampleFolder	= strIntermediateFolder;
		strGPMapMetaDataSampleFolder	= strGPMapMetaDataFolder;
	}
	else if(fOctreeDownSampling > 0)
	{
		// leaf size
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)

		// sub-folder
		std::stringstream ss;
		ss << "down_sampling_" << param << "/";
		strIntermediateSampleFolder	= strIntermediateFolder		+ ss.str();
		strGPMapMetaDataSampleFolder	= strGPMapMetaDataFolder	+ ss.str();
	}
	else
	{
		// sampling ratio
		std::cout << "Random sampling ratio: ";
		std::cin >> param;	// 0.5, 0.3, 0.2, 0.1

		// sub-folder
		std::stringstream ss;
		ss << "random_sampling_" << param << "/";
		strIntermediateSampleFolder	= strIntermediateFolder		+ ss.str();
		strGPMapMetaDataSampleFolder	= strGPMapMetaDataFolder	+ ss.str();
	}
	const std::string strLogFolder						(strGPMapMetaDataSampleFolder + "log/");
	create_directory(strGPMapMetaDataSampleFolder);
	create_directory(strLogFolder);

	// removed some parts
	bool fRemoved;
	std::cout << "Use data whose part are removed? (0/1) ";
	std::cin >> fRemoved;

	// [0] Show
	bool fShow = true;
	std::cout << "Would you like to see the octree? (0/1) "; std::cin >> fShow;
	if(fShow)
	{
		// downsampling
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)

		// file name
		std::stringstream ss;
		if(fRemoved)	ss << "_func_obs_removed_downsampled_" << param << ".pcd";
		else				ss << "_func_obs_downsampled_" << param << ".pcd";

		// Function Observations (Hit Points + Unit Ray Back Vectors) - All - Down Sampling
		PointNormalCloudPtr pAllSampledFuncObs(new PointNormalCloud());
		loadPointCloud<pcl::PointNormal>(pAllSampledFuncObs, strFileNameAll, strIntermediateSampleFolder, ss.str());

		// get bounding box
		pcl::PointXYZ min_pt, max_pt;
		getMinMaxPointXYZ<pcl::PointNormal>(*pAllSampledFuncObs, min_pt, max_pt);

		// GPMap
		typedef OctreeGPMapContainer<GaussianDistribution>	LeafT;
		typedef OctreeGPMap<GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs, LeafT> OctreeGPMapT;
		OctreeGPMapT gpmap(BLOCK_SIZE, 
								 NUM_CELLS_PER_AXIS, 
								 MIN_NUM_POINTS_TO_PREDICT, 
								 MAX_NUM_POINTS_TO_PREDICT, 
								 FLAG_INDEPENDENT_TEST_POSITIONS);

		// set bounding box
		gpmap.defineBoundingBox(min_pt, max_pt);

		// set input cloud
		gpmap.setInputCloud(pAllSampledFuncObs, GAP);

		// add points from the input cloud
		gpmap.addPointsFromInputCloud();

		// Octree Viewer
		OctreeViewer<pcl::PointNormal, OctreeGPMapT> octreeViewer(gpmap);
	}

	// [0] setting - Local GP or Glocal G
	bool fLocalGP;
	std::cout << "[Glocal GP (0) or Local GP (1)] ";
	std::cin >> fLocalGP;
	//if(!fLocalGP)
	//{
		// downsampling
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)
	//}

	// [1] Function Observations
	bool fRunFuncObs = true;
	std::cout << "[Function Observations] - Do you wish to run? (0/1) "; std::cin >> fRunFuncObs;
	if(fRunFuncObs)
	{
		//double			BLOCK_SIZE;						std::cout << "Block Size? ";															std::cin >> BLOCK_SIZE;
		//size_t			NUM_CELLS_PER_AXIS;			std::cout << "Number of cells per axis? ";										std::cin >> NUM_CELLS_PER_AXIS;
		//size_t			MAX_NUM_POINTS_TO_PREDICT;	std::cout << "Max Number of points for Gaussian process prediction? ";	std::cin >> MAX_NUM_POINTS_TO_PREDICT;

		std::stringstream ss;
		ss << "block_" << BLOCK_SIZE 
			<< "_cell_" << static_cast<float>(BLOCK_SIZE)/static_cast<float>(NUM_CELLS_PER_AXIS)
			<< "_m_" << MIN_NUM_POINTS_TO_PREDICT
			<< "_n_" << MAX_NUM_POINTS_TO_PREDICT
			<< "_gap_" << GAP;
		if(fRemoved)	ss << "_func_obs_removed";
		else				ss << "_func_obs";

		// filenames
		const std::string strObsFileName						= ss.str() + (fLocalGP ? "_local" : "_glocal");
		const std::string strObsFileNameSuffix				= fRemoved ? "_func_obs_removed.pcd" : "_func_obs.pcd";

		// train hyperparameters and build gpmaps
		if(fLocalGP)
		{
			// local hyperparameters
			// set default hyperparameters
			//const float sparse_ell	= 0.005f;	//0.141362;
			//const float matern_ell	= 0.005f;	//0.12342f;
			//const float sigma_f		= 1.f;		//0.235194f;
			//const float sigma_n		= 0.1f;		//0.00033809f;
			//const float sigma_nd		= 1.00000f;		// will not be used here

			// NO_RANDOM_SAMPLING, BLOCK_SIZE = 0.15, MAX_NUM_POINTS_TO_PREDICT = 200, ACRA 2014
			//const float sparse_ell	= 0.0539592f;
			//const float matern_ell	= 0.0326716f;
			//const float sigma_f		= 0.308823f;
			//const float sigma_n		= 0.00493079f;
			//const float sigma_nd		= 0.977637f;		// will not be used here

			// RANDOM_SAMPLING = 0.1, BLOCK_SIZE = 0.02, MAX_NUM_POINTS_TO_PREDICT = 100
			//const float sparse_ell	= 0.0990733f;
			//const float matern_ell	= 0.0779094f;
			//const float sigma_f		= 0.140968f;
			//const float sigma_n		= 0.000265014f;
			//const float sigma_nd		= 0.900084f;			// will not be used here
		
			//typedef	GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs>::Hyp		LocalHyp;
			//LocalHyp logLocalHyp;
			//logLocalHyp.cov(0) = log(sparse_ell);
			//logLocalHyp.cov(1) = log(matern_ell);
			//logLocalHyp.cov(2) = log(sigma_f);
			//logLocalHyp.lik(0) = log(sigma_n);
			//logLocalHyp.lik(1) = log(sigma_nd);

			// run
			//train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
			//													GP::CovSparseMaternisoDerObs, 
			//													GP::LikGaussDerObs, 
			//													GP::InfExactDerObs>(BLOCK_SIZE,							// block size
			//																			  NUM_CELLS_PER_AXIS,				// number of cells per each axie
			//																			  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
			//																			  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
			//																			  logLocalHyp,								// hyperparameters
			//																			  strObsFileNames,					// sequential observations input file name
			//																			  strFileNameAll,						// all-in-one observations input file name
			//																			  strIntermediateSampleFolder,	// input file name prefix
			//																			  strObsFileNameSuffix,				// input file name suffix
			//																			  GAP,									// gap
			//																			  MAX_ITER_BEFORE_UPDATE,			// number of iterations for training before update
			//																			  strGPMapMetaDataSampleFolder,	// output data folder
			//																			  strObsFileName,						// output file name prefix
			//																			  strLogFolder);						// log folder

			//// CovSEiso
			////const float ell		= 0.113708f;
			////const float sigma_f	= 1.02425f;
			////const float sigma_n	= 0.0112541f;			
			//float ell		= 0.5f;		std::cout << "ell = ";		std::cin >> ell;
			//float sigma_f	= 1.f;		std::cout << "sigma_f = "; std::cin >> sigma_f;
			//float sigma_n	= 0.01f;		std::cout << "sigma_n = "; std::cin >> sigma_n;		

			//typedef	GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovSEisoDerObs, GP::LikGaussDerObs>::Hyp		LocalHyp;
			//LocalHyp logLocalHyp;
			//logLocalHyp.cov(0) = log(ell);
			//logLocalHyp.cov(1) = log(sigma_f);
			//logLocalHyp.lik(0) = log(sigma_n);
			//logLocalHyp.lik(1) = log(sigma_nd);

			//// run
			//const std::string strObsFileNameSuffix	= "_func_obs_removed_downsampled_0.01.pcd";
			//const float GAP = 0.01;
			//train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
			//													GP::CovSEisoDerObs, 
			//													GP::LikGaussDerObs, 
			//													GP::InfExactDerObs>(BLOCK_SIZE,							// block size
			//																			  NUM_CELLS_PER_AXIS,				// number of cells per each axie
			//																			  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
			//																			  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
			//																			  logLocalHyp,								// hyperparameters
			//																			  strObsFileNames,					// sequential observations input file name
			//																			  strFileNameAll,						// all-in-one observations input file name
			//																			  strIntermediateSampleFolder,	// input file name prefix
			//																			  strObsFileNameSuffix,				// input file name suffix
			//																			  GAP,									// gap
			//																			  MAX_ITER_BEFORE_UPDATE,			// number of iterations for training before update
			//																			  strGPMapMetaDataSampleFolder,	// output data folder
			//																			  strObsFileName,						// output file name prefix
			//																			  strLogFolder);						// log folder

			// filenames
			std::stringstream ss;
			//ss << strIntermediateSampleFolder << strFileNameAll << "_func_obs_downsampled_" << param << ".pcd";
			ss << "_func_obs" << (fRemoved ? "_removed" : "") << "_downsampled_" << param << ".pcd";
			const std::string strObsFileNameSuffix	= ss.str();

			// CovRQiso
			//float ell		= 0.016659f;		std::cout << "ell = ";		std::cin >> ell;
			//float alpha		= 7.55446f;			std::cout << "alpha = ";	std::cin >> alpha;
			//float sigma_f	= 0.0117284f;		std::cout << "sigma_f = "; std::cin >> sigma_f;
			//float sigma_n	= 0.00186632f;		std::cout << "sigma_n = "; std::cin >> sigma_n;		
			const float ell		= 4.11029f;
			const float alpha		= 1.22971f;
			const float sigma_f	= 0.981831f;
			const float sigma_n	= 0.00289925f;	

			typedef	GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovRQisoDerObs, GP::LikGaussDerObs>::Hyp		LocalHyp;
			LocalHyp logLocalHyp;
			logLocalHyp.cov(0) = log(ell);
			logLocalHyp.cov(1) = log(alpha);
			logLocalHyp.cov(2) = log(sigma_f);
			logLocalHyp.lik(0) = log(sigma_n);
			logLocalHyp.lik(1) = log(1.f);

			// run
			//const float GAP = 0.01;
			train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
																GP::CovRQisoDerObs, 
																GP::LikGaussDerObs, 
																GP::InfExactDerObs>(BLOCK_SIZE,							// block size
																						  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																						  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																						  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																						  logLocalHyp,								// hyperparameters
																						  strObsFileNames,					// sequential observations input file name
																						  strFileNameAll,						// all-in-one observations input file name
																						  strIntermediateSampleFolder,	// input file name prefix
																						  strObsFileNameSuffix,				// input file name suffix
																						  GAP,									// gap
																						  MAX_ITER_BEFORE_UPDATE,			// number of iterations for training before update
																						  strGPMapMetaDataSampleFolder,	// output data folder
																						  strObsFileName,						// output file name prefix
																						  strLogFolder);						// log folder
		}
		else
		{
			// filenames
			std::stringstream ss;
			//ss << strIntermediateSampleFolder << strFileNameAll << "_func_obs_downsampled_" << param << ".pcd";
			ss << "_func_obs" << (fRemoved ? "_removed" : "") << "_downsampled_" << param << ".pcd";
			const std::string strGlobalObsFileNameSuffix	= ss.str();

			// global hyperparameters - local training data
			//const float ell		= 0.016659f;
			//const float alpha		= 7.55446f;
			//const float sigma_f	= 0.0117284f;
			//const float sigma_n	= 0.00186632f;		

			// global hyperparameters - global training data
			//const float ell		= 11.7067f;
			//const float alpha		= 7.89965f;
			//const float sigma_f	= 0.000911995f;
			//const float sigma_n	= 0.000502061f;	

			// global hyperparameters: global downsample = 0.01, nlZ = -4692.91
			//const float g_ell			= 3.51052f;
			//const float g_alpha		= 1.14496f;
			//const float g_sigma_f	= 0.993788f;
			//const float g_sigma_n	= 0.00999004f;	

			// global hyperparameters: global downsample = 0.02, nlZ = -1742.08
			const float g_ell			= 4.11029f;
			const float g_alpha		= 1.22971f;
			const float g_sigma_f	= 0.981831f;
			const float g_sigma_n	= 0.00289925f;	

			GP::MeanGlobalGP<float>::GlobalHyp logGlobalHyp;
			logGlobalHyp.cov(0) = log(g_ell);
			logGlobalHyp.cov(1) = log(g_alpha);
			logGlobalHyp.cov(2) = log(g_sigma_f);
			logGlobalHyp.lik(0) = log(g_sigma_n);
			logGlobalHyp.lik(1) = log(1.f);

			// local hyperparameters: global downsample = 0.01, -158166
			//const float l_sparse_ell	= 0.185507f;
			//const float l_matern_ell	= 0.258412f;
			//const float l_sigma_f		= 0.16042f;
			//const float l_sigma_n		= 0.00056761f;
			//const float l_sigma_nd		= 0.947003f;		// will not be used here

			// local hyperparameters: global downsample = 0.02, -165499
			const float l_sparse_ell	= 0.0310558f;
			const float l_matern_ell	= 0.0325278f;
			const float l_sigma_f		= 0.0614561f;
			const float l_sigma_n		= 0.000278377f;
			const float l_sigma_nd		= 0.00101494f;		// will not be used here

			typedef	GP::InfExactDerObs<float, GP::MeanGlobalGP, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs>::Hyp		LocalHyp;
			LocalHyp logLocalHyp;
			logLocalHyp.cov(0) = log(l_sparse_ell);
			logLocalHyp.cov(1) = log(l_matern_ell);
			logLocalHyp.cov(2) = log(l_sigma_f);
			logLocalHyp.lik(0) = log(l_sigma_n);
			logLocalHyp.lik(1) = log(l_sigma_nd);

			// run
			train_hyperparameters_and_build_gpmaps_using_MeanGlobalGP<GP::CovSparseMaternisoDerObs, 
																						 GP::LikGaussDerObs, 
																						 GP::InfExactDerObs>(BLOCK_SIZE,							// block size
																													NUM_CELLS_PER_AXIS,				// number of cells per each axie
																													MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																													MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																													logGlobalHyp,						// global hyperparameters
																													logLocalHyp,						// hyperparameters
																													strObsFileNames,					// sequential observations input file name
																													strFileNameAll,					// all-in-one observations input file name
																													strIntermediateSampleFolder,	// input file name prefix
																													strGlobalObsFileNameSuffix,	// global input file name suffix
																													strObsFileNameSuffix,			// input file name suffix
																													GAP,									// gap
																													MAX_ITER_BEFORE_UPDATE,			// number of iterations for training before update
																													strGPMapMetaDataSampleFolder,	// output data folder
																													strObsFileName,						// output file name prefix
																													strLogFolder);						// log folder
		}
	}

	// [2] Derivative Observations
	bool fRunDerObs = true;
	std::cout << "[Derivative Observations] - Do you wish to run? (0/1) "; std::cin >> fRunDerObs;
	if(fRunDerObs)
	{
		//double			BLOCK_SIZE;						std::cout << "Block Size? ";															std::cin >> BLOCK_SIZE;
		//size_t			NUM_CELLS_PER_AXIS;			std::cout << "Number of cells per axis? ";										std::cin >> NUM_CELLS_PER_AXIS;
		//size_t			MAX_NUM_POINTS_TO_PREDICT;	std::cout << "Max Number of points for Gaussian process prediction? ";	std::cin >> MAX_NUM_POINTS_TO_PREDICT;

		std::stringstream ss;
		ss << "block_" << BLOCK_SIZE 
			<< "_cell_" << static_cast<float>(BLOCK_SIZE)/static_cast<float>(NUM_CELLS_PER_AXIS)
			<< "_m_" << MIN_NUM_POINTS_TO_PREDICT
			<< "_n_" << MAX_NUM_POINTS_TO_PREDICT
			<< "_gap_" << GAP;
		if(fRemoved)	ss << "_der_obs_removed";
		else				ss << "_der_obs";

		// filenames
		const std::string strObsFileName						= ss.str() + (fLocalGP ? "_local" : "_glocal");
		const std::string strObsFileNameSuffix				= ss.str() + ".pcd";

		//// set default hyperparameters
		const float sparse_ell	= 0.0268108f;	
		const float matern_ell	= 0.0907317f;	
		const float sigma_f		= 0.0144318f;	
		const float sigma_n		= 4.31269e-007f;	// will not be used
		const float sigma_nd		= 0.017184f;	

		// BLOCK_SIZE = 0.009, MAX_NUM_POINTS_TO_PREDICT = 100
		//const float sparse_ell	= 0.0200197f;	
		//const float matern_ell	= 0.0311088f;	
		//const float sigma_f		= 0.00380452f;	
		//const float sigma_n		= 1.55892e-008f;	// will not be used
		//const float sigma_nd		= 0.0959042f;	

		// BLOCK_SIZE = 0.015, MAX_NUM_POINTS_TO_PREDICT = 100
		//// set default hyperparameters
		//const float sparse_ell	= 0.005f;	//0.0200197f;	
		//const float matern_ell	= 0.005f;	//0.028985f;	
		//const float sigma_f		= 1.f;		//0.00380452f;	
		//const float sigma_n		= 0.1f;		//5.34342e-009f;	// will not be used
		//const float sigma_nd		= 0.1f;		//0.0959042f;	

		// local hyperparameters
		typedef	GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs>::Hyp		LocalHyp;
		LocalHyp logLocalHyp;
		logLocalHyp.cov(0) = log(sparse_ell);
		logLocalHyp.cov(1) = log(matern_ell);
		logLocalHyp.cov(2) = log(sigma_f);
		logLocalHyp.lik(0) = log(sigma_n);
		logLocalHyp.lik(1) = log(sigma_nd);

		// train hyperparameters and build gpmaps
		train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
															GP::CovSparseMaternisoDerObs, 
															GP::LikGaussDerObs, 
															GP::InfExactDerObs>(BLOCK_SIZE,							// block size
																					  NUM_CELLS_PER_AXIS,				// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,		// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,		// max number of points to predict
																					  logLocalHyp,							// hyperparameters
																					  strObsFileNames,					// sequential observations input file name
																					  strFileNameAll,						// all-in-one observations input file name
																					  strIntermediateSampleFolder,	// input file name prefix
																					  strObsFileNameSuffix,				// input file name suffix
																					  GAP,									// gap
																					  MAX_ITER_BEFORE_UPDATE,			// number of iterations for training before update
																					  strGPMapMetaDataSampleFolder,	// output data folder
																					  strObsFileName,						// output file name prefix
																					  strLogFolder);						// log folder
	}

	// [3] Derivative Observations

	system("pause");
}

#endif