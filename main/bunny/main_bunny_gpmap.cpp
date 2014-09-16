#if 1
// Eigen
#include "serialization/eigen_serialization.hpp" // Eigen
// includes followings inside of it
//		- #define EIGEN_NO_DEBUG		// to speed up
//		- #define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
//		- #include <Eigen/Core>

// combinations
// ------------------------------------------------------------------------------------------------------------------------------------|
// | Coverage |                   Update                       |                             Observations                              |
// |===================================================================================================================================|
// |  Local   | Gaussian Distribution (No Update, Batch)       | All        Obs | Function   Obs (hit/empty points)                    |
// |  Global  | Independent Bayesian Committee Machines (iBCM) | Sequential Obs | Derivative Obs (virtual hit points, surface normals) |
// |  Glocal  |   Dependent Bayeisian Committee Machines (BCM) |                |                                                      |
// ------------------------------------------------------------------------------------------------------------------------------------|

// FSR  2013:   Local  + Gaussian Distribution + All Function Observations
// ACRA 2014:   Local  + iBCM                  + Sequential Function Observations
// (comparison)           BCM
// ICRA 2015:   Global + Gaussian Distribution + All Derivative Observations
//              Local  + Gaussian Distribution + All Function Observations
// (comparison) Global + Gaussian Distribution + All Derivative Observations
// TRO  2015:   Global + iBCM						  + Sequential Derivative Observations
//              Local  + iBCM						  + Sequential Function Observations

// GPMap
#include "gpmap/macro_gpmap.hpp"			// macro_gpmap
using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] Setting - GPMap constants
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

	// [0] Setting - Directory
	const std::string strDataFolder						("E:/Documents/GitHub/Data/");
	const std::string strDataName							("bunny");
	const std::string strInputFolder						(strDataFolder + strDataName + "/input/");
	const std::string strIntermediateFolder			(strDataFolder + strDataName + "/intermediate/");
	const std::string strSampledIntermediateFolder	(strIntermediateFolder + "random_sampling_0.1/");
	const std::string strGPMapFolder						(strDataFolder + strDataName + "/output/gpmap/");
	const std::string strGPMapMetaDataFolder			(strGPMapFolder + "meta_data/");
	const std::string strLogFolder						(strGPMapMetaDataFolder + "log/");
	create_directory(strGPMapFolder);
	create_directory(strGPMapMetaDataFolder);
	create_directory(strLogFolder);

	// [0] Setting - Observations
	const size_t NUM_OBSERVATIONS = 4; 
	const std::string strObsFileNames_[]		= {"bun000", "bun090", "bun180", "bun270"};
	const std::string strFileNameAll				=  "bunny_all";
	StringList strObsFileNameList(strObsFileNames_, strObsFileNames_ + NUM_OBSERVATIONS); 

	// [1] Setting - Combinations
	int combination = 3;
	std::cout << "[Combination] FSR 2013 (1) / ACRA 2014 (2) / ICRA 2015 (3) / TRO 2015 (4)? " << std::endl;
	std::cin >> combination;

	switch(combination)
	{
	// [1] FSR 2013
	case 1:
		{
			// input data
			bool fUseRemovedData;
			std::cout << "\tUse removed data? " << std::endl;
			std::cin >> fUseRemovedData;
			const std::string strInputLocalFileNameSuffix = fUseRemovedData ? "_func_obs_removed.pcd" : "_func_obs.pcd";
			bool fShow = true; std::cout << "Would you like to see local observations in the octree? (0/1) "; std::cin >> fShow;
			if(fShow) show(BLOCK_SIZE, strFileNameAll, strSampledIntermediateFolder, strInputLocalFileNameSuffix);

			// output data
			const std::string strOutputFileName = fUseRemovedData ? "ICRA_2015_Local_GP" : "FSR_2013";

			// local hyperparameters: -2.91161e6
			//const float l_sparse_ell	= 0.0539592f;
			//const float l_matern_ell	= 0.0326716f;
			//const float l_sigma_f		= 0.308823f;
			//const float l_sigma_n		= 0.00493079f;
			//const float l_sigma_nd		= 0.977637f;		// will not be used here

			// local hyperparameters: removed, -4.56377e6
			const float l_sparse_ell	= 0.0880937f;
			const float l_matern_ell	= 0.0649332f;
			const float l_sigma_f		= 0.185594f;
			const float l_sigma_n		= 0.00023501f;
			const float l_sigma_nd		= 0.977637f;		// will not be used here
		
			GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs>::Hyp logLocalHyp;
			logLocalHyp.cov(0) = log(l_sparse_ell);
			logLocalHyp.cov(1) = log(l_matern_ell);
			logLocalHyp.cov(2) = log(l_sigma_f);
			logLocalHyp.lik(0) = log(l_sigma_n);
			logLocalHyp.lik(1) = log(l_sigma_nd);

			const size_t MAX_NUM_POINTS_TO_PREDICT = 0;
			train_hyperparameters_and_build_gpmaps_batch_with_all_in_one_observations<GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>
				(LOCAL_GP,
				 BLOCK_SIZE,
				 NUM_CELLS_PER_AXIS,
				 MIN_NUM_POINTS_TO_PREDICT,
				 MAX_NUM_POINTS_TO_PREDICT,
				 logLocalHyp,
				 strFileNameAll,
				 strSampledIntermediateFolder,
				 strInputLocalFileNameSuffix,
				 GAP,
				 MAX_ITER_BEFORE_UPDATE,
				 strGPMapMetaDataFolder,
				 strOutputFileName,
				 strLogFolder);
			break;
		}
		
	// [2] ACRA 2014
	case 2:
		{
			// input data
			std::string strInputLocalFileNameSuffix	= "_func_obs.pcd";
			bool fShow = true; std::cout << "Would you like to see local observations in the octree? (0/1) "; std::cin >> fShow;
			if(fShow) show(BLOCK_SIZE, strFileNameAll, strSampledIntermediateFolder, strInputLocalFileNameSuffix);

			// output data
			std::string strOutputFileName					= "ACRA_2014";

			// local hyperparameters
			const float l_sparse_ell	= 0.0539592f;
			const float l_matern_ell	= 0.0326716f;
			const float l_sigma_f		= 0.308823f;
			const float l_sigma_n		= 0.00493079f;
			const float l_sigma_nd		= 0.977637f;		// will not be used here
		
			GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs>::Hyp logLocalHyp;
			logLocalHyp.cov(0) = log(l_sparse_ell);
			logLocalHyp.cov(1) = log(l_matern_ell);
			logLocalHyp.cov(2) = log(l_sigma_f);
			logLocalHyp.lik(0) = log(l_sigma_n);
			logLocalHyp.lik(1) = log(l_sigma_nd);

			train_hyperparameters_and_build_local_gpmaps_incrementally_with_sequential_observations<GP::MeanZeroDerObs, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>
				(BLOCK_SIZE,
				 NUM_CELLS_PER_AXIS,
				 MIN_NUM_POINTS_TO_PREDICT,
				 MAX_NUM_POINTS_TO_PREDICT,
				 logLocalHyp,
				 strFileNameAll,
				 strObsFileNameList,
				 strSampledIntermediateFolder,
				 strInputLocalFileNameSuffix,
				 GAP,
				 MAX_ITER_BEFORE_UPDATE,
				 strGPMapMetaDataFolder,
				 strOutputFileName,
				 strLogFolder);
			break;
		}
		
	// [3] ICRA 2015
	case 3:
		{
			// input data
			std::string strInputLocalFileNameSuffix	= "_func_obs_removed.pcd";
			std::string strInputGlobalFileNameSuffix = "_der_obs_removed_downsampled_0.02.pcd";
			bool fShow = true; std::cout << "Would you like to see global observations in the octree? (0/1) "; std::cin >> fShow;
			if(fShow) show(BLOCK_SIZE, strFileNameAll, strSampledIntermediateFolder, strInputGlobalFileNameSuffix);

			// output data
			std::string strOutputFileName					= "ICRA_2015";

			// global hyperparameters
			//const float g_ell			= 0.016659f;
			//const float g_alpha		= 7.55446f;
			//const float g_sigma_f		= 0.0117284f;
			//const float g_sigma_n		= 0.00186632f;
			//const float g_sigma_nd	= 0.01f;
			const float g_ell			= 0.0782042f;
			const float g_alpha		= 0.110977f;
			const float g_sigma_f	= 0.0351056f;
			const float g_sigma_n	= 0.000869302f;
			const float g_sigma_nd	= 0.0765257f;

			GP::MeanGlobalGP<float>::GlobalHyp logGlocalHyp;
			logGlocalHyp.cov(0) = log(g_ell);
			logGlocalHyp.cov(1) = log(g_alpha);
			logGlocalHyp.cov(2) = log(g_sigma_f);
			logGlocalHyp.lik(0) = log(g_sigma_n);
			logGlocalHyp.lik(1) = log(g_sigma_nd);

			// local hyperparameters
			//const float l_sparse_ell	= 0.0539592f;
			//const float l_matern_ell	= 0.0326716f;
			//const float l_sigma_f		= 0.308823f;
			//const float l_sigma_n		= 0.00493079f;
			//const float l_sigma_nd	= 0.977637f;		// will not be used here
			const float l_sparse_ell	= 0.0708181f;
			const float l_matern_ell	= 0.0326748f;
			const float l_sigma_f		= 0.087249f;
			const float l_sigma_n		= 0.000511312f;
			const float l_sigma_nd		= 0.977035f;		// will not be used here
		
			GP::GaussianProcess<float, GP::MeanGlobalGP, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>::Hyp logLocalHyp;
			logLocalHyp.cov(0) = log(l_sparse_ell);
			logLocalHyp.cov(1) = log(l_matern_ell);
			logLocalHyp.cov(2) = log(l_sigma_f);
			logLocalHyp.lik(0) = log(l_sigma_n);
			logLocalHyp.lik(1) = log(l_sigma_nd);

			train_hyperparameters_and_build_global_local_gpmaps_batch_with_all_in_one_observations<GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>
				(BLOCK_SIZE,
				 NUM_CELLS_PER_AXIS,
				 MIN_NUM_POINTS_TO_PREDICT,
				 logGlocalHyp,
				 logLocalHyp,
				 strFileNameAll,
				 strSampledIntermediateFolder,
				 strInputGlobalFileNameSuffix,
				 strInputLocalFileNameSuffix,
				 GAP,
				 MAX_ITER_BEFORE_UPDATE,
				 strGPMapMetaDataFolder,
				 strOutputFileName,
				 strLogFolder);
			break;
		}

	// [4] TRO 2015
	case 4:
		{
			// input data
			std::string strInputLocalFileNameSuffix	= "_func_obs_removed.pcd";
			std::string strInputGlobalFileNameSuffix = "_der_obs_removed_downsampled_0.02.pcd";
			bool fShow = true; std::cout << "Would you like to see global observations in the octree? (0/1) "; std::cin >> fShow;
			if(fShow) show(BLOCK_SIZE, strFileNameAll, strSampledIntermediateFolder, strInputGlobalFileNameSuffix);

			// output data
			std::string strOutputFileName					= "TRO_2015";

			// global hyperparameters
			const float g_ell			= 0.016659f;
			const float g_alpha		= 7.55446f;
			const float g_sigma_f	= 0.0117284f;
			const float g_sigma_n	= 0.00186632f;
			const float g_sigma_nd	= 0.01f;

			GP::MeanGlobalGP<float>::GlobalHyp logGlocalHyp;
			logGlocalHyp.cov(0) = log(g_ell);
			logGlocalHyp.cov(1) = log(g_alpha);
			logGlocalHyp.cov(2) = log(g_sigma_f);
			logGlocalHyp.lik(0) = log(g_sigma_n);
			logGlocalHyp.lik(1) = log(g_sigma_nd);

			// local hyperparameters
			const float l_sparse_ell	= 0.0539592f;
			const float l_matern_ell	= 0.0326716f;
			const float l_sigma_f		= 0.308823f;
			const float l_sigma_n		= 0.00493079f;
			const float l_sigma_nd		= 0.977637f;		// will not be used here
		
			GP::GaussianProcess<float, GP::MeanGlobalGP, GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>::Hyp logLocalHyp;
			logLocalHyp.cov(0) = log(l_sparse_ell);
			logLocalHyp.cov(1) = log(l_matern_ell);
			logLocalHyp.cov(2) = log(l_sigma_f);
			logLocalHyp.lik(0) = log(l_sigma_n);
			logLocalHyp.lik(1) = log(l_sigma_nd);

			train_hyperparameters_and_build_global_local_gpmaps_incrementally_with_sequential_observations<GP::CovSparseMaternisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs>
				(BLOCK_SIZE,
				 NUM_CELLS_PER_AXIS,
				 MIN_NUM_POINTS_TO_PREDICT,
				 MAX_NUM_POINTS_TO_PREDICT,
				 logGlocalHyp,
				 logLocalHyp,
				 strFileNameAll,
				 strObsFileNameList,
				 strSampledIntermediateFolder,
				 strInputGlobalFileNameSuffix,
				 strInputLocalFileNameSuffix,
				 GAP,
				 MAX_ITER_BEFORE_UPDATE,
				 strGPMapMetaDataFolder,
				 strOutputFileName,
				 strLogFolder);
			break;
		}
	}

	system("pause");
}

#endif