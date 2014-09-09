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
	//const size_t	MAX_NUM_POINTS_TO_PREDICT			= 100; // too small?
	//const size_t	MAX_NUM_POINTS_TO_PREDICT			= 300; // good?
	const size_t	MIN_NUM_POINTS_TO_PREDICT			= 2; // 10;
	const bool		FLAG_DEPENDENT_TEST_POSITIONS		= false;
	const bool		FLAG_INDEPENDENT_TEST_POSITIONS	= true;
	const float		GAP										= 0.001f;
	const int		MAX_ITER_BEFORE_UPDATE				= 0;	// 100
	typedef	GP::InfExactDerObs<float, GP::MeanZeroDerObs, GP::CovMaternisoDerObs, GP::LikGaussDerObs>::Hyp Hyp;

	// [0] setting - directory
	const std::string strInputDataFolder			("../../data/bunny/input/");
	const std::string strIntermediateDataFolder0	("../../data/bunny/intermediate/");
	const std::string strGPMapDataFolder0			("../../data/bunny/output/gpmap/");

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
	std::string strIntermediateDataFolder;
	std::string strGPMapDataFolder1;
	if(fOctreeDownSampling > 0)
	{
		// leaf size
		std::cout << "Down sampling leaf size: ";
		std::cin >> param; // 0.001(50%), 0.002(20%), 0.003(10%)

		// sub folder
		std::stringstream ss;
		ss << "down_sampling_" << param << "/";
		strIntermediateDataFolder	= strIntermediateDataFolder0	+ ss.str();
		strGPMapDataFolder1	= strGPMapDataFolder0	+ ss.str();
	}
	else if(fOctreeDownSampling < 0)
	{
		strIntermediateDataFolder	= strIntermediateDataFolder0	+ "original/";
		strGPMapDataFolder1	= strGPMapDataFolder0	+ "original/";
	}
	else
	{
		// sampling ratio
		std::cout << "Random sampling ratio: ";
		std::cin >> param;	// 0.5, 0.3, 0.2, 0.1

		// sub folder
		std::stringstream ss;
		ss << "random_sampling_" << param << "/";
		strIntermediateDataFolder	= strIntermediateDataFolder0	+ ss.str();
		strGPMapDataFolder1	= strGPMapDataFolder0	+ ss.str();
	}
	const std::string strOutputDataFolder			(strGPMapDataFolder1 + "meta_data/");
	const std::string strLogFolder					(strOutputDataFolder + "log/");
	create_directory(strGPMapDataFolder1);
	create_directory(strOutputDataFolder);
	create_directory(strLogFolder);


	// [1] Function Observations
	bool fRunFuncObs = true;
	std::cout << "[Function Observations] - Do you wish to run? (0/1)"; std::cin >> fRunFuncObs;
	if(fRunFuncObs)
	{
		//const double	BLOCK_SIZE = 0.02;
		//const size_t	NUM_CELLS_PER_AXIS = 10;
		//const size_t	MAX_NUM_POINTS_TO_PREDICT = 200;
		double			BLOCK_SIZE;						std::cout << "Block Size? ";															std::cin >> BLOCK_SIZE;
		size_t			NUM_CELLS_PER_AXIS;			std::cout << "Number of cells per axis? ";										std::cin >> NUM_CELLS_PER_AXIS;
		size_t			MAX_NUM_POINTS_TO_PREDICT;	std::cout << "Max Number of points for Gaussian process prediction? ";	std::cin >> MAX_NUM_POINTS_TO_PREDICT;
		std::stringstream ss;
		ss << "block_" << BLOCK_SIZE 
			<< "_cell_" << static_cast<float>(BLOCK_SIZE)/static_cast<float>(NUM_CELLS_PER_AXIS)
			<< "_m_" << MIN_NUM_POINTS_TO_PREDICT
			<< "_n_" << MAX_NUM_POINTS_TO_PREDICT
			<< "_gap_" << GAP;

		// filenames
		const std::string strObsFileName						= ss.str() + "_func_obs";
		const std::string strObsFileNameSuffix				= "_func_obs.pcd";

		// set default hyperparameters
		//const float sparse_ell	= 0.005f;	//0.141362;
		//const float matern_ell	= 0.005f;	//0.12342f;
		//const float sigma_f2		= 1.f;		//0.235194f;
		//const float sigma_n		= 0.1f;		//0.00033809f;
		//const float sigma_nd		= 1.00000f;		// will not be used here

		// NO_RANDOM_SAMPLING, BLOCK_SIZE = 0.15, MAX_NUM_POINTS_TO_PREDICT = 200
		const float sparse_ell	= 0.0539592f;
		const float matern_ell	= 0.0326716f;
		const float sigma_f2		= 0.308823f;
		const float sigma_n		= 0.00493079f;
		const float sigma_nd		= 0.977637f;		// will not be used here

		// RANDOM_SAMPLING = 0.1, BLOCK_SIZE = 0.02, MAX_NUM_POINTS_TO_PREDICT = 100
		//const float sparse_ell	= 0.0119139f;
		//const float matern_ell	= 3.20892e-005f;
		//const float sigma_f2		= 0.0178554f;
		//const float sigma_n		= 0.000628464f;
		//const float sigma_nd		= 3.36231f;			// will not be used here

		Hyp logHyp;
		logHyp.cov(0) = log(sparse_ell);
		logHyp.cov(1) = log(matern_ell);
		logHyp.cov(2) = log(sigma_f2);
		logHyp.lik(0) = log(sigma_n);
		logHyp.lik(1) = log(sigma_nd);

		// train hyperparameters and build gpmaps
		train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
															GP::CovMaternisoDerObs, 
															GP::LikGaussDerObs, 
															GP::InfExactDerObs>(BLOCK_SIZE,						// block size
																					  NUM_CELLS_PER_AXIS,			// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																					  logHyp,							// hyperparameters
																					  strObsFileNames,				// sequential observations input file name
																					  strFileNameAll,					// all-in-one observations input file name
																					  strIntermediateDataFolder,	// input file name prefix
																					  strObsFileNameSuffix,			// input file name suffix
																					  GAP,								// gap
																					  MAX_ITER_BEFORE_UPDATE,		// number of iterations for training before update
																					  strOutputDataFolder,			// output data folder
																					  strObsFileName,					// output file name prefix
																					  strLogFolder);					// log folder
	}

	// [2] Derivative Observations
	bool fRunDerObs = true;
	std::cout << "[Derivative Observations] - Do you wish to run? (0/1)"; std::cin >> fRunDerObs;
	if(fRunDerObs)
	{
		strIntermediateDataFolder += "search_radius_0.01/";

		//const double	BLOCK_SIZE = 0.02;
		//const size_t	NUM_CELLS_PER_AXIS = 10;
		//const size_t	MAX_NUM_POINTS_TO_PREDICT = 200;
		double			BLOCK_SIZE;						std::cout << "Block Size? ";															std::cin >> BLOCK_SIZE;
		size_t			NUM_CELLS_PER_AXIS;			std::cout << "Number of cells per axis? ";										std::cin >> NUM_CELLS_PER_AXIS;
		size_t			MAX_NUM_POINTS_TO_PREDICT;	std::cout << "Max Number of points for Gaussian process prediction? ";	std::cin >> MAX_NUM_POINTS_TO_PREDICT;
		std::stringstream ss;
		ss << "block_" << BLOCK_SIZE 
			<< "_cell_" << static_cast<float>(BLOCK_SIZE)/static_cast<float>(NUM_CELLS_PER_AXIS)
			<< "_m_" << MIN_NUM_POINTS_TO_PREDICT
			<< "_n_" << MAX_NUM_POINTS_TO_PREDICT
			<< "_gap_" << GAP;

		// filenames
		const std::string strObsFileName						= ss.str() + "_der_obs";
		const std::string strObsFileNameSuffix				= "_der_obs.pcd";

		//// set default hyperparameters
		const float sparse_ell	= 0.0268108f;	
		const float matern_ell	= 0.0907317f;	
		const float sigma_f2		= 0.0144318f;	
		const float sigma_n		= 4.31269e-007f;	// will not be used
		const float sigma_nd		= 0.017184f;	

		// BLOCK_SIZE = 0.009, MAX_NUM_POINTS_TO_PREDICT = 100
		//const float sparse_ell	= 0.0200197f;	
		//const float matern_ell	= 0.0311088f;	
		//const float sigma_f2		= 0.00380452f;	
		//const float sigma_n		= 1.55892e-008f;	// will not be used
		//const float sigma_nd		= 0.0959042f;	

		// BLOCK_SIZE = 0.015, MAX_NUM_POINTS_TO_PREDICT = 100
		//// set default hyperparameters
		//const float sparse_ell	= 0.005f;	//0.0200197f;	
		//const float matern_ell	= 0.005f;	//0.028985f;	
		//const float sigma_f2		= 1.f;		//0.00380452f;	
		//const float sigma_n		= 0.1f;		//5.34342e-009f;	// will not be used
		//const float sigma_nd		= 0.1f;		//0.0959042f;	

		Hyp logHyp;
		logHyp.cov(0) = log(sparse_ell);
		logHyp.cov(1) = log(matern_ell);
		logHyp.cov(2) = log(sigma_f2);
		logHyp.lik(0) = log(sigma_n);
		logHyp.lik(1) = log(sigma_nd);

		// train hyperparameters and build gpmaps
		train_hyperparameters_and_build_gpmaps<GP::MeanZeroDerObs,
															GP::CovMaternisoDerObs, 
															GP::LikGaussDerObs, 
															GP::InfExactDerObs>(BLOCK_SIZE,						// block size
																					  NUM_CELLS_PER_AXIS,			// number of cells per each axie
																					  MIN_NUM_POINTS_TO_PREDICT,	// min number of points to predict
																					  MAX_NUM_POINTS_TO_PREDICT,	// max number of points to predict
																					  logHyp,							// hyperparameters
																					  strObsFileNames,				// sequential observations input file name
																					  strFileNameAll,					// all-in-one observations input file name
																					  strIntermediateDataFolder,	// input file name prefix
																					  strObsFileNameSuffix,			// input file name suffix
																					  GAP,								// gap
																					  MAX_ITER_BEFORE_UPDATE,		// number of iterations for training before update
																					  strOutputDataFolder,			// output data folder
																					  strObsFileName,					// output file name prefix
																					  strLogFolder);					// log folder
	}

	// [3] Derivative Observations

	system("pause");
}

#endif