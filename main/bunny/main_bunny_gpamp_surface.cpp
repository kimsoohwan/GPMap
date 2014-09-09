#if 1

// STL
#include <string>
#include <vector>
#include <sstream>

// GPMap
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "iso_surface/iso_surface.hpp"		// IsoSurfaceExtraction
using namespace GPMap;

int main(int argc, char** argv)
{
	// [0] setting - directory
	const std::string strGPMapDataFolder	("../../data/bunny/output/gpmap/random_sampling_0.1/");
	const std::string strMetaDataFolder		(strGPMapDataFolder + "meta_data/");
	const std::string strOutputDataFolder0	(strGPMapDataFolder + "surface/");
	create_directory(strOutputDataFolder0);

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

	// surface reconstruction
	//const float	RESOLUTION = 0.001f;
	const float	RESOLUTION = 0.002f;

	bool fRunCustom;
	std::cout << "Run default(0) or custom(1)? ";
	std::cin >> fRunCustom;

	bool fMarchingCubes;
	std::cout << "Marching Cubes(0) or Marching Tetrahedron(1)? ";
	std::cin >> fMarchingCubes;
	std::cout << fMarchingCubes << std::endl;
	fMarchingCubes = !fMarchingCubes;

	std::string strMethodName;
	if(fMarchingCubes)	strMethodName = "_marching_cubes_";
	else						strMethodName = "_marching_etrahedrons_";

	if(!fRunCustom)
	{
		std::stringstream ss;
		ss << strOutputDataFolder0 << "default/";
		const std::string strOutputDataFolder			(ss.str());
		const std::string strLogFolder					(strOutputDataFolder + "log/");
		create_directory(strOutputDataFolder);
		create_directory(strLogFolder);

		// [1] Convert GPMaps to Surface
		for(size_t i = 0; i < strGPMapFileNames.size(); i++)
		{
			std::cout << "================[ " << strGPMapFileNames[i] << " ]================" << std::endl;

			// log file
			std::string	strLogFilePath = strLogFolder + strGPMapFileNames[i] + "_surface.log";
			LogFile logFile(strLogFilePath);

			// load GPMap
			pcl::PointCloud<pcl::PointNormal>::Ptr pPointCloudGPMap;
			loadPointCloud<pcl::PointNormal>(pPointCloudGPMap, strMetaDataFolder + strGPMapFileNames[i] + ".pcd");

			// iso-surface
			IsoSurfaceExtraction isoSurface(RESOLUTION);

			// convert GPMap to surface
			isoSurface.insertMeanVarFromGPMap(*pPointCloudGPMap);

			// zero-value iso-surface
			if(fMarchingCubes)		isoSurface.marchingcubes();			// marching cubes
			else							isoSurface.marchingTetrahedron();	// marching tetrahedron

			// min max
			const float minX = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_X);
			const float maxX = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_X);
			const float minY = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Y);
			const float maxY = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Y);
			const float minZ = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Z);
			const float maxZ = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Z);
			const float minV = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_V);
			const float maxV = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_V);

			logFile << "X: min = " << minX << ", max = " << maxX << std::endl;
			logFile << "Y: min = " << minY << ", max = " << maxY << std::endl;
			logFile << "Z: min = " << minZ << ", max = " << maxZ << std::endl;
			logFile << "V: min = " << minV << ", max = " << maxV << std::endl;
			logFile << std::endl;

			// color by Z
			{
				std::stringstream ss;
				ss << strOutputDataFolder << strGPMapFileNames[i] << "_Z_min_" << minZ << "_max_" << maxZ << ".ply";
				//ss << strOutputDataFolder << strMethodName << "_min_z_" << minZ << "_max_z_" << maxZ << ".ply";
				isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_Z, minZ, maxZ);
				logFile << "Saving " << ss.str() << " ... ";
				isoSurface.saveAsPLY(ss.str());
				logFile << "done." << std::endl;
			}

			// color by V
			{
				std::stringstream ss;
				ss << strOutputDataFolder << strGPMapFileNames[i] << "_V_min_" << minV << "_max_" << maxV << ".ply";
				//ss << strOutputDataFolder << strMethodName << "_min_z_" << minZ << "_max_z_" << maxZ << ".ply";
				isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_V, minV, maxV);
				logFile << "Saving " << ss.str() << " ... ";
				isoSurface.saveAsPLY(ss.str());
				logFile << "done." << std::endl;
			}
		}
	}
	else
	{
		// particular range
		while(true)
		{
			// color mode
			char  cColorMode;			std::cout << "color mode (q for quit): "; std::cin >> cColorMode;
			if(cColorMode == 'q' || cColorMode == 'Q') break;

			// color map
			float maxVarThld; std::cout << "max var thld: ";				std::cin >> maxVarThld;
			std::stringstream ss;
			ss << strOutputDataFolder0 << "max_var_" << maxVarThld << "/";
			const std::string strOutputDataFolder			(ss.str());
			const std::string strLogFolder					(strOutputDataFolder + "log/");
			create_directory(strOutputDataFolder);
			create_directory(strLogFolder);

			float minVarRange; std::cout << "min var color range: ";		std::cin >> minVarRange;
			float maxVarRange; std::cout << "max var color range: ";		std::cin >> maxVarRange;

			for(size_t i = 0; i < strGPMapFileNames.size(); i++)
			{
				std::cout << "================[ " << strGPMapFileNames[i] << " ]================" << std::endl;

				// log file
				std::string	strLogFilePath = strLogFolder + strGPMapFileNames[i] + "_surface.log";
				LogFile logFile(strLogFilePath);

				// load GPMap
				pcl::PointCloud<pcl::PointNormal>::Ptr pPointCloudGPMap;
				loadPointCloud<pcl::PointNormal>(pPointCloudGPMap, strMetaDataFolder + strGPMapFileNames[i] + ".pcd");

				// iso-surface
				IsoSurfaceExtraction isoSurface(RESOLUTION);

				// convert GPMap to surface
				isoSurface.insertMeanVarFromGPMap(*pPointCloudGPMap, maxVarThld);

				// zero-value iso-surface
				if(fMarchingCubes)		isoSurface.marchingcubes();			// marching cubes
				else							isoSurface.marchingTetrahedron();	// marching tetrahedron

				// min max
				//const float minX = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_X);
				//const float maxX = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_X);
				//const float minY = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Y);
				//const float maxY = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Y);
				const float minZ = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Z);
				const float maxZ = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_Z);
				//const float minV = isoSurface.getMinVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_V);
				//const float maxV = isoSurface.getMaxVertexValueForColor(IsoSurfaceExtraction::COLOR_BY_V);

				// set color
				//std::stringstream ss;
				//switch(cColorMode)
				//{
				//case 'x':
				//case 'X': 
				//	ss << strOutputDataFolder << strGPMapFileNames[i] << strMethodName << "X_min_" << minVarRange << "_max_" << maxVarThld << ".ply";
				//	isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_X);
				//	break;
				//case 'y':
				//case 'Y': 
				//	ss << strOutputDataFolder << strGPMapFileNames[i] << strMethodName << "Y_min_" << minVarRange << "_max_" << maxVarThld << ".ply";
				//	isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_Y);
				//	break;
				//case 'z':
				//case 'Z': 
				//	ss << strOutputDataFolder << strGPMapFileNames[i] << strMethodName << "Z_min_" << minVarRange << "_max_" << maxVarThld << ".ply";
				//	isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_Z);
				//	break;
				//case 'v':
				//case 'V': 
				//	ss << strOutputDataFolder << strGPMapFileNames[i] << strMethodName << "V_min_" << minVarRange << "_max_" << maxVarThld << ".ply";
				//	isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_V);
				//	break;
				//}
				//
				// save
				//logFile << "Saving " << ss.str() << " ... ";
				//isoSurface.saveAsPLY(ss.str());
				//logFile << "done." << std::endl;

				// z
				std::stringstream ss1;
				ss1 << strOutputDataFolder << strGPMapFileNames[i] << "_Z.ply";
				isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_Z);
				logFile << "Saving " << ss1.str() << " ... ";
				isoSurface.saveAsPLY(ss1.str());
				logFile << "done." << std::endl;

				// v
				std::stringstream ss2;
				ss2 << strOutputDataFolder << strGPMapFileNames[i] << "_V_min_" << minVarRange << "_max_" << maxVarRange << ".ply";
				isoSurface.setVertexColors(IsoSurfaceExtraction::COLOR_BY_V, minVarRange, maxVarThld);
				logFile << "Saving " << ss2.str() << " ... ";
				isoSurface.saveAsPLY(ss2.str());
				logFile << "done." << std::endl;
			}
		}
	}

	system("pause");
}

#endif