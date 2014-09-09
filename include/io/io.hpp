#ifndef _POINT_CLOUD_INPUT_OUTPUT_HPP_
#define _POINT_CLOUD_INPUT_OUTPUT_HPP_

// STL
#include <string>
#include <vector>
#include <fstream>

// PCL
#include <pcl/point_types.h>		// pcl::PointXYZ, pcl::Normal, pcl::PointNormal
#include <pcl/point_cloud.h>		// pcl::PointCloud
#include <pcl/common/common.h>	// pcl::getMinMax3D
#include <pcl/filters/filter.h>	// pcl::removeNaNFromPointCloud
#include <pcl/io/pcd_io.h>			// pcl::io::loadPCDFile, savePCDFile
#include <pcl/io/ply_io.h>			// pcl::io::loadPLYFile, savePLYFile

// GP
#include "gp.h"						// LogFile
using GP::LogFile;

// GPMap
#include "util/data_types.hpp"	// PointXYZVList
#include "util/filesystem.hpp"	// extractFileExtension

namespace GPMap {

template <typename PointT>
int loadPointCloud(typename pcl::PointCloud<PointT>::Ptr		&pPointCloud, 
						 const std::string								&strFilePath, 
						 const bool											fAccumulation = false)
{
	// point cloud
	typename pcl::PointCloud<PointT>::Ptr pTempCloud(new pcl::PointCloud<PointT>());

	// load the file based on the file extension
	int result = -2;
	const std::string strFileExtension(extractFileExtension(strFilePath));
	if(strFileExtension.compare(".pcd") == 0)		result = pcl::io::loadPCDFile<PointT>(strFilePath.c_str(), *pTempCloud);
	if(strFileExtension.compare(".ply") == 0)		result = pcl::io::loadPLYFile<PointT>(strFilePath.c_str(), *pTempCloud);
	

	// error 
	switch(result)
	{
		case -2:
		{
			PCL_ERROR("Unknown file extension!");
			system("pause");
			break;
		}
		
		case -1:
		{
			PCL_ERROR("Couldn't read the file!");
			system("pause");
			break;
		}
	}

    //remove NaN Points
    pcl::removeNaNFromPointCloud(*pTempCloud, *pTempCloud, std::vector<int>());

	// accumulate
	if(fAccumulation)		*pPointCloud += *pTempCloud;
	else						pPointCloud = pTempCloud;

	return pPointCloud->size();
}

template <typename PointT>
inline void loadPointCloud(typename pcl::PointCloud<PointT>::Ptr		&pPointCloud,
									const std::string									&strFileName, 
									const std::string									&strPrefix,
									const std::string									&strSuffix)
{
	// log file
	LogFile logFile;
	logFile << "Loading " << strFileName << " ... ";

	// load
	logFile << loadPointCloud<PointT>(pPointCloud, strPrefix + strFileName + strSuffix) << " points." << std::endl;
}

template <typename PointT>
void loadPointCloud(std::vector<typename pcl::PointCloud<PointT>::Ptr>		&pPointClouds,
						   const StringList													&strFileNames, 
						   const std::string													&strPrefix = std::string(),
						   const std::string													&strSuffix = std::string())
{
	// resize
	pPointClouds.resize(strFileNames.size());

	// load files
	for(size_t i = 0; i < strFileNames.size(); i++)
		loadPointCloud<PointT>(pPointClouds[i], strFileNames[i], strPrefix, strSuffix);
}

template <typename PointT>
inline void savePointCloud(const typename pcl::PointCloud<PointT>::ConstPtr	&pPointCloud,
									const std::string												&strFilePath, 
									const bool														fBinary = true)
{
	// save the file based on the file extension
	const std::string strFileExtension(extractFileExtension(strFilePath));
	if(strFileExtension.compare(".pcd") == 0)		pcl::io::savePCDFile<PointT>(strFilePath.c_str(), *pPointCloud, fBinary);
	if(strFileExtension.compare(".ply") == 0)		pcl::io::savePLYFile<PointT>(strFilePath.c_str(), *pPointCloud, fBinary);
}

template <typename PointT>
inline void savePointCloud(const typename pcl::PointCloud<PointT>::ConstPtr	&pPointCloud,
									const std::string												&strFileName, 
									const std::string												&strPrefix,
									const std::string												&strSuffix,				 
									const bool														fBinary = true)
{
	// log file
	LogFile logFile;
	logFile << "Saving " << strFileName << " ... ";

	// save
	savePointCloud<PointT>(pPointCloud, strPrefix + strFileName + strSuffix, fBinary);

	logFile << " done." << std::endl;
}

template <typename PointT>
void savePointCloud(const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&pPointClouds,
						   const StringList															&strFileNames, 
						   const std::string															&strPrefix = std::string(),
						   const std::string															&strSuffix = std::string(),				 
						   const bool																	fBinary = true)
{
	// save the files
	for(size_t i = 0; i < strFileNames.size(); i++)
		savePointCloud<PointT>(pPointClouds[i], strFileNames[i], strPrefix, strSuffix, fBinary);
}


void loadSensorPositionList(PointXYZVList			&sensorPositionList, 
									 const StringList		&strFileNames, 
									 const std::string	&strPrefix = std::string(),
									 const std::string	&strSuffix = std::string())
{
	// log file
	LogFile logFile;

	// resize
	sensorPositionList.resize(strFileNames.size());

	// for each file name
	for(size_t i = 0; i < strFileNames.size(); i++)
	{
		logFile << "Loading the sensor position of " << strFileNames[i] << " ... ";

		// file name
		const std::string strFileName = strPrefix + strFileNames[i] + strSuffix;

		// open
		std::ifstream fin(strFileName);
		fin >> sensorPositionList[i].x >> sensorPositionList[i].y >> sensorPositionList[i].z;
		
		logFile << " done." << std::endl;
	}
}

}

#endif