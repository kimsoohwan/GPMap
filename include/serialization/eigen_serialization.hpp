#ifndef _EIGEN_SERIALIZATION_HPP_
#define _EIGEN_SERIALIZATION_HPP_

// refer to 
// http://stackoverflow.com/questions/18382457/eigen-and-boostserialize
// http://stackoverflow.com/questions/12580579/how-to-use-boostserialization-to-save-eigenmatrix/12618789#12618789

// STL
#include <string>
#include <fstream>

// Boost - Filesystem
//#include <boost/filesystem.hpp>

// Boost - Serialization
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/array.hpp>

// Eigne
#define EIGEN_NO_DEBUG		// to speed up
#define EIGEN_USE_MKL_ALL	// to use Intel Math Kernel Library
#define EIGEN_DENSEBASE_PLUGIN "eigen_dense_base_addons.hpp"
// TODO: [NumericalIssue] Check if Intel is correctly intalled and running.
#include <Eigen/Core>

// GPMap
#include "util/filesystem.hpp"	// extractFileExtension

namespace GPMap {

//namespace boost {
//namespace serialization { 
//
////template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
//template<class Archive, typename _Scalar, int _Rows, int _Cols>
//void serialize(Archive &ar, 
//					//Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &t, 
//					Eigen::Matrix<_Scalar, _Rows, _Cols> &t, 
//					const unsigned int file_version) 
//{
//	// rows and cols
//	size_t rows = t.rows(), cols = t.cols();
//	ar & rows;
//	ar & cols;
//	
//	// resize
//	if(rows*cols != t.size()) t.resize(rows, cols);
//
//	// data
//	//ar & boost::serialization::make_array(t.data(), t.size());
//	for (size_t j = 0; j < cols; ++j )
//		for (size_t i = 0; i < rows; ++i )
//			ar & derived().coeff(i, j);
//}
//}
//}


template <typename T>
bool serialize(const T& data, const std::string& strFilePath)
{
	// boost path for the file extension
	//boost::filesystem::path p(strFilePath);
	//const std::string fileExtension(p.extension().string());
	const std::string fileExtension(extractFileExtension(strFilePath));

	// ascii
	if(fileExtension.compare(".txt") == 0 || fileExtension.compare(".dat") == 0)
	{
		// open an output file
		std::ofstream ofs(strFilePath.c_str());
		if (!ofs.is_open()) return false;

		// save
		{
			boost::archive::text_oarchive oa(ofs);
			oa << data;
		}
		
		// close the file
		ofs.close();
	}
	// binary
	else
	{
		// open a binary output file
		std::ofstream ofs(strFilePath.c_str(), std::ofstream::binary);
		if (!ofs.is_open()) return false;

		// save
		{
			boost::archive::binary_oarchive oa(ofs);
			oa << data;
		}

		// close the file
		ofs.close();
	}

	return true;
}

template <typename T>
bool deserialize(T& data, const std::string& strFilePath)
{
	// boost path for the file extension
	//boost::filesystem::path p(strFilePath);
	//const std::string fileExtension(p.extension().string());
	const std::string fileExtension(extractFileExtension(strFilePath));

	// ascii
	if(fileExtension.compare(".txt") == 0 || fileExtension.compare(".dat") == 0)
	{
		// open an input file
		std::ifstream ifs(strFilePath.c_str());
		if(!ifs.is_open()) return false;

		// load
		{
			boost::archive::text_iarchive ia(ifs);
			ia >> data;
		}

		// close the file
		ifs.close();
	}
	// binary
	else
	{
		// open an input file
		std::ifstream ifs(strFilePath.c_str(), std::ifstream::binary);
		if(!ifs.is_open()) return false;

		// load
		{
			boost::archive::binary_iarchive ia(ifs);
			ia >> data;
		}

		// close the file
		ifs.close();
	}

	return true;
}

}
#endif