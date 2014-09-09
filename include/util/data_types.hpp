#ifndef _GPMAP_DATA_TYPES_HPP_
#define _GPMAP_DATA_TYPES_HPP_

// STL
#include <vector>

// PCL
#include <pcl/point_types.h>

// OpenGP
#include <GP.h>

namespace GPMap {

// STL
typedef std::vector<std::string>								StringList;

// Eigen
/** @brief Data type for Matrix, MatrixPtr, MatrixConstPtr */
TYPE_DEFINE_MATRIX(float);

/** @brief Data type for Vector, VectorPtr, VectorConstPtr */
TYPE_DEFINE_VECTOR(float);

/** @brief Data type for CholeskyFactor, CholeskyFactorPtr, CholeskyFactorConstPtr */
TYPE_DEFINE_CHOLESKYFACTOR()

//PCL
/** @brief Type for a vector of pcl::PointXYZ points */
typedef std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >	PointXYZVList;
typedef boost::shared_ptr<PointXYZVList>													PointXYZVListPtr;
typedef boost::shared_ptr<const PointXYZVList>											PointXYZVListConstPtr;

/** @brief int */
typedef std::vector<int>										Indices;
typedef boost::shared_ptr<Indices>							IndicesPtr;
typedef boost::shared_ptr<const Indices>					IndicesConstPtr;

/** @brief PointXYZ */
typedef pcl::PointCloud<pcl::PointXYZ>						PointXYZCloud;
typedef PointXYZCloud::Ptr										PointXYZCloudPtr;
typedef PointXYZCloud::ConstPtr								PointXYZCloudConstPtr;
//typedef boost::shared_ptr<PointXYZCloud>				PointXYZCloudPtr;
//typedef boost::shared_ptr<const PointXYZCloud>		PointXYZCloudConstPtr;
typedef std::vector<PointXYZCloudPtr>						PointXYZCloudPtrList;
typedef std::vector<PointXYZCloudConstPtr>				PointXYZCloudConstPtrList;

/** @brief PointNormal */
typedef pcl::PointCloud<pcl::PointNormal>					PointNormalCloud;
typedef PointNormalCloud::Ptr									PointNormalCloudPtr;
typedef PointNormalCloud::ConstPtr							PointNormalCloudConstPtr;
//typedef boost::shared_ptr<PointNormalCloud>			PointNormalCloudPtr;
//typedef boost::shared_ptr<const PointNormalCloud>	PointNormalCloudConstPtr;
typedef std::vector<PointNormalCloudPtr>					PointNormalCloudPtrList;
typedef std::vector<PointNormalCloudConstPtr>			PointNormalCloudConstPtrList;

// OpenGP
//typedef GP::GaussianProcess<float, GP::MeanZeroDerObs, GP::CovSEisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs> GPType;
//typedef GP::DerivativeTrainingData<float>		DerivativeTrainingData;
//typedef GP::TestData<float>						TestData;

}

#endif