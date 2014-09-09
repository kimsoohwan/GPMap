#ifndef _GPMAP_TO_OCTOMAP_HPP_
#define _GPMAP_TO_OCTOMAP_HPP_

// STL
#include <string>
#include <vector>
#include <algorithm>					// min, max
#include <fstream>
#include <cmath>						// logf
#include <limits>						// std::numeric_limits<float>::digits10
											// std::numeric_limits<T>::min(), max()

// PCL
#include <pcl/point_types.h>		// pcl::PointXYZ, pcl::Normal, pcl::PointNormal
#include <pcl/point_cloud.h>		// pcl::PointCloud

// OctoMap
#include <octomap/octomap.h>
#include <octomap/octomap_timing.h>
#include <octomap/ColorOcTree.h>

// OpenGP
#include "GP.h"						// DlibScalar
using GP::LogFile;

// GPMap
#include "util/data_types.hpp"	// PointXYZCloud
#include "util/timer.hpp"			// Times
#include "util/color_map.hpp"		// ColorMap
#include "plsc/plsc.hpp"			// PLSC
namespace GPMap {

/** @brief Get min and max of means and variances of occupied cells in GPMap */
void getMinMaxMeanVarOfOccupiedCells(const pcl::PointCloud<pcl::PointNormal>	&pointCloudGPMap,
												 float &minMean,	float &maxMean,
												 float &minVar,		float &maxVar)
{
	// min, max
	minMean	= std::numeric_limits<float>::max();
	maxMean	= std::numeric_limits<float>::min();
	minVar	= std::numeric_limits<float>::max();
	maxVar	= std::numeric_limits<float>::min();		

	// for each point
	for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
	{
		// point
		const pcl::PointNormal &point = pointCloudGPMap.points[i];

		// PLSC
		const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

		// if occupied
		if(occupied_probabiliy > 0.5f)
		{
			// min, max
			minMean	= std::min<float>(minMean,		point.normal_x);
			maxMean	= std::max<float>(maxMean,		point.normal_x);
			minVar	= std::min<float>(minVar,		point.normal_y);
			maxVar	= std::max<float>(maxVar,		point.normal_y);
		}
	}

	// log
	LogFile logFile;
	logFile << "Min Mean: " << std::setprecision(std::numeric_limits<float>::digits10) << std::scientific << minMean << std::endl;
	logFile << "Max Mean: " << std::setprecision(std::numeric_limits<float>::digits10) << std::scientific << maxMean << std::endl;
	logFile << "Min Var: "  << std::setprecision(std::numeric_limits<float>::digits10) << std::scientific << minVar  << std::endl;
	logFile << "Max Var: "  << std::setprecision(std::numeric_limits<float>::digits10) << std::scientific << maxVar  << std::endl;
}

/** @brief Converting a PCL PointCloud to an octomap Pointcloud */
template <typename PointT>
void pcd2pc(const pcl::PointCloud<PointT>	&pointCloud,
				octomap::Pointcloud				&pc)
{
	// clear
	pc.clear();

	// add points
	for(size_t i = 0; i < pointCloud.size(); i++)
		pc.push_back(pointCloud.points[i].x, pointCloud.points[i].y, pointCloud.points[i].z);
}

class NO_COLOR {};
class COLOR {};
template <typename ColorT> struct OctomapType {};
template <> struct OctomapType<NO_COLOR>	{ typedef octomap::OcTree			OctomapT; };
template <> struct OctomapType<COLOR>		{ typedef octomap::ColorOcTree	OctomapT; };

template <typename ColorT>
class OctoMap
{
protected:
	typedef typename OctomapType<ColorT>::OctomapT	MyOctomapT;
public:
	/** @brief Constructor */
	OctoMap(const double resolution,
			  const bool	FLAG_SIMPLE_UPDATE = false)
		:	m_resolution(resolution),
			m_pOctree(new MyOctomapT(resolution)),
			FLAG_SIMPLE_UPDATE_(FLAG_SIMPLE_UPDATE)
	{
	}

	/** @brief Constructor */
	OctoMap(const double				resolution,
			  const std::string		strFileName,
			  const bool				FLAG_SIMPLE_UPDATE = false)
		:	m_resolution(resolution),
			m_pOctree(new MyOctomapT(resolution)),
			FLAG_SIMPLE_UPDATE_(FLAG_SIMPLE_UPDATE)
	{
		// load octomap
		m_pOctree->readBinary(strFileName);
	}

	/** @brief			Constructor 
	  * @details		Load the octomap from a point cloud
	  */
	OctoMap(const double											resolution,
			  const pcl::PointCloud<pcl::PointNormal>		&pointCloudGPMap,
			  const float											maxVarThld = std::numeric_limits<float>::max(),
			  const bool											fSetLogOddValue = true,
			  const bool											FLAG_SIMPLE_UPDATE = false)
		:	m_resolution(resolution),
			m_pOctree(new MyOctomapT(resolution)),
			FLAG_SIMPLE_UPDATE_(FLAG_SIMPLE_UPDATE)
	{
		GPMap2Octomap(pointCloudGPMap, maxVarThld, fSetLogOddValue);
	}

	/** @brief			Constructor 
	  * @details		Load the octomap from a point cloud
	  */
	OctoMap(const double											resolution,
			  const pcl::PointCloud<pcl::PointNormal>		&pointCloudGPMap,
			  const float											minVarRangeForColor,
			  const float											maxVarRangeForColor,
			  const float											maxVarThld = std::numeric_limits<float>::max(),
			  const bool											fSetLogOddValue = true,
			  const bool											FLAG_SIMPLE_UPDATE = false)
		:	m_resolution(resolution),
			m_pOctree(new MyOctomapT(resolution)),
			FLAG_SIMPLE_UPDATE_(FLAG_SIMPLE_UPDATE)
	{
		GPMap2Octomap(pointCloudGPMap, minVarRangeForColor, maxVarRangeForColor, maxVarThld, fSetLogOddValue);
	}

	/** @brief	Update a node of the octomap
	  * @return	Elapsed time (user/system/wall cpu times)
	  */
	inline void updateNode(const double x, const double y, const double z, const float log_odds_update, bool lazy_eval = false)
	{
		m_pOctree->updateNode(x, y, z, log_odds_update, lazy_eval);
	}

	/** @brief	Update a node of the octomap
	  * @return	Elapsed time (user/system/wall cpu times)
	  */
	inline void updateNode(const double x, const double y, const double z, bool occupied, bool lazy_eval = false)
	{
		m_pOctree->updateNode(x, y, z, occupied, lazy_eval);
	}

	/** @brief	Update the octomap with a point cloud
	  * @return	Elapsed time (user/system/wall cpu times)
	  */
	template <typename PointT1, typename PointT2>
	CPU_Times update(const typename pcl::PointCloud<PointT1>		&pointCloud,
						  const PointT2										&sensorPosition,
						  const double											maxrange = -1)
	{
		// robot position
		octomap::point3d robotPosition(sensorPosition.x, sensorPosition.y, sensorPosition.z);

		// point cloud
		octomap::Pointcloud pc;
		//for(size_t i = 0; i < pointCloud.size(); i++)
		//	pc.push_back(pointCloud.points[i].x, pointCloud.points[i].y, pointCloud.points[i].z);
		pcd2pc<PointT1>(pointCloud, pc);

		// timer - start
		CPU_Timer timer;

		// update
		if (FLAG_SIMPLE_UPDATE_)	m_pOctree->insertPointCloudRays(pc, robotPosition, maxrange);
		else								m_pOctree->insertPointCloud(pc, robotPosition, maxrange);

		// timer - end
		CPU_Times elapsed = timer.elapsed();

		// memory
		std::cout << "memory usage: "		<< m_pOctree->memoryUsage()		<< std::endl;
		std::cout << "leaf node count: " << m_pOctree->getNumLeafNodes() << std::endl;
		if(m_pOctree->memoryUsage() > 500000000) // 900000000
		{
			m_pOctree->toMaxLikelihood();
			m_pOctree->prune();
			std::cout << "after pruned - memory usage: "		<< m_pOctree->memoryUsage()		<< std::endl;
			std::cout << "after pruned - leaf node count: " << m_pOctree->getNumLeafNodes() << std::endl;
		}

		// return the elapsed time
		return elapsed;
	}

	/** @brief	Save the octomap as a binary file */
	bool save(const std::string &strFileNameWithoutExtension)
	{
		// check
		if(!m_pOctree) return false;

		// file name
		std::string strFileName;

		// *.ot
		if(FLAG_SIMPLE_UPDATE_)		strFileName = strFileNameWithoutExtension + "_simple.ot";
		else								strFileName = strFileNameWithoutExtension + ".ot";
		m_pOctree->write(strFileName);

		//// *.bt
		//if(FLAG_SIMPLE_UPDATE_)		strFileName = strFileNameWithoutExtension + "_simple_ml.bt";
		//else								strFileName = strFileNameWithoutExtension + "_ml.bt";
		//m_pOctree->toMaxLikelihood();
		//m_pOctree->prune();
		//m_pOctree->writeBinary(strFileName);

		std::cout << std::endl;
		return true;
	}

	/** @brief	Evaluate the octomap */
	template <typename PointT1, typename PointT2>
	bool evaluate(const std::vector<typename pcl::PointCloud<PointT1>::Ptr>			&pHitPointCloudPtrList,
					  const std::vector<PointT2, Eigen::aligned_allocator<PointT2> >	&sensorPositionList,
					  unsigned int		&num_points,
					  unsigned int		&num_voxels_correct,
					  unsigned int		&num_voxels_wrong,
					  unsigned int		&num_voxels_unknown,
					  const double		maxrange = -1)
	{
		// check size
		assert(pHitPointCloudPtrList.size() == sensorPositionList.size());

		// check memory
		if(!m_pOctree) return false;

		// initialization
		num_points = 0;
		num_voxels_correct = 0;
		num_voxels_wrong = 0;
		num_voxels_unknown = 0;

		// for each observation
		for(size_t i = 0; i < pHitPointCloudPtrList.size(); i++)
		{
			// robot position
			octomap::point3d robotPosition(sensorPositionList[i].x, sensorPositionList[i].y, sensorPositionList[i].z);

			// point cloud
			num_points += pHitPointCloudPtrList[i]->size();
			octomap::Pointcloud pc;
			//for(size_t j = 0; j < pHitPointCloudPtrList[i]->size(); j++)
			//	pc.push_back(pHitPointCloudPtrList[i]->points[j].x, pHitPointCloudPtrList[i]->points[i].y, pHitPointCloudPtrList[i]->points[j].z);
			pcd2pc<PointT1>(*(pHitPointCloudPtrList[i]), pc);

			// free/occupied cells
			octomap::KeySet free_cells, occupied_cells;
			m_pOctree->computeUpdate(pc, robotPosition, free_cells, occupied_cells, maxrange);
			
			// count free cells
			for(octomap::KeySet::iterator it = free_cells.begin(); it != free_cells.end(); ++it)
			{
				octomap::OcTreeNode* n = m_pOctree->search(*it);
				if(n)
				{
					if(m_pOctree->isNodeOccupied(n))	num_voxels_wrong++;
					else										num_voxels_correct++;
				}
				else											num_voxels_unknown++;
			}
			
			// count occupied cells
			for(octomap::KeySet::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it)
			{
				octomap::OcTreeNode* n = m_pOctree->search(*it);
				if(n)
				{
					if(m_pOctree->isNodeOccupied(n))	num_voxels_correct++;
					else										num_voxels_wrong++;
				}
				else											num_voxels_unknown++;
			}
		}

		return true;
	}

	/** @brief		Train hyperparameters of PLSC
	  *				by minimizing sum of negative log predictive probability
	  * @note		Note that it is not sum of negative log leave-one-out probability
	  *				which requires to predict GPR while training both PLSC parameters and GPR hyperparameters.
	  * @return		Minimum sum of log inference probabilities of occupied centers 
	 */
	GP::DlibScalar trainByMinimizingSumNegLogPredProb(const pcl::PointCloud<pcl::PointNormal>::ConstPtr	&pPointNormalCloudGPMap,
																	  float							&PLSC_alpha, 
																	  float							&PLSC_beta, 
																	  const bool					fConsiderBothOccupiedAndEmpty,
																	  const int						maxIter, 
																	  const GP::DlibScalar		minValue = 1e-15)
	{
		// set the member variables
		m_pPointNormalCloudGPMap			= pPointNormalCloudGPMap;
		m_fConsiderBothOccupiedAndEmpty	= fConsiderBothOccupiedAndEmpty;
		m_fTrainByMinimizingSumNegLogPredProb = true;

		// conversion from PLSC hyperparameters to a Dlib vector
		GP::DlibVector hypDlib;
		hypDlib.set_size(2);
		hypDlib(0, 0) = PLSC_alpha;
		hypDlib(1, 0) = PLSC_beta;

		// trainer
		GP::DlibScalar sumNegLogPredProb = GP::TrainerUsingApproxDerivatives<OctoMap>::train<GP::BOBYQA, GP::NoStopping>(hypDlib, *this, maxIter, minValue);

		// conversion from a Dlib vector to PLSC hyperparameters
		PLSC_alpha	= hypDlib(0, 0);
		PLSC_beta	= hypDlib(1, 0);

		// set the static variables
		PLSC::alpha	= PLSC_alpha;
		PLSC::beta	= PLSC_beta;

		// remove the GPMap
		m_pPointNormalCloudGPMap.reset();

		return sumNegLogPredProb;
	}

	/** @brief		Train hyperparameters of PLSC
	  *				by minimizing negative accuracy (=correct/(correct+wrong)) from evaluation
	  * @return		Minimum sum of log inference probabilities of occupied centers 
	  */
	GP::DlibScalar trainByMinimizingNegEvalAccu(const pcl::PointCloud<pcl::PointNormal>::ConstPtr			&pPointNormalCloudGPMap,
															  PointXYZCloudPtrList		*pHitPointCloudPtrList,
															  PointXYZVList				*pSensorPositionList,
															  float									&PLSC_alpha, 
															  float									&PLSC_beta, 
															  const int								maxIter, 
															  const GP::DlibScalar				minValue = 1e-15)
	{
		// set the member variables
		m_pPointNormalCloudGPMap	= pPointNormalCloudGPMap;
		m_pHitPointCloudPtrList		= pHitPointCloudPtrList;
		m_pSensorPositionList		= pSensorPositionList;
		m_fTrainByMinimizingSumNegLogPredProb = false;

		// conversion from PLSC hyperparameters to a Dlib vector
		GP::DlibVector hypDlib;
		hypDlib.set_size(2);
		hypDlib(0, 0) = PLSC_alpha;
		hypDlib(1, 0) = PLSC_beta;

		// trainer
		const long NPT = 4;
		GP::DlibScalar negEvalAccu = GP::TrainerUsingApproxDerivatives<OctoMap>::train<GP::BOBYQA, GP::NoStopping>(hypDlib, *this, maxIter, minValue, NPT);

		// conversion from a Dlib vector to PLSC hyperparameters
		PLSC_alpha	= hypDlib(0, 0);
		PLSC_beta	= hypDlib(1, 0);

		// set the static variables
		PLSC::alpha	= PLSC_alpha;
		PLSC::beta	= PLSC_beta;

		// remove the GPMap
		m_pPointNormalCloudGPMap.reset();

		return negEvalAccu;
	}


	/** @brief		Operator for optimizing hyperparameters 
	  * @return		Sum of log inference probabilities of occupied centers 
	  *				instead of Sum of log LOO (Leave-One-Out) probabilites
	  */
	GP::DlibScalar operator()(const GP::DlibVector &hypDlib) const
	{
		static size_t numCall = 0;

		// convert a Dlib vector to PLSC hyperparameters
		PLSC::alpha	= hypDlib(0, 0);
		PLSC::beta	= hypDlib(1, 0);

		// sum of negative log predictive probability
		if(m_fTrainByMinimizingSumNegLogPredProb)
		{
			// Sum of negative log marginalizations of all leaf nodes
			GP::DlibScalar sum_neg_log_occupied(0);
			GP::DlibScalar sum_neg_log_empty(0);

			// for each point
			for(size_t i = 0; i < m_pPointNormalCloudGPMap->points.size(); i++)
			{
				// point
				const pcl::PointNormal &point = m_pPointNormalCloudGPMap->points[i];

				// PLSC
				const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

				// if occupied
				if(occupied_probabiliy > 0.5f)	sum_neg_log_occupied -= logf(occupied_probabiliy);

				// else empty
				else										sum_neg_log_empty -= logf(1.f - occupied_probabiliy);
			}

			// sum of negative log probability
			GP::DlibScalar sum_neg_log_probability;
			if(m_fConsiderBothOccupiedAndEmpty)		sum_neg_log_probability = sum_neg_log_occupied + sum_neg_log_empty;
			else												sum_neg_log_probability = sum_neg_log_occupied;

			// log file
			LogFile logFile;
			logFile << "[" << numCall++ << "] (" << PLSC::alpha << ", "  << PLSC::beta  << ") : " << sum_neg_log_probability << std::endl;

			return sum_neg_log_probability;
		}

		// negative accuracy (=correct/(correct+wrong)) from evaluation

		// octomap
		OctoMap octomap(m_resolution, *m_pPointNormalCloudGPMap);

		// evaluate
		unsigned int num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown;
		octomap.evaluate<pcl::PointXYZ, pcl::PointXYZ>(*m_pHitPointCloudPtrList, *m_pSensorPositionList,
																	  num_points, num_voxels_correct, num_voxels_wrong, num_voxels_unknown);

		// negative accuracy
		GP::DlibScalar neg_accuracy = -static_cast<GP::DlibScalar>(num_voxels_correct)/static_cast<GP::DlibScalar>(num_voxels_correct + num_voxels_wrong);

		// log file
		LogFile logFile;
		logFile << "[" << numCall++ << "] (" << PLSC::alpha << ", "  << PLSC::beta  << ") : " << -neg_accuracy << " = "
				  << num_voxels_correct << " / (" << num_voxels_correct << " + " << num_voxels_wrong << ")" << std::endl;

		return neg_accuracy;
	}

protected:
	/** @brief	Convert a GPMap to an Octree or a ColorOctree based on PLSC */
	void GPMap2Octomap(const pcl::PointCloud<pcl::PointNormal>	&pointCloudGPMap,
							 const float										maxVarThld = std::numeric_limits<float>::max(),
							 const bool											fSetLogOddValue = true);

	/** @brief	Convert a GPMap to an Octree based on PLSC */
	void GPMap2Octomap(const pcl::PointCloud<pcl::PointNormal>	&pointCloudGPMap,
							 const float										minVarRangeForColor,
							 const float										maxVarRangeForColor,
							 const float										maxVarThld = std::numeric_limits<float>::max(),
							 const bool											fSetLogOddValue = true);

	/** @brief Log odd = log(p/(1-p)) */
	inline float logodd(const float p) const
	{
		return logf(p/(1-p));
	}

protected:
	/** @brief	Resolution */
	const double m_resolution;

	/** @brief	OctoMap */
	boost::shared_ptr<MyOctomapT>			m_pOctree;

	/** @brief	Flag for simple update */
	const bool FLAG_SIMPLE_UPDATE_;

	/** @brief	GPMap as a point cloud for training PLSC hyperparameters */
	pcl::PointCloud<pcl::PointNormal>::ConstPtr	m_pPointNormalCloudGPMap;

	/** @brief	Hit point cloud list for training PLSC hyperparameters */
	PointXYZCloudPtrList	*m_pHitPointCloudPtrList;

	/** @brief	Sensor position list for training PLSC hyperparameters */
	PointXYZVList			*m_pSensorPositionList;

	/** @brief	How to training PLSC hyperparameters */
	bool m_fTrainByMinimizingSumNegLogPredProb;

	/** @brief	Whether to consider both occupied and empty cells or occupied cells only
	  *			when training PLSC hyperparameters */
	bool	m_fConsiderBothOccupiedAndEmpty;
};

/** @brief	Convert a GPMap to an Octree based on PLSC */
template <>
void OctoMap<NO_COLOR>::GPMap2Octomap(const pcl::PointCloud<pcl::PointNormal>		&pointCloudGPMap,
												  const float											maxVarThld,
												  const bool											fSetLogOddValue)
{
	// set logodd value
	if(fSetLogOddValue)
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// set logodd value
			m_pOctree->setNodeValue(static_cast<double>(point.x),
											static_cast<double>(point.y), 
											static_cast<double>(point.z), 
											logodd(occupied_probabiliy));
		}
	}
	else
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// if occupied
			if(occupied_probabiliy > 0.5f)
			{
				// set occupied
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 true);
			}
			else
			{
				// set occupied
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 false);
			}
		}
	}
}

/** @brief	Convert a GPMap to a ColorOctree based on PLSC */
template <>
void OctoMap<COLOR>::GPMap2Octomap(const pcl::PointCloud<pcl::PointNormal>		&pointCloudGPMap,
											  const float											maxVarThld,
											  const bool											fSetLogOddValue)
{
	// get min, max
	float minMean, maxMean, minVar, maxVar;
	getMinMaxMeanVarOfOccupiedCells(pointCloudGPMap, minMean, maxMean, minVar, maxVar);

	// color map
	ColorMap colorMap(minVar, maxVar);
	unsigned char r, g, b;

	// set logodd value
	if(fSetLogOddValue)
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// set logodd value
			m_pOctree->setNodeValue(static_cast<double>(point.x),
											static_cast<double>(point.y), 
											static_cast<double>(point.z), 
											logodd(occupied_probabiliy));

			// color based on the variance
			colorMap.rgb(point.normal_y, r, g, b);

			// set color
			m_pOctree->setNodeColor(point.x, point.y, point.z, r, g, b);
		}
	}
	else
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// if occupied
			if(occupied_probabiliy > 0.5f)
			{
				// set occupied
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 true);
			}
			else
			{
				// set occupied
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 false);
			}

			// color based on the variance
			colorMap.rgb(point.normal_y, r, g, b);

			// set color
			m_pOctree->setNodeColor(point.x, point.y, point.z, r, g, b);
		}
	}
}

/** @brief	Convert a GPMap to a ColorOctree based on PLSC */
template <>
void OctoMap<COLOR>::GPMap2Octomap(const pcl::PointCloud<pcl::PointNormal>		&pointCloudGPMap,
											  const float											minVarRangeForColor,
											  const float											maxVarRangeForColor,
											  const float											maxVarThld,
											  const bool											fSetLogOddValue)
{
	// color map
	ColorMap colorMap(minVarRangeForColor, maxVarRangeForColor);
	unsigned char r, g, b;

	// set logodd value
	if(fSetLogOddValue)
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// set logodd value
			m_pOctree->setNodeValue(static_cast<double>(point.x),
											static_cast<double>(point.y), 
											static_cast<double>(point.z), 
											logodd(occupied_probabiliy));


			// color based on the variance
			colorMap.rgb(point.normal_y, r, g, b);

			// set color
			m_pOctree->setNodeColor(point.x, point.y, point.z, r, g, b);
		}
	}
	else
	{
		// for each point
		for(size_t i = 0; i < pointCloudGPMap.points.size(); i++)
		{
			// point
			const pcl::PointNormal &point = pointCloudGPMap.points[i];

			// variance check
			if(point.normal_y > maxVarThld) continue;

			// PLSC
			const float occupied_probabiliy = PLSC::occupancy(point.normal_x, point.normal_y);

			// if occupied
			if(occupied_probabiliy > 0.5f)
			{
				// set occupied
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 true);
			}
			else
			{
				// set empty
				m_pOctree->updateNode(static_cast<double>(point.x), 
											 static_cast<double>(point.y), 
											 static_cast<double>(point.z),
											 false);
			}

			// color based on the variance
			colorMap.rgb(point.normal_y, r, g, b);

			// set color
			m_pOctree->setNodeColor(point.x, point.y, point.z, r, g, b);
		}
	}
}

}

#endif