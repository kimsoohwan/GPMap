#ifndef _DATA_PARTITIONING_HPP_
#define _DATA_PARTITIONING_HPP_

// STL
#include <cmath>			// floor, ceil
#include <limits>			// std::numeric_limits<T>::min(), max()
#include <algorithm>		// std::min(), max(),  std::random_shuffle
#include <vector>			// std::vector

// PCL
//#include <pcl/point_types.h>
#include <pcl/point_cloud.h>				// PointCloud
#include <pcl/octree/octree.h>			// OctreePointCloudDensity
#include <pcl/octree/octree_impl.h>

namespace GPMap {

/** \brief @b Octree pointcloud density leaf node class
* \note This class is redefined for pcl::octree::OctreePointCloudDensityContainer
*/
template<typename DataT>
class OctreePointCloudDensityContainer
{
public:
   /** \brief Class initialization. */
   OctreePointCloudDensityContainer () : pointCounter_ (0)
   {
   }

   /** \brief Empty class deconstructor. */
   virtual ~OctreePointCloudDensityContainer ()
   {
   }

   /** \brief deep copy function */
   virtual OctreePointCloudDensityContainer *
   deepCopy () const
   {
      return (new OctreePointCloudDensityContainer (*this));
   }

   /** \brief Get size of container (number of DataT objects)
   * \return number of DataT elements in leaf node container.
   */
   size_t
   getSize () const
   {
      return 0;
   }

   /** \brief Read input data. Only an internal counter is increased.
      */
   void
   setData (const DataT&)
   {
      pointCounter_++;
   }

   /** \brief Returns a null pointer as this leaf node does not store any data.
      * \param[out] data_arg: reference to return pointer of leaf node DataT element (will be set to 0).
      */
   //void
   //getData (const DataT*& data_arg) const // SWAN: BUG!!!
   //{
   //   data_arg = 0;
   //}

	/** \brief Empty getData data vector implementation as this leaf node does not store any data.
	*/
  void
    getData (DataT &) const
   {
   }

   /** \brief Empty getData data vector implementation as this leaf node does not store any data. \
      */
   void
   getData (std::vector<DataT>&) const
   {
   }

   /** \brief Return point counter.
      * \return Amaount of points
      */
   unsigned int
   getPointCounter ()
   {
      return (pointCounter_);
   }

   /** \brief Empty reset leaf node implementation as this leaf node does not store any data. */
   void
   reset ()
   {
      pointCounter_ = 0;
   }

private:
   unsigned int pointCounter_;

};

class UniqueNonZeroInteger
{
public: 
	UniqueNonZeroInteger() : m_currVal(-1) {}
	int operator()() { return ++m_currVal; }

protected:
	int m_currVal;
};

bool random_data_partition(const std::vector<int>				&indices,
									const int								M, // maximum limit of the number of points in a leaf node of an octree
									std::vector<std::vector<int> >	&partitionedIndices,
									const bool								fSuffling = true)
{
	// if the maximum limit is less or equal to zero, do not divide!
	if(M <= 0) return false;

	// size
	const int N = indices.size();
	if(N <= M) return false;

	// suffled indices
	std::vector<int> suffledIndices(indices);
	if(fSuffling) std::random_shuffle(suffledIndices.begin(), suffledIndices.end());

	// partitioning
	partitionedIndices.clear();
	int from(0), to(-1);
	while(to < N-1)
	{
		// range
		from	= to + 1;
		to		= (N-1 < from + M - 1) ? N-1 : from + M - 1;

		// copy subset
		partitionedIndices.push_back(std::vector<int>(suffledIndices.begin() + from, suffledIndices.begin() + to + 1));
	}

	return true;
}

bool random_sampling(const std::vector<int>				&indices,
							const int								M, // maximum limit of the number of points in a leaf node of an octree
							std::vector<int>						&randomSampleIndices,
							const bool								fSuffling = true)
{
	// if the maximum limit is less or equal to zero, do not divide!
	if(M <= 0) return false;

	// size
	const int N = indices.size();
	if(N <= M) return false;

	// suffled indices
	std::vector<int> suffledIndices(indices);
	if(fSuffling) std::random_shuffle(suffledIndices.begin(), suffledIndices.end());

	// copy
	randomSampleIndices.resize(M);
	std::copy(suffledIndices.begin(), suffledIndices.begin() + M, randomSampleIndices.begin());

	return true;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
randomSampling(const typename pcl::PointCloud<PointT>::ConstPtr	&pCloud,
					const float														samplingRatio,
					const bool														fSuffling = true)
{
	// size
	const int N = static_cast<int>(pCloud->points.size());
	const int M = static_cast<int>(ceil(static_cast<float>(N)*samplingRatio));

	// random indices
	std::vector<int> indices(N);
	std::generate(indices.begin(), indices.end(), UniqueNonZeroInteger());
	std::vector<int> randomSampleIndices;
	random_sampling(indices, M, randomSampleIndices, fSuffling);

	// new point cloud
	typename pcl::PointCloud<PointT>::Ptr pSampledCloud(new pcl::PointCloud<PointT>());

	// resize
	pSampledCloud->header = pCloud->header;
	pSampledCloud->points.resize(M);

	// copy
	for (size_t i = 0; i < M; ++i)
	{
		pSampledCloud->points[i] = pCloud->points[randomSampleIndices[i]];
	}

	// resize
	pSampledCloud->height = 1;
	pSampledCloud->width  = static_cast<uint32_t>(M);

	return pSampledCloud;
}

template <typename PointT>
void randomSampling(const std::vector<typename pcl::PointCloud<PointT>::Ptr>		&cloudPtrList,
						  const float																	samplingRatio,
						  std::vector<typename pcl::PointCloud<PointT>::Ptr>				&filteredCloudPtrList)
{
	// reset
	filteredCloudPtrList.resize(cloudPtrList.size());

	// for each point cloud
	for(size_t i = 0; i < cloudPtrList.size(); i++)
	{
		filteredCloudPtrList[i] = randomSampling<PointT>(cloudPtrList[i], samplingRatio);
	}
}

//template <typename PointT>
//bool data_partition_by_octree(const typename pcl::PointCloud<PointT>::ConstPtr		&pPointCloud,
//										const double														octreeResolution,
//										const int															M, // maximum limit of the number of points in a leaf node of an octree
//										std::vector<typename pcl::PointCloud<PointT>::Ptr>	&partitionedPointCloudPtrList,
//										const bool														fSuffling = true)
//{
//	// point cloud check
//	assert(M > 0);
//	assert(pPointCloud);
//
//	// size
//	const int N = pPointCloud->points.size();
//	if(N < M) return false;
//
//	// octree
//	typedef pcl::octree::OctreePointCloudDensity<PointT, OctreePointCloudDensityContainer<int> > Octree;
//	Octree octree(octreeResolution);
//	octree.setInputCloud(pPointCloud);
//	octree.addPointsFromInputCloud();
//
//	// max number of points in a leaf node
//	int maxSize	= std::numeric_limits<int>::min();
//	Octree::LeafNodeIterator iter(octree);
//	while(*++iter)
//	{
//		// leaf node
//		Octree::LeafNode *pLeafNode = static_cast<Octree::LeafNode *>(iter.getCurrentOctreeNode());
//
//		// max
//		maxSize	= std::max<int>(maxSize, static_cast<int>(pLeafNode->getPointCounter()));
//	}
//
//	// if the max number of points in a leaf node is greater than the limit, do nothing
//	if(maxSize <= M) return false;
//
//	// number of partitions
//	const size_t K = ceil(static_cast<float>(maxSize) / static_cast<float>(M));
//
//	// suffled indices
//	std::vector<int> suffledIndices(N);
//	std::generate(suffledIndices.begin(), suffledIndices.end(), UniqueNonZeroInteger());
//	if(fSuffling) std::random_shuffle(suffledIndices.begin(), suffledIndices.end());
//
//	// partitioning
//	partitionedPointCloudPtrList.resize(K);
//	int from(0), to(-1);
//	int partitionIdx(0);
//	while(to < N-1)
//	{
//		// range
//		from	= to + 1;
//		to		= (N-1 < from + M - 1) ? N-1 : from + M - 1;
//
//		// indices
//		std::vector<int> indices(to - from + 1);
//		std::copy(suffledIndices.begin() + from, suffledIndices.begin() + to + 1, indices.begin());
//
//		// copy subset
//		partitionedPointCloudPtrList[partitionIdx++].reset(new pcl::PointCloud<PointT>(*pPointCloud, indices));
//	}
//}

}

#endif