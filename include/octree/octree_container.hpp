#ifndef _OCTREE_LEAF_NODE_HPP_
#define _OCTREE_LEAF_NODE_HPP_

// STL
#include <vector>

// PCL
#include <pcl/octree/octree.h>			// pcl::octree::OctreeContainerEmpty
#include <pcl/octree/octree_impl.h>

// Eigen
#include <Eigen/Dense>

// GPMap
#include "util/data_types.hpp"			// Matrix, MatrixPtr

namespace GPMap {

/** @brief Leaf node type */
template <typename GPMapContainer>
class OctreeGPMapContainer : public pcl::octree::OctreeContainerDataTVector<int>, 
									  public GPMapContainer
{
public:
	/** \brief Pushes a DataT element to internal DataT vector.
	  * \param[in] data reference to DataT element to be stored within leaf node.
	  */
	void setData(const int &data)
	{
		// if the index is negative, just create the node
		if(data < 0) return;

		// add to the int vector
		pcl::octree::OctreeContainerDataTVector<int>::setData(data);
	}
};

}

#endif