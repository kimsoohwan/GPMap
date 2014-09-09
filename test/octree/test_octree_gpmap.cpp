#if 0

#define _TEST_OCTREE_GPMAP

// GPMap
#include "serialization/eigen_serialization.hpp" // Eigen
#include "io/io.hpp"								// loadPointCloud, savePointCloud, loadSensorPositionList
#include "visualization/cloud_viewer.hpp"	// show
#include "data/training_data.hpp"			// genEmptyPointList
#include "features/surface_normal.hpp"		// estimateSurfaceNormals
#include "common/common.hpp"					// getMinMaxPointXYZ
#include "octree/octree_gpmap.hpp"			// OctreeGPMap
#include "octree/octree_viewer.hpp"			// OctreeViewer
#include "octomap/octomap.hpp"				// OctoMap
using namespace GPMap;

typedef OctreeGPMap<pcl::PointNormal, GP::MeanZeroDerObs, GP::CovSEisoDerObs, GP::LikGaussDerObs, GP::InfExactDerObs> OctreeGPMapType;

int main(int argc, char**argv)
{
	OctoMap o;

		// [1] load sensor positions
		//pcl::PointXYZ sensorPosition(7.2201273e-01, 2.5926464e-02, 1.6074278e-01);

		// [2] load surface normals
		PointNormalCloudPtr pPointNormalCloud(new PointNormalCloud());
		pcl::io::loadPCDFile<pcl::PointNormal>("../../data/input/bunny/bun000_normals.pcd", *pPointNormalCloud);

		// [3] bounding box
		pcl::PointXYZ min_pt, max_pt;
		getMinMaxPointXYZ<pcl::PointNormal>(*pPointNormalCloud, min_pt, max_pt);

		// [4] cropped point cloud list of four parts
		const size_t NUM_DATA = 4; 
		PointNormalCloudPtrList pointNormalCloudList(NUM_DATA);
		pcl::PointXYZ mid_pt((min_pt.x+max_pt.x)/2.f, 
									(min_pt.y+max_pt.y)/2.f,
									(min_pt.z+max_pt.z)/2.f);

		Eigen::Vector4f min_pt_temp[NUM_DATA], max_pt_temp[NUM_DATA];
		min_pt_temp[0].x() = min_pt.x;	min_pt_temp[0].y() = mid_pt.y;	min_pt_temp[0].z() = mid_pt.z;
		max_pt_temp[0].x() = max_pt.x;	max_pt_temp[0].y() = max_pt.y;	max_pt_temp[0].z() = max_pt.z;

		min_pt_temp[1].x() = min_pt.x;	min_pt_temp[1].y() = min_pt.y;	min_pt_temp[1].z() = mid_pt.z;
		max_pt_temp[1].x() = max_pt.x;	max_pt_temp[1].y() = mid_pt.y;	max_pt_temp[1].z() = max_pt.z;

		min_pt_temp[2].x() = min_pt.x;	min_pt_temp[2].y() = min_pt.y;	min_pt_temp[2].z() = min_pt.z;
		max_pt_temp[2].x() = max_pt.x;	max_pt_temp[2].y() = mid_pt.y;	max_pt_temp[2].z() = mid_pt.z;

		min_pt_temp[3].x() = min_pt.x;	min_pt_temp[3].y() = mid_pt.y;	min_pt_temp[3].z() = min_pt.z;
		max_pt_temp[3].x() = max_pt.x;	max_pt_temp[3].y() = max_pt.y;	max_pt_temp[3].z() = mid_pt.z;


		// crop
		size_t nTotalPoints(0);
		for(size_t i = 0; i < NUM_DATA; i++)
		{
			pointNormalCloudList[i] = cropBox<pcl::PointNormal>(pPointNormalCloud, min_pt_temp[i], max_pt_temp[i]);
			nTotalPoints += pointNormalCloudList[i]->size();
		}
		assert(pPointNormalCloud->size() == nTotalPoints);
		//show<pcl::PointNormal>("Cropped Bunny000", pointNormalCloudList, 0.01);		
		
		// [5] octree-based GPMap
		// x:0.07, y:0.1 z:0.15
		const double	BLOCK_SIZE				= 0.003; // 0.01
		const size_t	NUM_CELLS_PER_AXIS	= 3;		// cell size: 0.001
		const bool		INDEPENDENT_BCM		= true;
		const bool		POINT_DUPLICATION		= false;
		const float		GAP						= 0.001;
		OctreeGPMapType gpmap(BLOCK_SIZE, NUM_CELLS_PER_AXIS, INDEPENDENT_BCM, POINT_DUPLICATION);
		gpmap.defineBoundingBox(min_pt, max_pt);

		// [7] Update
		// hyperparameters
		const float ell			= 0.107467f;		// 0.107363f;
		const float sigma_f		= 0.99968f;			//0.99985f;
		const float sigma_n		= 0.00343017f;		// 0.0034282f;
		const float sigma_nd		= 0.0985929f;		// 0.0990157f;
		OctreeGPMapType::Hyp logHyp;
		logHyp.cov(0) = log(ell);
		logHyp.cov(1) = log(sigma_f);
		logHyp.lik(0) = log(sigma_n);
		logHyp.lik(1) = log(sigma_nd);

		// for each observation
		for(size_t i = 0; i < NUM_DATA; i++)
		{
			std::cout << "observation: " << i << std::endl;
			//gpmap.setInputCloud(pointNormalCloudList[i], GAP);
			gpmap.setInputCloud(pointNormalCloudList[i]);
			gpmap.addPointsFromInputCloud();
			gpmap.update(logHyp);
			OctreeViewer<pcl::PointNormal, OctreeGPMapType> octree_viewer(gpmap);
		}

		system("pause");

	return 0;
}

#endif