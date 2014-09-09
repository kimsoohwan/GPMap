#ifndef _SHOW_POINT_CLOUD_HPP_
#define _SHOW_POINT_CLOUD_HPP_

// STL
#include <vector>
#include <string>
#include <sstream>

// Boost Thread
#include <boost/thread/thread.hpp> 

// PCL
#include <pcl/point_cloud.h>						// pcl::PointCloud
#include <pcl/common/common.h>					// pcl::getMinMax3D
#include <pcl/visualization/cloud_viewer.h>	// pcl::visualization::PCLVisualizer

// GPMap
#include "filter/filters.hpp"

namespace GPMap {

template <typename PointT>
void addPointCloudNormals(pcl::visualization::PCLVisualizer					&viewer, 
								  typename pcl::PointCloud<PointT>::ConstPtr		&pPointCloud, 
								  const double												scale, 
								  const std::string										&strName);

template <>
void addPointCloudNormals<pcl::PointXYZ>(pcl::visualization::PCLVisualizer				&viewer, 
													  pcl::PointCloud<pcl::PointXYZ>::ConstPtr	&pPointCloud, 
													  const double											scale, 
													  const std::string									&strName)
{
	return;
}

template <>
void addPointCloudNormals<pcl::PointNormal>(pcl::visualization::PCLVisualizer					&viewer, 
														  pcl::PointCloud<pcl::PointNormal>::ConstPtr	&pPointCloud, 
														  const double												scale, 
														  const std::string										&strName)
{
	if(scale <= 0.0) return;
	viewer.addPointCloudNormals<pcl::PointNormal>(pPointCloud, 1, scale, strName);
}

template <typename PointT>
void show(const std::string												&strWindowName,
			 const typename pcl::PointCloud<PointT>::ConstPtr		&pPointCloud,
			 const double														scale = 0.1,
			 const float														downSampleLeafSize = 0.f,
			 const bool															fBlackBackground = true,
			 const bool															fDrawAxis = true,
			 const bool															fDrawBoundingbox = true)
{
	// visualizer
	pcl::visualization::PCLVisualizer viewer(strWindowName);

	// background
	if(fBlackBackground)		viewer.setBackgroundColor(0, 0, 0);
	else							viewer.setBackgroundColor(1, 1, 1);

	// axis
	if(fDrawAxis)	viewer.addCoordinateSystem(1.0);

	// points
	typename pcl::PointCloud<PointT>::ConstPtr pTempPointCloud;
	if(downSampleLeafSize > 0)		pTempPointCloud = downSampling<PointT>(pPointCloud, downSampleLeafSize);
	else									pTempPointCloud = pPointCloud;

	// draw points
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> hHeightColor(pTempPointCloud, "z");
	viewer.addPointCloud<PointT>(pTempPointCloud, hHeightColor, "cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

	// draw normal vectors
	addPointCloudNormals<PointT>(viewer, pTempPointCloud, scale, "normals");

	// draw bounding box
	if(fDrawBoundingbox)
	{
		PointT min_pt, max_pt;
		pcl::getMinMax3D(*pPointCloud, min_pt, max_pt);
		viewer.addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z);
		std::stringstream ss_cube;
		ss_cube << "min: " << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << endl;
		ss_cube << "max: " << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << endl;
		ss_cube << "size: " << max_pt.x - min_pt.x << ", " << max_pt.y - min_pt.y << ", " << max_pt.z - min_pt.z << endl;

		// text
		viewer.addText(ss_cube.str(), 10, 100, "text_cube");
	}

	// camera
	viewer.resetCameraViewpoint("cloud");

	// pause
	//viewer.spin();
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	// close
	viewer.close();
}

template <typename PointT>
void show(const std::string														&strWindowName,
			 const std::vector<typename pcl::PointCloud<PointT>::Ptr>	&pPointClouds,
			 const double																scale = 0.1,
			 const float																downSampleLeafSize = 0.f,
			 const bool																	fBlackBackground = true,
			 const bool																	fDrawAxis = true,
			 const bool																	fDrawBoundingbox = true,
			 const bool																	fDrawPreviousPointCloud = true)
{
	// visualizer
	pcl::visualization::PCLVisualizer viewer(strWindowName);

	// background
	if(fBlackBackground)		viewer.setBackgroundColor(0, 0, 0);
	else							viewer.setBackgroundColor(1, 1, 1);

	// axis
	if(fDrawAxis)	viewer.addCoordinateSystem(1.0);

	// load files
	for(size_t i = 0; i < pPointClouds.size(); i++)
	{
		std::stringstream ss_name; 
		ss_name << (i+1) << " / " << pPointClouds.size();

		// points
		typename pcl::PointCloud<PointT>::ConstPtr pTempPointCloud;
		if(downSampleLeafSize > 0)		pTempPointCloud = downSampling<PointT>(pPointClouds[i], downSampleLeafSize);
		else									pTempPointCloud = pPointClouds[i];

		// draw points
		const std::string strPointCloudName = std::string("cloud: ") + ss_name.str();
		//pcl::visualization::PointCloudColorHandlerCustom<PointT> hColor(pTempPointCloud, 255, 0, 0); // current: red
		pcl::visualization::PointCloudColorHandlerGenericField<PointT> hColor(pTempPointCloud, "z");	// height
		viewer.addPointCloud<PointT>(pTempPointCloud, hColor, strPointCloudName);
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, strPointCloudName);

		// draw normal vectors
		const std::string strNormalCloudName = std::string("normals: ") + ss_name.str();
		if(scale > 0) addPointCloudNormals<PointT>(viewer, pTempPointCloud, scale, strNormalCloudName);

		// draw bounding box
		if(fDrawBoundingbox)
		{
			PointT min_pt, max_pt;
			pcl::getMinMax3D(*pPointClouds[i], min_pt, max_pt);
			viewer.removeShape("cube");
			viewer.addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z);
			std::stringstream ss_cube;
			ss_cube << "min: " << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << endl;
			ss_cube << "max: " << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << endl;
			ss_cube << "size: " << max_pt.x - min_pt.x << ", " << max_pt.y - min_pt.y << ", " << max_pt.z - min_pt.z << endl;

			// text
			if(i == 0)
			{
				viewer.addText(strPointCloudName, 10, 60, "text_cloud");
				viewer.addText(ss_cube.str(), 10, 100, "text_cube");
			}
			else
			{
				viewer.removeShape("text_cloud");
				viewer.removeShape("text_cube");
				viewer.addText(strPointCloudName, 10, 60, "text_cloud");
				viewer.addText(ss_cube.str(), 10, 100, "text_cube");
			}
		}

		// camera
		if(i == 0)	viewer.resetCameraViewpoint(strPointCloudName);

		// pause
		viewer.spin();

		// next
		if(fDrawPreviousPointCloud)
		{
			pcl::visualization::PointCloudColorHandlerCustom<PointT> grayColorHandle(pTempPointCloud, 200, 200, 200); // previous: gray
			viewer.updatePointCloud<PointT>(pTempPointCloud, grayColorHandle, strPointCloudName);
		}
		else
		{
			viewer.removeShape(strPointCloudName);
			if(scale > 0) viewer.removeShape(strNormalCloudName);
		}
	}

	// close
	viewer.close();
}


template <typename PointT>
void compare(const std::vector<typename pcl::PointCloud<PointT>::Ptr> &pPointClouds1,
				 const std::vector<typename pcl::PointCloud<PointT>::Ptr> &pPointClouds2,
				 const double scale = 0.0)
{
	// visualizer
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	int viewport1(0), viewport2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, viewport1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, viewport2);
	viewer.setBackgroundColor(0, 0, 0, viewport1);
	viewer.setBackgroundColor(0, 0, 0, viewport2);
	viewer.addCoordinateSystem(1.0, viewport1);
	viewer.addCoordinateSystem(1.0, viewport2);

	// load files
	for(size_t i = 0; i < pPointClouds1.size(); i++)
	{
		std::stringstream name1; name1 << i << "_1";
		std::stringstream name2; name2 << i << "_2";

		// points
		pcl::visualization::PointCloudColorHandlerCustom<PointT> redColorHandle1(pPointClouds1[i], 255, 0, 0); // current: red
		pcl::visualization::PointCloudColorHandlerCustom<PointT> redColorHandle2(pPointClouds2[i], 255, 0, 0); // current: red
		viewer.addPointCloud<PointT>(pPointClouds1[i], redColorHandle1, name1.str(), viewport1);
		viewer.addPointCloud<PointT>(pPointClouds2[i], redColorHandle2, name2.str(), viewport2);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name1.str());
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name2.str());

		// normal vectors
		if(scale > 0)
		{
			viewer.addPointCloudNormals<PointT>(pPointClouds1[i], 1, scale, name1.str() + "_normals", viewport1);
			viewer.addPointCloudNormals<PointT>(pPointClouds2[i], 1, scale, name2.str() + "_normals", viewport2);
		}

		// text and camera
		if(i == 0)
		{
			viewer.addText(name1.str(), 10, 60, "text1");
			viewer.addText(name2.str(), 10, 60, "text2");
			viewer.resetCameraViewpoint(name1.str());
			viewer.resetCameraViewpoint(name2.str());
		}
		else
		{
			viewer.updateText(name1.str(), 10, 60, "text1");
			viewer.updateText(name2.str(), 10, 60, "text2");
		}

		// pause
		viewer.spin();

		// next
		pcl::visualization::PointCloudColorHandlerCustom<PointT> grayColorHandle1(pPointClouds1[i], 200, 200, 200); // previous: gray
		pcl::visualization::PointCloudColorHandlerCustom<PointT> grayColorHandle2(pPointClouds2[i], 200, 200, 200); // previous: gray
		viewer.updatePointCloud<PointT>(pPointClouds1[i], grayColorHandle1, name1.str());
		viewer.updatePointCloud<PointT>(pPointClouds2[i], grayColorHandle2, name2.str());
	}

	// close
	viewer.close();
}

}

#endif