#ifndef _OCTREE_VIEWER_HPP_
#define _OCTREE_VIEWER_HPP_

// Boost
#include <boost/thread/thread.hpp>

// PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/common/common.h>
//#include <pcl/octree/octree.h>
//#include <pcl/octree/octree_impl.h>
//#include <pcl/octree/octree_key.h>
#include <pcl/filters/filter.h>

namespace GPMap {

//=============================
// Displaying cubes is very long!
// so we limit their numbers.
//const int MAX_DISPLAYED_CUBES(15000);
const size_t MAX_DISPLAYED_CUBES(1500000);
//=============================
//typedef pcl::octree::OctreePointCloud<pcl::PointNormal> OctreeT;
//typedef OctreeGPMap<pcl::PointNormal> OctreeT;
//typedef OctreeGPMap OctreeT;

template <typename PointT, typename OctreeT>
class OctreeViewer
{
public:
	OctreeViewer(OctreeT &octree)
		: viz("Octree visualizator"),
		  m_pOriginalPointCloud(octree.getInputCloud()),
		  m_octree(octree),
		  m_pVoxelCenterPointCloud(new pcl::PointCloud<pcl::PointXYZ>()),
		  m_fDrawWithCubesOrCenterPoints(true),
		  m_fDisplayOriginalPointsWithCubes(true),
		  m_fWireframe(true)
		  //,		  m_occupancyThreshold(0.9f)
		  , x(0.045f), y(-0.015f), z(0.082f), r(0.035f)
	{		
		//register keyboard callbacks
		viz.registerKeyboardCallback(&OctreeViewer::keyboardEventOccurred, *this, 0);
		
		//key legends
		viz.addText("Keys:",														 0, 170, 0.0, 1.0, 0.0, "keys_t");
		viz.addText("c -> Toggle Point/Cube representation",			10, 155, 0.0, 1.0, 0.0, "key_d_t");
		viz.addText("p -> Show/Hide original cloud",						10, 140, 0.0, 1.0, 0.0, "key_x_t");
		viz.addText("s/w -> Surface/Wireframe representation",		10, 125, 0.0, 1.0, 0.0, "key_sw_t");
		//viz.addText("i/k -> Increase/decrease occupancy threshold", 10, 110, 0.0, 1.0, 0.0, "key_th_t");
		
		//show m_octree at default depth
		extractCenterPoints();
		
		//reset camera
		viz.resetCameraViewpoint("cloud");
		
		//run main loop
		run();
	}


  //========================================================

	/* @brief		Callback to interact with the keyboard 
	 * @details		j: screeshot
	 */

	void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *)
	{
		if			(event.getKeySym() == "c" && event.keyDown())	{ m_fDrawWithCubesOrCenterPoints = !m_fDrawWithCubesOrCenterPoints; update(); }
		else if	(event.getKeySym() == "p" && event.keyDown())	{ m_fDisplayOriginalPointsWithCubes = !m_fDisplayOriginalPointsWithCubes; update(); }
		else if	(event.getKeySym() == "w" && event.keyDown())	{ if(!m_fWireframe)	m_fWireframe = true;		update(); }
		else if	(event.getKeySym() == "s" && event.keyDown())	{ if(m_fWireframe)	m_fWireframe = false;	update(); }
		//else if	(event.getKeySym() == "l" && event.keyDown())	{ m_occupancyThreshold += 0.02;	extractCenterPoints(); }
		//else if	(event.getKeySym() == "k" && event.keyDown())	{ m_occupancyThreshold -= 0.02;	extractCenterPoints(); }
		else if	(event.getKeySym() == "x" && event.keyDown() &&  event.isAltPressed())	{ x -= 0.001f;	std::cout << "x = " << setprecision(5) << x << std::endl; update(); }
		else if	(event.getKeySym() == "x" && event.keyDown() && !event.isAltPressed())	{ x += 0.001f;	std::cout << "x = " << setprecision(5) << x << std::endl; update(); }
		else if	(event.getKeySym() == "y" && event.keyDown() &&  event.isAltPressed())	{ y -= 0.001f;	std::cout << "y = " << setprecision(5) << y << std::endl; update(); }
		else if	(event.getKeySym() == "y" && event.keyDown() && !event.isAltPressed())	{ y += 0.001f;	std::cout << "y = " << setprecision(5) << y << std::endl; update(); }
		else if	(event.getKeySym() == "z" && event.keyDown() &&  event.isAltPressed())	{ z -= 0.001f;	std::cout << "z = " << setprecision(5) << z << std::endl; update(); }
		else if	(event.getKeySym() == "z" && event.keyDown() && !event.isAltPressed())	{ z += 0.001f;	std::cout << "z = " << setprecision(5) << z << std::endl; update(); }
		else if	(event.getKeySym() == "d" && event.keyDown() &&  event.isAltPressed())	{ r -= 0.001f;	std::cout << "r = " << setprecision(5) << r << std::endl; update(); }
		else if	(event.getKeySym() == "d" && event.keyDown() && !event.isAltPressed())	{ r += 0.001f;	std::cout << "r = " << setprecision(5) << r << std::endl; update(); }
	}
	
	
	/* @brief Graphic loop for the viewer */
	void run()
	{
		while (!viz.wasStopped())
		{
			//main loop of the visualizer
			viz.spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}

	/* @brief Helper function that draw info for the user on the viewer */
	void displayLegend(const bool fDrawCubes)
	{
		// cubes or points
		char dataDisplay[256];
		sprintf(dataDisplay, "Displaying data as %s", (fDrawCubes) ? ("CUBES") : ("POINTS"));
		viz.removeShape("disp_t");
		viz.addText(dataDisplay, 0, 60, 1.0, 0.0, 0.0, "disp_t");
		
		// voxel size
		char size[256];
		viz.removeShape("size_t");
		sprintf(size, "Voxel size: %.4fm [%d voxels]", m_octree.getResolution(), m_pVoxelCenterPointCloud->size());
		viz.addText(size, 0, 45, 1.0, 0.0, 0.0, "size_t");

		// original points
		viz.removeShape("org_t");
		if (m_fDisplayOriginalPointsWithCubes) viz.addText("Displaying original cloud", 0, 30, 1.0, 0.0, 0.0, "org_t");

		// threshold
		//viz.removeShape("thld_t");
		//sprintf(size, "Threshold: %.2f", m_occupancyThreshold);
		//viz.addText(size, 0, 15, 1.0, 0.0, 0.0, "thld_t");
	}

	/* @brief Visual update. Create visualizations and add them to the viewer */
	void update()
	{
		// remove existing shapes from visualizer
		clearView();
		
		// prevent the display of too many cubes
		bool fDrawCubes = m_fDrawWithCubesOrCenterPoints && (m_pVoxelCenterPointCloud->size() <= MAX_DISPLAYED_CUBES);
		
		// display legend
		displayLegend(fDrawCubes);

		// draw voxels
		if(fDrawCubes)
		{
			//draw octree as cubes
			drawCubes();
			
			// display original points with cubes
			if (m_fDisplayOriginalPointsWithCubes)
			{
				//add original cloud in visualizer
				pcl::visualization::PointCloudColorHandlerGenericField<PointT> hColor(m_pOriginalPointCloud, "z");
				viz.addPointCloud<PointT>(m_pOriginalPointCloud, hColor, "cloud");
			}
		}

		// display voxel center points
		else
		{
			//add current cloud in visualizer
			pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> hColor(m_pVoxelCenterPointCloud,"z");
			viz.addPointCloud<pcl::PointXYZ>(m_pVoxelCenterPointCloud, hColor, "cloud");
		}

		// sphere
		viz.removeShape("sphere");
		viz.addSphere(pcl::PointXYZ(x, y, z), r, 1, 0, 0, "sphere");
	}
	
	/* @brief remove dynamic objects from the viewer */
	void clearView()
	{
		//remove cubes if any
		vtkRenderer *renderer = viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();
		while(renderer->GetActors()->GetNumberOfItems() > 0)
			renderer->RemoveActor(renderer->GetActors()->GetLastActor());
		
		//remove point clouds if any
		viz.removePointCloud("cloud");
	}
	
	/* @brief Create a vtkSmartPointer object containing a cube */
	vtkSmartPointer<vtkPolyData> GetCuboid(double minX, double maxX, double minY, double maxY, double minZ, double maxZ)
	{
		vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
		cube->SetBounds(minX, maxX, minY, maxY, minZ, maxZ);
		return cube->GetOutput();
	}
	
	/* @brief display octree cubes via vtk-functions */
	void drawCubes()
	{
		if(m_pVoxelCenterPointCloud->size() == 0) return;

		//get the renderer of the visualizer object
		vtkRenderer *renderer = viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();

		// poly data
		vtkSmartPointer<vtkAppendPolyData> treeWireframe = vtkSmartPointer<vtkAppendPolyData>::New();

		// create cubes for each voxel center point with a fixed size
		const double s = m_octree.getResolution() / 2.0;
		for (size_t i = 0; i < m_pVoxelCenterPointCloud->points.size(); i++)
		{
			const double x = m_pVoxelCenterPointCloud->points[i].x;
			const double y = m_pVoxelCenterPointCloud->points[i].y;
			const double z = m_pVoxelCenterPointCloud->points[i].z;
			
			treeWireframe->AddInput(GetCuboid(x - s, x + s, y - s, y + s, z - s, z + s));
		}

		// tree actor
		vtkSmartPointer<vtkActor> treeActor = vtkSmartPointer<vtkActor>::New();

		// dataset mapper
		vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
		mapper->SetInput(treeWireframe->GetOutput());
		treeActor->SetMapper(mapper);
		
		// color, line width and wireframe/surface
		treeActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
		treeActor->GetProperty()->SetLineWidth(2);
		if(m_fWireframe)
		{
			treeActor->GetProperty()->SetRepresentationToWireframe();
			treeActor->GetProperty()->SetOpacity(0.35);
		}
		else
			treeActor->GetProperty()->SetRepresentationToSurface();
		
		// add to the renderer
		renderer->AddActor(treeActor);
	}

	bool isNotIsolatedVoxel(const pcl::octree::OctreeKey &key)
	{
		// if it is on the boundary
		// TODO: maxKey_
		if(key.x == 0 || key.y == 0 || key.z == 0) return true;

		// check if the node is surrounded with occupied nodes
		if(!m_octree.existLeaf(key.x+1, key.y,   key.z  ))		return true;
		if(!m_octree.existLeaf(key.x-1, key.y,   key.z  ))		return true;
		if(!m_octree.existLeaf(key.x,   key.y+1, key.z  ))		return true;
		if(!m_octree.existLeaf(key.x,   key.y-1, key.z  ))		return true;
		if(!m_octree.existLeaf(key.x,   key.y,   key.z+1))		return true;
		if(!m_octree.existLeaf(key.x,   key.y,   key.z-1))		return true;

		return false;
	}

	/* @brief Extracts all the center points of occupied voxels */
	void extractCenterPoints()
	{
#if 1
		// voxel centers
		//m_octree.getOccupiedVoxelCenters(m_pVoxelCenterPointCloud->points);
		//m_octree.getOccupiedVoxelCenters(m_pVoxelCenterPointCloud->points, true);
		m_octree.getOccupiedVoxelCenters(m_pVoxelCenterPointCloud->points, false);
		//m_octree.getOccupiedCellCenters(m_pVoxelCenterPointCloud->points, m_occupancyThreshold, true);
#else
		// clear
		m_pVoxelCenterPointCloud->clear();

		// iterator
		OctreeT::LeafNodeIterator iter(m_octree);
		pcl::PointXYZ pt;
		Eigen::Vector3f min_pt, max_pt;
		size_t nIsolated(0);
		while(*++iter)
		{
			// get bounds
			m_octree.getVoxelBounds(iter, min_pt, max_pt);

			// middle point
			pt.x = (min_pt.x() + max_pt.x()) / 2.0f;
			pt.y = (min_pt.y() + max_pt.y()) / 2.0f;
			pt.z = (min_pt.z() + max_pt.z()) / 2.0f;

			// check if its neighbor is also occupied
			// push back
			if(isNotIsolatedVoxel(iter.getCurrentOctreeKey()))	m_pVoxelCenterPointCloud->push_back(pt);
			else																nIsolated++;
		}
		std::cout << "isolated voxels: " << nIsolated << std::endl;
		std::cout << "non-isolated voxels: " << m_pVoxelCenterPointCloud->size() << std::endl;
#endif

		// update the scene
		update();
	}

protected:
	//visualizer
	pcl::visualization::PCLVisualizer viz;

	//original cloud
	typename pcl::PointCloud<PointT>::ConstPtr &m_pOriginalPointCloud;

	//displayed_cloud
	typename pcl::PointCloud<pcl::PointXYZ>::Ptr m_pVoxelCenterPointCloud;

	//octree
	OctreeT &m_octree;
	
	//bool to decide if we display points or cubes
	bool m_fDrawWithCubesOrCenterPoints, m_fDisplayOriginalPointsWithCubes, m_fWireframe;

	// threshold
	//float m_occupancyThreshold;
	float x, y, z, r;
};

}

#endif