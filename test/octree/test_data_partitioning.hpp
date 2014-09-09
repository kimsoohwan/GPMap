#ifndef _TEST_DATA_PARTITIONING_HPP_
#define _TEST_DATA_PARTITIONING_HPP_

// Google Test
#include "gtest/gtest.h"

// GPMap
#include "octree/data_partitioning.hpp"
using namespace GPMap;

TEST(Octree, DataPartitioning)
{
	// indices 1
	const int N1 = 10;
	std::vector<int> indices1(N1);
	std::generate(indices1.begin(), indices1.end(), UniqueNonZeroInteger());
	std::cout << std::endl;
	for(int i = 0; i < N1; i++)
	{
		EXPECT_EQ(i, indices1[i]);
	}

	// indices 2
	const int N2 = 20;
	std::vector<int> indices2(N2);
	std::generate(indices2.begin(), indices2.end(), UniqueNonZeroInteger());
	std::cout << std::endl;
	for(int i = 0; i < N2; i++)
	{
		EXPECT_EQ(i, indices2[i]);
	}

	// indices
	const int N = 30;
	std::vector<int> indices(N);
	std::generate(indices.begin(), indices.end(), UniqueNonZeroInteger());

	// data partition
	const int M = 8;
	std::vector<std::vector<int> > partitionedIndices;

	// with out suffling
	if(random_data_partition(indices, M, partitionedIndices, false))
	{
		for(size_t i = 0; i < partitionedIndices.size(); i++)
		{
			std::cout << "Partition " << i << ": ";
			for(size_t j = 0; j < partitionedIndices[i].size(); j++)
			{
				std::cout << partitionedIndices[i][j] << ", ";
			}
			std::cout << std::endl;
		}
	}

	// with suffling
	if(random_data_partition(indices, M, partitionedIndices, true))
	{
		for(size_t i = 0; i < partitionedIndices.size(); i++)
		{
			std::cout << "Partition " << i << ": ";
			for(size_t j = 0; j < partitionedIndices[i].size(); j++)
			{
				std::cout << partitionedIndices[i][j] << ", ";
			}
			std::cout << std::endl;
		}
	}
}

TEST(Octree, DataSampling)
{
	// indices
	const int N = 10;
	std::vector<int> indices(N);
	std::generate(indices.begin(), indices.end(), UniqueNonZeroInteger());

	// random sampling
	const int M = 5;
	std::vector<int> randomSampleIndices;
	random_sampling(indices, M, randomSampleIndices, false);

	// check
	EXPECT_EQ(M, randomSampleIndices.size());
	for(int i = 0; i < M; i++)
	{
		EXPECT_EQ(i, randomSampleIndices[i]);
	}

	// random sampling
	std::vector<int> randomSampleIndices2;
	random_sampling(indices, M, randomSampleIndices);
	for(size_t i = 0; i < randomSampleIndices.size(); i++)
	{
		std::cout << randomSampleIndices[i] << ", ";
	}
	std::cout << std::endl;
}

TEST(Octree, RandomSampling)
{
	// original point cloud
	const int N = 10;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < N; i++)
	{
		pCloud->push_back(pcl::PointXYZ(i, i, i));
	}

	// sampled point cloud
	const float samplingRatio = 0.4;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pRandomlySampledCloud = randomSampling<pcl::PointXYZ>(pCloud, samplingRatio, false);
	const int M = static_cast<int>(ceil(static_cast<float>(N)*samplingRatio));
	EXPECT_EQ(M, pRandomlySampledCloud->points.size());
	for (int i = 0; i < M; ++i)
	{
		pRandomlySampledCloud->points[i].x = pCloud->points[i].x;
		pRandomlySampledCloud->points[i].y = pCloud->points[i].y;
		pRandomlySampledCloud->points[i].z = pCloud->points[i].z;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr pRandomlySampledCloud2 = randomSampling<pcl::PointXYZ>(pCloud, samplingRatio, true);
	for (int i = 0; i < M; ++i)
	{
		std::cout << pRandomlySampledCloud2->points[i].x << ", "
					 << pRandomlySampledCloud2->points[i].y << ", "
					 << pRandomlySampledCloud2->points[i].z << std::endl;
	}

}

#endif