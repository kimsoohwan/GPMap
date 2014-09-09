#ifndef _TEST_EIGEN_SERIALIZATION_HPP_
#define _TEST_EIGEN_SERIALIZATION_HPP_

// Boost - Shared Pointer
#include <boost/shared_ptr.hpp>

// Google Test
#include "gtest/gtest.h"

// GPMap
#include "serialization/eigen_serialization.hpp"
using namespace GPMap;

TEST(TestEigen, Serialization)
{
	// vector
	Eigen::VectorXf vec1(100);
	vec1.setRandom();
	serialize(vec1, "vec.txt");

	Eigen::VectorXf vec2;
	deserialize(vec2, "vec.txt");

	EXPECT_TRUE(vec1.isApprox(vec2));

	// matrix
	Eigen::MatrixXf mat1(100, 200);
	mat1.setRandom();
	serialize(mat1, "mat.bin");

	Eigen::MatrixXf mat2;
	deserialize(mat2, "mat.bin");

	EXPECT_TRUE(mat1.isApprox(mat2));

	// shared pointer
	boost::shared_ptr<Eigen::MatrixXf> pMat1(new Eigen::MatrixXf(100, 200));
	pMat1->setRandom();
	serialize(*pMat1, "mat2.bin");

	boost::shared_ptr<Eigen::MatrixXf> pMat2(new Eigen::MatrixXf());
	deserialize(*pMat2, "mat2.bin");

	EXPECT_TRUE((*pMat1).isApprox(*pMat2));
}

#endif