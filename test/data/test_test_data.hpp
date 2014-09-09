#ifndef _TEST_TEST_DATA_HPP_
#define _TEST_TEST_DATA_HPP_

// Google Test
#include "gtest/gtest.h"

// GPMap
#include "data/test_data.hpp"
using namespace GPMap;

class TestTestData : public ::testing::Test
{
public:
	/** @brief	Constructor. */
	TestTestData()
		: N(3),
		  CELL_SIZE(0.1f),
		  pXs1(new Matrix(N*N*N, 3))
	{
		// origin
		min_pt << 0.5377f, 1.8339f, -2.2588f;

		// test data
		(*pXs1) << 0.587700000000000f,  1.883900000000000f, -2.208800000000000f, 
					  0.587700000000000f,  1.883900000000000f, -2.108800000000000f, 
					  0.587700000000000f,  1.883900000000000f, -2.008800000000000f, 
					  0.587700000000000f,  1.983900000000000f, -2.208800000000000f, 
					  0.587700000000000f,  1.983900000000000f, -2.108800000000000f, 
					  0.587700000000000f,  1.983900000000000f, -2.008800000000000f, 
					  0.587700000000000f,  2.083900000000000f, -2.208800000000000f, 
					  0.587700000000000f,  2.083900000000000f, -2.108800000000000f, 
					  0.587700000000000f,  2.083900000000000f, -2.008800000000000f, 
					  0.687700000000000f,  1.883900000000000f, -2.208800000000000f, 
					  0.687700000000000f,  1.883900000000000f, -2.108800000000000f, 
					  0.687700000000000f,  1.883900000000000f, -2.008800000000000f, 
					  0.687700000000000f,  1.983900000000000f, -2.208800000000000f, 
					  0.687700000000000f,  1.983900000000000f, -2.108800000000000f, 
					  0.687700000000000f,  1.983900000000000f, -2.008800000000000f, 
					  0.687700000000000f,  2.083900000000000f, -2.208800000000000f, 
					  0.687700000000000f,  2.083900000000000f, -2.108800000000000f, 
					  0.687700000000000f,  2.083900000000000f, -2.008800000000000f, 
					  0.787700000000000f,  1.883900000000000f, -2.208800000000000f, 
					  0.787700000000000f,  1.883900000000000f, -2.108800000000000f, 
					  0.787700000000000f,  1.883900000000000f, -2.008800000000000f, 
					  0.787700000000000f,  1.983900000000000f, -2.208800000000000f, 
					  0.787700000000000f,  1.983900000000000f, -2.108800000000000f, 
					  0.787700000000000f,  1.983900000000000f, -2.008800000000000f, 
					  0.787700000000000f,  2.083900000000000f, -2.208800000000000f, 
					  0.787700000000000f,  2.083900000000000f, -2.108800000000000f, 
					  0.787700000000000f,  2.083900000000000f, -2.008800000000000f;
	}

protected:
	const size_t		N;				// number of cells per axis
	const float			CELL_SIZE;	// cell size
	Eigen::Vector3f	min_pt;		// min point of the block

	/** @brief The Function Training inputs. */
	MatrixPtr pXs1;
};


/** @brief Test for xyz2row and row2xyz */
TEST_F(TestTestData, IndexTest)
{
	// row2xyz
	// expected
	size_t idx1 = 0;					// expected
	size_t idx2, ix2, iy2, iz2;	// actual
	for(size_t ix1 = 0; ix1 < N; ix1++)
	{
		for(size_t iy1 = 0; iy1 < N; iy1++)
		{
			for(size_t iz1 = 0; iz1 < N; iz1++)
			{
				// xyz2row
				idx2 = xyz2row(N, ix1, iy1, iz1); // actual
				EXPECT_TRUE(idx1 == idx2);

				// row2xyz
				row2xyz(N, idx1, ix2, iy2, iz2); // actual
				EXPECT_TRUE(ix1 == ix2 && iy1 == iy2 && iz1 == iz2);

				// next index
				idx1++;
			}
		}
	}
}

/** @brief Test for meshGrid */
TEST_F(TestTestData, DataTest)
{
	// actual
	MatrixPtr pXs2;
	meshGrid(min_pt, N, CELL_SIZE, pXs2);

	// comparison
	EXPECT_TRUE(pXs1->isApprox(*pXs2));
}

/** @brief Test for meshGrid */
TEST_F(TestTestData, ShiftTest)
{
	// actual
	MatrixPtr pXs2;
	meshGrid(Eigen::Vector3f(0.f, 0.f, 0.f), N, CELL_SIZE, pXs2);
	Matrix minValue(1, 3); 
	minValue << min_pt.x(), min_pt.y(), min_pt.z();
	pXs2->noalias() = (*pXs2) + minValue.replicate(N*N*N, 1);

	// comparison
	EXPECT_TRUE(pXs1->isApprox(*pXs2));
}
#endif