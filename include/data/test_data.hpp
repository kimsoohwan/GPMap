#ifndef _GPMAP_TEST_DATA_HPP_
#define _GPMAP_TEST_DATA_HPP_

// GPMap
#include "util/data_types.hpp" // Matrix, MatrixPtr, MatrixConstPtr

namespace GPMap {

inline size_t xyz2row(const size_t n, const size_t ix, const size_t iy, const size_t iz)
{
	return (ix*n*n + iy*n + iz);
}

inline void row2xyz(const size_t n, size_t row, size_t &ix, size_t &iy, size_t &iz)
{
	ix = static_cast<size_t>(static_cast<double>(row)/static_cast<double>(n*n));
	row -= ix*n*n;
	iy = static_cast<size_t>(static_cast<double>(row)/static_cast<double>(n));
	iz = row - iy*n;
}

void meshGrid(const Eigen::Vector3f		&min_pt,
				  const size_t					n,				// number of grid per axis
				  const float					gridSize,
				  MatrixPtr						&pXs)
{
	assert(n > 0);

	// matrix size
	if(!pXs || pXs->rows() != (n*n*n) || pXs->cols() != 3)
		pXs.reset(new Matrix(n*n*n, 3));

	// generate mesh grid in the order of x, y, z
	int row = 0;
	float x = min_pt.x() + gridSize * static_cast<float>(0.5f);
	for(size_t ix = 0; ix < n; ix++, x += gridSize)
	{
		float y = min_pt.y() + gridSize * static_cast<float>(0.5f);
		for(size_t iy = 0; iy < n; iy++, y += gridSize)
		{
			float z = min_pt.z() + gridSize * static_cast<float>(0.5f);
			for(size_t iz = 0; iz < n; iz++, z += gridSize)
			{
				(*pXs)(row, 0) = x;
				(*pXs)(row, 1) = y;
				(*pXs)(row, 2) = z;
				row++;
			}
		}
	}
}

}
#endif