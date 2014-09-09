#ifndef _GAUSSIAN_DISTRIBUTION_HPP_
#define _GAUSSIAN_DISTRIBUTION_HPP_

// STL
#include <cmath>

// GP
#include "GP.h"						// LogFile, Epsilon
using GP::LogFile;
using GP::Epsilon;

// GPMap
#include "util/data_types.hpp"	// MatrixPtr, VectorPtr

namespace GPMap {

/** @class Gaussian Distribution */
class GaussianDistribution
{
public:
	/** @brief Comparison Operator */
	inline bool operator==(const GaussianDistribution &other) const
	{
		// memory check
		if(!isInitialized() || !other.isInitialized()) return false;

		// size check
		if(D() != other.D() ||
			isIndependent() != other.isIndependent()) return false;

		// compare data only 
		return (m_pMean->isApprox(*(other.m_pMean)) &&
				   m_pCov->isApprox(*(other.m_pCov)));
	}

	/** @brief Comparison Operator */
	inline bool operator!=(const GaussianDistribution &other) const
	{
		return !((*this) == other);
	}

	/** @brief Initialization check */
	inline bool isInitialized() const
	{
		return (m_pMean		&&		// memory for mean
				  m_pCov			&&		// memory for cov
				  m_pMean->size() == m_pCov->rows()); // dimension
	}

	/** @brief Get the number of dimensions */
	inline size_t D() const
	{
		// memory check
		assert(isInitialized());

		return m_pMean->size();
	}

	/** @brief Check independent GaussianDistribution */
	inline size_t isIndependent() const
	{
		// memory check
		assert(isInitialized());

		// flag
		const bool fIsIndependent = (m_pCov->cols() == 1);

		// if dependent, the cov should be square
		if(!fIsIndependent)	assert(m_pCov->rows() == m_pCov->cols());

		return fIsIndependent;
	}

	/** @brief Get means and variances */
	bool get(VectorPtr &pMean, MatrixPtr &pVar) const
	{
		// memory check 
		//assert(isInitialized());
		if(!isInitialized()) return false;

		// memory allocation
		if(!pMean || pMean->size() != m_pMean->size())	
			pMean.reset(new Vector(m_pMean->size()));

		if(!pVar  || pVar->rows()  != m_pCov->rows()
					 || pVar->cols()  != 1)			
			pVar.reset(new Matrix(m_pCov->rows(), 1));

		// just copy
		*pMean = *m_pMean;
		if(m_pCov->cols() == 1)		*pVar	= *m_pCov;
		else								*pVar = m_pCov->diagonal();

		return true;
	}

	/** @brief Update the mean and [co]variance */
	void update(const VectorConstPtr &pMean, const MatrixConstPtr &pCov)
	{
		// memory check
		assert(pMean && pMean->size() > 0 && 
				 pCov  && pCov->rows()  > 0 &&
							 pCov->cols()  > 0 &&
				 pMean->size() == pCov->rows() &&
				 (pCov->cols() == 1 || pCov->rows() == pCov->cols()));

		// initialization
		if(!isInitialized()						|| 
			pMean->size() != m_pMean->size() ||
			pCov->rows()  != m_pCov->rows()	||
			pCov->cols()  != m_pCov->cols())
		{
			// memory allocation
			m_pMean.reset(new Vector(pMean->size()));
			m_pCov.reset(new Matrix(pCov->rows(), pCov->cols()));
		}

		// just copy
		*m_pMean	= *pMean;
		*m_pCov	= *pCov;
	}

protected:
	/** @brief	Mean vector */
	VectorPtr m_pMean;

	/** @brief	Covariance matrix or variance vector */
	MatrixPtr m_pCov;
};

}


#endif