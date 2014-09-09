#ifndef _BAYESIAN_COMMITTEE_MACHINE_HPP_
#define _BAYESIAN_COMMITTEE_MACHINE_HPP_

// STL
#include <cmath>

// GP
#include "GP.h"						// LogFile, Epsilon
using GP::LogFile;
using GP::Epsilon;

// GPMap
#include "util/data_types.hpp"	// MatrixPtr, VectorPtr

namespace GPMap {

/** @brief Bayesian Committee Machine */
class BCM
{
public:
	/** @brief Comparison Operator */
	inline bool operator==(const BCM &other) const
	{
		// memory check
		if(!isInitialized() || !other.isInitialized()) return false;

		// size check
		if(D() != other.D() ||
			isIndependent() != other.isIndependent()) return false;

		// compare data only 
		return (m_pSumOfWeightedMeans->isApprox(*(other.m_pSumOfWeightedMeans)) &&
						  m_pSumOfInvCovs->isApprox(*(other.m_pSumOfInvCovs)));
	}

	/** @brief Comparison Operator */
	inline bool operator!=(const BCM &other) const
	{
		return !((*this) == other);
	}

	/** @brief Initialization check */
	inline bool isInitialized() const
	{
		return (m_pSumOfWeightedMeans		&&		// memory for mean
				  m_pSumOfInvCovs				&&		// memory for cov
				  m_pSumOfWeightedMeans->size() == m_pSumOfInvCovs->rows()); // dimension
	}

	/** @brief Get the number of dimensions */
	inline size_t D() const
	{
		// memory check
		assert(isInitialized());

		return m_pSumOfWeightedMeans->size();
	}

	/** @brief Check independent BCM */
	inline size_t isIndependent() const
	{
		// memory check
		assert(isInitialized());

		// flag
		const bool fIsIndependent = (m_pSumOfInvCovs->cols() == 1);

		// if dependent, the cov should be square
		if(!fIsIndependent)	assert(m_pSumOfInvCovs->rows() == m_pSumOfInvCovs->cols());

		return fIsIndependent;
	}

	/** @brief Reset the prior inverse covariance matrix */
	static void resetPrior()
	{
		assert(m_fInvCov0);
		m_fInvCov0 = 0;
		m_pInvCov0.reset();
	}

	/** @brief Set the prior inverse covariance matrix */
	static void setPrior(const MatrixConstPtr &pCov)
	{
		assert(!m_fInvCov0);
		assert(pCov);

		// memory allocation
		m_pInvCov0.reset(new Matrix(pCov->rows(), pCov->cols()));

		// variance vector
		if(pCov->cols() == 1)
		{
			// inv(Sigma)
			m_pInvCov0->noalias() = pCov->cwiseInverse();

			// make it stable
			const float inv_eps = 1.f / Epsilon<float>::value;
			for(int row = 0; row < pCov->rows(); row++)
			{
				if((*pCov)(row, 0) < Epsilon<float>::value)	(*m_pInvCov0)(row, 0) = inv_eps;
			}
		}

		// covariance matrix
		else
		{
			assert(pCov->rows() == pCov->cols());

			// cholesky factor of the covariance matrix
			CholeskyFactor L(*pCov);

			int num_iters(-1);
			float factor;
			while(L.info() != Eigen::/*ComputationInfo::*/Success)
			{
				num_iters++;
				factor = powf(10.f, static_cast<float>(num_iters)) * Epsilon<float>::value;
				L.compute(*pCov + factor * Matrix::Identity(pCov->rows(), pCov->cols()));
			}
			if(num_iters > 0)
			{
				LogFile logFile;
				logFile << "BCM::Set::Iter: " << num_iters << "(" << factor << ")" << std::endl;
			}
//			if(L.info() != Eigen::/*ComputationInfo::*/Success)
//			{
//				GP::Exception e;
//				switch(L.info())
//				{
//					case Eigen::/*ComputationInfo::*/NumericalIssue :
//					{
//						e = "BCM::Set::NumericalIssue";
//						break;
//					}
//					case Eigen::/*ComputationInfo::*/NoConvergence :
//					{
//						e = "BCM::Set::NoConvergence";
//						break;
//					}
//#if EIGEN_VERSION_AT_LEAST(3,2,0)
//					case Eigen::/*ComputationInfo::*/InvalidInput :
//					{
//						e = "BCM::Set::InvalidInput";
//						break;
//					}
//#endif
//				}
//				throw e;
//			}

			// dimension
			const size_t dim = pCov->rows();

			// inv(Sigma)
#if EIGEN_VERSION_AT_LEAST(3,2,0)
			m_pInvCov0->noalias()	= L.solve(Matrix::Identity(dim, dim));	// (LL')*inv(Cov) = I
#else
			(*m_pInvCov0)				= L.solve(Matrix::Identity(dim, dim));				// (LL')*inv(Cov) = I
#endif
		}

		m_fInvCov0 = true;
	}

	/** @brief Get means and variances */
	bool get(VectorPtr &pMean, MatrixPtr &pVar) const
	{
		// memory check 
		//assert(isInitialized());
		if(!isInitialized()) return false;

		// memory allocation
		if(!pMean || pMean->size() != m_pSumOfWeightedMeans->size())	
			pMean.reset(new Vector(m_pSumOfWeightedMeans->size()));

		if(!pVar  || pVar->rows()  != m_pSumOfInvCovs->rows()
					 || pVar->cols()  != 1)			
			pVar.reset(new Matrix(m_pSumOfInvCovs->rows(), 1));

		// variance vector
		if(isIndependent())
		{
			// variance
			pVar->noalias() = m_pSumOfInvCovs->cwiseInverse();

			// make it stable
			const float inv_eps = 1.f / Epsilon<float>::value;
			for(int row = 0; row < m_pSumOfInvCovs->rows(); row++)
			{
				if((*m_pSumOfInvCovs)(row, 0) < Epsilon<float>::value)	(*pVar)(row, 0) = inv_eps;
			}

			// mean
			pMean->noalias() = pVar->cwiseProduct(*m_pSumOfWeightedMeans);
		}

		// covariance matrix
		else
		{
			// cholesky factor of the covariance matrix
			CholeskyFactor L(*m_pSumOfInvCovs);

			int num_iters(-1);
			float factor;
			while(L.info() != Eigen::/*ComputationInfo::*/Success)
			{
				num_iters++;
				factor = powf(10.f, static_cast<float>(num_iters)) * Epsilon<float>::value;
				L.compute(*m_pSumOfInvCovs + factor * Matrix::Identity(m_pSumOfInvCovs->rows(), m_pSumOfInvCovs->cols()));
			}
			if(num_iters > 0)
			{
				LogFile logFile;
				logFile << "BCM::Get::Iter: " << num_iters << "(" << factor << ")" << std::endl;
			}
//			if(L.info() != Eigen::/*ComputationInfo::*/Success)
//			{
//				GP::Exception e;
//				switch(L.info())
//				{
//					case Eigen::/*ComputationInfo::*/NumericalIssue :
//					{
//						e = "BCM::Get::NumericalIssue";
//						break;
//					}
//					case Eigen::/*ComputationInfo::*/NoConvergence :
//					{
//						e = "BCM::Get::NoConvergence";
//						break;
//					}
//#if EIGEN_VERSION_AT_LEAST(3,2,0)
//					case Eigen::/*ComputationInfo::*/InvalidInput :
//					{
//						e = "BCM::Get::InvalidInput";
//						break;
//					}
//#endif
//				}
//				throw e;
//			}

			// Sigma
#if EIGEN_VERSION_AT_LEAST(3,2,0)
			pVar->noalias()	= L.solve(Matrix::Identity(m_pSumOfInvCovs->rows(), m_pSumOfInvCovs->cols())).diagonal();	// (LL')*inv(Cov) = I
#else
			(*pVar)				= L.solve(Matrix::Identity(m_pSumOfInvCovs->rows(), m_pSumOfInvCovs->cols())).diagonal();	// (LL')*inv(Cov) = I
#endif

			// mean
#if EIGEN_VERSION_AT_LEAST(3,2,0)
			pMean->noalias()	= L.solve(*m_pSumOfWeightedMeans);	// (LL')*x = mean
#else
			(*pMean)				= L.solve(*m_pSumOfWeightedMeans);	// (LL')*x = mean
#endif
		}

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
		if(!isInitialized())
		{
			// memory allocation
			m_pSumOfWeightedMeans.reset(new Vector(pMean->size()));
			m_pSumOfInvCovs.reset(new Matrix(pCov->rows(), pCov->cols()));

			// set zero
			m_pSumOfWeightedMeans->setZero();
			//m_pSumOfInvCovs->setZero();
			(*m_pSumOfInvCovs) = (*m_pInvCov0);
		}
		else
		{
			// check size
			assert(pMean->size() == m_pSumOfWeightedMeans->size());
			assert(pCov->rows() == m_pSumOfInvCovs->rows() &&
					 pCov->cols() == m_pSumOfInvCovs->cols());
		}

		// temporary variables
		Vector weightedMean(pMean->size());				// weighted mean
		Matrix invCov(pCov->rows(), pCov->cols());	// inverted cov

		// variance vector
		if(isIndependent())
		{
			// inv(Sigma)
			invCov.noalias() = pCov->cwiseInverse();

			// make it stable
			const float inv_eps = 1.f / Epsilon<float>::value;
			for(int row = 0; row < pCov->rows(); row++)
			{
				if((*pCov)(row, 0) < Epsilon<float>::value)	invCov(row, 0) = inv_eps;
			}

			// inv(Sigma)*mean
			weightedMean.noalias() = invCov.cwiseProduct(*pMean);
		}

		// covariance matrix
		else
		{
			// cholesky factor of the covariance matrix
			CholeskyFactor L(*pCov);

			int num_iters(-1);
			float factor;
			while(L.info() != Eigen::/*ComputationInfo::*/Success)
			{
				num_iters++;
				factor = powf(10.f, static_cast<float>(num_iters)) * Epsilon<float>::value;
				L.compute(*pCov + factor * Matrix::Identity(pCov->rows(), pCov->cols()));
			}
			if(num_iters > 0)
			{
				LogFile logFile;
				logFile << "BCM::Update::Iter: " << num_iters << "(" << factor << ")" << std::endl;
			}
//			if(L.info() != Eigen::/*ComputationInfo::*/Success)
//			{
//				GP::Exception e;
//				switch(L.info())
//				{
//					case Eigen::/*ComputationInfo::*/NumericalIssue :
//					{
//						e = "BCM::Update::NumericalIssue";
//						break;
//					}
//					case Eigen::/*ComputationInfo::*/NoConvergence :
//					{
//						e = "BCM::Update::NoConvergence";
//						break;
//					}
//#if EIGEN_VERSION_AT_LEAST(3,2,0)
//					case Eigen::/*ComputationInfo::*/InvalidInput :
//					{
//						e = "BCM::Update::InvalidInput";
//						break;
//					}
//#endif
//				}
//				throw e;
//			}

			// dimension
			const size_t dim = D();

			// inv(Sigma)
#if EIGEN_VERSION_AT_LEAST(3,2,0)
			invCov.noalias()	= L.solve(Matrix::Identity(dim, dim));	// (LL')*inv(Cov) = I
#else
			invCov				= L.solve(Matrix::Identity(dim, dim));				// (LL')*inv(Cov) = I
#endif

			// inv(Sigma)*mean
#if EIGEN_VERSION_AT_LEAST(3,2,0)
			weightedMean.noalias()	= L.solve(*pMean);	// (LL')x = b
#else
			weightedMean				= L.solve(*pMean);	// (LL')x = b
#endif
		}

		// add up
		(*m_pSumOfInvCovs)			+= invCov;			// sum of inverted covariance matrices or variance vectors
		(*m_pSumOfWeightedMeans)	+= weightedMean;	// sum of weighted means
		if(m_fInvCov0) (*m_pSumOfInvCovs) -= (*m_pInvCov0);	// zero variance
	}

protected:
	/** @brief	Sum of weighted means with its inverse covariance matrix 
	  * @detail	\f$\mathbf\mu_* = \mathbf\Sigma_*\left(\sum_{k=1}^K \mathbf\Sigma_k^{-1}\mathbf\mu_k\right)\f$
	  */
	VectorPtr m_pSumOfWeightedMeans;

	/** @brief Sum of inverse covariance matrices
	  * @detail	\f$\mathbf\Sigma_* = \left(\sum_{k=1}^K \mathbf\Sigma_k^{-1} - (K-1)\mathbf\Simga_0^{-1}\right)^{-1}\f$
	  */
	MatrixPtr m_pSumOfInvCovs;

	/** @brief Prior inverse covariance matrices */
	static MatrixPtr	m_pInvCov0;
	static bool			m_fInvCov0;
};

MatrixPtr	BCM::m_pInvCov0;
bool			BCM::m_fInvCov0 = false;

}


#endif