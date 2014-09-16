#ifndef _BAYESIAN_COMMITTEE_MACHINE_WITH_PRIOR_HPP_
#define _BAYESIAN_COMMITTEE_MACHINE_WITH_PRIOR_HPP_


// GPMap
#include "bcm.hpp"	// BCM

namespace GPMap {

/** @brief Bayesian Committee Machine */
class BCM_With_Prior : public BCM
{
public:

	/** @brief Reset the prior inverse covariance matrix */
	static void resetPrior()
	{
		//assert(m_fPriorSetted);
		if(!m_fPriorSetted) return;
		m_pInvCov0.reset();
		m_fPriorSetted = false;
	}

	/** @brief Set the prior inverse covariance matrix */
	static void setPrior(const MatrixConstPtr &pCov)
	{
		//assert(!m_fPriorSetted);
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

		m_fPriorSetted = true;
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
			if(m_fPriorSetted)		(*m_pSumOfInvCovs) = (*m_pInvCov0);
			else							m_pSumOfInvCovs->setZero();
		}
		else
		{
			// check size
			assert(pMean->size() == m_pSumOfWeightedMeans->size());
			assert(pCov->rows() == m_pSumOfInvCovs->rows() &&
					 pCov->cols() == m_pSumOfInvCovs->cols());
		}

		// BCM
		BCM::update(pMean, pCov);

		// add the prior
		if(m_fPriorSetted) (*m_pSumOfInvCovs) -= (*m_pInvCov0);	// zero variance
	}

protected:

	/** @brief Prior inverse covariance matrices */
	static MatrixPtr	m_pInvCov0;
	static bool			m_fPriorSetted;
};

MatrixPtr	BCM_With_Prior::m_pInvCov0;
bool			BCM_With_Prior::m_fPriorSetted = false;

}


#endif