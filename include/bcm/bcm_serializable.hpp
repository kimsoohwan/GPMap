#ifndef _BAYESIAN_COMMITTEE_MACHINE_SERIALIZABLE_HPP_
#define _BAYESIAN_COMMITTEE_MACHINE_SERIALIZABLE_HPP_

// STL
#include <string>
#include <sstream>
#include <fstream>

// Boost
#include <boost/serialization/split_member.hpp>
#include <boost/filesystem.hpp>	// oost::filesystem::exists, remove

// GPMap
#include "bcm.hpp"
#include "serialization/eigen_serialization.hpp"	// serialize, deserialize

namespace GPMap {

class BCM_Serializable : public BCM
{
public:
	/** @brief Default Constructor */
	BCM_Serializable()
		: m_fDumped(false)
	{
	}

	/** @brief Default Destructor */
	virtual ~BCM_Serializable()
	{
		// remove temporary file
		if(m_fDumped)
		{
			if(boost::filesystem::exists(mem2str()))
				boost::filesystem::remove(mem2str());
		} 
	}

	/** @brief Get mean and [co]variance */
	inline bool get(VectorPtr &pMean, MatrixPtr &pVar)
	{
		// load if necessary
		load();

		// get
		const bool ret = BCM::get(pMean, pVar);

		// dump
		dump();

		return ret;
	}

	/** @brief Update the mean and [co]variance */
	void update(const VectorConstPtr &pMean, const MatrixConstPtr &pCov)
	{
		// load if necessary
		load();

		// update
		BCM::update(pMean, pCov);

		// dump
		dump();
	}

	///** @brief Comparison Operator */
	//inline bool operator==(BCM_Serializable &other)
	//{
	//	// load the data
	//	load();
	//	other.load();

	//	// compare
	//	const bool comparison = static_cast<BCM>(*this) == static_cast<BCM>(other);

	//	// dump
	//	dump();
	//	other.dump();

	//	return comparison;
	//}

	///** @brief Comparison Operator */
	//inline bool operator!=(const BCM_Serializable &other)
	//{
	//	return !((*this) == other);
	//}

protected:
	/** @brief Convert the unique memory address to string */
	inline std::string mem2str() const
	{
		const void *address = static_cast<const void*>(this);
		std::stringstream ss;
		ss << address;  
		return ss.str(); 
	}

	/** @brief Dump all the data */
	bool dump()
	{
		// check initialized
		if(!isInitialized()) return false;

		// save
		if(!GPMap::serialize(*this, mem2str())) return false;

		// deallocate memories
		m_pSumOfWeightedMeans.reset();
		m_pSumOfInvCovs.reset();

		// turn the flag on
		m_fDumped = true;

		return true;
	}

	/** @brief Load all the data */
	bool load()
	{
		// flag check
		if(!m_fDumped) return false;

		// load
		if(!GPMap::deserialize(*this, mem2str())) return false;

		// turn the flag off
		m_fDumped = false;

		return true;
	}

protected:
	/** @brief	Boost Serialization */
	friend class boost::serialization::access;

	/** @brief	Save */
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const
	{
		// initialization flag
		const bool fInitialized = isInitialized();
		ar & fInitialized;

		// data
		if(fInitialized)
		{
			// dimension
			const size_t dim = D();
			ar & dim;

			// independency flag
			const bool fIndependent = isIndependent();
			ar & fIndependent;

			// mean
			ar & (*m_pSumOfWeightedMeans);

			// cov
			ar & (*m_pSumOfInvCovs);
		}
	}

	/** @brief	Load */
	template<class Archive>
	void load(Archive & ar, const unsigned int version)
	{
		// initialization flag
		bool fInitialized;
		ar & fInitialized;

		// data
		if(fInitialized)
		{
			// dimension
			size_t dim;
			ar & dim;

			// independency flag
			bool fIndependent;
			ar & fIndependent;

			// memory allocation
			if(!m_pSumOfWeightedMeans || m_pSumOfWeightedMeans->size() != dim)
			{
				m_pSumOfWeightedMeans.reset(new Vector(dim));
			}
			if(fIndependent)
			{
				if(!m_pSumOfInvCovs || m_pSumOfInvCovs->rows() != dim || m_pSumOfInvCovs->cols() != 1)
					m_pSumOfInvCovs.reset(new Matrix(dim, 1));
			}
			else
			{
				if(!m_pSumOfInvCovs || m_pSumOfInvCovs->rows() != dim || m_pSumOfInvCovs->cols() != dim)
					m_pSumOfInvCovs.reset(new Matrix(dim, dim));
			}

			// mean
			ar & (*m_pSumOfWeightedMeans);

			// cov
			ar & (*m_pSumOfInvCovs);
		}
	}

	BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
	/** @brief	Flag for serialization */
	bool				m_fDumped;
};


}

#endif