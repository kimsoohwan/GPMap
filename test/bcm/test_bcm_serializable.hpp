#ifndef _TEST_BCM_SERIALIZATION_HPP_
#define _TEST_BCM_SERIALIZATION_HPP_

// Google Test
#include "gtest/gtest.h"

// GPMap
#include "bcm/bcm_serializable.hpp"
using namespace GPMap;

#include "bcm/test_bcm.hpp"

class TestBCMSerialization : public ::testing::Test,
									  public TestBCMData,
									  public BCM_Serializable
{
};

/** @brief Update by mean vectors and covariance matrices */
TEST_F(TestBCMSerialization, Serialization)
{
	update(pMean1, pCov1);
	//dump("bcm.txt");
	load();

	BCM bcm;
	bcm.update(pMean1, pCov1);
	EXPECT_TRUE(static_cast<BCM>(*this) == bcm);
}

/** @brief Update by mean vectors and covariance matrices */
TEST_F(TestBCMSerialization, CovTest)
{
	// pointer should be NULL initially
	EXPECT_FALSE(m_pSumOfWeightedMeans);
	EXPECT_FALSE(m_pSumOfInvCovs);

	// prediction 1
	update(pMean1, pCov1);	//dump("bcm.txt");
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs1));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov1));
	dump();	//dump("bcm.txt");

	// prediction 2
	update(pMean2, pCov2);
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs2));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov2));
	dump();	//	dump("bcm.txt");

	// prediction 3
	update(pMean3, pCov3);
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs3));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov3));
	dump();	//	dump("bcm.txt");

	// final
	VectorPtr pMean;
	MatrixPtr pCov;
	get(pMean, pCov);
	EXPECT_TRUE(pCov->isApprox(pCovFinal->diagonal())); // get a variance!!!
	EXPECT_TRUE(pMean->isApprox(*pMeanByCovFinal));
}

/** @brief Update by mean vectors and variance vectors */
TEST_F(TestBCMSerialization, VarTest)
{
	// pointer should be NULL initially
	EXPECT_FALSE(m_pSumOfWeightedMeans);
	EXPECT_FALSE(m_pSumOfInvCovs);

	// prediction 1
	update(pMean1, pVar1);	// dump("bcm.txt");
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar1));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar1));
	dump();	//	dump("bcm.txt");

	// prediction 2
	update(pMean2, pVar2);
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar2));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar2));
	dump();	//	dump("bcm.txt");

	// prediction 3
	update(pMean3, pVar3);
	load();
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar3));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar3));
	dump();	//	dump("bcm.txt");

	// final
	VectorPtr pMean;
	MatrixPtr pCov;
	get(pMean, pCov);
	EXPECT_TRUE(pCov->isApprox(*pVarFinal));
	EXPECT_TRUE(pMean->isApprox(*pMeanByVarFinal));
}

#endif