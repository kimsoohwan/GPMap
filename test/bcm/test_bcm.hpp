#ifndef _TEST_BCM_HPP_
#define _TEST_BCM_HPP_

// Google Test
#include "gtest/gtest.h"

// GPMap
#include "bcm/bcm.hpp"
using namespace GPMap;

class TestBCMData
{
public:
	TestBCMData()
		: EPS_SOLVER(1e-5f),
		  D(4),
		  pMean1(new Vector(D)), pMean2(new Vector(D)), pMean3(new Vector(D)),
		  pSumOfWeightedMeansByCov1(new Vector(D)), pSumOfWeightedMeansByCov2(new Vector(D)), pSumOfWeightedMeansByCov3(new Vector(D)),
		  pSumOfWeightedMeansByVar1(new Vector(D)), pSumOfWeightedMeansByVar2(new Vector(D)), pSumOfWeightedMeansByVar3(new Vector(D)),
		  pMeanByCovFinal(new Vector(D)), pMeanByVarFinal(new Vector(D)),

		  pCov1(new Matrix(D, D)), pCov2(new Matrix(D, D)), pCov3(new Matrix(D, D)),
		  pSumOfInvCovs1(new Matrix(D, D)), pSumOfInvCovs2(new Matrix(D, D)), pSumOfInvCovs3(new Matrix(D, D)), 
		  pCovFinal(new Matrix(D, D)),

		  pVar1(new Matrix(D, 1)), pVar2(new Matrix(D, 1)), pVar3(new Matrix(D, 1)),
		  pSumOfInvVar1(new Matrix(D, 1)), pSumOfInvVar2(new Matrix(D, 1)), pSumOfInvVar3(new Matrix(D, 1)),
		  pVarFinal(new Matrix(D, 1))
	{
		// predictions 1
		(*pMean1) << 0.034400000000000f, 0.438700000000000f, 0.381600000000000f, 0.765500000000000f;
		(*pVar1)  << 2.250000000000000f, 2.250000000000000f, 2.250000000000000f, 2.250000000000000f;
		(*pCov1)  << 2.250000000000000f,  1.172075789902540f,  0.079002611998964f,  0.300544591865789f,
						 1.172075789902540f,  2.250000000000000f,  0.499333922441860f,  0.352442509452200f,
						 0.079002611998964f,  0.499333922441860f,  2.250000000000000f,  0.455259868374894f,
						 0.300544591865789f,  0.352442509452200f,  0.455259868374894f,  2.250000000000000f;

		// predictions 12
		(*pMean2) << 0.795200000000000f, 0.186900000000000f, 0.489800000000000f, 0.445600000000000f;
		(*pVar2)  << 2.250000000000000f, 2.250000000000000f, 2.250000000000000f, 2.250000000000000f;
		(*pCov2)  <<  2.250000000000000f,  0.834976912859432f,  1.222327583240585f,  0.766086766254449f,
						  0.834976912859432f,  2.250000000000000f,  0.391495865690848f,  0.582607113981600f,
						  1.222327583240585f,  0.391495865690848f,  2.250000000000000f,  1.108455135636441f,
						  0.766086766254449f,  0.582607113981600f,  1.108455135636441f,  2.250000000000000f;

		// predictions 3
		(*pMean3) << 0.473300000000000f, 0.351700000000000f, 0.830800000000000f, 0.585300000000000f;
		(*pVar3)  << 2.250000000000000f, 2.250000000000000f, 2.250000000000000f, 2.250000000000000f;
		(*pCov3)  << 2.250000000000000f,  1.064881618043477f,  1.872832547714784f,  0.279747512475859f,
						 1.064881618043477f,  2.250000000000000f,  1.128165470718026f,  0.467834909028893f,
						 1.872832547714784f,  1.128165470718026f,  2.250000000000000f,  0.242703587730994f,
						 0.279747512475859f,  0.467834909028894f,  0.242703587730994f,  2.250000000000000f;

		// intermediate state 1
		(*pSumOfInvCovs1) <<  0.619696750138153f, -0.329363327734956f,  0.060105934637053f, -0.043346083997247f,
									-0.329363327734956f,  0.649011169776005f, -0.125956394344468f, -0.032181274185984f,
									 0.060105934637053f, -0.125956394344468f,  0.487894053334891f, -0.087018092618678f,
									-0.043346083997247f, -0.032181274185984f, -0.087018092618678f,  0.472882366909877f;
		(*pSumOfWeightedMeansByCov1) << -0.133419126314966f, 0.200691376235431f, 0.066378594805593f, 0.313176317451327f;

		(*pSumOfInvVar1) <<  0.444444444444444f, 0.444444444444444f, 0.444444444444444f, 0.444444444444444f;
		(*pSumOfWeightedMeansByVar1) <<  0.015288888888889f, 0.194977777777778f, 0.169600000000000f, 0.340222222222222f;
		
		// intermediate state 2
		(*pSumOfInvCovs2) <<  1.330542494083842f, -0.528005028759328f, -0.280198489111929f, -0.066291534197481f,
									-0.528005028759328f,  1.182363848018523f, -0.060159366499103f, -0.135066314029981f,
									-0.280198489111929f, -0.060159366499103f,  1.239485970790945f, -0.358456768963841f,
									-0.066291534197481f, -0.135066314029981f, -0.358456768963841f,  1.085503414325298f;
		(*pSumOfWeightedMeansByCov2) <<  0.217813675987715f, 0.128796921628552f, 0.055242628535271f, 0.415734156559709f;

		(*pSumOfInvVar2) <<   0.888888888888889f, 0.888888888888889f, 0.888888888888889f, 0.888888888888889f;
		(*pSumOfWeightedMeansByVar2) <<   0.368711111111111f, 0.278044444444444f, 0.387288888888889f, 0.538266666666667f;
		
		// intermediate state 3
		(*pSumOfInvCovs3) <<  2.800136222118345f, -0.630516297512465f, -1.448224773397999f, -0.101701609715652f,
									-0.630516297512465f,  1.802841176561167f, -0.275912406538220f, -0.228061711913289f,
									-1.448224773397999f, -0.275912406538220f,  2.761546522989980f, -0.332554682460582f,
									-0.101701609715652f, -0.228061711913289f, -0.332554682460582f,  1.550892678648664f;
		(*pSumOfWeightedMeansByCov3) << -0.113802579939685f, 0.064822382530542f, 0.706223841998233f, 0.660179876256773f;

		(*pSumOfInvVar3) <<  1.333333333333333f, 1.333333333333333f, 1.333333333333333f, 1.333333333333333f;
		(*pSumOfWeightedMeansByVar3) <<  0.579066666666667f, 0.434355555555555f, 0.756533333333333f, 0.798400000000000f;

		// final
		(*pCovFinal) << 0.625496178024017f,  0.297533715879916f,  0.377715030623976f,  0.165763190696905f,
							 0.297533715879916f,  0.719535498373899f,  0.249457633054505f,  0.178810863676681f,
							 0.377715030623976f,  0.249457633054505f,  0.608229187682328f,  0.191873641511251f,
							 0.165763190696905f,  0.178810863676681f,  0.191873641511251f,  0.723097599446245f;
		(*pMeanByCovFinal) <<  0.324288648374911f, 0.307002162720697f, 0.529402563798328f, 0.605606891455783f;

		(*pVarFinal) <<   0.750000000000000f, 0.750000000000000f, 0.750000000000000f, 0.750000000000000f;
		(*pMeanByVarFinal) <<  0.434300000000000f, 0.325766666666667f, 0.567400000000000f, 0.598800000000000f;
	}

protected:
	/** @brief EPS for inversion */
	const float EPS_SOLVER;

	/** @brief Number of dimensions */
	const int D;

	/** @brief predictions */
	VectorPtr pMean1, pMean2, pMean3;
	MatrixPtr pCov1, pCov2, pCov3;
	MatrixPtr pVar1, pVar2, pVar3;

	/** @brief intermediate state */
	MatrixPtr pSumOfInvCovs1, pSumOfInvCovs2, pSumOfInvCovs3;
	VectorPtr pSumOfWeightedMeansByCov1, pSumOfWeightedMeansByCov2, pSumOfWeightedMeansByCov3;

	MatrixPtr pSumOfInvVar1, pSumOfInvVar2, pSumOfInvVar3;
	VectorPtr pSumOfWeightedMeansByVar1, pSumOfWeightedMeansByVar2, pSumOfWeightedMeansByVar3;

	/** @brief final state */
	MatrixPtr pCovFinal;
	VectorPtr pMeanByCovFinal;
	MatrixPtr pVarFinal;
	VectorPtr pMeanByVarFinal;
};

class TestBCM : public ::testing::Test,
					 public TestBCMData,
					 public BCM
{
};

/** @brief Update by mean vectors and covariance matrices */
TEST_F(TestBCM, CovTest)
{
	// pointer should be NULL initially
	EXPECT_FALSE(m_pSumOfWeightedMeans);
	EXPECT_FALSE(m_pSumOfInvCovs);

	// prediction 1
	update(pMean1, pCov1);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs1));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov1));

	// prediction 2
	update(pMean2, pCov2);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs2));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov2));

	// prediction 3
	update(pMean3, pCov3);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvCovs3));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByCov3));

	// final
	VectorPtr pMean;
	MatrixPtr pCov;
	get(pMean, pCov);
	EXPECT_TRUE(pCov->isApprox(pCovFinal->diagonal())); // get a variance!!! 
	EXPECT_TRUE(pMean->isApprox(*pMeanByCovFinal));
}

/** @brief Update by mean vectors and variance vectors */
TEST_F(TestBCM, VarTest)
{
	// pointer should be NULL initially
	EXPECT_FALSE(m_pSumOfWeightedMeans);
	EXPECT_FALSE(m_pSumOfInvCovs);

	// prediction 1
	update(pMean1, pVar1);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar1));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar1));

	// prediction 2
	update(pMean2, pVar2);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar2));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar2));

	// prediction 3
	update(pMean3, pVar3);
	EXPECT_TRUE(m_pSumOfInvCovs->isApprox(*pSumOfInvVar3));
	EXPECT_TRUE(m_pSumOfWeightedMeans->isApprox(*pSumOfWeightedMeansByVar3));

	// final
	VectorPtr pMean;
	MatrixPtr pCov;
	get(pMean, pCov);
	EXPECT_TRUE(pCov->isApprox(*pVarFinal));
	EXPECT_TRUE(pMean->isApprox(*pMeanByVarFinal));
}

#endif