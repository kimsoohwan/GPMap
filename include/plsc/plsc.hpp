#ifndef _PROBABILISTIC_LEAST_SQAURE_CLASSIFICATION_HPP_
#define _PROBABILISTIC_LEAST_SQAURE_CLASSIFICATION_HPP_

// STL 
#include <cmath> // sqrt

// GPMap
//#include "util/data_types.hpp"	// Vector

namespace GPMap {
	
// constants
const float a1(0.254829592f);
const float a2(-0.284496736f);
const float a3(1.421413741f);
const float a4(-1.453152027f);
const float a5(1.061405429f);
const float p(0.3275911f);
const float sqrt2(sqrt(2.f));

inline float normcdf(float x)
{

	// Save the sign of x
	float sign = 1.f;
	if (x < 0) sign = -1.f;
	x = fabs(x)/sqrt2;

	// A&S formula 7.1.26
	const float t = 1.f/(1.f + p*x);
	const float y = 1.f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

	return 0.5f * (1.f + sign*y);
}

//class PLSC
//{
//public:
//	/** @brief Probability of the point being occupied */
//	//static inline float occupancy(const float mu, const float sigma)
//	//{
//	//	return normcdf((mean - mu) / sqrt(sigma + var));
//	//}
//	static inline float occupancy(const float mu, const float sigma)
//	{
//		return 1.f - normcdf((mean - mu) / sqrt(sigma + var));
//	}
//
//public:
//	static float mean;	// mean of the profit likelihood
//	static float var;		// variance of the profit likelihood
//	//static Vector hyp;
//};
//
//float PLSC::mean = 0.05f;
//float PLSC::var = 0.0001f;

class PLSC
{
public:
	/** @brief Probability of the point being occupied */
	//static inline float occupancy(const float mean, const float var)
	//{
	//	return normcdf((alpha*mean + beta) / sqrt(1 + alpha*alpha*var));
	//}
	//static inline float occupancy(const float mean, const float var)
	//{
	//	return normcdf((mean - beta) / powf(sqrt(var), alpha));
	//}
	//static inline float occupancy(const float mean, const float var)
	//{
	//	return normcdf(mean / (sqrt(var)*alpha + beta));
	//}
	//static inline float occupancy(const float mean, const float var)
	//{
	//	return normcdf((mean - beta) / (sqrt(var)*alpha));
	//}
	static inline float occupancy(const float mu, const float var)
	{
		//return 1.f - normcdf((alpha - mu) / sqrt(var + beta));
		return normcdf((mu - alpha) / sqrt(var + beta));
	}
public:
	static float alpha;
	static float beta;
};

float PLSC::alpha		= 1.f;
float PLSC::beta		= 0.f;

}

#endif
