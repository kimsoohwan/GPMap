#ifndef _GPMAP_RANDOM_HPP_
#define _GPMAP_RANDOM_HPP_

namespace GPMap {

/** @brief	Fisher-Yates shuffle (select random m out of n) */
template<class bidiiter>
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random)
{
	size_t left = std::distance(begin, end);
	while(num_random--)
	{
		bidiiter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}
	return begin;
}

}

#endif