#include "serialization/test_eigen_serialization.hpp"
#include "data/test_test_data.hpp"
#include "bcm/test_bcm.hpp"
#include "serialization/test_eigen_serialization.hpp"
#include "data/test_test_data.hpp"
#include "bcm/test_bcm.hpp"
#include "bcm/test_bcm_serializable.hpp"
#include "plsc/test_plsc.hpp"
#include "octree/test_data_partitioning.hpp"

//#include "octree/test_octree_gpmap.hpp"

int main(int argc, char** argv) 
{ 
	// Initialize test environment
	::testing::InitGoogleTest(&argc, argv);
		
	// Test
	int ret = RUN_ALL_TESTS(); 

	system("pause");
	return ret;
}