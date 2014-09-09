GPMap
=====

Gaussian process occupancy mapping and reconstruction in C++

## References
* Soohwan Kim and Jonghyuk Kim, "GPmap: A Unified Framework for Robotic Mapping Based on Sparse Gaussian Processes," Proceedings of International Conference on Field and Service Robots, 2013. ([link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0CCUQFjAA&url=http%3A%2F%2Fwww.araa.asn.au%2Ffsr%2Ffsr2013%2Fpapers%2Fpap141s2.pdf&ei=pFHTU7ntNIL3oATDjYC4Dw&usg=AFQjCNFWj4u1IusbwkZX4YDm4xh9Pja6WA&sig2=ubwuFaqs39g4xWJFyFXD5A&bvm=bv.71778758,d.cGU))

## Dependencies
* [GPMap](https://github.com/kimsoohwan/GPMap)
	* [Boost 1.50.0](http://www.boost.org/) for file systems
	* [Google Test 1.7.0](https://code.google.com/p/googletest/) for unit test (optional)
	* [OpenGP](https://github.com/kimsoohwan/OpenGP)
		* [Eigen 3.2.0](http://eigen.tuxfamily.org/) for linear algebra
		* [Boost 1.50.0](http://www.boost.org/) for shared pointers
		* [Dlib 18.3](http://dlib.net/) for hyperparameter training
		* [Google Test 1.7.0](https://code.google.com/p/googletest/) for unit test (optional)
	* [PCL 1.6.0](http://pointclouds.org/)
		* [Boost 1.50.0](http://www.boost.org/) for shared pointers and threads
		* [Eigen 3.0.5](http://eigen.tuxfamily.org/)
		* [FLANN 1.7.1](http://www.cs.ubc.ca/research/flann/)
		* [VTK 5.8.0](http://www.vtk.org/)