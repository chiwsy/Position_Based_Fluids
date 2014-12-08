Position Based Fluids
=====================

=====================
Introduction
=====================
* The first stage of this project has already been done with Beiling Lu and Lei Radium Yang. 
	* A serial version implemented the basic algorithms
	* Obj loader for 3D models
	* New constraints for collision detection and interactivity
	* Surface reconstruction

	[Check the demo for stage 1](http://youtu.be/UF9xwl5-nlQ)
	
	Some images to show our project:
	
	![Bunny Cup](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/OBJ1.png)
	![Bunny Drop](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/RealTime3.png)
	![Dragon Fountain](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/OBJ2.png)
	![Beer bottle](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/OBJ3.png)
	![Pour Beer](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/UserInteraction1.png)
	![Pour Beer](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Suyang_Beiling_Lei/final/PNG/UserInteraction2.png)
	
* In the second stage, I would like to perform the following tasks:
	* CUDA version of PBF  Caution:
		* ~~The physical system has problem so the cup is leaking!!! :(~~
		* The transparency is tested for future surface reconstruction. :p
		
	![Bunny Cup New](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/PBF2_20141128.gif)
	
	* Accelerated algorithms in density estimation
		* Here is a image to show the performance of GPU version vs CPU version. Our data can be find in [performance.xls](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/performance.xls).
		
	![GPU vs. CPU data](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/PerformanceData.png)
		
	![GPU vs. CPU](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/Performance.png)
	
	![GPU perf vs. BlockSize](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/GPUperformance.png)
	
	* ~~Floating objects~~ Only rigidbody fluid coupling implemented:
	
	![rigidBody Fall view](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/rigidBodyFall.gif)
	![rigidBody Fall view2](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/rigidBodyFall2.gif)
	
	![water tight cube](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/WaterTightCube.gif)
	
	![dragon in water](https://github.com/chiwsy/Position_Based_Fluids/blob/master/PBF_Stage2/Images/DragonInWater.gif)
	
	* ~~Real-time surface reconstruction~~

=====================
Acknowledgement
=====================
This project is based on the paper [_Position_ _Based_ _Fluids_](http://mmacklin.com/pbf_sig_preprint.pdf) by Miles Macklin and Matthias Muller from NVidia. All the algorithms are designed by the authors if no specifications. In order to implement the rigidbody physics, I referred to the [_Meshless Deformations Based on Shape Matching_](https://www.cs.drexel.edu/~david/Classes/Papers/MeshlessDeformations_SIG05.pdf). And the overall program framework is based on the paper [_Unified Particle Physics for Real-Time Applications_](http://mmacklin.com/uppfrta_preprint.pdf).
