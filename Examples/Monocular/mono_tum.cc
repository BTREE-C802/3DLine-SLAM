/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
//#include<chrono>//会发生未知错误

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

int main(int argc, char **argv)
{
	
	if(argc != 7)
	{
	    cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
	    return 1;
	}
	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	//argv[4]为L3DPPing部分输入文件地址
	//argv[5]为L3DPPing部分输出文件名
	ORB_SLAM2::System SLAM(argv[1],argv[2],argv[3],argv[4],argv[5],ORB_SLAM2::System::MONOCULAR,true);
	
	/*
	// Info: reads only the _first_ 3D model in the NVM file!
	TCLAP::CmdLine cmd("LINE3D++");

	TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images (if not specified, path in .nvm file is expected to be correct)", false, "", "string");
	cmd.add(inputArg);

	TCLAP::ValueArg<std::string> nvmArg("m", "nvm_file", "full path to the VisualSfM result file (.nvm)", true, ".", "string");
	cmd.add(nvmArg);

	TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> input_folder+'/Line3D++/')", false, "", "string");
	cmd.add(outputArg);

	TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
	cmd.add(scaleArg);

	TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
	cmd.add(neighborArg);

	TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
	cmd.add(sigma_A_Arg);

	TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
	cmd.add(sigma_P_Arg);

	TCLAP::ValueArg<float> epipolarArg("e", "min_epipolar_overlap", "minimum epipolar overlap for matching", false, L3D_DEF_EPIPOLAR_OVERLAP, "float");
	cmd.add(epipolarArg);

	TCLAP::ValueArg<int> knnArg("k", "knn_matches", "number of matches to be kept (<= 0 --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
	cmd.add(knnArg);

	TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
	cmd.add(segNumArg);

	TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
	cmd.add(visibilityArg);

	TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
	cmd.add(diffusionArg);

	TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
	cmd.add(loadArg);

	TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
	cmd.add(collinArg);

	TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
	cmd.add(cudaArg);

	TCLAP::ValueArg<bool> ceresArg("c", "use_ceres", "use CERES (for 3D line optimization)", false, L3D_DEF_USE_CERES, "bool");
	cmd.add(ceresArg);

	TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
	cmd.add(constRegDepthArg);
    	// read arguments
	cmd.parse(argc,argv);
	std::string inputFolder = inputArg.getValue().c_str();
	std::string nvmFile = nvmArg.getValue().c_str();
	*/
	
	// Retrieve paths to images
	vector<string> vstrImageFilenames;
	vector<double> vTimestamps;
	string strFile = string(argv[3])+"/rgb.txt";
	SLAM.LoadImages(strFile, vstrImageFilenames, vTimestamps);//实例化之后才能使用
	int nImages = vstrImageFilenames.size();

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Start processing sequence ..." << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	// Main loop
	cv::Mat im;
	for(int ni=0; ni<nImages; ni++)
	{
	    // Read image from file
	    im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
	    double tframe = vTimestamps[ni];

	    if(im.empty())
	    {
		cerr << endl << "Failed to load image at: "
		    << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
		return 1;
	    }

	    // Pass the image to the SLAM system
	    clock_t start = clock();
	    SLAM.TrackMonocular(im,tframe);
	    clock_t end   = clock();
	    double ttrack = (double)(end - start) / CLOCKS_PER_SEC;
	    vTimesTrack[ni]=ttrack;

	    // Wait to load the next frame
	    double T=0;
	    if(ni<nImages-1)
		T = vTimestamps[ni+1]-tframe;
	    else if(ni>0)
		T = tframe-vTimestamps[ni-1];

	    if(ttrack<T)
		usleep((T-ttrack)*1e6);
	}

	// Stop all threads
	SLAM.Shutdown();

	// Tracking time statistics
	sort(vTimesTrack.begin(),vTimesTrack.end());
	float totaltime = 0;
	for(int ni=0; ni<nImages; ni++)
	{
	    totaltime+=vTimesTrack[ni];
	}
	cout << "-------" << endl << endl;
	cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
	cout << "mean tracking time: " << totaltime/nImages << endl;

	// Save camera trajectory
	SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
	SLAM.SaveKeyFrameDataForLine3D(argv[3],"vsfm_result.nvm");
	cv::waitKey(0);
	return 0;
}