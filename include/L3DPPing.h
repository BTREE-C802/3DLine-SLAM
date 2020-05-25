/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef L3DPPing_H_
#define L3DPPing_H_

// check libs
#include "configLIBS.h"

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include "Converter.h"

// std
#include <iostream>
#include <fstream>

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"
#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "MapPoint.h"


// INFO:
// This executable reads VisualSfM results (*.nvm) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the nvm file, you need to use the _original_
// (distorted) images!
namespace ORB_SLAM2
{
	class Viewer;
	class FrameDrawer;
	class Map;
	class LocalMapping;
	class LoopClosing;
	class Tracking;
	class System;
	class L3DPPing
	{
		public:
		int Run();
		boost::mutex display_text_mutex_;
		L3DPPing(Map *pMap,const string &strImageFolder,const string &strInputFolder,const string &strNvmFile);
		void SetTracker(Tracking* pTracker);
		void SetLocalMapper(LocalMapping* pLocalMapper);
		void SetLoopCloser(LoopClosing* pLoopCloser);
		void printf(const char* arg1);
		std::string outputFolder ;
		int maxWidth = 640;//scaleArg.getValue();
		unsigned int neighbors = 10;//std::max(neighborArg.getValue(),2);
		bool diffusion = false;//diffusionArg.getValue();
		bool loadAndStore = true;//loadArg.getValue();
		float collinearity = -1.0f;//collinArg.getValue();
		bool useGPU = false;//cudaArg.getValue();
		bool useCERES = false;//ceresArg.getValue();
		float epipolarOverlap = 0.25f;//fmin(fabs(epipolarArg.getValue()),0.99f);
		float sigmaA = 10.0f;//fabs(sigma_A_Arg.getValue());//默认10.0f
		float sigmaP = 5.0f;//sigma_P_Arg.getValue();//默认2.5f
		int kNN = 10;//knnArg.getValue();
		unsigned int maxNumSegments = 3000;//segNumArg.getValue();
		unsigned int visibility_t = 3;//visibilityArg.getValue();
		float constRegDepth = -1.0f;//constRegDepthArg.getValue();

		Map* mpMapData;
		int camID;//关键帧的全局地址，已经剔除的关键帧也占有位置
		//std::vector<std::string> cams_imgFilenames;//关键帧的图像文件名
		//std::vector<float> cams_focals;//相机焦距
		//std::vector<Eigen::Matrix3d> cams_rotation;//相机旋转矩阵
		//std::vector<Eigen::Vector3d> cams_translation;//相机平移向量
		//std::vector<Eigen::Vector3d> cams_centers;//相机中心点
		//std::vector<float> cams_distortion;//相机扭曲系数
		//vector<std::list<unsigned int>> cams_worldpointIDs;//每一个关键帧里看到的世界点的全局ID
		//vector<std::vector<double>> cams_worldpointDepths;//每一个关键帧里看到的所有世界点的深度
		
		protected:
		//在L3DPPing内部使用的KeyFrame数据库，为及时KeyFrame
		KeyFrame* mpKeyFrameData;
		LoopClosing* mpLoopCloser;
		Tracking* mpTracker;
		LocalMapping* mpLocalMapper;
		std::string ImageFolder;
		std::string inputFolder;
		std::string nvmFile;
		int current_size_lists_;
		//std::map< long unsigned int,struct List_Factor > mmCurrentKeyFrameList;//关键帧一览表，当前的，及时的就是当前的
		//std::map< long unsigned int,struct List_Factor > mmReferenceKeyFrameList;//关键帧一览，表参考的，使用完便变成参考的
	};

}
#endif //L3DPPing_H_