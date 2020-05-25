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

// add libs
#include "L3DPPing.h"
#include "System.h"
#include "KeyFrame.h"
#include "time.h"

using namespace cv;
using namespace Eigen;
using namespace std;
// INFO:
// This executable reads VisualSfM results (*.nvm) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the nvm file, you need to use the _original_
// (distorted) images!
namespace ORB_SLAM2
{
	L3DPPing::L3DPPing( Map *pMap,const string &strImageFolder,const string &strInputFolder,const string &strNvmFile ):
	mpMapData(pMap),ImageFolder(strImageFolder),inputFolder(strInputFolder),nvmFile(strNvmFile)
	{
		if(outputFolder.length() == 0)
		outputFolder = inputFolder+"/Line3D++/";
	}
	
	int L3DPPing::Run()
	{
	  	std::cout << "********L3DPPING线程开始运行********" << std::endl;
		std::cout << "L3DPP inputFolder is : " << inputFolder << std::endl;//L3DPPing部分输入文件地址
		std::cout << "L3DPP NVM file name is : " << nvmFile << std::endl;//L3DPPing部分输出文件名
		// create output directory
		boost::filesystem::path dir(L3DPPing::outputFolder);
		boost::filesystem::create_directory(dir);
		// check if NVM file exists
		boost::filesystem::path nvm(nvmFile);
		if(!boost::filesystem::exists(nvm))
		{
			display_text_mutex_.lock();
			std::cerr << "NVM file " << nvmFile << " does not exist!" << std::endl;
			display_text_mutex_.unlock();
			ofstream out(nvmFile);
			out.close();
		}
		bool use_full_image_path = false;
		if(inputFolder.length() == 0)
		{
			// parse input folder from .nvm file
			use_full_image_path = true;
			inputFolder = nvm.parent_path().string();
		}
		// create Line3D++ object
		loadAndStore = false;
		L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
		maxNumSegments,true,useGPU);
		//初始化值
		mpLocalMapper->mbNewKFList = false;
		Line3D->matched_.clear();//第一次运行，则初始化清除，其他情况不再清零。除非再次初始化
		std::ofstream f_time;
		f_time.open("./TimeRecord.txt",std::ios::app);//为输入输出打开，如无文件则创建，不覆盖
		if(!f_time.fail())
		{
			f_time << "开始记录时间: " << std::endl;
		}
		f_time << fixed;
		int count_loop = 0;
		int Reconstruction_count_loop = 0;
		std::map<int,int> KeyFrame_number;
		std::map<int,double> Matching_time;
		std::map<int,double> Reconstruction_time;
		std::chrono::steady_clock::time_point t_zero = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point t_end;
		while(1)
		{	
			int size_temp = mpLocalMapper->mmCurrentKeyFrameList.size();
			if(mpLocalMapper->mbNewKFList == 0)
			{
				this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			if(mpLocalMapper->mbNewKFList)
			{	
				mpLocalMapper->mbKFListBusy = true;//开始要使用一览表了，所以状态设定为忙
				//Before deleted KeyFrames
				Line3D->views2worldpoints_.clear();//需要重新统计
				Line3D->worldpoints2views_.clear();//重新计算,清除帧看到的世界点
				Line3D->Delete_camID_.clear();
				Line3D->Add_camID_.clear();
				//删除一览表中剔除的帧----------------------------------------
				std::cout << std::endl;
				std::cout << std::endl;
				std::cout << "-----------------------------------------------!" << std::endl;
				std::cout << "[L3DPPing] " << "[0] 开始新一次运算 ===========================" << std::endl;
				//f_time << "while循环第" <<count_loop << "次:"<< endl;
				f_time << "3D重构循环第" << Reconstruction_count_loop << "次:"<< endl;
				if( mpLocalMapper->mmReferenceKeyFrameList.empty() != true )//不为空，则不是第一次运行
				{	
					std::map<long unsigned int,struct List_Factor>::iterator itr_C = mpLocalMapper->mmCurrentKeyFrameList.begin();
					std::map<long unsigned int,struct List_Factor>::iterator itr_R = mpLocalMapper->mmReferenceKeyFrameList.begin();
					for(;itr_R != mpLocalMapper->mmReferenceKeyFrameList.end();itr_R++,itr_C++)//以参考表为准，因为如果以当前为准可能有一些新来的还没处理
					{
						if(itr_C->first == itr_R->first && itr_C->second.List_LFflag == 0 && itr_R->second.List_LFflag != 0)
						{
							Line3D->deleteImage(itr_C->first);
						}
					}
				}
				display_text_mutex_.lock();
				std::cout << "[L3DPPing] " << "(deleteImage) 全局删减完成------------------!" << std::endl;
				display_text_mutex_.unlock();
				
				//加入图像----------------------------------------
				std::map<long unsigned int,struct List_Factor>::iterator itr_C = mpLocalMapper->mmCurrentKeyFrameList.begin();
				std::map<long unsigned int,struct List_Factor>::iterator itr_R = mpLocalMapper->mmReferenceKeyFrameList.begin();
				for( ; itr_C != mpLocalMapper->mmCurrentKeyFrameList.end(); itr_C++ )
				{
					camID = itr_C->first;
					if(mpLocalMapper->cams_worldpointDepths[camID].size() > 0 &&
					    camID >= mpLocalMapper->mmReferenceKeyFrameList.size() &&
					    itr_C->second.List_LFflag == 1)
					{
						int poDepths = (int)(mpLocalMapper->cams_worldpointDepths[camID].size());		
						display_text_mutex_.lock();
						//std::cout << "cams_worldpointDepths[" << camID << "] = "<< poDepths << std::endl;
						display_text_mutex_.unlock();
						// parse filename
						std::string fname = mpLocalMapper->cams_imgFilenames[camID];
						boost::filesystem::path img_path(fname);
						// load image
						cv::Mat image;
						//std::cout << ImageFolder << fname << std::endl;
						if(use_full_image_path)
							image = cv::imread(ImageFolder+"/rgb/"+fname,CV_LOAD_IMAGE_GRAYSCALE);
						else
							image = cv::imread(ImageFolder+"/rgb/"+img_path.filename().string(),CV_LOAD_IMAGE_GRAYSCALE);
						// setup intrinsics
						float px = float(image.cols)/2.0f;
						float py = float(image.rows)/2.0f;
						float f = mpLocalMapper->cams_focals[camID];
						Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
						K(0,0) = f;
						K(1,1) = f;
						K(0,2) = px;
						K(1,2) = py;
						K(2,2) = 1.0;
						// undistort (if necessary)
						float d = mpLocalMapper->cams_distortion[camID];
						cv::Mat img_undist;
						if(fabs(d) > L3D_EPS)
						{
							// undistorting
							Eigen::Vector3d radial(-d,0.0,0.0);
							Eigen::Vector2d tangential(0.0,0.0);
							Line3D->undistortImage(image,img_undist,radial,tangential,K);
						}
						else
						{
							// already undistorted
							img_undist = image;
						}
						// median point depth
						std::sort(mpLocalMapper->cams_worldpointDepths[camID].begin(),mpLocalMapper->cams_worldpointDepths[camID].end());
						size_t med_pos = floor(mpLocalMapper->cams_worldpointDepths[camID].size()/2);//向下取整
						float med_depth = mpLocalMapper->cams_worldpointDepths[camID].at(med_pos);
						//Image add to system
						Line3D->addImage(camID,img_undist,K,mpLocalMapper->cams_rotation[camID],
								mpLocalMapper->cams_translation[camID],
								med_depth,mpLocalMapper->cams_worldpointIDs[camID]); 
					}
				}
				display_text_mutex_.lock();
				std::cout << "[L3DPPing] " << "(addImage)添加图片完成------------------!" << std::endl;
				display_text_mutex_.unlock();
				
				//更新图像信息
				int good_keyframes_number = 0;
				itr_C = mpLocalMapper->mmCurrentKeyFrameList.begin();
				for( ; itr_C != mpLocalMapper->mmCurrentKeyFrameList.end(); itr_C++ )
				{
					camID = itr_C->first;
					if(itr_C->second.List_LFflag == 1 )//&& camID < mpLocalMapper->mmReferenceKeyFrameList.size() )
					{		
						// median point depth
						std::sort(mpLocalMapper->cams_worldpointDepths[camID].begin(),mpLocalMapper->cams_worldpointDepths[camID].end());
						size_t med_pos = floor(mpLocalMapper->cams_worldpointDepths[camID].size()/2);//向下取整
						float med_depth = mpLocalMapper->cams_worldpointDepths[camID].at(med_pos);
						Line3D->UpdataImage(camID,mpLocalMapper->cams_rotation[camID],
								mpLocalMapper->cams_translation[camID],med_depth,
								mpLocalMapper->cams_worldpointIDs[camID]);
						good_keyframes_number++;
					}
				}
				f_time << "good_keyframes_number: " << good_keyframes_number << endl;
				KeyFrame_number.insert(make_pair(Reconstruction_count_loop,good_keyframes_number));
				display_text_mutex_.lock();
				std::cout << "[L3DPPing] " << "(UpdataImage)全局更新完成------------------!" << std::endl << std::endl;
				display_text_mutex_.unlock();
				
				//打开开关
				mpLocalMapper->mmReferenceKeyFrameList = mpLocalMapper->mmCurrentKeyFrameList;//将当前的一览表备份
				mpLocalMapper->mbNewKFList = false;//标志位清零,这时候才允许改变一览表,在其他地方改变会出错
				mpLocalMapper->mbKFListBusy = false;//一览表不需要使用了，所以状态设定为不忙，可以更新数据了
				
				//match images
				std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
				std::chrono::duration<double> time_used = std::chrono::duration_cast<chrono::duration<double>>(t1-t_zero);
				f_time << "当前时刻: " << time_used.count() << "s" << endl;
				Line3D->matchImages(sigmaP,sigmaA,neighbors,epipolarOverlap,kNN,constRegDepth);
				std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
				time_used = std::chrono::duration_cast<chrono::duration<double>>(t2-t1);
				f_time << "matchImage use time: " << time_used.count() << "s" << endl;
				Matching_time.insert(make_pair(Reconstruction_count_loop,(double)time_used.count()));
				// compute result
				Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useCERES);
				std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
				time_used = std::chrono::duration_cast<chrono::duration<double>>(t3-t2);
				f_time << "reconstruct3Dlines use time: " << time_used.count() << "s" << endl;
				Reconstruction_time.insert(make_pair(Reconstruction_count_loop,(double)time_used.count()));
				// save end result
				std::vector< std::vector<L3DPP::FinalLine3D> > result;
				Line3D->get3Dlines(result);
				//将保存的结果转接至map中
				mpMapData->Cope3DLines(result);
				Reconstruction_count_loop++;	
				if(Reconstruction_count_loop == 110)
				{
					std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
					time_used = std::chrono::duration_cast<chrono::duration<double>>(t_end-t_zero);
					f_time << "程序运行时间: " << time_used.count() << "s" << endl;
					for(int i = 0; i<=2; i++)
					{
						f_time << endl;
						f_time << "有效循环次数: " << Reconstruction_count_loop << endl;
						for(int j = 0; j<Reconstruction_count_loop; j++)
						{
							switch(i)
							{
								case 0:
									f_time << KeyFrame_number[j] << endl;
									 break;
								case 1:
									f_time << Matching_time[j] << endl;
									 break;
								case 2:
									f_time << Reconstruction_time[j] << endl;
									 break;
								default:
									break;
							}
						}
						f_time << "*******************************" << endl;
						f_time << endl;
					}
				}
			}
			count_loop++;
		}
		f_time << endl;
	}

	void L3DPPing::SetTracker(Tracking* pTracker)
	{
		mpTracker = pTracker;
	}
	
	void L3DPPing::SetLocalMapper(LocalMapping* pLocalMapper)
	{
		mpLocalMapper = pLocalMapper;
	}
	
	void L3DPPing::SetLoopCloser(LoopClosing* pLoopCloser)
	{
		  mpLoopCloser = pLoopCloser;
	}
	
}
