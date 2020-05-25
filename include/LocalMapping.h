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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "L3DPPing.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;
class L3DPPing;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    void SetL3DPPer(L3DPPing* pL3DPPer);
    
    // Main function
    void Run();
    int LocalMapperToL3DPPer(vector<KeyFrame*>);//将LocalMapping线程中得到的数据转换兵迁移到L3DPPing线程中

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    std::mutex mMutexNewKFs;
    std::mutex mMutexCullingKFs;
    boost::mutex display_text_mutex_;

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    vector< struct IDandTS > vKFCullingIDandTimeStamp;//要裁剪的关键帧的ID与时间戳
    void KFchangeGuide( unsigned int MaxKFstoL3DPP );
    bool mbNewKFList;//新的关键帧一览表已经发布，可以进行下载使用了
    bool mbKFListBusy;
    std::map< long unsigned int,struct List_Factor > mmKeyFrameList;//关键帧一览表
    vector<std::list<unsigned int>> cams_worldpointIDs;//每一个关键帧里看到的世界点的全局ID
    vector<std::vector<double>> cams_worldpointDepths;//每一个关键帧里看到的所有世界点的深度
    int current_size_lists;
    int camID;//关键帧的全局地址，已经剔除的关键帧也占有位置
    std::vector<std::string> cams_imgFilenames;//关键帧的图像文件名
    std::vector<float> cams_focals;//相机焦距
    std::vector<Eigen::Matrix3d> cams_rotation;//相机旋转矩阵
    std::vector<Eigen::Vector3d> cams_translation;//相机平移向量
    std::vector<cv::Mat> cams_centers;//相机中心点
    std::vector<float> cams_distortion;//相机扭曲系数
    std::map< long unsigned int,struct List_Factor > mmCurrentKeyFrameList;//关键帧一览表，当前的，及时的就是当前的
    std::map< long unsigned int,struct List_Factor > mmReferenceKeyFrameList;//关键帧一览，表参考的，使用完便变成参考的

protected:
    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;
    L3DPPing* mpL3DPPer;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
    
    bool mbKeyFrameCullingDone;//关键帧裁剪步骤是否执行标志，1为已执行，0为复位状态
    
};

struct IDandTS
{
	long unsigned int ID;
	double TS;
};

struct List_Factor
{
	double List_TS;
	bool List_flag;
	bool List_LFflag;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
