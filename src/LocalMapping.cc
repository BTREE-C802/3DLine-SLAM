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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "KeyFrameDatabase.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::SetL3DPPer(L3DPPing* pL3DPPer)
{
    mpL3DPPer=pL3DPPer;
}

void LocalMapping::Run()
{

    mbFinished = false;
    mbKeyFrameCullingDone = false;//关键帧裁剪状态复位，被后续程序使用后也会复位为0
    mbNewKFList = false;//刚进入LocalMapping程序，要将记号复位

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
		//生成关键帧时间戳与与位姿变化剧烈系数的map表，并给一些构建直线图部分程序作引导的信息赋值
		KFchangeGuide( 40 );
            }
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    unique_lock<mutex> lock(mMutexFinish);
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    unique_lock<mutex> lock(mMutexFinish);
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    unique_lock<mutex> lock(mMutexFinish);
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbStopRequested = true;
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    unique_lock<mutex> lock(mMutexCullingKFs);//裁剪进程加锁，独占所有权
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)//初始关键帧？
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();//得到这一关键帧的地图点

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
	{
	    //std::cout << "<<--KeyFrameCulling: " << to_string(pKF->mTimeStamp);
            pKF->SetBadFlag();
	    struct IDandTS IDandTS_;
	    IDandTS_.ID = pKF->mnId;
	    IDandTS_.TS = pKF->mTimeStamp;
	    vKFCullingIDandTimeStamp.push_back(IDandTS_);
	    //vKFCullingTimeStamp.push_back(pKF->mTimeStamp);
	    mbKeyFrameCullingDone = true;//关键帧裁剪已经执行了，需要告诉接下来的步骤状态
	    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
	    /*
	    display_text_mutex_.lock();
	    std::cout << "[KeyFrameCulling] KeyFrame remain: " << (int)vpKFs.size() << std::endl;
	    display_text_mutex_.unlock();
	    */
	}
    }
}

void LocalMapping::KFchangeGuide( unsigned int MaxKFstoL3DPP )
{
	unique_lock<mutex> lock(mMutexFinish);
	unique_lock<mutex> lock1(mMutexAccept);
	unique_lock<mutex> lock2(mMutexNewKFs);//确保关键帧这时候没有在被添加
	unique_lock<mutex> lock3(mMutexCullingKFs);//确保关键帧这时候没有在被裁剪
	unsigned int MaxKFstoL3DPP_ = MaxKFstoL3DPP;//需求关键帧数目，不指加入到一览表中的关键帧数目
	unsigned int CurrentKFsize;//当前关键帧数目
	if( mbKeyFrameCullingDone == true && mbNewKFList == false )
	{
	  	mbKeyFrameCullingDone = false;//复位
		//std::cout << " Need to renew KeyFrame List！"  ;
		vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();//KeyFrames的数目
		CurrentKFsize = vpKFs.size();
		//std::cout  << "    " << " KeyFrame.size: " << (int)vpKFs.size();
		vector<struct IDandTS>::iterator iter =  vKFCullingIDandTimeStamp.begin();
		for( iter; iter != vKFCullingIDandTimeStamp.end(); iter++ )
		{
			List_Factor M;
			M.List_flag = 0;
			M.List_TS = iter->TS;
			mmKeyFrameList[iter->ID] = M;
		}
		for( int nKFs = 0; nKFs < (int)vpKFs.size(); nKFs ++ )
		{
			KeyFrame* pKF = vpKFs[nKFs];
			List_Factor M1;
			M1.List_flag = 1;
			M1.List_TS = pKF->mTimeStamp;
			M1.List_LFflag = 1;//默认为0，即不为局部关键帧，还需要后续程序进一步判断
			mmKeyFrameList.insert(make_pair(pKF->mnId, M1));
		}
		//紧接着处理后续需要的世界点/位姿/图像名等信息
		//std::cout << "[LocalMapperToL3DPPer] 已裁剪数: " << (int)vKFCullingIDandTimeStamp.size() << 
		//" 一览表尺寸: " << (int)mmKeyFrameList.size() << std::endl;
		//将LocalMapper线程中发布的关键帧一览表复制到当前线程中
		current_size_lists = (int)mmKeyFrameList.size();//当前关键帧一览表的第一个元素的尺寸
		// read camera data (sequentially)
		cams_imgFilenames.resize(current_size_lists);//关键帧的图像文件名
		cams_focals.resize(current_size_lists);//相机焦距重新定义尺寸，一部分数据会保留
		cams_rotation.resize(current_size_lists);//相机旋转矩阵重新定义尺寸，一部分数据会保留
		cams_translation.resize(current_size_lists);//相机平移向量重新定义尺寸，一部分数据会保留
		cams_centers.resize(current_size_lists);//相机中心点重新定义尺寸，一部分数据会保留
		cams_distortion.resize(current_size_lists);//相机扭曲系数重新定义尺寸，一部分数据会保留
		
		//世界点相关初始化清零
		cams_worldpointIDs.clear();
		cams_worldpointDepths.clear();
		cams_worldpointIDs.resize(current_size_lists);
		cams_worldpointDepths.resize(current_size_lists);
		
		sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
		int cam_num = 0;
		for(map< long unsigned int,struct List_Factor >::iterator its = mmKeyFrameList.begin(); its != mmKeyFrameList.end(); its ++)
		{	
			KeyFrame* pKF;//声明一个KeyFrame的类，这个时候还没赋值，所以不能对其操作
			if(its->second.List_flag == 1)//关键帧一览表的第二个值为1，说明此时间戳的图像仍然是关键帧
			{
				pKF = vpKFs[cam_num];
				camID = pKF->mnId;
				cam_num++;
			}
			else continue;
			if(pKF->isBad())
			continue;
			if(pKF->mTimeStamp!= its->second.List_TS)
			{
				std::cerr << "关键帧数据与关键帧数据一览表中的数据不匹配" << std::endl;
			}
			string ScrImageName = to_string(pKF->mTimeStamp)+".png";
			// image filename，对前面线程数据进行转换并传递
			std::string filename;
			filename = ScrImageName;
			float focal_length = 517.0;
			float CameraDistortion = 0.0f;//设定一个很小的扭曲系数，但是大于1 e-12，则需要进行扭曲矫正，以后可能需要修改完善
			// rotation amd translation
			cv::Mat R3d = pKF->GetRotation();//旋转矩阵的转置，去掉转置看看
			cv::Mat C = pKF->GetCameraCenter();
			Eigen::Matrix3d R;
			R = ORB_SLAM2::Converter::toMatrix3d(R3d);
			Eigen::Vector3d C3d;
			C3d = ORB_SLAM2::Converter::toVector3d(C);
			Eigen::Vector3d t = -R*C3d;
			// focal_length,quaternion,center,distortion
			cams_imgFilenames[camID] = filename;
			cams_focals[camID] = focal_length;
			cams_distortion[camID] = CameraDistortion;
			cams_rotation[camID] = R;
			cams_centers[camID] = C;
			cams_translation[camID] = t;
		}
		//下面是世界点与其深度的信息处理，每一帧都有大量的世界点数据
		vector<MapPoint*> vpAllKFMapPoint = mpMap->GetAllMapPoints();//获取所有地图点
		for(int i = 0;i<(int)vpAllKFMapPoint.size();i++)
		{
			MapPoint* pMP = vpAllKFMapPoint[i];
			cv::Mat MPcoordinates = pMP->GetWorldPos();//pos3D为每一个世界点的三维坐标
			const map<KeyFrame*,size_t> System_Observ = pMP->GetObservations();
			map<KeyFrame*,size_t>::const_iterator mit=System_Observ.begin(), mend=System_Observ.end();
			for(; mit!=mend; mit++)
			{
				KeyFrame* pKF = mit->first;
				if(pKF->isBad() && pKF->mnId >= current_size_lists)
				{
					continue;
				}
				int camID_1 = pKF->mnId;//这里的canID和外循环中的canID不相同
				cams_worldpointIDs[camID_1].push_back(i);
				double WPDepth = cv::norm(MPcoordinates-cams_centers[camID_1]);
				cams_worldpointDepths[camID_1].push_back (WPDepth);
			}
		}
		//zip-mmKeyFrameList//压缩关键帧，确保最大关键帧数目不超过最大值
		bool Tmp_flag;
		if(mmKeyFrameList.size() != 0)
		{
			if(mmCurrentKeyFrameList.size() == 0)
			{
				mmCurrentKeyFrameList = mmKeyFrameList;
				Tmp_flag = 1;//表示第一次进入
			}
			else
			{
				Tmp_flag = 0;//不为空
			}
			map< long unsigned int,struct List_Factor >::iterator its = mmKeyFrameList.begin();
			map< long unsigned int,struct List_Factor >::iterator its1 = mmCurrentKeyFrameList.begin();
			int j = 0;//计数，1的个数，即当作关键帧的个数
			long int camID;
			long int DZ_camID = 0;
			ORB_SLAM2::List_Factor K,CK;
			//cout<< "mmCurrentKeyFrameList.size" << mmCurrentKeyFrameList.size() << endl;
			//f << " " << endl;
			//f << "mmKeyFrameList.size:" << mmKeyFrameList.size() << endl;
			for(its; its != mmKeyFrameList.end(); )
			{
				camID = its->first;
				bool Tmp_end_flag = 0;
				K = its->second;
				if( mmCurrentKeyFrameList.find(camID) == mmCurrentKeyFrameList.end() )
				{
					Tmp_end_flag = 1;
					CK.List_flag = K.List_flag;
					CK.List_LFflag = 0;
				}
				else
				{
					Tmp_end_flag = 0;
					CK = its1->second;
				}
				//cout << "Tmp_end_flag: " << Tmp_end_flag << endl;
				if(K.List_flag == 1)
				{
					if( Tmp_flag == 1 || CK.List_LFflag == 1 || Tmp_end_flag == 1)
					{
						CK.List_flag = K.List_flag;
						CK.List_LFflag = 1;
						CK.List_TS = K.List_TS;
						j++;//计数加一
					}
					else
					{
						CK.List_flag = K.List_flag;
						CK.List_LFflag = 0;
						CK.List_TS = K.List_TS;
						//cout <<  2 << " " << camID << " " << CK.List_flag << " " << CK.List_LFflag << " " << endl;
					}
				}
				else
				{
					CK.List_flag = 0;
					CK.List_LFflag = 0;
					CK.List_TS = K.List_TS;
					//cout <<  3 << " " << camID << " " << CK.List_flag << " " << CK.List_LFflag << " " << endl;
				}
				if(Tmp_end_flag == 1)
				{mmCurrentKeyFrameList.insert(make_pair(camID,CK));}//插入只适用于不存在的值
				else
				{mmCurrentKeyFrameList[camID] = CK;}//存在不能被再次插入，只能修改
				if( j >= 20 && CK.List_LFflag == 1 )
				{
					if( j == 20 && DZ_camID == 0 )
					{ DZ_camID = camID;}
					else
					{
						cv::Mat R1 = Converter::toCvMat(cams_rotation[DZ_camID]);//参考帧即第20帧的旋转矩阵
						cv::Mat R2 = Converter::toCvMat(cams_rotation[camID]);//当前要判断帧的旋转矩阵
						vector<float> q1 = Converter::toQuaternion(R1);//转换为四元数
						vector<float> q2 = Converter::toQuaternion(R2);//转换为四元数
						Eigen::Quaterniond Q1,Q2;
						Q1.x() = q1[0];Q1.y()= q1[1];Q1.z()= q1[2];Q1.w()=q1[3];
						Q2.x() = q2[0];Q2.y()= q2[1];Q2.z()= q2[2];Q2.w()=q2[3];
						//double theta1 = min( PI - fabs(acos(Q1.w())*2.0), fabs(acos(Q1.w())*2.0) );
						Q1 = Q1.conjugate();//反向旋转，四元数共轭即反向旋转，具体为实部不变，虚部相反数，虚部则为转轴
						Eigen::Quaterniond Q3 = Q2*(Q1);//Q2与未反向前Q1之间的旋转
						//double theta2 = min( PI - fabs(acos(Q2.w())*2.0), fabs(acos(Q2.w())*2.0) );
						double theta3 = min( PI*2.0 - fabs(acos(Q3.w())*2.0), fabs(acos(Q3.w())*2.0) );
						//cout << "DZ_camID:" << DZ_camID << " " << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << endl;
						//cout << "当前camID:" << camID << " " << q2[0] << " " << q2[1] << " " << q2[2] << " " << q2[3] << endl;
						//cout << "theta1: " << theta1 << endl;
						//cout << "theta2: " << theta2 << endl;
						//cout << "theta3: " << theta3 << endl;
						cv::Mat locate1 = cams_centers[DZ_camID];
						cv::Mat locate2 = cams_centers[camID];
						cv::Mat distance = locate2 - locate1;
						//cout << "参考DZ_camID位置:" << DZ_camID <<  " " << locate1.at<float>(0,0) << " " << locate1.at<float>(0,1) << " " << locate1.at<float>(0,2) << endl;
						//cout << "当前camID位置:" << camID << " " << locate2.at<float>(0,0) << " " << locate2.at<float>(0,1) << " " << locate2.at<float>(0,2) << endl;
						double kinematic = theta3 + double( cv::norm(distance) );
						//cout << "移动距离:" << distance.at<float>(0,0) << " " << distance.at<float>(0,1) << " " << distance.at<float>(0,2) << " 取范数: " << cv::norm(distance) << " 运动度: " << kinematic << endl;
						if( kinematic >= 0.4)//满足则判断为新的可用关键帧
						{
							DZ_camID = camID;//更新关键帧地址
							//cout << "当前帧运动度满足: " << camID << " j= " << j << endl;
						}
						else
						{
							if( mmCurrentKeyFrameList.find(camID) != mmCurrentKeyFrameList.end()
							     && mmCurrentKeyFrameList.find(camID+1) == mmCurrentKeyFrameList.end() )
							{
								CK.List_flag = mmCurrentKeyFrameList[DZ_camID].List_flag;
								CK.List_LFflag = 0;
								CK.List_TS = mmCurrentKeyFrameList[DZ_camID].List_TS;
								mmCurrentKeyFrameList[DZ_camID] = CK;//存在不能被再次插入，只能修改
								CK.List_flag = mmCurrentKeyFrameList[camID].List_flag;
								CK.List_LFflag = 1;
								CK.List_TS = mmCurrentKeyFrameList[camID].List_TS;	
								mmCurrentKeyFrameList[camID] = CK;//存在不能被再次插入，只能修改
								DZ_camID = camID;//更新关键帧地址
								//cout << "当前帧强制满足: " << camID << " j= " << j << endl;
							}
							else
							{
							  	CK.List_flag = mmCurrentKeyFrameList[camID].List_flag;
								CK.List_LFflag = 0;
								CK.List_TS = mmCurrentKeyFrameList[camID].List_TS;
								mmCurrentKeyFrameList[camID] = CK;//存在不能被再次插入，只能修改
								//cout << "当前帧不满足: " << camID << " j= " << j << endl;
								j--;
							}
						}
					}/////////////////判断距离
				}
				its++;
				its1++;
			}
			its1 = mmCurrentKeyFrameList.begin();
			for(its1; its1 != mmCurrentKeyFrameList.end(); its1++)
			{
				camID = its1->first;
				CK = its1->second;
				if( j >= 21 && CK.List_LFflag == 1 )
				{
					CK.List_LFflag = 0;
					mmCurrentKeyFrameList[camID] = CK;
					j--;
				}
				if( j == 20)
				break;
			}
			/*
			//将各帧之间运动度情况记录下来
			std::ofstream f;
			f.open("./MovementLists.txt",std::ios::app);//为输入输出打开，如无文件则创建，不覆盖
			if(!f.fail())
			{
				f << "mmCurrentKeyFrameList.size:" << mmCurrentKeyFrameList.size() << endl;
			}
			f << fixed;
			DZ_camID = -1;
			camID = 0;
			its1 = mmCurrentKeyFrameList.begin();
			for(its1; its1 != mmCurrentKeyFrameList.end(); its1++)
			{
				CK = its1->second;
				if(CK.List_LFflag == 1&& DZ_camID == -1)
				{ 
					DZ_camID = its1->first;
					camID = DZ_camID;
					continue;
				}
				else if(CK.List_LFflag == 0)
					continue;
					else
					{
						camID = its1->first;
						f << DZ_camID << "/" << camID << ":";
						cv::Mat R1 = Converter::toCvMat(cams_rotation[DZ_camID]);//参考帧即第20帧的旋转矩阵
						cv::Mat R2 = Converter::toCvMat(cams_rotation[camID]);//当前要判断帧的旋转矩阵
						vector<float> q1 = Converter::toQuaternion(R1);//转换为四元数
						vector<float> q2 = Converter::toQuaternion(R2);//转换为四元数
						Eigen::Quaterniond Q1,Q2;
						Q1.x() = q1[0];Q1.y()= q1[1];Q1.z()= q1[2];Q1.w()=q1[3];
						Q2.x() = q2[0];Q2.y()= q2[1];Q2.z()= q2[2];Q2.w()=q2[3];
						Q1 = Q1.conjugate();//反向旋转，四元数共轭即反向旋转，具体为实部不变，虚部相反数，虚部则为转轴
						Eigen::Quaterniond Q3 = Q2*(Q1);//Q2与未反向前Q1之间的旋转
						double theta3 = min( PI*2.0 - fabs(acos(Q3.w())*2.0), fabs(acos(Q3.w())*2.0) );
						cv::Mat locate1 = cams_centers[DZ_camID];
						cv::Mat locate2 = cams_centers[camID];
						cv::Mat distance = locate2 - locate1;
						double kinematic = theta3 + double( cv::norm(distance) );
						f << theta3 << " " << double( cv::norm(distance) ) << " " <<kinematic << endl;
					}
				DZ_camID = camID;
			}
			f << " " << endl;
			//将各帧之间运动度情况记录下来
			std::ofstream f1;
			f1.open("./AllKFMovementLists.txt",std::ios::app);//为输入输出打开，如无文件则创建，不覆盖
			if(!f1.fail())
			{
				f1 << "mmKeyFrameList.size:" << mmKeyFrameList.size() << endl;
			}
			f1 << fixed;
			DZ_camID = -1;
			camID = 0;
			its1 = mmKeyFrameList.begin();
			for(its1; its1 != mmKeyFrameList.end(); its1++)
			{
				CK = its1->second;
				if(CK.List_LFflag == 1&& DZ_camID == -1)
				{ 
					DZ_camID = its1->first;
					camID = DZ_camID;
					continue;
				}
				else if(CK.List_LFflag == 0)
					continue;
					else
					{
						camID = its1->first;
						f1 << DZ_camID << "/" << camID << ":";
						cv::Mat R1 = Converter::toCvMat(cams_rotation[DZ_camID]);//参考帧即第20帧的旋转矩阵
						cv::Mat R2 = Converter::toCvMat(cams_rotation[camID]);//当前要判断帧的旋转矩阵
						vector<float> q1 = Converter::toQuaternion(R1);//转换为四元数
						vector<float> q2 = Converter::toQuaternion(R2);//转换为四元数
						Eigen::Quaterniond Q1,Q2;
						Q1.x() = q1[0];Q1.y()= q1[1];Q1.z()= q1[2];Q1.w()=q1[3];
						Q2.x() = q2[0];Q2.y()= q2[1];Q2.z()= q2[2];Q2.w()=q2[3];
						Q1 = Q1.conjugate();//反向旋转，四元数共轭即反向旋转，具体为实部不变，虚部相反数，虚部则为转轴
						Eigen::Quaterniond Q3 = Q2*(Q1);//Q2与未反向前Q1之间的旋转
						double theta3 = min( PI*2.0 - fabs(acos(Q3.w())*2.0), fabs(acos(Q3.w())*2.0) );
						cv::Mat locate1 = cams_centers[DZ_camID];
						cv::Mat locate2 = cams_centers[camID];
						cv::Mat distance = locate2 - locate1;
						double kinematic = theta3 + double( cv::norm(distance) );
						f1 << theta3 << " " << double( cv::norm(distance) ) << " " <<kinematic << endl;
					}
				DZ_camID = camID;
			}
			f1 << " " << endl;
			*/
		}
		
		/*
		//输出mmCurrentKeyFrameList和mmKeyFrameList一览表
		std::ofstream f;
		f.open("./Lists.txt",std::ios::app);//为输入输出打开，如无文件则创建，不覆盖
		if(!f.fail())
		{
			f << "mmCurrentKeyFrameList.size:" << mmCurrentKeyFrameList.size() << endl;
		}
		f << fixed;
		map< long unsigned int,struct List_Factor >::iterator its1 = mmCurrentKeyFrameList.begin();
		for(its1; its1!= mmCurrentKeyFrameList.end(); its1++)
		{
			long int camID = its1->first;
			ORB_SLAM2::List_Factor CK = its1->second;
			f << camID << " " << CK.List_flag << " "  << CK.List_LFflag << " " << CK.List_TS << "   ";//setprecision(7)设定浮点型数据小数点后几位个数
			Eigen::Matrix3d R = cams_rotation[camID];
			f<< R(0,0) << " " << R(0,1)<< " " << R(0,2)<< " " << R(1,0)<< " " << R(1,1)
			<< " " << R(1,2)<< " " << R(2,0) << " " << R(2,1)<< " " << R(2,2)<< endl;
		}
		f << " " << endl;
		f << "mmKeyFrameList.size:" << mmKeyFrameList.size() << endl;
		map< long unsigned int,struct List_Factor >::iterator its = mmKeyFrameList.begin();
		for(its; its!= mmKeyFrameList.end(); its++)
		{
			long int camID = its->first;
			ORB_SLAM2::List_Factor K = its->second;
			f << camID << " " << K.List_flag << " "  << K.List_LFflag << "  ";//setprecision(7)设定浮点型数据小数点后几位个数
		}
		f << endl;
		f << " " << endl;
		*/
		mbNewKFList = true;//发布。一览表标志记号标为1，表示可以下载使用了
	}
	else if(mbKFListBusy == false)
	{
		//紧接着处理后续需要的世界点/位姿/图像名等信息
		//将LocalMapper线程中发布的关键帧一览表复制到当前线程中
		vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();//KeyFrames的数目
		sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
		for(int cam_num = 0; cam_num < vpKFs.size(); cam_num++)
		{
			KeyFrame* pKF;//声明一个KeyFrame的类，这个时候还没赋值，所以不能对其操作
			pKF = vpKFs[cam_num];
			camID = pKF->mnId;
			if(mmKeyFrameList.find(camID) != mmKeyFrameList.end())
			{
				cv::Mat R3d = pKF->GetRotation();//旋转矩阵的转置，去掉转置看看
				cv::Mat C = pKF->GetCameraCenter();
				Eigen::Matrix3d R;
				R = ORB_SLAM2::Converter::toMatrix3d(R3d);
				Eigen::Vector3d C3d;
				C3d = ORB_SLAM2::Converter::toVector3d(C);
				Eigen::Vector3d t = -R*C3d;
				// focal_length,quaternion,center,distortion
				cams_translation[camID] = t;
				cams_rotation[camID] = R;
				cams_centers[camID] = C;
			}
		}
	}
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
