#include "line3D.h"

namespace L3DPP
{
    //------------------------------------------------------------------------------
    Line3D::Line3D(const std::string& output_folder, const bool load_segments,
                   const int max_img_width,
                   const unsigned int max_line_segments,
                   const bool neighbors_by_worldpoints,
                   const bool use_GPU) :
        data_folder_(output_folder+"/L3D++_data/"), load_segments_(load_segments),
        max_image_width_(max_img_width), max_line_segments_(max_line_segments),
        neighbors_by_worldpoints_(neighbors_by_worldpoints)
    {
        //设置参数
        // set params
        num_lines_total_ = 0;//线条总数初始化为0
        med_scene_depth_ = L3D_EPS;//初始化值为预处理值=1e-12
        med_scene_depth_lines_ = 0.0f;
        translation_ = Eigen::Vector3d(0,0,0);

        // default默认
        collinearity_t_ = L3D_DEF_COLLINEARITY_T;//-1.0f
        num_neighbors_ = L3D_DEF_MATCHING_NEIGHBORS;//10
        epipolar_overlap_ = L3D_DEF_EPIPOLAR_OVERLAP;//0.25f
        kNN_ = L3D_DEF_KNN;//10
        sigma_p_ = L3D_DEF_SCORING_POS_REGULARIZER;//2.5f
        sigma_a_ = L3D_DEF_SCORING_ANG_REGULARIZER;//10.0f
        const_regularization_depth_ = -1.0f;
        two_sigA_sqr_ = 2.0f*sigma_a_*sigma_a_;
        perform_RDD_ = false;
        use_CERES_ = false;
        max_iter_CERES_ = L3D_DEF_CERES_MAX_ITER;//250
        visibility_t_ = 3;
	estimated_position3D_num_ = 0;//进入3D直线估计的数目

        if(sigma_p_ < L3D_EPS)//sigma_p_小于1e-12
        {
            // fixed sigma_p in world-coords
            fixed3Dregularizer_ = true;//修订3D正则化矩阵为1
            sigma_p_ = fabs(sigma_p_);//求绝对值
        }
        else//sigma_p_大于1e-12
        {
            // regularizer in pixels (scale unknown)
            fixed3Dregularizer_ = false;//修订3D正则化矩阵为0
            sigma_p_ = fmax(0.1f,sigma_p_);//0.1f与sigma_p_中较大的那一个
        }

#ifdef L3DPP_CUDA
        useGPU_ = use_GPU;
#else
        useGPU_ = false;
#endif //L3DPP_CUDA

        prefix_ = "[L3D++] ";//前缀[L3D++]
        prefix_err_ = prefix_+"ERROR: ";//前缀+ERROR：
        prefix_wng_ = prefix_+"WARNING: ";//前缀+WARNING:

	//创造输出目录
        // create output directory
        boost::filesystem::path dir(data_folder_);//dir路径为data_folder_，默认为/L3D++_data/
        boost::filesystem::create_directory(dir);//创建目录

	//prefix_为前缀[L3D++]
	display_text_mutex_.lock();
        std::cout << std::endl;
        std::cout << prefix_ << "//////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << prefix_ << "Line3D++ - http://www.icg.tugraz.at/ - AerialVisionGroup" << std::endl;
        std::cout << prefix_ << "(c) 2015, Manuel Hofer" << std::endl;
        std::cout << prefix_ << "published under the GNU General Public License" << std::endl;
        std::cout << prefix_ << "//////////////////////////////////////////////////////////////////////" << std::endl;
        display_text_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    Line3D::~Line3D()
    {
        // delete views
        std::map<unsigned int,L3DPP::View*>::iterator it = views_.begin();
        for(; it!=views_.end(); ++it)
        {
            delete it->second;
        }
    }

    //畸变矫正------------------------------------------------------------------------------
    void Line3D::undistortImage(const cv::Mat& inImg, cv::Mat& outImg,
                                const Eigen::Vector3d& radial_coeffs,
                                const Eigen::Vector2d& tangential_coeffs,
                                const Eigen::Matrix3d& K)
    {
        cv::Mat I = cv::Mat_<double>::eye(3,3);
        cv::Mat cvK = cv::Mat_<double>::zeros(3,3);
        cvK.at<double>(0,0) = K(0,0);
        cvK.at<double>(1,1) = K(1,1);
        cvK.at<double>(0,2) = K(0,2);
        cvK.at<double>(1,2) = K(1,2);
        cvK.at<double>(2,2) = 1.0;

        cv::Mat cvDistCoeffs(5,1,CV_64FC1,cv::Scalar(0));
        cvDistCoeffs.at<double>(0) = radial_coeffs.x();
        cvDistCoeffs.at<double>(1) = radial_coeffs.y();
        cvDistCoeffs.at<double>(2) = tangential_coeffs.x();
        cvDistCoeffs.at<double>(3) = tangential_coeffs.y();
        cvDistCoeffs.at<double>(4) = radial_coeffs.z();

        cv::Mat undistort_map_x;
        cv::Mat undistort_map_y;

        cv::initUndistortRectifyMap(cvK,cvDistCoeffs,I,cvK,cv::Size(inImg.cols, inImg.rows),
                                    undistort_map_x.type(), undistort_map_x, undistort_map_y );
        cv::remap(inImg,outImg,undistort_map_x,undistort_map_y,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
    }

    //------------------------------------------------------------------------------
    void Line3D::addImage(const unsigned int camID, cv::Mat& image,
                          const Eigen::Matrix3d& K, const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t, const float median_depth,
                          const std::list<unsigned int>& wps_or_neighbors,
                          const std::vector<cv::Vec4f>& line_segments)
    {
        // check size
        if(std::max(image.cols,image.rows) < L3D_DEF_MIN_IMG_WIDTH)//如果图像的长宽中的较大值小于800
        {
	    //显示图像太小，不合适，需要超过800行宽的图像
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "image is too small for reliable results: " << std::max(image.cols,image.rows);
            std::cout << "px (larger side should be >= " << L3D_DEF_MIN_IMG_WIDTH << "px)" << std::endl;
            display_text_mutex_.unlock();
            return;
        }

        // check ID
        view_reserve_mutex_.lock();
        if(views_reserved_.find(camID) != views_reserved_.end())
        {
	    //ID已经在使用
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "camera ID [" << camID << "] already in use!" << std::endl;
            display_text_mutex_.unlock();
            view_reserve_mutex_.unlock();
            return;
        }
        else
        {
            // reserve(储存)
            views_reserved_.insert(camID);
	    Add_camID_.insert(camID);//将新加入的帧放入记录值中
        }

        if(views_reserved_.size() == 1)
        {
            display_text_mutex_.lock();
            std::cout << std::endl << prefix_ << "[1] ADDING IMAGES ================================" << std::endl;
            display_text_mutex_.unlock();
        }
        view_reserve_mutex_.unlock();

	//查验世界点
        // check worldpoints
        if(wps_or_neighbors.size() == 0)
        {
            display_text_mutex_.lock();
            if(neighbors_by_worldpoints_)
                std::cout << prefix_err_ << "view [" << camID << "] has no worldpoints!" << std::endl;
            else
                std::cout << prefix_err_ << "view [" << camID << "] has no visual neighbors!" << std::endl;

            display_text_mutex_.unlock();

            return;
        }

        // detect segments
        L3DPP::DataArray<float4>* lines = NULL;
        if(line_segments.size() == 0)
        {
            // detect segments using LSD algorithm
	    double t = (double)cv::getTickCount();
            lines = detectLineSegments(camID,image);
	    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	    std::cout<< prefix_ <<"(detectLine])Time cost in line extraction: "<<t<<"s"<<std::endl;
        }
        else
        {
            // use given segments
            lines = new L3DPP::DataArray<float4>(line_segments.size(),1);
            for(size_t i=0; i<line_segments.size(); ++i)
            {
                cv::Vec4f coords = line_segments[i];
                float4 coordsf4;
                coordsf4.x = coords(0); coordsf4.y = coords(1);
                coordsf4.z = coords(2); coordsf4.w = coords(3);
                lines->dataCPU(i,0)[0] = coordsf4;
            }
        }

        if(lines == NULL)
        {
            display_text_mutex_.lock();
            std::cout << prefix_wng_ << "no line segments found in image [" << camID << "]!" << std::endl;
            display_text_mutex_.unlock();
            return;
        }

        // create view
        L3DPP::View* v = new L3DPP::View(camID,lines,K,R,t,image.cols,image.rows,median_depth);
        view_mutex_.lock();

        display_text_mutex_.lock();
        std::cout << prefix_ << "adding view [" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << camID;
        std::cout << "]: #line number=" << lines->width();
        std::cout << " [" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << views_.size() << "]" << std::endl;
        display_text_mutex_.unlock();

        views_[camID] = v;
        view_order_.push_back(camID);
        matches_[camID] = std::vector<std::list<L3DPP::Match> >(lines->width());
        num_matches_[camID] = 0;
        processed_[camID] = false;
        visual_neighbors_[camID] = std::set<unsigned int>();
        num_lines_total_ += lines->width();
        views_avg_depths_[camID] = (fmax(median_depth,L3D_EPS));//这个是针对views_avg_depths_为map类型的情况
	//views_avg_depths_.push_back(fmax(median_depth,L3D_EPS));//这里针对是vector类型
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::processWPlist(const unsigned int camID, const std::list<unsigned int>& wps)
    {
        std::list<unsigned int>::const_iterator it = wps.begin();
        for(; it!=wps.end(); ++it)
        {
            unsigned int wpID = *it;
            worldpoints2views_[wpID].push_back(camID);
        }
        num_worldpoints_[camID] = wps.size();
	//views2worldpoints_[camID].clear();//清除这一帧看到的世界点
        views2worldpoints_[camID] = wps;
    }

    //------------------------------------------------------------------------------
    void Line3D::setVisualNeighbors(const unsigned int camID, const std::list<unsigned int>& neighbors)
    {
        fixed_visual_neighbors_[camID] = neighbors;
    }

    //------------------------------------------------------------------------------
    L3DPP::DataArray<float4>* Line3D::detectLineSegments(const unsigned int camID, const cv::Mat& image)
    {
        // check image format
        cv::Mat imgGray;
        if(image.type() == CV_8UC3)
        {
            // convert to grayscale
            cv::cvtColor(image,imgGray,CV_RGB2GRAY);
        }
        else if(image.type() == CV_8U)
        {
            imgGray = image.clone();
        }
        else
        {
            display_text_mutex_.lock();
            std::cout << prefix_err_ << "image type not supported! must be CV_8U (gray) or CV_8UC3 (RGB)!" << std::endl;
            display_text_mutex_.unlock();
            return NULL;
        }

        // check image size
        int max_dim = std::max(imgGray.rows,imgGray.cols);
        float upscale_x = 1.0f;
        float upscale_y = 1.0f;
        unsigned int new_width = imgGray.cols;
        unsigned int new_height = imgGray.rows;

        cv::Mat imgResized;
        if(max_image_width_ > 0 && max_dim > max_image_width_)
        {
            // rescale
            float s = float(max_image_width_)/float(max_dim);
            cv::resize(imgGray,imgResized,cv::Size(),s,s);

            upscale_x = float(imgGray.cols)/float(imgResized.cols);
            upscale_y = float(imgGray.rows)/float(imgResized.rows);

            new_width = imgResized.cols;
            new_height = imgResized.rows;
        }
        else
        {
            imgResized = imgGray.clone();
        }
        // see if lines already exist
        L3DPP::DataArray<float4>* segments = NULL;
        std::stringstream str;
        if(load_segments_)
        {
            str << data_folder_ << "segments_L3D++_" << camID << "_" << new_width << "x" << new_height << "_" << L3D_DEF_MAX_NUM_SEGMENTS << ".bin";

            boost::filesystem::path file(str.str());
            if(boost::filesystem::exists(file))
            {
                segments = new L3DPP::DataArray<float4>();
                L3DPP::serializeFromFile(str.str(),*segments);
                return segments;
            }
        }
        /*
	//*****这里是使用OpenCV里面的LSD函数检测直线段*****
	#ifndef L3DPP_OPENCV3
		cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetectorPtr(cv::LSD_REFINE_ADV);
	#else
		cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);
	#endif //L3DPP_LSD_EXT
		std::vector<cv::Vec4f> detections;
		display_text_mutex_.lock();
		std::cout << "lsd is ready to run!" << std::endl;
		display_text_mutex_.unlock();
		lsd->detect(imgResized,detections);
	//***********************************************
	*/
	
        // detect line segments
        //第一步：extract lines, compute their descriptors
	LineDescriptor lineDesc;
	ScaleLines linesInImage;
	std::vector<unsigned int> matchResult;
	lineDesc.GetLineDescriptor(imgResized,linesInImage);
	if(!linesInImage.size())
	{
		display_text_mutex_.lock();
		std::cout<<"Error:[linesInImage'size] is equal to 0!"<<std::endl;
		display_text_mutex_.unlock();
		return NULL;
	}
	//第二步：将图像中检测的直线段信息剥离多余信息，只使用直线段的两个端点
        float diag = sqrtf(float(image.rows*image.rows)+float(image.cols*image.cols));//对角线长度
        float min_len = diag*L3D_DEF_MIN_LINE_LENGTH_FACTOR;//乘以因子，默认0.005f

        L3DPP::lines2D_sorted_by_length sorted;
        for(size_t i=0; i<linesInImage.size(); ++i)
        {
	    //监测到的直线
            cv::Vec4f data;
	    data(0) = float(linesInImage[i][0].startPointX);
	    data(1) = float(linesInImage[i][0].startPointY);
	    data(2) = float(linesInImage[i][0].endPointX);
	    data(3) = float(linesInImage[i][0].endPointY);
	    //缩放到原图像大小尺寸
            L3DPP::SegmentData2D seg2D;
            seg2D.p1x_ = data(0)*upscale_x;
            seg2D.p1y_ = data(1)*upscale_y;
            seg2D.p2x_ = data(2)*upscale_x;
            seg2D.p2y_ = data(3)*upscale_y;
	    //计算直线像素长度
            float dx = seg2D.p1x_-seg2D.p2x_;
            float dy = seg2D.p1y_-seg2D.p2y_;
            seg2D.length_ = sqrtf(dx*dx + dy*dy);
	    //像素长度大于最小长度则加以储存，合适直线段
            if(seg2D.length_ > min_len)//min_len = 对角线长度*因子
                sorted.push(seg2D);
        }
	//处理
        if(sorted.size() > 0)
        {
            // convert to dataArray转换为数据序列
            if(sorted.size() < max_line_segments_)
                segments = new L3DPP::DataArray<float4>(sorted.size(),1);
            else
                segments = new L3DPP::DataArray<float4>(max_line_segments_,1);

            unsigned int pos = 0;
            while(!sorted.empty() && pos < segments->width())
            {
                L3DPP::SegmentData2D segData = sorted.top();
                float4 coordsf4;
                coordsf4.x = segData.p1x_; coordsf4.y = segData.p1y_;
                coordsf4.z = segData.p2x_; coordsf4.w = segData.p2y_;
                segments->dataCPU(pos,0)[0] = coordsf4;
                sorted.pop();
                ++pos;
            }
            // save
            if(load_segments_)
            {
                L3DPP::serializeToFile(str.str(),*segments);
            }
            return segments;
        }
        return NULL;
    }
    
    	//删除从关键帧一览表中剔除掉的图像
	bool Line3D::deleteImage(const unsigned int camID)
	{
		// check ID
		view_reserve_mutex_.lock();
		if(views_reserved_.find(camID) != views_reserved_.end())//需要删除的这帧在views服务内
		{
			//ID已经在使用,则是需要删除的已经被剔除的关键帧
			num_lines_total_ -= (views_[camID]->getLineNum());//总的监测到的直线数量减去要剔除的这帧所检测到的直线数
			std::vector<unsigned int>::iterator locate= std::find(view_order_.begin(),view_order_.end(),camID);
			view_order_.erase(locate);//删除所剔除帧的位置
			views_avg_depths_.erase(camID);//view的平均深度表减去当前帧的记录值，针对的是map类型才能用
			//matches_.erase(camID);
			num_matches_.erase(camID);
			processed_.erase(camID);
			visual_neighbors_.erase(camID);
			num_worldpoints_.erase(camID);
			views2worldpoints_.erase(camID);
			views_reserved_.erase(camID);
			//views_.erase(camID);views_不删除camID那一类，后续还有用，只是会占用大量内存
			Delete_camID_.insert(camID);
			display_text_mutex_.lock();
			std::cout << prefix_ << "camera ID [" << camID << "] 已删除------------------!" << std::endl;
			display_text_mutex_.unlock();
			view_reserve_mutex_.unlock();
			return true;
		}
		else
		{
			display_text_mutex_.lock();
			std::cout << prefix_err_ << "camera ID [" << camID << "] non_existent!" << std::endl;
			display_text_mutex_.unlock();
			view_reserve_mutex_.unlock();
			return false;
		}
	}

	//更新图像信息
	void Line3D::UpdataImage(const unsigned int camID, 
		const Eigen::Matrix3d& R,const Eigen::Vector3d& t, const float median_depth,
		const std::list<unsigned int>& wps_or_neighbors)
	{
		// check ID
		view_mutex_.lock();
		if(views_.find(camID) != views_.end())
		{
			//更新 view
			views_avg_depths_[camID] = (fmax(median_depth,L3D_EPS));//map类型才能用
			//views_avg_depths_.push_back(fmax(median_depth,L3D_EPS));//这里针对是vector类型
			views_[camID]->UpdateView(camID,R,t,median_depth);
			view_reserve_mutex_.lock();
			if( views_reserved_.find(camID) != views_reserved_.end() )
			{
				if(neighbors_by_worldpoints_)
				{
					//查验世界点
					// check worldpoints
					if(wps_or_neighbors.size() == 0)
					{
						display_text_mutex_.lock();
						if(neighbors_by_worldpoints_)
							std::cout << prefix_err_ << "view [" << camID << "] has no worldpoints!" << std::endl;
						else
							std::cout << prefix_err_ << "view [" << camID << "] has no visual neighbors!" << std::endl;
						display_text_mutex_.unlock();
						return;
					}
					else
					{
						// process worldpoint list
						processWPlist(camID,wps_or_neighbors);
					}
				}
				else
				{
				    // neighbors explicitely given
				    setVisualNeighbors(camID,wps_or_neighbors);
				}
			}
			else
			{
				//ID已经在使用
				display_text_mutex_.lock();
				std::cout << prefix_err_ << "Can not find camID[" << camID << "] in views_reserved_" << std::endl;
				display_text_mutex_.unlock();
				view_reserve_mutex_.unlock();
				view_mutex_.unlock();
				return;
			}
			view_reserve_mutex_.unlock();
		}
		view_mutex_.unlock();
	}

    //------------------------------------------------------------------------------
    //sigma_position默认2.5f
    //sigma_angle默认10.0f
    //num_neighbors默认10
    //epipolar_overlap默认0.25f
    //kNN默认10
    //const_regularization_depth默认值-1.0f
    void Line3D::matchImages(const float sigma_position, const float sigma_angle,
                             const unsigned int num_neighbors, const float epipolar_overlap,
                             const int kNN, const float const_regularization_depth)
    {
        // no new views can be added in the meantime! 
        view_reserve_mutex_.lock();
        view_mutex_.lock();
	display_text_mutex_.lock();
        std::cout << std::endl << prefix_ << "[2] LINE MATCHING ================================" << std::endl;
	display_text_mutex_.unlock();
        if(views_.size() == 0)
        {
		display_text_mutex_.lock();
		std::cout << prefix_wng_ << "no images to match! forgot to add them?" << std::endl;
		display_text_mutex_.unlock();
		view_mutex_.unlock();
		view_reserve_mutex_.unlock();
		return;
        }

        // check params
        num_neighbors_ = std::max(int(num_neighbors),2);//10
        sigma_p_ = sigma_position;//2.5
        sigma_a_ = fmin(fabs(sigma_angle),90.0f);//10
        two_sigA_sqr_ = 2.0f*sigma_a_*sigma_a_;
        epipolar_overlap_ = fmin(fabs(epipolar_overlap),0.99f);//0.25
        kNN_ = kNN;//10
        const_regularization_depth_ = const_regularization_depth;//-1.0f

        if(sigma_p_ < 0.0f)
        {
		// fixed sigma_p in world-coords
		fixed3Dregularizer_ = true;
		sigma_p_ = fabs(sigma_p_);
        }
        else//满足
	{
		// regularizer in pixels (scale unknown)
		fixed3Dregularizer_ = false;
		sigma_p_ = fmax(0.1f,sigma_p_);//得到2.5
        }

        // reset
        //matched_.clear();
        estimated_position3D_.clear();
        entry_map_.clear();

        // compute spatial regularizer
        if(!fixed3Dregularizer_)//满足
	{
		display_text_mutex_.lock();
		std::cout << prefix_ << "computing spatial regularizers... [" << sigma_p_ << " px]" << std::endl;//10px
		display_text_mutex_.unlock();
	}
        else
	{
		display_text_mutex_.lock();
		std::cout << prefix_ << "computing spatial regularizers... [" << sigma_p_ << " m]" << std::endl;
		display_text_mutex_.unlock();
	}
        med_scene_depth_ = const_regularization_depth_;//-1.0f
        if(const_regularization_depth_ < 0.0f && fixed3Dregularizer_ && views_avg_depths_.size() > 0)//这里由于第二个条件不满足，所以不运行接下来的内容
        {
		// compute median scene depth
		//std::sort(views_avg_depths_.begin(),views_avg_depths_.end());//这里原先views_avg_depths_的类型是vector,可以用sort函数排序
		med_scene_depth_ = views_avg_depths_[views_avg_depths_.size()/2];//这里是map类型，不可用sort排序，但是要取中值就要另谋他法
		display_text_mutex_.lock();
		std::cout << prefix_ << "median_scene_depth = " << med_scene_depth_ << std::endl;
		display_text_mutex_.unlock();
        }
        
        // translate reconstruction (for better numerical stability)
        translate();//将视图跳整到平均视角处
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<view_order_.size(); ++i)
        {
		unsigned int camID = view_order_[i];
		//std::cout << "camID = " << camID << std::endl;
		if(!fixed3Dregularizer_)
		{
		    views_[camID]->computeSpatialRegularizer(sigma_p_);//计算正则化系数
		    //std::cout << "****k_: " << views_[camID]->k() << "  ****" << std::endl;
		}
		else
		    views_[camID]->update_k(sigma_p_,med_scene_depth_);
		if( matches_.find(camID) == matches_.end() )//如果是第一次运行或者这一帧没有匹配
		{
			//reset matches
			matches_[camID] = std::vector<std::list<L3DPP::Match> >(views_[camID]->num_lines());
			num_matches_[camID] = 0;
			processed_[camID] = false;
		}
        }
        // find visual neighbors
        display_text_mutex_.lock();
        std::cout << prefix_ << "computing visual neighbors...     [" << num_neighbors_ << " imgs.]" << std::endl;
        std::cout << prefix_ << "starting to match " << views_.size() << " images..." << std::endl;
	display_text_mutex_.unlock();

//暂时失效，运行成功后打开，因为可以开多个线程
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<view_order_.size(); ++i)
        {
		unsigned int camID = view_order_[i];
		if(fixed_visual_neighbors_.find(camID) != fixed_visual_neighbors_.end())
		{
			if(visual_neighbors_[camID].size() == 0)
			{
				// fixed neighbors
				std::list<unsigned int>::iterator n_it = fixed_visual_neighbors_[camID].begin();
				for(; n_it!=fixed_visual_neighbors_[camID].end(); ++n_it)
				{
					if(views_.find(*n_it) != views_.end())
					visual_neighbors_[camID].insert(*n_it);
				}
			}
		}
		else
		{
			// compute neighbors from WP overlap
			findVisualNeighborsFromWPs(camID);
		}
        }
        
        // match images
        display_text_mutex_.lock();
        std::cout << prefix_ << "computing matches..." << std::endl;
	display_text_mutex_.unlock();
	double t1 = (double)cv::getTickCount();
        computeMatches();
	t1 = ((double)cv::getTickCount() - t1)/cv::getTickFrequency();
	std::cout<<"<Compute Matches> run time is: "<<t1<<"s"<<std::endl;
	
	//更新位姿子
	update_Matches_and_Estimated_position3D();
        // translate back
        untranslate();

        view_mutex_.unlock();
        view_reserve_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::translate()
    {
        if(views_.size() == 0)
            return;

        // find median x,y,z coordinates
        std::vector<std::vector<double> > coords(3);
        translation_ = Eigen::Vector3d(0,0,0);

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<3; ++i)
        {
            std::vector<unsigned int>::const_iterator it = view_order_.begin();
            for(; it != view_order_.end(); ++it)
            {
		L3DPP::View* v = views_[*it];
                double val = (v->C())(i);
                if(fabs(val) > L3D_EPS)
                    coords[i].push_back(val);
            }

            if(coords[i].size() > 0)
            {
                std::sort(coords[i].begin(),coords[i].end());
                translation_(i) = coords[i][coords[i].size()/2];
            }
        }
	display_text_mutex_.lock();
        std::cout << prefix_ << "translation: ";
        std::cout << -translation_(0) << " ";
        std::cout << -translation_(1) << " ";
        std::cout << -translation_(2) << std::endl;
	display_text_mutex_.unlock();
        // apply translation to views and 3D lines
        performTranslation(-translation_);
    }

    //------------------------------------------------------------------------------
    void Line3D::untranslate()
    {
	display_text_mutex_.lock();
        std::cout << prefix_ << "translating back..." << std::endl;
	display_text_mutex_.unlock();
        // untranslate back to the original coordinates
        performTranslation(translation_);
    }

    //------------------------------------------------------------------------------
    void Line3D::performTranslation(const Eigen::Vector3d t)
    {
        // translate views
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<view_order_.size(); ++i)
        {
            views_[view_order_[i]]->translate(t);//这里使用的是平移函数
        }

        // translate available 3D lines
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D L = lines3D_[i];
            std::list<L3DPP::Segment3D>::iterator it = L.collinear3Dsegments_.begin();
            for(; it!=L.collinear3Dsegments_.end(); ++it)
            {
                (*it).translate(t);
            }

            L.underlyingCluster_.translate(t);
            lines3D_[i] = L;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::findVisualNeighborsFromWPs(const unsigned int camID)
    {
        if(visual_neighbors_.find(camID) != visual_neighbors_.end())
        {
            // reset
            visual_neighbors_[camID].clear();
            std::map<unsigned int,unsigned int> commonWPs;
            std::list<unsigned int>::const_iterator wp_it = views2worldpoints_[camID].begin();
	    for(; wp_it!=views2worldpoints_[camID].end(); ++wp_it)
            {
                // iterate over worldpoints
                unsigned int wpID = *wp_it;
                std::list<unsigned int>::const_iterator view_it = worldpoints2views_[wpID].begin();
                for(; view_it!=worldpoints2views_[wpID].end(); ++view_it)
                {
                    // all views are potential neighbors
                    unsigned int vID = *view_it;
                    if(vID != camID)
                    {
                        if(commonWPs.find(vID) == commonWPs.end())
                        {
                            commonWPs[vID] = 1;
                        }
                        else
                        {
                            ++commonWPs[vID];
                        }
                    }
                }
            }
            if(commonWPs.size() == 0)
                return;
            // find visual neighbors
            std::set<unsigned int> used_neighbors;
            std::list<L3DPP::VisualNeighbor> neighbors;
            L3DPP::View* v = views_[camID];
            std::map<unsigned int,unsigned int>::const_iterator c_it = commonWPs.begin();
            for(; c_it!=commonWPs.end(); ++c_it)
            {
                unsigned int vID = c_it->first;
                unsigned int num_common_wps = c_it->second;
                VisualNeighbor vn;
                vn.camID_ = vID;
                vn.score_ = 2.0f*float(num_common_wps)/float(num_worldpoints_[camID]+num_worldpoints_[vID]);
                vn.axisAngle_ = v->opticalAxesAngle(views_[vID]);
                vn.distance_score_ = v->distanceVisualNeighborScore(views_[vID]);

                // check baseline
                if(vn.axisAngle_ < 1.571f && num_common_wps > 4) // ~ PI/2
                {
                    neighbors.push_back(vn);
                }
            }
            // sort by score
            neighbors.sort(L3DPP::sortVisualNeighborsByScore);

            // reduce to best neighbors
            if(neighbors.size() > num_neighbors_)
            {
                // copy neighbors
                std::list<L3DPP::VisualNeighbor> neighbors_tmp = neighbors;

                // get max score
                float score_t = 0.80f*neighbors.front().score_;
                unsigned int num_bigger_t = 0;

                // count the number of highly similar views
                std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
                while(nit!=neighbors.end() && (*nit).score_ > score_t)
                {
                    ++num_bigger_t;
                    ++nit;
                }

                neighbors.resize(num_bigger_t);

                // resort based on projective_score and world_point_score
                neighbors.sort(L3DPP::sortVisualNeighborsByDistScore);

                if(neighbors.size() > num_neighbors_/2)
                    neighbors.resize(num_neighbors_/2);

                // combine
                neighbors.splice(neighbors.end(),neighbors_tmp);
            }
            // highscore neighbors -> store in visual neighbor map
            float min_baseline = v->getSpecificSpatialReg(0.5f)*v->median_depth();
            //min_baseline = 0.1f;这里设置值可能有误
            std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
            while(nit!=neighbors.end() && used_neighbors.size() < num_neighbors_)
            {
                L3DPP::VisualNeighbor vn = *nit;
                L3DPP::View* v2 = views_[vn.camID_];

                // check baseline
                if(used_neighbors.find(vn.camID_) == used_neighbors.end() && v->baseLine(v2) > min_baseline)
                {
                    std::set<unsigned int>::const_iterator u_it = used_neighbors.begin();
                    bool valid = true;
                    for(; u_it!=used_neighbors.end() && valid; ++u_it)
                    {
                        if(!(v->baseLine(views_[*u_it]) > min_baseline))
                            valid = false;
                    }
                    if(valid)
                        used_neighbors.insert(vn.camID_);
                }
                ++nit;
            }
            visual_neighbors_[camID] = used_neighbors;
	    /*
	    std::cout << "visual_neighbors_[" << camID << "]: ";
	    std::set< unsigned int >::const_iterator it = used_neighbors.begin();
	    for( ; it != used_neighbors.end(); ++it )
	    {
		  std::cout << "[" << *it << "] ";
	    }
	    std::cout << std::endl;
	    */
        }
    }
    
    //------------------------------------------------------------------------------
    void Line3D::computeMatches()
    {
        std::map<unsigned int,std::set<unsigned int> >::iterator it = visual_neighbors_.begin();
        for(; it!=visual_neighbors_.end(); ++it)
        {    
	    //不再删除记录，说明还需要计算
            if(useGPU_)
                std::cout << prefix_ << "@GPU: ";
            else
                std::cout << prefix_ << "@CPU: ";
	      
            std::cout << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << it->first << "] --> ";
	    
            // init GPU data
            if(useGPU_)
                initSrcDataGPU(it->first);
	    
	    //double t3 = (double)cv::getTickCount();
	    std::set<unsigned int>::const_iterator n_it = it->second.begin();
            for(; n_it!=it->second.end(); ++n_it)
            {
                if(matched_[it->first].find(*n_it) == matched_[it->first].end())
                {
                    // not yet matched
                    std::cout << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << *n_it << "] ";

                    // compute fundamental matrix
                    Eigen::Matrix3d F = getFundamentalMatrix(views_[it->first],views_[*n_it]);
                    // matching
                    if(useGPU_)
                        matchingGPU(it->first,*n_it,F);
                    else
		    {
			 //double t2 = (double)cv::getTickCount();
                         matchingCPU(it->first,*n_it,F);
			 //t2 = ((double)cv::getTickCount() - t2)/cv::getTickFrequency();
			 //std::cout << "<matchingCPU> run time is: " << t2 << "s" << std::endl;
		    }
                    // set matched
                    matched_[it->first].insert(*n_it);
                    matched_[*n_it].insert(it->first);
                }
            }
            std::cout << "done!" << std::endl;
	    //t3 = ((double)cv::getTickCount() - t3)/cv::getTickFrequency();
	    //std::cout << "<MatchingCPU count all> run time is: " << t3 << "s" << std::endl;
	    // check matches for orientation
            if(L3D_DEF_CHECK_MATCH_ORIENTATION)
            {
                checkMatchOrientation(it->first);//检查匹配子的方向，过滤掉边缘的直线
            }
            
            // scoring
            float valid_f;
            if(useGPU_)
                scoringGPU(it->first,valid_f);
            else
	    {
		//double t5 = (double)cv::getTickCount();
                scoringCPU(it->first,valid_f);//计算每个匹配子分数
		//t5 = ((double)cv::getTickCount() - t5)/cv::getTickFrequency();
		//std::cout << "<scoringCPU> run time is: " << t5 << "s" << std::endl;
	    }
            std::cout << prefix_ << "scoring: " << "clusterable_segments = " << int(valid_f*100) << "%";
            std::cout << std::endl;
            // cleanup GPU data
            if(useGPU_)
                removeSrcDataGPU(it->first);
	    
	    //删除此关键帧内无效匹配子
	    updateMatch(it->first);
	    
            // store inverse matches
            storeInverseMatches(it->first);
	    
            // filter invalid matches
            filterMatches(it->first);
	    std::cout << prefix_ << "当前所得estimated_position3D_.size:" << estimated_position3D_.size() << std::endl;
	    
            // set processed
            processed_[it->first] = true;
            std::cout << prefix_ << "#matches: ";
            std::cout << std::setfill(' ') << std::setw(L3D_DISP_MATCHES) << num_matches_[it->first] << std::endl;
            std::cout << prefix_ << "median_depth: " << views_[it->first]->median_depth() << std::endl;
        }
        /*
        // DEBUG: save all remaining matches
        std::vector<L3DPP::Segment3D> all_matches;
        std::map<unsigned int,std::vector<std::list<L3DPP::Match> > >::iterator dbg_it = matches_.begin();
        for(; dbg_it!=matches_.end(); ++dbg_it)
        {
            L3DPP::View* v = views_[dbg_it->first];
            for(size_t i=0; i<dbg_it->second.size(); ++i)
            {
                std::list<L3DPP::Match>::iterator dbg_it2 = dbg_it->second.at(i).begin();
                for(; dbg_it2!=dbg_it->second.at(i).end(); ++dbg_it2)
                {
                    L3DPP::Match m = *dbg_it2;
                    L3DPP::Segment3D seg3D = v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
                    all_matches.push_back(seg3D);
                }
            }
        }
        saveTempResultAsSTL(data_folder_,"all",all_matches);

        // DEBUG: save best hypotheses
        std::vector<L3DPP::Segment3D> best_matches;
        for(size_t i=0; i<estimated_position3D_.size(); ++i)
        {
            best_matches.push_back(estimated_position3D_[i].first);
        }
        saveTempResultAsSTL(data_folder_,"best",best_matches);
        */
    }

    //------------------------------------------------------------------------------
    void Line3D::checkMatchOrientation(const unsigned int src)
    {
        if(matches_.find(src) == matches_.end())//没有匹配到其他帧
            return;

	//需要注意的是匹配子的方向检查是可以复用的，如果已经检查且方向在范围内，则下次也有同样的结果
        unsigned int num_matches_before = num_matches_[src];
        unsigned int num_matches = 0;
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
	    std::list<L3DPP::Match> remaining;
	    std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
	    for(; it!=matches_[src][i].end(); ++it)
            {
		if( (*it).match_orientation_ == 0 )
		{
			L3DPP::Match m = *it;
			// unproject
			L3DPP::Segment3D seg3D = unprojectMatch(m);
			// check angle
			double ang = views_[m.src_camID_]->segmentQualityAngle(seg3D,m.src_segID_);
			if(ang > L3D_PI_1_32 && ang < L3D_PI_31_32 )
			{
				(*it).match_orientation_ = true;
				remaining.push_back(m);
			}
		}
		else
		{
		  	L3DPP::Match m = *it;
			remaining.push_back(m);
		}
            }
            matches_[src][i] = remaining;
	    
            match_mutex_.lock();
            num_matches += matches_[src][i].size();
            match_mutex_.unlock();
        }

        num_matches_[src] = num_matches;

        float perc = 0;
        if(num_matches_before > 0)
            perc = float(num_matches)/float(num_matches_before)*100.0f;

        std::cout << prefix_ << "filter matches by orientation... ";
        std::cout << num_matches_before << " --> " << num_matches;
        std::cout << " (~" << int(perc) << "%)" << std::endl;
    }
    
    void Line3D::updateMatch(const unsigned int src)
    {
	    if(matches_.find(src) == matches_.end())//没有匹配到其他帧
            return;

	//需要注意的是匹配子的方向检查是可以复用的，如果已经检查且方向在范围内，则下次也有同样的结果
        unsigned int num_matches_before = num_matches_[src];
        unsigned int num_matches = 0;
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
	    std::list<L3DPP::Match> remaining;
	    std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
	    for(; it!=matches_[src][i].end(); ++it)
            {
                L3DPP::Match m = *it;
		std::set<unsigned int>::iterator locate = Delete_camID_.find(m.tgt_camID_);//不计算对象帧(tgt)为即将要删除的匹配子
		if( locate == Delete_camID_.end() )
		{
		      remaining.push_back(m);
		}
            }
            matches_[src][i] = remaining;
	    
            match_mutex_.lock();
            num_matches += matches_[src][i].size();
            match_mutex_.unlock();
        }
        num_matches_[src] = num_matches;

        float perc = 0;
        if(num_matches_before > 0)
            perc = float(num_matches)/float(num_matches_before)*100.0f;

        std::cout << prefix_ << "Delete Invalid Matches... ";
        std::cout << num_matches_before << " --> " << num_matches;
        std::cout << " (~" << int(perc) << "%)" << std::endl;
    }

    //计算两帧之间的基础矩阵-----------------------------------------
    Eigen::Matrix3d Line3D::getFundamentalMatrix(L3DPP::View* src, L3DPP::View* tgt)
    {
        // check if it already exists
        if(fundamentals_[src->id()].find(tgt->id()) != fundamentals_[src->id()].end())
        {
            return fundamentals_[src->id()][tgt->id()];
        }
        else if(fundamentals_[tgt->id()].find(src->id()) != fundamentals_[tgt->id()].end())
        {
            Eigen::Matrix3d Ft = fundamentals_[tgt->id()][src->id()].transpose();
            return Ft;
        }

        // compute new fundamental matrix
        Eigen::Matrix3d K1 = src->K();
        Eigen::Matrix3d R1 = src->R();
        Eigen::Vector3d t1 = src->t();

        Eigen::Matrix3d K2 = tgt->K();
        Eigen::Matrix3d R2 = tgt->R();
        Eigen::Vector3d t2 = tgt->t();

        Eigen::Matrix3d R = R2 * R1.transpose();
        Eigen::Vector3d t = t2 - R * t1;

        Eigen::Matrix3d T(3,3);
        T(0,0) = 0.0;    T(0,1) = -t.z(); T(0,2) = t.y();
        T(1,0) = t.z();  T(1,1) = 0.0;    T(1,2) = -t.x();
        T(2,0) = -t.y(); T(2,1) = t.x();  T(2,2) = 0.0;

        Eigen::Matrix3d E = T * R;
        Eigen::Matrix3d F = K2.transpose().inverse() * E * K1.inverse();

        fundamentals_[src->id()][tgt->id()] = F;

        return F;
    }

    //------------------------------------------------------------------------------
    void Line3D::matchingCPU(const unsigned int src, const unsigned int tgt,
                             const Eigen::Matrix3d& F)
    {
        L3DPP::View* v_src = views_[src];
        L3DPP::View* v_tgt = views_[tgt];
        L3DPP::DataArray<float4>* lines_src = v_src->lines();
        L3DPP::DataArray<float4>* lines_tgt = v_tgt->lines();
	//std::printf("src.linewidth:%d,tgt.linewidth:%d\n",lines_src->width(),lines_tgt->width());
        unsigned int num_matches = 0;
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int r=0; r<lines_src->width(); ++r)
        {
            int new_matches = 0;
            // source line
            Eigen::Vector3d p1(lines_src->dataCPU(r,0)[0].x,
                               lines_src->dataCPU(r,0)[0].y,1.0);
            Eigen::Vector3d p2(lines_src->dataCPU(r,0)[0].z,
                               lines_src->dataCPU(r,0)[0].w,1.0);

            // epipolar lines//组成p1,p2在tgt像平面的一条线
            Eigen::Vector3d epi_p1 = F*p1;//p1射线端点投影大tgt像平面的极线
            Eigen::Vector3d epi_p2 = F*p2;//p2射线端点投影大tgt像平面的极线

            // use priority queue when kNN > 0
            L3DPP::pairwise_matches scored_matches;
            for(size_t c=0; c<lines_tgt->width(); ++c)
            {
                // target line
                Eigen::Vector3d q1(lines_tgt->dataCPU(c,0)[0].x,
                                   lines_tgt->dataCPU(c,0)[0].y,1.0);
                Eigen::Vector3d q2(lines_tgt->dataCPU(c,0)[0].z,
                                   lines_tgt->dataCPU(c,0)[0].w,1.0);
                Eigen::Vector3d l2 = q1.cross(q2);
                // intersect叠加求交
                Eigen::Vector3d p1_proj = l2.cross(epi_p1);//直线与直线的叉积为交点
                Eigen::Vector3d p2_proj = l2.cross(epi_p2);

                if(fabs(p1_proj.z()) > L3D_EPS && fabs(p2_proj.z()) > L3D_EPS)
                {
                    // normalize
                    p1_proj /= p1_proj.z();
                    p2_proj /= p2_proj.z();

		    if( p1_proj.x() < 0 || p1_proj.x() > max_image_width_ 
		      || p1_proj.y() < 0 || p1_proj.y() > max_image_width_
		      || p2_proj.x() < 0 || p2_proj.x() > max_image_width_ 
		      || p2_proj.y() < 0 || p2_proj.y() > max_image_width_)
		    {
			continue;//投影线已经超出图像界限，默认不合格匹配
		    }

                    // check overlap
                    std::vector<Eigen::Vector3d> collinear_points(4);
                    collinear_points[0] = p1_proj;
                    collinear_points[1] = p2_proj;
                    collinear_points[2] = q1;
                    collinear_points[3] = q2;
                    float score = mutualOverlap(collinear_points);

                    if(score > epipolar_overlap_)
                    {
                        // triangulate
                        Eigen::Vector2d depths_src = triangulationDepths(src,p1,p2,
                                                                         tgt,q1,q2);
                        Eigen::Vector2d depths_tgt = triangulationDepths(tgt,q1,q2,
                                                                         src,p1,p2);
                        if(depths_src.x() > L3D_EPS && depths_src.y() > L3D_EPS &&
                                depths_tgt.x() > L3D_EPS && depths_tgt.y() > L3D_EPS)
                        {
                            // potential match
                            L3DPP::Match M;
                            M.src_camID_ = src;
                            M.src_segID_ = r;
                            M.tgt_camID_ = tgt;
                            M.tgt_segID_ = c;
                            M.overlap_score_ = score;
                            M.score3D_ = 0.0f;
                            M.depth_p1_ = depths_src.x();
                            M.depth_p2_ = depths_src.y();
                            M.depth_q1_ = depths_tgt.x();
                            M.depth_q2_ = depths_tgt.y();
			    M.match_orientation_ = false;//初始化为没有检查方向
			    M.match_valid_ = false;//初始化值为无效值
                            if(kNN_ > 0)
                            {
                                // kNN matching
                                scored_matches.push(M);
                            }
                            else
                            {
                                // all matches are used
                                matches_[src][r].push_back(M);
                                ++new_matches;
                            }
                        }
                    }
                }
            }
            // push kNN matches into list
            if(kNN_ > 0)
            {
                while(new_matches < kNN_ && !scored_matches.empty())
                {
                    matches_[src][r].push_back(scored_matches.top());
                    scored_matches.pop();
                    ++new_matches;
                }
            }
            match_mutex_.lock();
            num_matches += new_matches;
            match_mutex_.unlock();
        }
        num_matches_[src] += num_matches;
    }

    //------------------------------------------------------------------------------
    void Line3D::initSrcDataGPU(const unsigned int src)
    {
#ifdef L3DPP_CUDA
        // upload
        L3DPP::View* v1 = views_[src];
        v1->lines()->upload();
        v1->RtKinvGPU()->upload();
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::removeSrcDataGPU(const unsigned int src)
    {
#ifdef L3DPP_CUDA
        // cleanup
        L3DPP::View* v1 = views_[src];
        v1->lines()->removeFromGPU();
        v1->RtKinvGPU()->removeFromGPU();
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::matchingGPU(const unsigned int src, const unsigned int tgt,
                             const Eigen::Matrix3d& F)
    {
#ifdef L3DPP_CUDA
        // INFO: src data must be on GPU! initSrcDataGPU(src)
        L3DPP::View* v1 = views_[src];

        // upload segments to GPU
        L3DPP::View* v2 = views_[tgt];
        v2->lines()->upload();

        // move F to GPU
        L3DPP::DataArray<float>* F_GPU = NULL;
        eigen2dataArray(F_GPU,F);
        F_GPU->upload();

        // move RtKinv to GPU
        v2->RtKinvGPU()->upload();

        // match segments on GPU
        unsigned int num_matches = L3DPP::match_lines_GPU(v1->lines(),v2->lines(),F_GPU,
                                                          v1->RtKinvGPU(),v2->RtKinvGPU(),
                                                          v1->C_GPU(),v2->C_GPU(),
                                                          &(matches_[src]),src,tgt,
                                                          epipolar_overlap_,kNN_);

        num_matches_[src] += num_matches;

        // cleanup
        v2->lines()->removeFromGPU();
        v2->RtKinvGPU()->removeFromGPU();
        delete F_GPU;

#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    bool Line3D::pointOnSegment(const Eigen::Vector3d& x, const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& p2)
    {
        Eigen::Vector2d v1(p1.x()-x.x(),p1.y()-x.y());
        Eigen::Vector2d v2(p2.x()-x.x(),p2.y()-x.y());
        return (v1.dot(v2) < L3D_EPS);
    }

    //------------------------------------------------------------------------------
    float Line3D::mutualOverlap(const std::vector<Eigen::Vector3d>& collinear_points)
    {
        float overlap = 0.0f;

        if(collinear_points.size() != 4)
            return 0.0f;

        Eigen::Vector3d p1 = collinear_points[0];
        Eigen::Vector3d p2 = collinear_points[1];
        Eigen::Vector3d q1 = collinear_points[2];
        Eigen::Vector3d q2 = collinear_points[3];

        if(pointOnSegment(p1,q1,q2) || pointOnSegment(p2,q1,q2) ||
                pointOnSegment(q1,p1,p2) || pointOnSegment(q2,p1,p2))
        {
            // find outer distance and inner points
            float max_dist = 0.0f;
            size_t outer1 = 0;
            size_t inner1 = 1;
            size_t inner2 = 2;
            size_t outer2 = 3;

            for(size_t i=0; i<3; ++i)
            {
                for(size_t j=i+1; j<4; ++j)
                {
                    float dist = (collinear_points[i]-collinear_points[j]).norm();
                    if(dist > max_dist)
                    {
                        max_dist = dist;
                        outer1 = i;
                        outer2 = j;
                    }
                }
            }

            if(max_dist < 1.0f)
                return 0.0f;

            if(outer1 == 0)
            {
                if(outer2 == 1)
                {
                    inner1 = 2;
                    inner2 = 3;
                }
                else if(outer2 == 2)
                {
                    inner1 = 1;
                    inner2 = 3;
                }
                else
                {
                    inner1 = 1;
                    inner2 = 2;
                }
            }
            else if(outer1 == 1)
            {
                inner1 = 0;
                if(outer2 == 2)
                {
                    inner2 = 3;
                }
                else
                {
                    inner2 = 2;
                }
            }
            else
            {
                inner1 = 0;
                inner2 = 1;
            }

            overlap = (collinear_points[inner1]-collinear_points[inner2]).norm()/max_dist;

        }
        return overlap;//返回重叠率
    }

    //三角化得深度------------------------------------------------------------------------------
    Eigen::Vector2d Line3D::triangulationDepths(const unsigned int src_camID, const Eigen::Vector3d& p1,
                                                const Eigen::Vector3d& p2, const unsigned int tgt_camID,
                                                const Eigen::Vector3d& line_q1, const Eigen::Vector3d& line_q2)
    {
        L3DPP::View* v_src = views_[src_camID];
        L3DPP::View* v_tgt = views_[tgt_camID];

        // rays through points
        Eigen::Vector3d C1 = v_src->C();
        Eigen::Vector3d ray_p1 = v_src->getNormalizedRay(p1);
        Eigen::Vector3d ray_p2 = v_src->getNormalizedRay(p2);

        // plane
        Eigen::Vector3d C2 = v_tgt->C();
        Eigen::Vector3d ray_q1 = v_tgt->getNormalizedRay(line_q1);
        Eigen::Vector3d ray_q2 = v_tgt->getNormalizedRay(line_q2);
        Eigen::Vector3d n = ray_q1.cross(ray_q2);
        n.normalize();

        if(fabs(ray_p1.dot(n)) < L3D_EPS || fabs(ray_p2.dot(n)) < L3D_EPS)
            return Eigen::Vector2d(-1,-1);

        double d1 = (C2.dot(n) - n.dot(C1)) / (n.dot(ray_p1));
        double d2 = (C2.dot(n) - n.dot(C1)) / (n.dot(ray_p2));
        return Eigen::Vector2d(d1,d2);
    }

    //------------------------------------------------------------------------------
    void Line3D::sortMatches(const unsigned int src)
    {
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
            matches_[src][i].sort(L3DPP::sortMatchesByIDs);
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::scoringCPU(const unsigned int src, float& valid_f)
    {
        // init
        valid_f = 0.0f;
        L3DPP::View* v = views_[src];
        float k = v->k();
        unsigned int num_valid = 0;
        // iterative scoring
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
            bool valid_match_exists = false;
            std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)
            {
	      	L3DPP::Match M = *it;
		std::set<unsigned int>::iterator locate = Delete_camID_.find(M.tgt_camID_);
		if( locate == Delete_camID_.end() )//当前匹配子的配对帧不为已经删除的帧
		{
			// unproject once
			L3DPP::Segment3D M3D = v->unprojectSegment(M.src_segID_,M.depth_p1_,M.depth_p2_);
			// compute spatial regularizers
			float reg1,reg2;
			float sig1 = M.depth_p1_*k;
			float sig2 = M.depth_p2_*k;
			reg1 = 2.0f*sig1*sig1;
			reg2 = 2.0f*sig2*sig2;
			// compute spatial regularizers (tgt)
			float sig1_tgt = views_[M.tgt_camID_]->regularizerFrom3Dpoint(M3D.P1());
			float sig2_tgt = views_[M.tgt_camID_]->regularizerFrom3Dpoint(M3D.P2());
			reg1 = 0.5f*(reg1 + 2.0f*sig1_tgt*sig1_tgt);
			reg2 = 0.5f*(reg2 + 2.0f*sig2_tgt*sig2_tgt);		
			if( (*it).score3D_ != 0 )//证明已经算过了的，有分值
			{
				float score3D;
				std::map<unsigned int,float> adds_score_per_cam;//新增的帧对于当前匹配子的分值储存记录容器
				std::map<unsigned int,float> deletes_score_per_cam;//删除的帧对于当前匹配子的分值储存记录容器
				std::list<L3DPP::Match>::const_iterator it2 = matches_[src][i].begin();
				for(; it2!=matches_[src][i].end(); ++it2)
				{
				    L3DPP::Match M2 = *it2;				
				    if(M.tgt_camID_ != M2.tgt_camID_)
				    {		
					if( Add_camID_.size() != 0 )
					{
						std::set<unsigned int>::iterator locate1 = Add_camID_.find(M2.tgt_camID_);//将新增的帧的分值直接加入M中，即为M的分值
						if( locate1 != Add_camID_.end() )
						{
							// compute similarity
							float sim = similarityForScoring(M,M2,M3D,reg1,reg2);
							if(adds_score_per_cam.find(M2.tgt_camID_) != adds_score_per_cam.end())
							{
							    if(sim > adds_score_per_cam[M2.tgt_camID_])
							    {
								score3D -= adds_score_per_cam[M2.tgt_camID_];
								score3D += sim;
								adds_score_per_cam[M2.tgt_camID_] = sim;
							    }
							}
							else
							{
							    score3D += sim;
							    adds_score_per_cam[M2.tgt_camID_] = sim;
							}
						}
					}
					if( Delete_camID_.size() != 0 )
					{
					      std::set<unsigned int>::iterator locate2 = Delete_camID_.find(M2.tgt_camID_);//将要删除的匹配子的以前对M的分值计算，再在原分值减去这个分值
					      if( locate2 != Delete_camID_.end() )
					      {
						  // compute similarity
						  float sim = similarityForScoring(M,M2,M3D,reg1,reg2);
						  if(deletes_score_per_cam.find(M2.tgt_camID_) != deletes_score_per_cam.end())
						  {
						      if(sim > deletes_score_per_cam[M2.tgt_camID_])
						      {
							  score3D -= deletes_score_per_cam[M2.tgt_camID_];
							  score3D += sim;
							  deletes_score_per_cam[M2.tgt_camID_] = sim;
						      }
						  }
						  else
						  {
						      score3D += sim;
						      deletes_score_per_cam[M2.tgt_camID_] = sim;
						  }
					      }
					}
				    }
				}
				std::map<unsigned int,float>::const_iterator add_score = adds_score_per_cam.begin();
				for( ; add_score != adds_score_per_cam.end(); add_score++ )
				{
					(*it).score3D_ += add_score->second;
				}
				std::map<unsigned int,float>::const_iterator delete_score = deletes_score_per_cam.begin();
				for( ; delete_score != deletes_score_per_cam.end(); delete_score++ )
				{
					(*it).score3D_ -= delete_score->second;
				}
				if((*it).score3D_ > L3D_DEF_MIN_BEST_SCORE_3D)
				{
				    valid_match_exists = true;
				}
			}
			else//新的匹配子
			{
				std::map<unsigned int,float> score_per_cam;
				std::list<L3DPP::Match>::const_iterator it2 = matches_[src][i].begin();
				for(; it2!=matches_[src][i].end(); ++it2)
				{
				    L3DPP::Match M2 = *it2;
				    if(M.tgt_camID_ != M2.tgt_camID_)
				    {
					std::set<unsigned int>::iterator locate2 = Delete_camID_.find(M2.tgt_camID_);//不计算即将要删除的匹配子
					if( locate2 == Delete_camID_.end() )
					{
						// compute similarity
						float sim = similarityForScoring(M,M2,M3D,reg1,reg2);
						if(score_per_cam.find(M2.tgt_camID_) != score_per_cam.end())
						{
						    if(sim > score_per_cam[M2.tgt_camID_])
						    {
							(*it).score3D_ -= score_per_cam[M2.tgt_camID_];
							(*it).score3D_ += sim;
							score_per_cam[M2.tgt_camID_] = sim;
						    }
						}
						else
						{
						    (*it).score3D_ += sim;
						    score_per_cam[M2.tgt_camID_] = sim;
						}
					}
				    }
				}
				if((*it).score3D_ > L3D_DEF_MIN_BEST_SCORE_3D)
				{
				    valid_match_exists = true;
				}
			}
		}
            }

            if(valid_match_exists)
            {
                scoring_mutex_.lock();
                ++num_valid;
                scoring_mutex_.unlock();
            }
        }

        // check number of segments with valid matches
        valid_f = float(num_valid)/float(v->num_lines());
    }

    //------------------------------------------------------------------------------
    void Line3D::scoringGPU(const unsigned int src, float& valid_f)
    {
#ifdef L3DPP_CUDA
        // INFO: src data must be on GPU! initSrcDataGPU(src) -> remove afterwards!

        // init
        valid_f = 0.0f;
        L3DPP::View* v = views_[src];
        float k = v->k();

        if(num_matches_[src] == 0)
            return;

        // sort matches by ids first
        sortMatches(src);

        // find start and end indices
        L3DPP::DataArray<int2>* ranges = new L3DPP::DataArray<int2>(v->num_lines(),1);
        unsigned int offset = 0;
        for(size_t i=0; i<v->num_lines(); ++i)
        {
            if(matches_[src][i].size() > 0)
            {
                ranges->dataCPU(i,0)[0] = make_int2(offset,offset+matches_[src][i].size()-1);
                offset += matches_[src][i].size();
            }
            else
            {
                // no matches for this segment
                ranges->dataCPU(i,0)[0] = make_int2(-1,-1);
            }
        }

        // store matches in array
        L3DPP::DataArray<float4>* matches = new L3DPP::DataArray<float4>(num_matches_[src],1);
        L3DPP::DataArray<float2>* regularizers_tgt = new L3DPP::DataArray<float2>(num_matches_[src],1);
        L3DPP::DataArray<float>* scores = new L3DPP::DataArray<float>(num_matches_[src],1,true);

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
            int offset = ranges->dataCPU(i,0)[0].x;
            if(offset >= 0)
            {
                int id = 0;
                std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
                for(; it!=matches_[src][i].end(); ++it,++id)
                {
                    L3DPP::Match m = *it;
                    matches->dataCPU(offset+id,0)[0] = make_float4(i,m.tgt_camID_,
                                                                   m.depth_p1_,m.depth_p2_);
                    L3DPP::Segment3D s3D = v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
                    regularizers_tgt->dataCPU(offset+id,0)[0] = make_float2(views_[m.tgt_camID_]->regularizerFrom3Dpoint(s3D.P1()),
                                                                            views_[m.tgt_camID_]->regularizerFrom3Dpoint(s3D.P2()));
                }
            }
        }

        // upload
        ranges->upload();
        matches->upload();
        regularizers_tgt->upload();

        unsigned int num_valid = 0;

        // score on GPU
        L3DPP::score_matches_GPU(v->lines(),matches,ranges,scores,regularizers_tgt,
                                 v->RtKinvGPU(),v->C_GPU(),
                                 two_sigA_sqr_,k,L3D_DEF_MIN_SIMILARITY_3D);
        scores->download();

        // write back
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
            bool valid_match_exists = false;
            int offset = ranges->dataCPU(i,0)[0].x;
            if(offset >= 0)
            {
                int id = 0;
                std::list<L3DPP::Match>::iterator it = matches_[src][i].begin();
                for(; it!=matches_[src][i].end(); ++it,++id)
                {
                    // get score
                    float score = scores->dataCPU(offset+id,0)[0];

                    // update
                    (*it).score3D_ = score;

                    if(score > L3D_DEF_MIN_BEST_SCORE_3D)
                    {
                        valid_match_exists = true;
                    }
                }
            }

            if(valid_match_exists)
            {
                scoring_mutex_.lock();
                ++num_valid;
                scoring_mutex_.unlock();
            }
        }

        // check number of segments with valid matches
        valid_f = float(num_valid)/float(v->num_lines());

        // cleanup
        delete ranges;
        delete matches;
        delete scores;
        delete regularizers_tgt;
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    float Line3D::similarityForScoring(const L3DPP::Match& m1, const L3DPP::Match& m2,
                                       const L3DPP::Segment3D& seg3D1,
                                       const float reg1, const float reg2)
    {
        L3DPP::Segment3D seg3D2 = unprojectMatch(m2,true);
	
        if(seg3D1.length() < L3D_EPS || seg3D2.length() < L3D_EPS)
            return 0.0f;

        // positional similarity
        float sim_p = 0.0f;
        if(m1.src_camID_ == m2.src_camID_ && m1.src_segID_ == m2.src_segID_)
        {
            // local similarity
            float d1 = m1.depth_p1_-m2.depth_p1_;
            float d2 = m1.depth_p2_-m2.depth_p2_;

            sim_p = fmin(expf(-d1*d1/reg1),expf(-d2*d2/reg2));
        }
        else
	    return 0.0f;
        
        // angular similarity
        float angle = angleBetweenSeg3D(seg3D1,seg3D2,true);
        float sim_a = expf(-angle*angle/two_sigA_sqr_);

        float sim = fmin(sim_a,sim_p);
        if(sim > L3D_DEF_MIN_SIMILARITY_3D)
            return sim;
        else
            return 0.0f;
    }

    //------------------------------------------------------------------------------
    float Line3D::similarity(const L3DPP::Segment2D& seg1, const L3DPP::Segment2D& seg2,
                             const bool truncate)
    {
        // check for 3D estimates
        if(entry_map_.find(seg1) == entry_map_.end())
        {
            return 0.0f;
        }

        size_t ent1 = entry_map_[seg1];
        std::pair<L3DPP::Segment3D,L3DPP::Match> data1 = estimated_position3D_[ent1];
        L3DPP::Segment3D s1 = data1.first;
        L3DPP::Match m1 = data1.second;

        return similarity(s1,m1,seg2,truncate);
    }

    //------------------------------------------------------------------------------
    float Line3D::similarity(const L3DPP::Segment3D& s1, const L3DPP::Match& m1,
                             const L3DPP::Segment2D& seg2, const bool truncate)
    {
        // check for 3D estimates
        if(entry_map_.find(seg2) == entry_map_.end())
        {
            return 0.0f;
        }

        size_t ent2 = entry_map_[seg2];
        std::pair<L3DPP::Segment3D,L3DPP::Match> data2 = estimated_position3D_[ent2];
        L3DPP::Segment3D s2 = data2.first;
        L3DPP::Match m2 = data2.second;

        if(s1.length() < L3D_EPS || s2.length() < L3D_EPS)
            return 0.0f;

        L3DPP::View* v1 = views_[m1.src_camID_];
        L3DPP::View* v2 = views_[m2.src_camID_];

        // angular similarity
        float angle = angleBetweenSeg3D(s1,s2,true);
        float sim_a = expf(-angle*angle/two_sigA_sqr_);

        // cutoff depths
        float cutoff1 = v1->median_depth();
        float cutoff2 = v2->median_depth();

        if(med_scene_depth_lines_ > L3D_EPS)
        {
            cutoff1 = fmin(cutoff1,med_scene_depth_lines_);
            cutoff2 = fmin(cutoff2,med_scene_depth_lines_);
        }

        // positional similarity
        float d11 = s2.distance_Point2Line(s1.P1());
        float d12 = s2.distance_Point2Line(s1.P2());
        float d21 = s1.distance_Point2Line(s2.P1());
        float d22 = s1.distance_Point2Line(s2.P2());

        float reg11,reg12,reg21,reg22;
        float sig11;
        if(m1.depth_p1_ > cutoff1)
            sig11 = cutoff1*v1->k();
        else
            sig11 = m1.depth_p1_*v1->k();

        float sig12;
        if(m1.depth_p2_ > cutoff1)
            sig12 = cutoff1*v1->k();
        else
            sig12 = m1.depth_p2_*v1->k();

        reg11 = 2.0f*sig11*sig11;
        reg12 = 2.0f*sig12*sig12;

        float sig21;
        if(m2.depth_p1_ > cutoff2)
            sig21 = cutoff2*v2->k();
        else
            sig21 = m2.depth_p1_*v2->k();

        float sig22;
        if(m2.depth_p2_ > cutoff2)
            sig22 = cutoff2*v2->k();
        else
            sig22 = m2.depth_p2_*v2->k();

        reg21 = 2.0f*sig21*sig21;
        reg22 = 2.0f*sig22*sig22;

        float sim_p1 = fmin(expf(-d11*d11/reg11),expf(-d12*d12/reg12));
        float sim_p2 = fmin(expf(-d21*d21/reg21),expf(-d22*d22/reg22));

        float sim_p = fmin(sim_p1,sim_p2);

        float sim = fmin(sim_a,sim_p);

        if(truncate)
        {
            if(sim > L3D_DEF_MIN_SIMILARITY_3D)
                return sim;
            else
                return 0.0f;
        }
        return sim;
    }

    //------------------------------------------------------------------------------
    L3DPP::Segment3D Line3D::unprojectMatch(const L3DPP::Match& m, const bool src)
    {
        if(src)
        {
            L3DPP::View* v = views_[m.src_camID_];
            return v->unprojectSegment(m.src_segID_,m.depth_p1_,m.depth_p2_);
        }
        else
        {
            L3DPP::View* v = views_[m.tgt_camID_];
            return v->unprojectSegment(m.tgt_segID_,m.depth_q1_,m.depth_q2_);
        }
    }

    //------------------------------------------------------------------------------
    float Line3D::angleBetweenSeg3D(const L3DPP::Segment3D& s1, const L3DPP::Segment3D& s2,
                                    const bool undirected)
    {
        float dot_p = s1.dir().dot(s2.dir());
        float angle = acos(fmax(fmin(dot_p,1.0f),-1.0f))/M_PI*180.0f;

        if(undirected && angle > 90.0f)
        {
            angle = 180.0f-angle;
        }

        return angle;
    }

    //------------------------------------------------------------------------------
    //更新估计位姿中的匹配子的深度等信息，同时原匹配子也要一同更新
    void Line3D::update_Matches_and_Estimated_position3D()
    {
	entry_map_.clear();
	for(int i=0; i<estimated_position3D_.size(); ++i)
        {
            L3DPP::Segment3D seg3D = estimated_position3D_[i].first;
            L3DPP::Match m = estimated_position3D_[i].second;
	    if( views_.find(m.src_camID_) != views_.end()
	        &&views_.find(m.tgt_camID_) != views_.end()
	        && m.score3D_ > L3D_DEF_MIN_BEST_SCORE_3D)
            {
		L3DPP::View* v_src = views_[m.src_camID_];
		L3DPP::View* v_tgt = views_[m.tgt_camID_];
		
		L3DPP::DataArray<float4>* lines_src = v_src->lines();
		L3DPP::DataArray<float4>* lines_tgt = v_tgt->lines();
		
		unsigned int r = m.src_segID_;
		unsigned int c = m.tgt_segID_;
		
		Eigen::Vector3d p1(lines_src->dataCPU(r,0)[0].x, lines_src->dataCPU(r,0)[0].y,1.0);
                Eigen::Vector3d p2(lines_src->dataCPU(r,0)[0].z, lines_src->dataCPU(r,0)[0].w,1.0);
		
		Eigen::Vector3d q1(lines_tgt->dataCPU(c,0)[0].x,lines_tgt->dataCPU(c,0)[0].y,1.0);
                Eigen::Vector3d q2(lines_tgt->dataCPU(c,0)[0].z,lines_tgt->dataCPU(c,0)[0].w,1.0);
		
		Eigen::Vector2d depths_src = triangulationDepths(m.src_camID_,p1,p2,
								  m.tgt_camID_,q1,q2);
		Eigen::Vector2d depths_tgt = triangulationDepths(m.tgt_camID_,q1,q2,
								  m.src_camID_,p1,p2);
				
		//将矫正后的结果赋值回原来的匹配子中
		if(depths_src.x() > L3D_EPS && depths_src.y() > L3D_EPS &&
			depths_tgt.x() > L3D_EPS && depths_tgt.y() > L3D_EPS)
		{
			m.depth_p1_ = depths_src.x();
			m.depth_p2_ = depths_src.y();
			m.depth_q1_ = depths_tgt.x();
			m.depth_q2_ = depths_tgt.y();
			L3DPP::Segment2D seg(m.src_camID_,r);
			L3DPP::Segment3D seg3D = unprojectMatch(m,true);
			best_match_mutex_.lock();
			entry_map_[seg] = update_estimated_position3D_.size();
			update_estimated_position3D_.push_back(std::pair<L3DPP::Segment3D,L3DPP::Match>(seg3D,m));
			best_match_mutex_.unlock();
		}
            }
	}
	estimated_position3D_.clear();
	estimated_position3D_.insert( estimated_position3D_.begin(),update_estimated_position3D_.begin(),update_estimated_position3D_.end() );
        update_estimated_position3D_.clear();//使用完即丢弃释放内存
    }
    
    //------------------------------------------------------------------------------
    void Line3D::filterMatches(const unsigned int src)
    {
        // filter and find median depth
        std::vector<float> depths;
        // compute maximum score for this view
        float max_score = 0.0f;
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)
            {
                max_score = fmax(max_score,(*it).score3D_);
            }
        }
        // scores must be at least a certain percentage of the best
        float score_lim = L3D_DEF_MIN_BEST_SCORE_PERC*max_score;
        std::cout << prefix_ << "scoring: max_score = " << max_score << std::endl;
        unsigned int num_valid = 0;
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<matches_[src].size(); ++i)
        {
            L3DPP::Match best_match;
            best_match.score3D_ = 0.0f;
            std::list<L3DPP::Match> matches = matches_[src][i];
            matches_[src][i].clear();
            std::list<L3DPP::Match>::iterator it = matches.begin();
	    unsigned int num_match_valid = 0;
            for(; it != matches.end(); ++it)
            {
                if((*it).score3D_ > 0.0f && (*it).score3D_ > score_lim)
                {
		    matches_[src][i].push_back(*it);
		    num_match_valid++;
                    if((*it).score3D_ > best_match.score3D_)
		    {
                        best_match = (*it);
		    }
                }
            }
            scoring_mutex_.lock();
            num_valid += num_match_valid;
            scoring_mutex_.unlock();

            // store best match as estimated 3D position
            if(best_match.score3D_ > L3D_DEF_MIN_BEST_SCORE_3D)
            {
		L3DPP::Segment2D seg(src,i);
		L3DPP::Segment3D seg3D = unprojectMatch(best_match,true);
		best_match_mutex_.lock();
		entry_map_[seg] = estimated_position3D_.size();
		estimated_position3D_.push_back(std::pair<L3DPP::Segment3D,L3DPP::Match>(seg3D,best_match));
		// store depths
		depths.push_back(best_match.depth_p1_);
		depths.push_back(best_match.depth_p2_);
		best_match_mutex_.unlock();
            }
        }
        //num_matches_[src] = num_valid;
        // median depth
        float med_depth = L3D_EPS;
        if(depths.size() > 0)
        {
            std::sort(depths.begin(),depths.end());
            med_depth = depths[depths.size()/2];
        }

        if(!fixed3Dregularizer_)
            views_[src]->update_median_depth(med_depth,-1.0f,med_scene_depth_);
        else
            views_[src]->update_median_depth(med_depth,sigma_p_,med_scene_depth_);
    }

    //------------------------------------------------------------------------------
    void Line3D::storeInverseMatches(const unsigned int src)
    {
        for(size_t i=0; i<matches_[src].size(); ++i)
        {
            std::list<L3DPP::Match>::const_iterator it = matches_[src][i].begin();
            for(; it!=matches_[src][i].end(); ++it)
            {
                L3DPP::Match m = *it;
                if(m.score3D_ > 0.0f && !processed_[m.tgt_camID_])
                {
                    L3DPP::Match m_inv;
                    m_inv = m;
                    m_inv.src_camID_ = m.tgt_camID_;
                    m_inv.src_segID_ = m.tgt_segID_;
                    m_inv.tgt_camID_ = m.src_camID_;
                    m_inv.tgt_segID_ = m.src_segID_;
                    m_inv.depth_p1_ = m.depth_q1_;
                    m_inv.depth_p2_ = m.depth_q2_;
                    m_inv.depth_q1_ = m.depth_p1_;
                    m_inv.depth_q2_ = m.depth_p2_;
                    m_inv.score3D_ = 0.0f;
		    m_inv.match_orientation_ = true;
		    m_inv.match_valid_ = false;

                    matches_[m.tgt_camID_][m.tgt_segID_].push_back(m_inv);
                    ++num_matches_[m.tgt_camID_];
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::reconstruct3Dlines(const unsigned int visibility_t, const bool perform_diffusion,
                                    const float collinearity_t, const bool use_CERES,
                                    const unsigned int max_iter_CERES)
    {
        // no views can be added during reconstruction!
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        std::cout <<std::endl << prefix_ << "[3] RECONSTRUCTION ===============================" << std::endl;

        if(estimated_position3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no clusterable segments! forgot to match lines?" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // init
        max_iter_CERES_ = max_iter_CERES;
        visibility_t_ = std::max(int(visibility_t),3);
        clusters3D_.clear();
        lines3D_.clear();
        float prev_collin_t = collinearity_t_;
        collinearity_t_ = collinearity_t;

#ifdef L3DPP_CUDA
        perform_RDD_ = (perform_diffusion && useGPU_);
        if(perform_diffusion && !useGPU_)
            std::cout << prefix_err_ << "diffusion only possible when GPU mode enabled! using graph clustering instead..." << std::endl;
#else
        perform_RDD_ = false;
        if(perform_diffusion)
            std::cout << prefix_err_ << "diffusion not possible without CUDA! using graph clustering instead..." << std::endl;
#endif //L3DPP_CUDA

#ifdef L3DPP_CERES
        use_CERES_ = use_CERES;
#else
        use_CERES_ = false;
        if(use_CERES)
            std::cout << prefix_err_ << "CERES was not found! no optimization will be performed..." << std::endl;
#endif

        std::cout << prefix_ << "reconstructing 3D lines... [diffusion=" << perform_RDD_ << ", CERES=" << use_CERES_ << "]" << std::endl;

        // translate
        translate();

        // find collinear segments (if not already done)
        if(collinearity_t_ > L3D_EPS && (prev_collin_t < L3D_EPS || fabs(prev_collin_t-collinearity_t_) > L3D_EPS))
        {
            std::cout << prefix_ << "find collinear segments... [" << collinearity_t_ <<" px]" << std::endl;
            findCollinearSegments();
        }

        // compute median scene depth for lines
        std::vector<float> scene_depths_lines;
        for(std::map<unsigned int,L3DPP::View*>::const_iterator vit=views_.begin(); vit!=views_.end(); ++vit)
        {
	    std::vector<unsigned int>::iterator s = find(view_order_.begin(),view_order_.end(),vit->second->id());
            if(vit->second->median_depth() > L3D_EPS && s != view_order_.end() )
                scene_depths_lines.push_back(vit->second->median_depth());
        }

        if(scene_depths_lines.size() > 0)
        {
            std::sort(scene_depths_lines.begin(),scene_depths_lines.end());
            med_scene_depth_lines_ = scene_depths_lines[scene_depths_lines.size()/2];
        }
        else
        {
            med_scene_depth_lines_ = 0.0f;
        }

        // compute affinity matrix
        std::cout << prefix_ << "computing affinity matrix..." << std::endl;
        computingAffinityMatrix();

	std::cout << prefix_ << "num_lines_total_: ";
        std::cout << num_lines_total_ << std::endl;
	
        std::cout << prefix_ << "A: ";
        std::cout << "#entries=" << A_.size() << ", #rows=" << global2local_.size();

        unsigned int perc = float(global2local_.size())/float(num_lines_total_)*100.0f;
        std::cout << " [~" << perc << "%]" << std::endl;

        // perform diffusion
        if(perform_RDD_)
        {
            std::cout << prefix_ << "matrix diffusion..." << std::endl;
            performRDD();
        }

        // cluster matrix
        std::cout << prefix_ << "clustering segments..." << std::endl;
        clusterSegments();

        global2local_.clear();
        local2global_.clear();

        // optimize
        if(use_CERES_)
        {
            std::cout << prefix_ << "optimizing 3D lines..." << std::endl;
            optimizeClusters();
        }

        // compute final 3D segments
        std::cout << prefix_ << "computing final 3D lines..." << std::endl;
        computeFinal3Dsegments();

        clusters3D_.clear();

        // filter tiny (noisy) segments
        std::cout << prefix_ << "filtering tiny segments..." << std::endl;
        filterTinySegments();

        std::cout << prefix_ << "3D lines: total=" << lines3D_.size() << std::endl;
	
        // untranslate
        untranslate();
	project3DsegmentOn2D(lines3D_);
	for(int c=0; All_lines3D_.size()>1 && c<All_lines3D_.size()-1; c++)//最后一次更新不计算
	{
		//std::cout << std::endl;
		//std::cout << "测试! & All_lines3D_[" << c << "].size=" << All_lines3D_[c].size() << std::endl;	
		//std::cout << std::endl;
		std::vector<L3DPP::FinalLine3D> &Tmp_lines3D = All_lines3D_[c];
		clusters3D_.clear();
		for(int i=0; i<Tmp_lines3D.size(); i++)
		{	
			L3DPP::LineCluster3D& cluster = Tmp_lines3D[i].underlyingCluster_;
			if( cluster.residuals()->size() >= 2)
			{
				cv::vector<unsigned int> camID;
				cv::vector<unsigned int> segID;
				cv::vector<float> Coord;
				cv::vector<float> Coord_v;
				cv::Mat x3D;
				cv::Mat x3D_v;
			  	std::list<L3DPP::Segment2D>::const_iterator it = cluster.residuals()->begin();
			  	for( int count = 0; it!=cluster.residuals()->end(); it++ )
				{
					if(views_.find(it->camID()) != views_.end());
					{
						camID.push_back(it->camID());
						Coord.push_back(cluster.line_points[it->segID()].x);
						Coord.push_back(cluster.line_points[it->segID()].y);
						Coord_v.push_back(cluster.line_points[it->segID()].z);
						Coord_v.push_back(cluster.line_points[it->segID()].w);
						count++;
						if( count==4 )
						break;
					}
				}
				if(camID.size()==4)
				{
				        unsigned int camID1=camID[0],camID2=camID[1],camID3=camID[2],camID4=camID[3];	
					cv::Mat KRt1(3,4,CV_32F);
					cv::Mat KRt2(3,4,CV_32F);
					cv::Mat KRt3(3,4,CV_32F);
					cv::Mat KRt4(3,4,CV_32F);
					KRt1 = ORB_SLAM2::Converter::toCvMat(views_[camID1]->KRt());
					KRt2 = ORB_SLAM2::Converter::toCvMat(views_[camID2]->KRt());
					KRt3 = ORB_SLAM2::Converter::toCvMat(views_[camID3]->KRt());
					KRt4 = ORB_SLAM2::Converter::toCvMat(views_[camID4]->KRt());
					cv::KeyPoint p1;
					cv::KeyPoint p2;
					cv::KeyPoint p3;
					cv::KeyPoint p4;
					p1.pt.x = Coord[0];
					p1.pt.y = Coord[1];
					p2.pt.x = Coord[2];
					p2.pt.y = Coord[3];
					p3.pt.x = Coord[4];
					p3.pt.y = Coord[5];
					p4.pt.x = Coord[6];
					p4.pt.y = Coord[7];
					Triangulate_svd(p1, p2, p3, p4, KRt1, KRt2, KRt3, KRt4,x3D);
					cv::KeyPoint p5;
					cv::KeyPoint p6;
					cv::KeyPoint p7;
					cv::KeyPoint p8;
					p5.pt.x = Coord_v[0];
					p5.pt.y = Coord_v[1];
					p6.pt.x = Coord_v[2];
					p6.pt.y = Coord_v[3];
					p7.pt.x = Coord_v[4];
					p7.pt.y = Coord_v[5];
					p8.pt.x = Coord_v[6];
					p8.pt.y = Coord_v[7];
					Triangulate_svd(p5, p6, p7, p8, KRt1, KRt2, KRt3, KRt4, x3D_v);
					L3DPP::Segment3D seg3D1 = Tmp_lines3D[i].underlyingCluster_.seg3D();
					Eigen::Vector3d P_1 = seg3D1.P1();
					Eigen::Vector3d P_2 = seg3D1.P2();
					std::list<L3DPP::Segment3D>::iterator it_seg = Tmp_lines3D[i].collinear3Dsegments_.begin();
					Eigen::Vector3d P_3 = (*it_seg).P1();
					Eigen::Vector3d P_4 = (*it_seg).P2();
					//std::cout << "underlyingCluster内:" << P_1 << std::endl << P_2 << std::endl;
					//std::cout << "collinear3Dsegments内:" << P_3 << std::endl << P_4 << std::endl;
					L3DPP::Segment3D seg = L3DPP::Segment3D(ORB_SLAM2::Converter::toVector3d(x3D),
															   ORB_SLAM2::Converter::toVector3d(x3D_v));
					//Tmp_lines3D[i].underlyingCluster_.update3Dline(seg);
					//(*it_seg).P1_ = ORB_SLAM2::Converter::toVector3d(x3D);
					//(*it_seg).P2_ = ORB_SLAM2::Converter::toVector3d(x3D_v);
					//std::cout << "最小二乘计算值:" 
					//<< ORB_SLAM2::Converter::toVector3d(x3D) << std::endl 
					//<< ORB_SLAM2::Converter::toVector3d(x3D_v) << std::endl;
				}
			}
			clusters3D_.push_back(Tmp_lines3D[i].underlyingCluster_);
		}
		
		#ifdef L3DPP_OPENMP
		#pragma omp parallel for
	#endif //L3DPP_OPENMP
		for(int i=0; i<clusters3D_.size(); ++i)
		{
			std::list<L3DPP::Segment3D> collinear = findCollinearSegments_return(clusters3D_[i]);
			if(collinear.size() > 0)
			{
				Tmp_lines3D[i].collinear3Dsegments_ = collinear;
			}
		}
	}
        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::findCollinearSegments()
    {
        if(collinearity_t_ > L3D_EPS)
        {
            std::map<unsigned int,L3DPP::View*>::iterator it=views_.begin();
            unsigned int i=0;
            for(; it!=views_.end(); ++it,++i)
            {
                it->second->findCollinearSegments(collinearity_t_,useGPU_);

                if(i%10 == 0)
                {
                    if(i != 0)
                        std::cout << std::endl;

                    std::cout << prefix_;
                }

                std::cout << "[" << std::setfill('0') << std::setw(L3D_DISP_CAMS) << it->first << "] ";
            }
            std::cout << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::computingAffinityMatrix()
    {
        // reset
        A_.clear();
        global2local_.clear();
        local2global_.clear();
        localID_ = 0;
        used_.clear();

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<estimated_position3D_.size(); ++i)
        {
            L3DPP::Segment3D seg3D = estimated_position3D_[i].first;
            L3DPP::Match m = estimated_position3D_[i].second;
            L3DPP::Segment2D seg2D(m.src_camID_,m.src_segID_);
            bool found_aff = false;
            int id1 = -1;

            // iterate over matches
            std::list<L3DPP::Match>::const_iterator m_it = matches_[m.src_camID_][m.src_segID_].begin();
            for(; m_it != matches_[m.src_camID_][m.src_segID_].end(); ++m_it)
            {
                L3DPP::Match m2 = *m_it;
                L3DPP::Segment2D seg2D2(m2.tgt_camID_,m2.tgt_segID_);

                float sim = similarity(seg3D,m,seg2D2,false);

                if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D2))
                {
                    // check IDs
                    if(id1 < 0)
                        id1 = getLocalID(seg2D);

                    int id2 = getLocalID(seg2D2);

                    // push into affinity matrix
                    aff_mat_mutex_.lock();

                    CLEdge e;
                    e.i_ = id1;
                    e.j_ = id2;
                    e.w_ = sim;
                    A_.push_back(e);
                    e.i_ = id2;
                    e.j_ = id1;
                    A_.push_back(e);
                    found_aff = true;

                    aff_mat_mutex_.unlock();

                    // add links to potentially collinear segments to tgt
                    if(collinearity_t_ > L3D_EPS)
                    {
                        L3DPP::View* v = views_[seg2D2.camID()];
                        std::list<unsigned int> coll = v->collinearSegments(seg2D2.segID());

                        std::list<unsigned int>::const_iterator cit = coll.begin();
                        for(; cit!=coll.end(); ++cit)
                        {
                            L3DPP::Segment2D seg2D2_coll(seg2D2.camID(),*cit);

                            float sim = similarity(seg3D,m,seg2D2_coll,false);

                            if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D2_coll))
                            {
                                // check IDs
                                int id2 = getLocalID(seg2D2_coll);

                                // push into affinity matrix
                                aff_mat_mutex_.lock();

                                CLEdge e;
                                e.i_ = id1;
                                e.j_ = id2;
                                e.w_ = sim;
                                A_.push_back(e);
                                e.i_ = id2;
                                e.j_ = id1;
                                A_.push_back(e);

                                aff_mat_mutex_.unlock();
                            }
                        }
                    }
                }
            }

            // add links to potentially collinear segments
            if(found_aff && id1 >= 0 && collinearity_t_ > L3D_EPS)
            {
                L3DPP::View* v = views_[seg2D.camID()];
                std::list<unsigned int> coll = v->collinearSegments(seg2D.segID());

                std::list<unsigned int>::const_iterator cit = coll.begin();
                for(; cit!=coll.end(); ++cit)
                {
                    L3DPP::Segment2D seg2D_coll(seg2D.camID(),*cit);

                    float sim = similarity(seg3D,m,seg2D_coll,false);

                    if(sim > L3D_DEF_MIN_AFFINITY && unused(seg2D,seg2D_coll))
                    {
                        // check IDs
                        int id2 = getLocalID(seg2D_coll);

                        // push into affinity matrix
                        aff_mat_mutex_.lock();

                        CLEdge e;
                        e.i_ = id1;
                        e.j_ = id2;
                        e.w_ = sim;
                        A_.push_back(e);
                        e.i_ = id2;
                        e.j_ = id1;
                        A_.push_back(e);

                        aff_mat_mutex_.unlock();
                    }
                }
            }
        }

        // cleanup
        used_.clear();
    }

    //------------------------------------------------------------------------------
    bool Line3D::unused(const Segment2D &seg1, const Segment2D &seg2)
    {
        bool unused = true;

        // check if used
        aff_used_mutex_.lock();
        if(used_[seg1].find(seg2) != used_[seg1].end())
        {
            // already used
            unused = false;
        }
        else
        {
            // not yet used
            used_[seg1].insert(seg2);
            used_[seg2].insert(seg1);
        }
        aff_used_mutex_.unlock();

        return unused;
    }

    //------------------------------------------------------------------------------
    int Line3D::getLocalID(const Segment2D &seg)
    {
        int id;
        aff_id_mutex_.lock();
        if(global2local_.find(seg) == global2local_.end())
        {
            id = localID_;
            ++localID_;

            global2local_[seg] = id;
            local2global_[id] = seg;
        }
        else
        {
            id = global2local_[seg];
        }
        aff_id_mutex_.unlock();
        return id;
    }

    //------------------------------------------------------------------------------
    void Line3D::performRDD()
    {
#ifdef L3DPP_CUDA
        // create sparse GPU matrix
        L3DPP::SparseMatrix* W = new L3DPP::SparseMatrix(A_,global2local_.size());

        // perform RDD
        L3DPP::replicator_dynamics_diffusion_GPU(W,prefix_);

        // update affinities (symmetrify)
        W->download();
        A_.clear();

        std::map<int,std::map<int,float> > entries;
        for(unsigned int i=0; i<W->entries()->width(); ++i)
        {
            int s1 = W->entries()->dataCPU(i,0)[0].x;
            int s2 = W->entries()->dataCPU(i,0)[0].y;
            float w12 = W->entries()->dataCPU(i,0)[0].z;

            float w21 = w12;
            if(entries[s2].find(s1) != entries[s2].end())
            {
                // other one already processed
                w21 = entries[s2][s1];
            }

            float w = fmin(w12,w21);

            entries[s1][s2] = w;
            entries[s2][s1] = w;
        }

        std::map<int,std::map<int,float> >::const_iterator it = entries.begin();
        for(; it!=entries.end(); ++it)
        {
            std::map<int,float>::const_iterator it2 = it->second.begin();
            for(; it2!=it->second.end(); ++it2)
            {
                CLEdge e;
                e.i_ = it->first;
                e.j_ = it2->first;
                e.w_ = it2->second;
                A_.push_back(e);
            }
        }

        // cleanup
        delete W;
#endif //L3DPP_CUDA
    }

    //------------------------------------------------------------------------------
    void Line3D::clusterSegments()
    {
        // init
        clusters3D_.clear();
        lines3D_.clear();

        if(A_.size() == 0)
            return;

        // graph clustering
        L3DPP::CLUniverse* u = L3DPP::performClustering(A_,global2local_.size(),3.0f);

        // clustering done
        A_.clear();

        //process clusters
        std::map<int,std::list<L3DPP::Segment2D> > cluster2segments;
        std::map<int,std::map<unsigned int,bool> > cluster2cameras;
        std::vector<int> unique_clusters;

        std::map<int,L3DPP::Segment2D>::const_iterator it = local2global_.begin();
        for(; it!=local2global_.end(); ++it)
        {
            int clID = u->find(it->first);
            L3DPP::Segment2D seg = it->second;

            if(cluster2segments.find(clID) == cluster2segments.end())
                unique_clusters.push_back(clID);

            // store segment
            cluster2segments[clID].push_back(seg);
            // store camera
            cluster2cameras[clID][seg.camID()] = true;
        }
        delete u;

        if(cluster2segments.size() == 0)
        {
            std::cout << prefix_wng_ << "no clusters found..." << std::endl;
            return;
        }

        std::cout << prefix_ << "clusters: ";
        std::cout << "total=" << cluster2segments.size() << ", ";

        // create 3D lines for valid clusters
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<unique_clusters.size(); ++i)
        {
            int clID = unique_clusters[i];

            if(cluster2cameras[clID].size() >= visibility_t_)
            {
                // create 3D line cluster
                L3DPP::LineCluster3D LC = get3DlineFromCluster(cluster2segments[clID]);
		//std::cout << clID << " ";
                if(LC.size() > 0)
                {
                    // 3D line valid --> store in list
                    cluster_mutex_.lock();
                    clusters3D_.push_back(LC);
                    cluster_mutex_.unlock();
                }
            }
        }
        std::cout << "valid=" << clusters3D_.size();

        unsigned int perc = float(clusters3D_.size())/float(cluster2segments.size())*100;
        std::cout << " [~" << perc << "%]";

        std::cout << std::endl;
    }

    //------------------------------------------------------------------------------
    L3DPP::LineCluster3D Line3D::get3DlineFromCluster(const std::list<L3DPP::Segment2D>& cluster)
    {
        // create scatter matrix
        Eigen::Vector3d P(0,0,0);
        int n = cluster.size()*2;
        Eigen::MatrixXd L_points(3,n);

        std::list<L3DPP::Segment2D>::const_iterator it = cluster.begin();
        unsigned int reference_cam = 0;
        float max_len_2D = 0.0f;
        for(size_t i=0; it!=cluster.end(); ++it,i+=2)
        {
            // get 3D hypothesis
            size_t pos = entry_map_[*it];
            L3DPP::Segment3D hyp3D = estimated_position3D_[pos].first;

            P += hyp3D.P1();
            P += hyp3D.P2();

            L_points(0,i) = hyp3D.P1().x();
            L_points(1,i) = hyp3D.P1().y();
            L_points(2,i) = hyp3D.P1().z();

            L_points(0,i+1) = hyp3D.P2().x();
            L_points(1,i+1) = hyp3D.P2().y();
            L_points(2,i+1) = hyp3D.P2().z();

            // check 2D length -> max length defines reference view (for filtering later on)
            Eigen::Vector4f coords = views_[(*it).camID()]->getLineSegment2D((*it).segID());
            float length_sqr = (coords(0)-coords(2))*(coords(0)-coords(2)) + (coords(1)-coords(3))*(coords(1)-coords(3));
            if(length_sqr > max_len_2D)
            {
                max_len_2D = length_sqr;
                reference_cam = (*it).camID();
            }
        }

        // center of gravity
        P /= double(n);

        // direction
        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n,n)-(1.0/(double)(n))*Eigen::MatrixXd::Constant(n,n,1.0);
        Eigen::MatrixXd Scat = L_points*C*L_points.transpose();

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Scat, Eigen::ComputeThinU);

        Eigen::MatrixXd U;
        Eigen::VectorXd S;

        U = svd.matrixU();
        S = svd.singularValues();

        int maxPos;
        S.maxCoeff(&maxPos);

        Eigen::Vector3d dir = Eigen::Vector3d(U(0, maxPos), U(1, maxPos), U(2, maxPos));
        dir.normalize();

        // initial 3D line for cluster
        L3DPP::Segment3D initial3Dline(P-dir,P+dir);
        L3DPP::LineCluster3D LC = L3DPP::LineCluster3D(initial3Dline,cluster,reference_cam);

        return LC;
    }

    //------------------------------------------------------------------------------
    L3DPP::Segment3D Line3D::project2DsegmentOnto3Dline(const L3DPP::Segment2D& seg2D,
                                                        const L3DPP::Segment3D& seg3D,
                                                        bool& success)
    {
        // tgt line
        Eigen::Vector3d P = seg3D.P1();
        Eigen::Vector3d u = seg3D.dir();

        // src line 1
        L3DPP::View* v = views_[seg2D.camID()];
        Eigen::Vector3d Q = v->C();
        Eigen::Vector3d v1 = v->getNormalizedLinePointRay(seg2D.segID(),true);

        // src line 2
        Eigen::Vector3d v2 = v->getNormalizedLinePointRay(seg2D.segID(),false);

        Eigen::Vector3d w = P-Q;

        // vals
        double a = u.dot(u);
        double b1 = u.dot(v1);
        double b2 = u.dot(v2);
        double c1 = v1.dot(v1);
        double c2 = v2.dot(v2);
        double d = u.dot(w);
        double e1 = v1.dot(w);
        double e2 = v2.dot(w);

        double denom1 = a*c1 - b1*b1;
        double denom2 = a*c2 - b2*b2;

        if(fabs(denom1) > L3D_EPS && fabs(denom2) > L3D_EPS)
        {
            success = true;
            double s1 = (b1*e1 - c1*d)/denom1;
            double s2 = (b2*e2 - c2*d)/denom2;
            return L3DPP::Segment3D(P+s1*u,P+s2*u);
        }
        else
        {
            // projection not possible
            success = false;
            return L3DPP::Segment3D();
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::optimizeClusters()
    {
#ifdef L3DPP_CERES
        L3DPP::LineOptimizer opt(views_,&clusters3D_,max_iter_CERES_,prefix_);
        opt.optimize();
#endif //L3DPP_CERES
    }

    //------------------------------------------------------------------------------
    void Line3D::computeFinal3Dsegments()
    {
        // iterate over clusters and find all valid collinear segments
#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<clusters3D_.size(); ++i)
        {
            std::list<L3DPP::Segment3D> collinear = findCollinearSegments_return(clusters3D_[i]);
	    //std::cout << "*" << collinear.size() << " " << clusters3D_.size() << " collinearity_t_:" << collinearity_t_ << std::endl;
            if(collinear.size() > 0)
            {
                L3DPP::FinalLine3D final;
                final.collinear3Dsegments_ = collinear;
                final.underlyingCluster_ = clusters3D_[i];

                cluster_mutex_.lock();
                lines3D_.push_back(final);
                cluster_mutex_.unlock();
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::filterTinySegments()
    {
        // remove 3D segments that are too small
        size_t valid_before = lines3D_.size();
        if(valid_before == 0)
            return;

#ifdef L3DPP_OPENMP
        #pragma omp parallel for
#endif //L3DPP_OPENMP
        for(int i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::View* v = views_[lines3D_[i].underlyingCluster_.reference_view()];

            std::list<L3DPP::Segment3D> filteredSegments;
            std::list<L3DPP::Segment3D>::const_iterator it = lines3D_[i].collinear3Dsegments_.begin();
            for(; it!=lines3D_[i].collinear3Dsegments_.end(); ++it)
            {
                if(v->projectedLongEnough(*it))
                    filteredSegments.push_back(*it);
            }
            lines3D_[i].collinear3Dsegments_ = filteredSegments;
        }

        // remove invalid lines
        std::vector<L3DPP::FinalLine3D> lines3D;
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            if(lines3D_[i].collinear3Dsegments_.size() > 0)
                lines3D.push_back(lines3D_[i]);
        }
        lines3D_ = lines3D;

        size_t valid_after = lines3D_.size();

        std::cout << prefix_ << "removed lines: " << valid_before-valid_after << std::endl;
    }

    //------------------------------------------------------------------------------
    std::list<L3DPP::Segment3D> Line3D::findCollinearSegments_return(const L3DPP::LineCluster3D& cluster)
    {
        // project onto 3D line
        std::list<L3DPP::Segment3D> collinear_segments;
        Eigen::Vector3d COG = 0.5*(cluster.seg3D().P1()+cluster.seg3D().P2());
        std::list<L3DPP::Segment2D>::const_iterator it = cluster.residuals()->begin();

        std::list<L3DPP::PointOn3DLine> linePoints;
        std::vector<Eigen::Vector3d> pts(cluster.residuals()->size()*2);

        float distToCOG = 0.0f;
        Eigen::Vector3d border;

        size_t pID = 0;
        for(size_t id=0; it!=cluster.residuals()->end(); ++it,++id,pID+=2)
        {
            // project onto 3D line
            bool success;
            L3DPP::Segment3D proj = project2DsegmentOnto3Dline(*it,cluster.seg3D(),success);
            if(success)
            {
                // create line points
                L3DPP::PointOn3DLine p1,p2;

                p1.lineID_ = id;
                p1.pointID_ = pID;
                p1.camID_ = (*it).camID();
                pts[pID] = proj.P1();
                linePoints.push_back(p1);

                float d = (proj.P1()-COG).norm();
                if(d > distToCOG)
                {
                    distToCOG = d;
                    border = proj.P1();
                }

                p2.lineID_ = id;
                p2.pointID_ = pID+1;
                p2.camID_ = (*it).camID();
                pts[pID+1] = proj.P2();
                linePoints.push_back(p2);

                d = (proj.P2()-COG).norm();
                if(d > distToCOG)
                {
                    distToCOG = d;
                    border = proj.P2();
                }
            }
        }
        // check number of projected lines/points
        if(linePoints.size() < 6)
	{
            return collinear_segments;
	}

        // sort by distance to border
        std::list<L3DPP::PointOn3DLine>::iterator lit = linePoints.begin();
        for(; lit!=linePoints.end(); ++lit)
        {
            (*lit).distToBorder_ = (pts[(*lit).pointID_]-border).norm();
        }
        linePoints.sort(L3DPP::sortPointsOn3DLine);

        // iterate and create 3D segments
        std::map<size_t,unsigned int> open;
        std::map<size_t,bool> open_lines;
        bool opened = false;
        Eigen::Vector3d current_start(0,0,0);
        lit = linePoints.begin();
        for(; lit!=linePoints.end(); ++lit)
        {
            L3DPP::PointOn3DLine pt = *lit;
            if(open_lines.find(pt.lineID_) == open_lines.end())
            {
                // opening
                open_lines[pt.lineID_] = true;
                if(open.find(pt.camID_) == open.end())
                    open[pt.camID_] = 1;
                else
                    ++open[pt.camID_];
            }
            else
            {
                // closing
                open_lines.erase(pt.lineID_);

                --open[pt.camID_];

                if(open[pt.camID_] == 0)
                    open.erase(pt.camID_);
            }

            if(opened && open.size() < 3)
            {
                L3DPP::Segment3D l(current_start,pts[pt.pointID_]);
                collinear_segments.push_back(l);
                opened = false;
            }
            else if(!opened && open.size() >= 3)
            {
                current_start = pts[pt.pointID_];
                opened = true;
            }
        }
        return collinear_segments;
    }
    
    void Line3D::project3DsegmentOn2D(std::vector<L3DPP::FinalLine3D>& lines3D_)
    {		
	std::vector<L3DPP::FinalLine3D>::iterator it_L= lines3D_.begin();
	for(int i=0; i<lines3D_.size(); i++)
	{	
		std::list<L3DPP::Segment3D>::const_iterator it_3DL = lines3D_[i].collinear3Dsegments_.begin();
		//std::cout << P1.x() << " " << P1.y() << " " << P1.z() << " " << std::endl;
		//std::cout << P2.x() << " " << P2.y() << " " << P2.z() << " " << std::endl;

		L3DPP::LineCluster3D& cluster = lines3D_[i].underlyingCluster_;
		//Eigen::Vector3d P1 = cluster.seg3D().P1();
		//Eigen::Vector3d P2 = cluster.seg3D().P2();
		Eigen::Vector3d P1 = (*it_3DL).P1();
		Eigen::Vector3d P2 = (*it_3DL).P2();
		
		if( cluster.residuals()->size() >= 2)
		{
			unsigned int camID;
			unsigned int segID;
			std::list<L3DPP::Segment2D>::const_iterator it = cluster.residuals()->begin();
			for( int count = 0; it!=cluster.residuals()->end(); it++ )
			{
				camID = it->camID();
				segID = it->segID();
				Eigen::Matrix<double,3,4> KRt = views_[camID]->KRt();
				Eigen::Vector4f coords = views_[camID]->getLineSegment2D(segID);
				Eigen::Vector4d P(P1(0),P1(1),P1(2),1);
				Eigen::Vector3d q = KRt*P;
				float xn = float( q[0] / q[2] );
				float yn = float( q[1] / q[2] );
				//std::cout << KRt << std::endl;
				//std::cout << "原值:" << coords(0) << " " << coords(1) << " " << coords(2) << " " << coords(3) << std::endl;
				//std::cout << "结果:" << xn << " " << yn << std::endl;
				float length1 = (coords(0)-xn)*(coords(0)-xn) + (coords(1)-yn)*(coords(1)-yn);
				float length2 = (coords(2)-xn)*(coords(2)-xn) + (coords(3)-yn)*(coords(3)-yn);
				if(length1 <= length2)
				{
					float4 a;
					a.x = coords(0); a.y = coords(1); a.z = coords(2); a.w = coords(3);
					cluster.line_points.insert(std::map<unsigned int,float4>::value_type ( segID,a )); 
				}
				else
				{
					float4 a;
					a.x = coords(2); a.y = coords(3); a.z = coords(0); a.w = coords(1);
					cluster.line_points.insert(std::map<unsigned int,float4>::value_type ( segID,a )); 
				}
			}
		}
	}
    }

    //------------------------------------------------------------------------------
    void Line3D::get3Dlines(std::vector< std::vector<L3DPP::FinalLine3D> >& result)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();
	All_lines3D_.push_back(lines3D_);
	result = All_lines3D_;
	std::cout << prefix_ << "参与 " << All_lines3D_.size() << "次 运算!" << std::endl;
        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }
    
    void Line3D::Triangulate_svd(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::KeyPoint &kp3, const cv::KeyPoint &kp4,
				 const cv::Mat &P1, const cv::Mat &P2, const cv::Mat &P3, const cv::Mat &P4,
				 cv::Mat &x3D) 
    { 
	cv::Mat A(8,4,CV_32F);
	A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0); 
	A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
	A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0); 
	A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1); 
	A.row(4) = kp3.pt.x*P3.row(2)-P3.row(0); 
	A.row(5) = kp3.pt.y*P3.row(2)-P3.row(1);
	A.row(6) = kp4.pt.x*P4.row(2)-P4.row(0); 
	A.row(7) = kp4.pt.y*P4.row(2)-P4.row(1); 
	cv::Mat u,w,vt; 
	cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV); 
	x3D = vt.row(3).t(); 
	x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
    }
    
    //------------------------------------------------------------------------------
    void Line3D::saveResultAsSTL(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".stl";

        std::ofstream file;
        file.open(filename.c_str());

        file << "solid lineModel" << std::endl;

        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                char x1[50];
                char y1[50];
                char z1[50];

                char x2[50];
                char y2[50];
                char z2[50];

                sprintf(x1,"%e",P1.x());
                sprintf(y1,"%e",P1.y());
                sprintf(z1,"%e",P1.z());

                sprintf(x2,"%e",P2.x());
                sprintf(y2,"%e",P2.y());
                sprintf(z2,"%e",P2.z());

                file << " facet normal 1.0e+000 0.0e+000 0.0e+000" << std::endl;
                file << "  outer loop" << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "   vertex " << x2 << " " << y2 << " " << z2 << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "  endloop" << std::endl;
                file << " endfacet" << std::endl;
            }
        }

        file << "endsolid lineModel" << std::endl;
        file.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::saveTempResultAsSTL(const std::string& output_folder,
                                     const std::string& suffix,
                                     const std::vector<L3DPP::Segment3D>& lines3D)
    {
        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+"__"+suffix+".stl";

        std::ofstream file;
        file.open(filename.c_str());

        file << "solid lineModel" << std::endl;

        for(size_t i=0; i<lines3D.size(); ++i)
        {
            L3DPP::Segment3D current = lines3D[i];

            Eigen::Vector3d P1 = current.P1();
            Eigen::Vector3d P2 = current.P2();

            char x1[50];
            char y1[50];
            char z1[50];

            char x2[50];
            char y2[50];
            char z2[50];

            sprintf(x1,"%e",P1.x());
            sprintf(y1,"%e",P1.y());
            sprintf(z1,"%e",P1.z());

            sprintf(x2,"%e",P2.x());
            sprintf(y2,"%e",P2.y());
            sprintf(z2,"%e",P2.z());

            file << " facet normal 1.0e+000 0.0e+000 0.0e+000" << std::endl;
            file << "  outer loop" << std::endl;
            file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
            file << "   vertex " << x2 << " " << y2 << " " << z2 << std::endl;
            file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
            file << "  endloop" << std::endl;
            file << " endfacet" << std::endl;
        }

        file << "endsolid lineModel" << std::endl;
        file.close();
    }

    //------------------------------------------------------------------------------
    void Line3D::saveResultAsOBJ(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".obj";

        std::ofstream file;
        file.open(filename.c_str());

        size_t lineID = 0;
        size_t pointID = 1;
        std::map<size_t,size_t> lines2points;
        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2,++lineID,pointID+=2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                file << "v " << P1.x() << " " << P1.y() << " " << P1.z() << std::endl;
                file << "v " << P2.x() << " " << P2.y() << " " << P2.z() << std::endl;

                lines2points[lineID] = pointID;
            }
        }

        std::map<size_t,size_t>::const_iterator it = lines2points.begin();
        for(; it!=lines2points.end(); ++it)
        {
            file << "l " << it->second << " " << it->second+1 << std::endl;
        }

        file.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsTXT(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".txt";

        std::ofstream file;
        file.open(filename.c_str());

        for(size_t i=0; i<lines3D_.size(); ++i)
        {
            L3DPP::FinalLine3D current = lines3D_[i];

            if(current.collinear3Dsegments_.size() == 0)
                continue;

            // write 3D segments
            file << current.collinear3Dsegments_.size() << " ";
            std::list<L3DPP::Segment3D>::const_iterator it2 = current.collinear3Dsegments_.begin();
            for(; it2!=current.collinear3Dsegments_.end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).P1();
                Eigen::Vector3d P2 = (*it2).P2();

                file << P1.x() << " " << P1.y() << " " << P1.z() << " ";
                file << P2.x() << " " << P2.y() << " " << P2.z() << " ";
            }

            // write 2D residuals
            file << current.underlyingCluster_.residuals()->size() << " ";
            std::list<L3DPP::Segment2D>::const_iterator it3 = current.underlyingCluster_.residuals()->begin();
            for(; it3!=current.underlyingCluster_.residuals()->end(); ++it3)
            {
                file << (*it3).camID() << " " << (*it3).segID() << " ";
                Eigen::Vector4f coords = getSegmentCoords2D(*it3);
                file << coords(0) << " " << coords(1) << " ";
                file << coords(2) << " " << coords(3) << " ";
            }

            file << std::endl;
        }

        file.close();

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsBIN(const std::string& output_folder)
    {
        view_mutex_.lock();
        view_reserve_mutex_.lock();

        if(lines3D_.size() == 0)
        {
            std::cout << prefix_wng_ << "no 3D lines to save!" << std::endl;
            view_reserve_mutex_.unlock();
            view_mutex_.unlock();
            return;
        }

        // get filename
        std::string filename = output_folder+"/"+createOutputFilename()+".bin";

        // serialize
        L3DPP::serializeToFile(filename,lines3D_);

        view_reserve_mutex_.unlock();
        view_mutex_.unlock();
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::rotationFromRPY(const double roll, const double pitch,
                                            const double yaw)
    {
        const Eigen::Matrix3d Rx
          = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const Eigen::Matrix3d Ry
          = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Matrix3d Rz
          = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

        const Eigen::Matrix3d R = Rz * Ry * Rx;

        return R;
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::rotationFromQ(const double Qw, const double Qx,
                                          const double Qy, const double Qz)
    {
        double n = Qw*Qw + Qx*Qx + Qy*Qy + Qz*Qz;

        double s;
        if(fabs(n) < L3D_EPS)
        {
            s = 0;
        }
        else
        {
            s = 2.0/n;
        }

        double wx = s*Qw*Qx; double wy = s*Qw*Qy; double wz = s*Qw*Qz;
        double xx = s*Qx*Qx; double xy = s*Qx*Qy; double xz = s*Qx*Qz;
        double yy = s*Qy*Qy; double yz = s*Qy*Qz; double zz = s*Qz*Qz;

        Eigen::Matrix3d R;
        R(0,0) = 1.0 - (yy + zz); R(0,1) = xy - wz;         R(0,2) = xz + wy;
        R(1,0) = xy + wz;         R(1,1) = 1.0 - (xx + zz); R(1,2) = yz - wx;
        R(2,0) = xz - wy;         R(2,1) = yz + wx;         R(2,2) = 1.0 - (xx + yy);
        return R;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f Line3D::getSegmentCoords2D(const L3DPP::Segment2D& seg2D)
    {
        Eigen::Vector4f coords(0,0,0,0);
        if(views_.find(seg2D.camID()) != views_.end())
        {
            coords = views_[seg2D.camID()]->getLineSegment2D(seg2D.segID());
        }
        return coords;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector4f Line3D::getSegmentCoords2D(const unsigned int camID,
                                               const unsigned int segID)
    {
        return getSegmentCoords2D(L3DPP::Segment2D(camID,segID));
    }

    //------------------------------------------------------------------------------
    void Line3D::eigen2dataArray(L3DPP::DataArray<float>* &DA, const Eigen::MatrixXd& M)
    {
        DA = new L3DPP::DataArray<float>(M.cols(),M.rows());
        for(size_t y=0; y<size_t(M.rows()); ++y)
            for(size_t x=0; x<size_t(M.cols()); ++x)
                DA->dataCPU(x,y)[0] = M(y,x);
    }

    //------------------------------------------------------------------------------
    void Line3D::decomposeProjectionMatrix(const Eigen::MatrixXd P_in,
                                           Eigen::Matrix3d& K_out,
                                           Eigen::Matrix3d& R_out,
                                           Eigen::Vector3d& t_out)
    {
        if(P_in.rows() != 3 && P_in.cols() != 4)
        {
            std::cout << "P is not a 3x4 matrix! (" << P_in.rows() << "x" << P_in.cols() << ")" << std::endl;
            return;
        }

        K_out = P_in.block<3,3>(0,0);

        // get affine matrix (rq-decomposition of M)
        // See Hartley & Zissermann, p552 (1st ed.)
        double h = std::sqrt((K_out(2,1))*(K_out(2,1)) + (K_out(2,2))*(K_out(2,2)));
        double s =  K_out(2,1) / h;
        double c = -K_out(2,2) / h;

        Eigen::Matrix3d Rx;
        Rx.setZero();
        Rx(0,0) =  1;
        Rx(1,1) =  c; Rx(2,2) = c;
        Rx(1,2) = -s; Rx(2,1) = s;

        K_out = K_out * Rx;

        h = sqrt((K_out(2,0))*(K_out(2,0)) + (K_out(2,2))*(K_out(2,2)));
        s =  K_out(2,0) / h;
        c = -K_out(2,2) / h;

        Eigen::Matrix3d Ry;
        Ry.setZero();
        Ry(1,1) =  1;
        Ry(0,0) =  c; Ry(2,2) = c;
        Ry(0,2) = -s; Ry(2,0) = s;

        K_out = K_out * Ry;

        h = sqrt((K_out(1,0)*K_out(1,0)) + (K_out(1,1)*K_out(1,1)));
        s =  K_out(1,0) / h;
        c = -K_out(1,1) / h;

        Eigen::Matrix3d Rz;
        Rz.setZero();
        Rz(2,2) =  1;
        Rz(0,0) =  c; Rz(1,1) = c;
        Rz(0,1) = -s; Rz(1,0) = s;

        K_out = K_out * Rz;

        Eigen::Matrix3d Sign = Eigen::Matrix3d::Identity(3,3);

        if (K_out(0,0) < 0) Sign(0,0) = -1;
        if (K_out(1,1) < 0) Sign(1,1) = -1;
        if (K_out(2,2) < 0) Sign(2,2) = -1;

        K_out = K_out * Sign; // change signum of columns

        R_out = Rx * Ry * Rz * Sign;
        R_out.transposeInPlace();

        Eigen::Vector3d P4;
        P4 = P_in.block<3,1>(0,3);

        t_out = K_out.inverse() * P4;

        K_out *= 1.0 / K_out(2,2); // normalize, such that lower-right element is 1
    }

    //------------------------------------------------------------------------------
    std::string Line3D::createOutputFilename()
    {
        std::stringstream str;
        str << "Line3D++__";

        if(max_image_width_ > 0)
            str << "W_" << max_image_width_ << "__";
        else
            str << "W_FULL__";

        str << "N_" << num_neighbors_ << "__";

        str << "sigmaP_" << sigma_p_ << "__";
        str << "sigmaA_" << sigma_a_ << "__";

        str << "epiOverlap_" << epipolar_overlap_ << "__";

        if(kNN_ > 0)
            str << "kNN_" << kNN_ << "__";

        if(collinearity_t_ > L3D_EPS)
            str << "COLLIN_" << collinearity_t_ << "__";

        if(fixed3Dregularizer_)
        {
            str << "FXD_SIGMA_P__";

            if(const_regularization_depth_ > 0.0f)
                str << "REG_DEPTH_" << const_regularization_depth_ << "__";
        }

        if(perform_RDD_)
            str << "DIFFUSION__";

        if(use_CERES_)
            str << "OPTIMIZED__";

        str << "vis_" << visibility_t_;
        return str.str();
    }
}
