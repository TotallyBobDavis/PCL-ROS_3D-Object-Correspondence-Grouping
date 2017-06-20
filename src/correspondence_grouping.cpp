#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h> 
#include <pcl/io/io.h>
#include <pcl_ros/point_cloud.h>
#include <iostream>
#include <signal.h>	

#include <sensor_msgs/JointState.h>
#include "segbot_arm_perception/TabletopPerception.h"
#include <segbot_arm_manipulation/arm_utils.h>
#include "segbot_arm_manipulation/TabletopGraspAction.h"

#include <ros/callback_queue.h>
#include <pcl/filters/filter.h>
#include <pcl/surface/mls.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/normal_3d.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/planar_region.h>
#include <pcl/impl/point_types.hpp>

typedef pcl::PointXYZ PointType; // Add and remove RGB as needed
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;

typedef pcl::SHOT352 DescriptorType;
//~ typedef pcl::VFHSignature308 DescriptorType;
//~ typedef pcl::FPFHSignature33 DescriptorType;

std::vector<std::string> model_filenames;
std::string scene_filename_;
std::vector<std::string> models_recognized;
std::vector<std::string> cmodels_recognized;
bool turn_visualization_on (true);
ros::Subscriber sub;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);    
float model_ss_ (0.01f);	// (default = 0.01f)
float scene_ss_ (0.03f);	// (default = 0.03f)
float rf_rad_ (0.015f);		// Suggestion: recommended increase for better accuracy (default = 0.015f)
float descr_rad_ (0.02f);	// Suggestion: adjust as needed (default = 0.02f) | Apparently 0.06 is the best search radius?
float cg_size_ (0.01f);		// Suggestion: decrease for better accuracy (default = 0.01f)
float cg_thresh_ (5.0f);	// Suggestion: increase for better accuracy (default = 5.0f)

bool use_resampling_smoothing_ (false);

//true if Ctrl-C is pressed
bool g_caught_sigint (false);

// Suggestion: Adjust the descriptor radius and downsampling radius

/* what happens when ctr-c is pressed */
void sig_handler(int sig) {
  g_caught_sigint = true;
  ROS_INFO("caught sigint, init shutdown sequence...");
  ros::shutdown();
  exit(1);
};

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping          - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "     -s:                     Resample and smooth out the scene" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[], std::string filename)
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

 /*
  //Model & scene filenames (manually adjust the model filenames to be checked for in the scene)
  std::vector<int> filenames;
  // filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

  if (filenames.size () <= 0)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }
  */

  // Converts model_filename_ to vector of strings that contains the model names
  //for(int models = filenames[0]; models <= argc - 1; models++)
  //{
  //	model_filenames.push_back(argv[models]);
  //}
  
  // ADDING OR REMOVING PCD MODELS TO BE SEARCHED FOR THROUGH THE KINECT

  // Manually add models to be searched for in the scene here (push_back the filenames onto the vector)
  model_filenames.push_back(filename); 

  // This is the file that is stores the scene that the Kinect (or rosbag) is displaying (do not edit)
  scene_filename_ = "scene.pcd";

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-s"))
  {
    use_resampling_smoothing_ = true;
  }
  
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}


void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud)
{ 
  pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
  
  pcl::fromROSMsg (*cloud, *temp_cloud);
  
  std::string prefix_ = "scene";
  
  if ((temp_cloud->width * temp_cloud->height) == 0)
        return;

      ROS_INFO ("Received %d data points in frame %s with the following fields: %s",
                (int)temp_cloud->width * temp_cloud->height,
                temp_cloud->header.frame_id.c_str (),
                pcl::getFieldsList (*temp_cloud).c_str ());

      std::stringstream ss;
      //ss << prefix_ << cloud->header.stamp << ".pcd";
      ss << scene_filename_;
      ROS_INFO ("Data saved to %s", ss.str ().c_str ());

      	//~ pcl::io::savePCDFile (ss.str (), *temp_cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
	pcl::io::savePCDFile(ss.str(), *temp_cloud);
}

int
main (int argc, char *argv[])
{
  ros::init (argc, argv, "correspondence_grouping");
  ros::NodeHandle nh("~");
  signal(SIGINT, sig_handler);
  
  std::string filename;
  nh.getParam("file", filename); // DO NOT PUT QUOTES FOR THE PRIVATE PARAMETER! EXAMPLE: _file:=milk 
 
  filename += ".pcd";
  
  // Change the topic if it no longer utilizes nav_kinect and replace it with the new camera
  //~ sub = nh.subscribe<sensor_msgs::PointCloud2> ("/nav_kinect/depth_registered/points", 1, cloud_cb);
  //~ sub = nh.subscribe<sensor_msgs::PointCloud2> ("/xtion_camera/depth_registered/points", 100, cloud_cb);
  //~ sub = nh.subscribe<sensor_msgs::PointCloud2> ("/table_object_detection_node/plane_cloud", 100, cloud_cb);
  
  // Use SegBot's own pipeline to segment out the tabletop
  segbot_arm_perception::TabletopPerception::Response table_scene = segbot_arm_manipulation::getTabletopScene(nh);
 
  if ((int)table_scene.cloud_clusters.size() == 0){
			ROS_WARN("No objects found on table. The end...");
			exit(1);
  }
  
  int largest_pc_index = -1;
  int largest_num_points = -1;
  for (unsigned int i = 0; i < table_scene.cloud_clusters.size(); i++){
	
	int num_points_i = table_scene.cloud_clusters[i].height* table_scene.cloud_clusters[i].width;

	if (num_points_i > largest_num_points){
		largest_num_points = num_points_i;
		largest_pc_index = i;
	}
  }
  
  //
  //  Saving scene.pcd Without Using the Subscriber
  //
  pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
  
  pcl::fromROSMsg (table_scene.cloud_clusters[largest_pc_index], *temp_cloud);
  
  //~ std::string prefix_ = "scene";
  //~ 
  //~ if ((temp_cloud->width * temp_cloud->height) == 0)
        //~ return 0;
//~ 
      //~ ROS_INFO ("Received %d data points in frame %s with the following fields: %s",
                //~ (int)temp_cloud->width * temp_cloud->height,
                //~ temp_cloud->header.frame_id.c_str (),
                //~ pcl::getFieldsList (*temp_cloud).c_str ());
//~ 
      //~ std::stringstream ss;
      //~ //ss << prefix_ << cloud->header.stamp << ".pcd";
      //~ ss << scene_filename_;
      //~ ROS_INFO ("Data saved to %s", ss.str ().c_str ());

      	//~ pcl::io::savePCDFile (ss.str (), *temp_cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
	//~ pcl::io::savePCDFile(ss.str(), *temp_cloud);
  
  ros::Rate loop_rate(300); // 1Hz = 1 second? (1 / Hz = seconds?)
  
  parseCommandLine (argc, argv, filename);

  for(int num = 0; num < table_scene.cloud_clusters.size(); num++)
  {
	  
	  //~ ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.75));
	
	  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
	  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
	  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
	  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
		
	  pcl::copyPointCloud(*temp_cloud, *scene);
	  
	  //
	  //  Load clouds
	  //
	  if (pcl::io::loadPCDFile ("/home/totallybobdavis/catkin_ws/src/correspondence_grouping/pcd_files/" + model_filenames[num], *model) < 0)
	  {
	    std::cout << "Error loading model cloud." << std::endl;
	    showHelp (argv[0]);
	    return (-1);
	  }
	  //~ if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
	  //~ {
	    //~ std::cout << "Error loading scene cloud." << std::endl;
	    //~ showHelp (argv[0]);
	    //~ return (-1);
	  //~ }
	  //~ else // MLS Surface Reconstruction To Smooth and Resample Noisy Data
	  //~ {
		  if (use_resampling_smoothing_)
		  {
			  std::vector<int> indices;
			  pcl::PointCloud<PointType>::Ptr scene1 (new pcl::PointCloud<PointType> ());
			  pcl::removeNaNFromPointCloud(*scene, *scene1, indices);		  
			  
			  // Load input file into a PointCloud<T> with an appropriate type
			  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
			  pcl::copyPointCloud(*scene1, *cloud);	
			  
			  // Create a KD-Tree
			  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
			  
			  // Output has the PointNormal type in order to store the normals calculated by MLS
			  pcl::PointCloud<pcl::PointNormal> mls_points;
			  
			  // Init object (second point type is for the normals, even if unused)
			  pcl::MovingLeastSquares<PointType, pcl::PointNormal> mls;
			  
			  mls.setComputeNormals (true);
			  
			  // Set Parameters
			  mls.setInputCloud (cloud);
			  mls.setPolynomialFit (true);
			  mls.setSearchMethod (tree);
			  mls.setSearchRadius (0.03);
			  
			  // Reconstruct
			  mls.process (mls_points);
			  
			  // Save output
			  pcl::copyPointCloud(mls_points, *scene);
		  }
		
	  //~ }

	  //
	  //  Set up resolution invariance
	  //
	  if (use_cloud_resolution_)
	  {
	    float resolution = static_cast<float> (computeCloudResolution (model));
	    if (resolution != 0.0f)
	    {
	      model_ss_   *= resolution;
	      scene_ss_   *= resolution;
	      rf_rad_     *= resolution;	
	      descr_rad_  *= resolution;
	      cg_size_    *= resolution;
	    }

	    std::cout << "Model resolution:       " << resolution << std::endl;
	    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
	    std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
	    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
	    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
	    std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
	  }

	  //
	  //  Compute Normals (default = NormalEstimationOMP)
	  //
	  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	  norm_est.setKSearch (10); // Default = 10
	  norm_est.setNumberOfThreads(8);
	  norm_est.setInputCloud (model);
	  norm_est.compute (*model_normals);

	  norm_est.setInputCloud (scene);
	  norm_est.compute (*scene_normals);
		
		  //~ std::vector<int> indices;
		  //~ pcl::PointCloud<PointType>::Ptr model1 (new pcl::PointCloud<PointType> ());
		  //~ pcl::removeNaNFromPointCloud(*model, *model1, indices);
		  //~ pcl::copyPointCloud(*model1, *model);	
		  //~ 
		  //~ pcl::PointCloud<PointType>::Ptr scene1 (new pcl::PointCloud<PointType> ());
		  //~ pcl::removeNaNFromPointCloud(*scene, *scene1, indices);
		  //~ pcl::copyPointCloud(*scene1, *scene);		  
		//~ 
		  //~ pcl::NormalEstimation<PointType, NormalType> norm_est;
		  //~ norm_est.setInputCloud(model);
		  //~ norm_est.setRadiusSearch(0.03);
		  //~ pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
		  //~ norm_est.setSearchMethod(kdtree);
		  //~ norm_est.compute(*model_normals);
		  //~ 
		  //~ norm_est.setInputCloud(scene);
		  //~ norm_est.compute(*scene_normals);
	  
	  //
	  //  Plane Segmentation Using OrganizedMultiPlaneSegmentation
	  //
	  //~ // Segment planes
	  //~ pcl::OrganizedMultiPlaneSegmentation< PointType, NormalType, pcl::Label > mps;
	  //~ mps.setMinInliers (100); // 10000 inliers
	  //~ mps.setAngularThreshold (0.017453 * 2.0); // 2 degrees
	  //~ mps.setDistanceThreshold (0.02); // 2cm
	  //~ mps.setInputNormals (scene_normals);
	  //~ mps.setInputCloud (scene);
	  //~ std::vector< pcl::PlanarRegion<PointType>, Eigen::aligned_allocator<pcl::PlanarRegion<PointType> > > regions;
	  //~ mps.segmentAndRefine (regions);
	  //~ ROS_INFO ("%d", regions.size());
	  //~ 
	  //~ for (size_t i = 0; i < regions.size (); i++)
	  //~ {
		//~ 
		//~ Eigen::Vector3f centroid = regions[i].getCentroid ();
	    //~ Eigen::Vector4f model = regions[i].getCoefficients ();
	    //~ pcl::PointCloud<PointType> boundary_cloud;
	    //~ boundary_cloud.width = 1;
		//~ boundary_cloud.points = regions[i].getContour ();
		//~ boundary_cloud.height = boundary_cloud.points.size();
		//~ printf ("Centroid: (%f, %f, %f)\n  Coefficients: (%f, %f, %f, %f)\n Inliers: %d\n",
			  //~ centroid[0], centroid[1], centroid[2],
			  //~ model[0], model[1], model[2], model[3],
			  //~ boundary_cloud.points.size ());
		//~ 
		//~ pcl::io::savePCDFile("test" + boost::lexical_cast<std::string>(i) + ".pcd", boundary_cloud);
	  //~ }
	  
	  //
	  //  Downsample Clouds to Extract keypoints (default = on is using SHOTEstimationOMP)
	  //
	  pcl::UniformSampling<PointType> uniform_sampling;
	  uniform_sampling.setInputCloud (model);
	  uniform_sampling.setRadiusSearch (model_ss_);
//	  uniform_sampling.filter (*model_keypoints); // Function no longer works - uncomment when fix is found
	  pcl::PointCloud<int> keypointIndices1;
	  uniform_sampling.compute(keypointIndices1);
	  pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints);
	  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " <<      model_keypoints->size () << std::endl;


	  uniform_sampling.setInputCloud (scene);
	  uniform_sampling.setRadiusSearch (scene_ss_);
//	  uniform_sampling.filter (*scene_keypoints); // Function no longer works - uncomment when fix is found
	  pcl::PointCloud<int> keypointIndices2;
	  uniform_sampling.compute(keypointIndices2);
	  pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints);
	  std::cout << "Scene total points: " << scene->size ()  << "; Selected Keypoints: " << scene_keypoints->size () << std::endl; 


	  //
	  // Removing NaN's
	  //
	  //~ std::vector<int> indices2;
	  //~ pcl::PointCloud<PointType>::Ptr model_keypoints2 (new pcl::PointCloud<PointType> ());
	  //~ pcl::PointCloud<PointType>::Ptr scene_keypoints2 (new pcl::PointCloud<PointType> ());
	  //~ pcl::PointCloud<NormalType>::Ptr model_normals2 (new pcl::PointCloud<NormalType> ());
	  //~ pcl::PointCloud<NormalType>::Ptr scene_normals2 (new pcl::PointCloud<NormalType> ());
	  //~ 
	  //~ pcl::removeNaNFromPointCloud(*model_keypoints, *model_keypoints2, indices2);
	  //~ pcl::removeNaNFromPointCloud(*model_normals, *model_normals2, indices2);
	  //~ 
	  //~ pcl::removeNaNFromPointCloud(*scene_keypoints, *scene_keypoints2, indices2);
	  //~ pcl::removeNaNFromPointCloud(*scene_normals, *scene_normals2, indices2);
	  
	   //
	  //  Compute Descriptor for keypoints
	  //
	  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	  //~ pcl::FPFHEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	  //~ pcl::OURCVFHEstimation<PointType, NormalType, DescriptorType> descr_est;
	  //~ 
	  //~ descr_est.setSearchMethod(kdtree);
	  //~ descr_est.setInputCloud(model);
	  //~ // descr_est.setInputCloud(model_keypoints);
	  //~ descr_est.setInputNormals(model_normals);
	  //~ descr_est.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees
	  //~ descr_est.setCurvatureThreshold(1.0);
	  //~ descr_est.setNormalizeBins(false); // True = scale does not matter (I think?)
	  //~ descr_est.setAxisRatio(0.8);
	  //~ descr_est.compute(*model_descriptors);
	  //~ 
	  //~ descr_est.setInputCloud(scene);
	  //~ // descr_est.setInputCloud(scene_keypoints);
	  //~ descr_est.setInputNormals(scene_normals);
	  //~ descr_est.compute(*scene_descriptors);
	  
	  //
	  // DEFAULT - SHOTEstimationOMP's Functions and Parameters
	  //
	  descr_est.setRadiusSearch (descr_rad_);

	  descr_est.setInputCloud (model_keypoints);
	  descr_est.setInputNormals (model_normals);
	  descr_est.setSearchSurface (model);
	  descr_est.compute (*model_descriptors);

	  descr_est.setInputCloud (scene_keypoints);
	  //~ descr_est.setInputCloud(scene);
	  descr_est.setInputNormals (scene_normals);
	  descr_est.setSearchSurface (scene);
	  descr_est.compute (*scene_descriptors);

	  
	  //
	  //  Find Model-Scene Correspondences with KdTree
	  //
	  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	  pcl::KdTreeFLANN<DescriptorType> match_search;
	  match_search.setInputCloud (model_descriptors);
	
	  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	  for (size_t i = 0; i < scene_descriptors->size (); ++i)
	  {
	    std::vector<int> neigh_indices (1);
	    std::vector<float> neigh_sqr_dists (1);
	    bool allow_match (true);
	    //~ for(size_t x = 0; x < scene_descriptors->at (i).descriptorSize() && allow_match; x++)
	    //~ {			
			if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs (SHOT352 uses .descriptor[0])
			{
				allow_match = false;
			}
		//~ }
		
		if (allow_match)
		{			
			int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
			if(found_neighs == 1  && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 = default (SHOT descriptor distances are between 0 and 1 by design)
			{
			  pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			  model_scene_corrs->push_back (corr);
			}
		}
	  }
	 
	  // Replacing above section of code with this in attempt to improve correspondences (failure)
	  //~ for (size_t i = 0; i < scene_descriptors->size (); ++i) 
	  //~ { 
		//~ std::vector<int> neigh_indices (10); // Default = 10 (nearest N features as correspondences)
		//~ std::vector<float> neigh_sqr_dists (10); // Default = 10 (nearest N features as correspondences)
		//~ if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs 
		//~ { 
		  //~ continue; 
		//~ } 
		//~ int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 10, neigh_indices, neigh_sqr_dists); 
		//~ for(int k = 0; k < found_neighs; k++) 
		//~ { 
			//~ if(found_neighs == 1 && neigh_sqr_dists[k] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design) 
			//~ { 
					//~ pcl::Correspondence corr (neigh_indices[k], static_cast<int> (i), neigh_sqr_dists[k]); 
					//~ model_scene_corrs->push_back (corr); 
			//~ } 
		//~ } 
	  //~ } 
	  
	  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
	  
	  if(model_scene_corrs->size () >= (model_keypoints->size() / 2)) // (model->size() / 2)) 
	  {
		ROS_INFO("Correlated model name: %s", model_filenames[num].c_str());
		cmodels_recognized.push_back(model_filenames[num].c_str());
	  }
	  

	  
	  //
	  //  Actual Clustering
	  //
	  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	  std::vector<pcl::Correspondences> clustered_corrs;

	  //  Using Hough3D
	  if (use_hough_)
	  {
	    //
	    //  Compute (Keypoints) Reference Frames only for Hough
	    //
	    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

	    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
	    rf_est.setFindHoles (true);
	    rf_est.setRadiusSearch (rf_rad_);

	    rf_est.setInputCloud (model_keypoints);
	    //~ rf_est.setInputCloud (model);
	    rf_est.setInputNormals (model_normals);
	    rf_est.setSearchSurface (model);
	    rf_est.compute (*model_rf);

	    rf_est.setInputCloud (scene_keypoints);
	    //~ rf_est.setInputCloud(scene);
	    rf_est.setInputNormals (scene_normals);
	    rf_est.setSearchSurface (scene);
	    rf_est.compute (*scene_rf);

	    //  Clustering
	    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
	    clusterer.setHoughBinSize (cg_size_);
	    clusterer.setHoughThreshold (cg_thresh_);
	    clusterer.setUseInterpolation (true);
	    clusterer.setUseDistanceWeight (false);

	    clusterer.setInputCloud (model_keypoints);
	    //~ clusterer.setInputCloud (model);
	    clusterer.setInputRf (model_rf);
	    clusterer.setSceneCloud (scene_keypoints);
	    //~ clusterer.setSceneCloud (scene);
	    clusterer.setSceneRf (scene_rf);
	    clusterer.setModelSceneCorrespondences (model_scene_corrs);

	    clusterer.cluster (clustered_corrs);
	    clusterer.recognize (rototranslations, clustered_corrs);
	  }
	  else // Using GeometricConsistency
	  {
	    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
	    gc_clusterer.setGCSize (cg_size_);
	    gc_clusterer.setGCThreshold (cg_thresh_);

	    gc_clusterer.setInputCloud (model_keypoints);
	    gc_clusterer.setSceneCloud (scene_keypoints);
	    //~ gc_clusterer.setInputCloud (model);
	    //~ gc_clusterer.setSceneCloud (scene);
	    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

	    gc_clusterer.cluster (clustered_corrs);
	    gc_clusterer.recognize (rototranslations, clustered_corrs);
	  }

	  //
	  //  Output results
	  //
	  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
	  for (size_t i = 0; i < rototranslations.size (); ++i)
	  {
	    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
	    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;
	     
	    // 
	    if(clustered_corrs[i].size() >= (model_keypoints->size() / 2)) // (model->size() / 2)) // (model_keypoints->size() / 2))
	    {
		ROS_INFO("Recognized model name: %s", model_filenames[num].c_str());
		models_recognized.push_back(model_filenames[num].c_str());
	    }	

	    // Print the rotation matrix and translation vector
	    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
	    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

	    printf ("\n");
	    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
	    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
	    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
	    printf ("\n");
	    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
	  }

	  //
	  //  Visualization
	  //
	  
	  if(turn_visualization_on)
	  {
		  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
		  viewer.addPointCloud (scene, "scene_cloud");

		  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
		  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

		  if (show_correspondences_ || show_keypoints_)
		  {
		    //  We are translating the model so that it doesn't end in the middle of the scene representation
		    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
		    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

		    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
		    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
		  }

		  if (show_keypoints_)
		  {
		    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
		    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
		    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

		    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
		    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
		    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
		  }

		  for (size_t i = 0; i < rototranslations.size (); ++i)
		  {
		    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
		    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

		    std::stringstream ss_cloud;
		    ss_cloud << "instance" << i;

		    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
		    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

		    if (show_correspondences_)
		    {
		      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
		      {
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
			PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
		      }

		    }
		  }

		  while (!viewer.wasStopped ())
		  {
		    viewer.spinOnce ();
		  }
	  }	  
	  
	  //~ if(turn_visualization_on)
	  //~ {
		  //~ 
	  //~ }

	  loop_rate.sleep();
  }

  ROS_INFO("Models correlated:"); 
  for(int modelsFound = 0; modelsFound < cmodels_recognized.size(); modelsFound++)
  {
	ROS_INFO("%s", cmodels_recognized[modelsFound].c_str());
  }

  ROS_INFO("Models recognized:");
  for(int modelsUsed = 0; modelsUsed < models_recognized.size(); modelsUsed++)
  {
	ROS_INFO("%s", models_recognized[modelsUsed].c_str());
  }
  return (0);
}
