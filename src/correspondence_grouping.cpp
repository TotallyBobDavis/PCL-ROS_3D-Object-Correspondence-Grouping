#include <iostream>
#include <signal.h>	

#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h> 

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/normal_3d.h>

#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/obj_rec_ransac.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <pcl/common/transforms.h>

#include <pcl/console/parse.h>

#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/JointState.h>

#include "segbot_arm_perception/TabletopPerception.h"
#include <segbot_arm_manipulation/arm_utils.h>
#include "segbot_arm_manipulation/TabletopGraspAction.h"

#include <pcl/filters/filter.h>

#include <pcl/surface/mls.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/planar_region.h>

#include <pcl/impl/point_types.hpp>

#include <pcl/registration/icp.h>


typedef pcl::PointXYZ PointType; // Add and remove RGB as needed
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;

typedef pcl::SHOT352 DescriptorType;
//~ typedef pcl::SHOT1344 DescriptorType;
typedef pcl::VFHSignature308 GlobalDescriptorType;
//~ typedef pcl::FPFHSignature33 DescriptorType;

struct CloudStyle
{
    double r;
    double g;
    double b;
    double size;

    CloudStyle (double r,
                double g,
                double b,
                double size) :
        r (r),
        g (g),
        b (b),
        size (size)
    {
    }
};

CloudStyle style_white (255.0, 255.0, 255.0, 4.0);
CloudStyle style_red (255.0, 0.0, 0.0, 3.0);
CloudStyle style_green (0.0, 255.00, 0.0, 5.0);
CloudStyle style_cyan (93.0, 200.0, 217.0, 4.0);
CloudStyle style_violet (255.0, 0.0, 255.0, 8.0);

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

// Adjust the following parameters to best fit the scene (implementation of segmentation means model_ss == scene_ss under normal circumstances)
float model_ss_ (0.01f);	// (default = 0.01f)
float scene_ss_ (0.03f);	// (default = 0.03f) (scene segments out individual objects now - make model_ss == scene_ss?)
float rf_rad_ (0.015f);		// Suggestion: recommended increase for better accuracy (default = 0.015f)
float descr_rad_ (0.06f);	// Suggestion: adjust as needed (default = 0.02f) | Apparently 0.06 is the best search radius?
float cg_size_ (0.01f);		// The size of the cubes that are created around each keypoint during hough. If two points match also all the points inside of their bins will match, so the smaller the value the better match you will find.Suggestion: decrease for better accuracy (default = 0.01f)
float cg_thresh_ (5.0f);	// The number of good votes to identify the model. The higher the better the results; Suggestion: increase for better accuracy (default = 5.0f) 
int icp_max_iter_ (5);		// Suggestion: increase to allow more transformations (default = 5)
float icp_corr_distance_ (0.005f); 	
float hv_clutter_reg_ (5.0f); 	// Clutter Regularizer (default 5.0)
float hv_inlier_th_ (0.005f);	// Inlier threshold (default 0.005)
float hv_occlusion_th_ (0.01f);	// Occlusion threshold (default 0.01)
float hv_rad_clutter_ (0.03f);	// Clutter radius (default 0.03)
float hv_regularizer_ (3.0f);	// Regularizer value (default 3.0)
float hv_rad_normals_ (0.05);	// Normals radius (default 0.05)
bool hv_detect_clutter_ (true);	//	TRUE if clutter detect enabled (default true)
bool use_resampling_smoothing_ (false);
bool flip_model_scene (false);
bool use_uniform_sampling (true);
bool obj_rec_RANSAC (false);
bool show_issues (false);

float min_scale = 0.03f; 
int nr_octaves = 8; 
int nr_scales_per_octave = 8; 
float min_contrast = 0.03f; 

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
  std::cout << "     -f:					 Flip the scene and model (target and source)" << std::endl;
  std::cout << "     -v:					 Turn off visualization for simple correspondence" << std::endl;
  std::cout << "	 -u:					 Turn off uniform sampling (use SIFT instead)" << std::endl;
  std::cout << "     -o:					 Turn on ObjRecRANSAC" << std::endl;
  std::cout << "	 -e:					 Show issues and ideas" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
  std::cout << "     --icp_max_iter val:          ICP max iterations number (default " << icp_max_iter_ << ")" << std::endl;
  std::cout << "     --icp_corr_distance val:     ICP correspondence distance (default " << icp_corr_distance_ << ")" << std::endl << std::endl;
  std::cout << "     --hv_clutter_reg val:        Clutter Regularizer (default " << hv_clutter_reg_ << ")" << std::endl;
  std::cout << "     --hv_inlier_th val:          Inlier threshold (default " << hv_inlier_th_ << ")" << std::endl;
  std::cout << "     --hv_occlusion_th val:       Occlusion threshold (default " << hv_occlusion_th_ << ")" << std::endl;
  std::cout << "     --hv_rad_clutter val:        Clutter radius (default " << hv_rad_clutter_ << ")" << std::endl;
  std::cout << "     --hv_regularizer val:        Regularizer value (default " << hv_regularizer_ << ")" << std::endl;
  std::cout << "     --hv_rad_normals val:        Normals radius (default " << hv_rad_normals_ << ")" << std::endl;
  std::cout << "     --hv_detect_clutter val:     TRUE if clutter detect enabled (default " << hv_detect_clutter_ << ")" << std::endl << std::endl;
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
  if (pcl::console::find_switch (argc, argv, "-f"))
  {
    flip_model_scene = true;
  }
  if (pcl::console::find_switch (argc, argv, "-v"))
  {
    turn_visualization_on = false;
  }
  if (pcl::console::find_switch (argc, argv, "-u"))
  {
    use_uniform_sampling = false;
  }
  if (pcl::console::find_switch (argc, argv, "-o"))
  {
    obj_rec_RANSAC = true;
  }
  if (pcl::console::find_switch (argc, argv, "-e"))
  {
    show_issues = true;
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

  pcl::console::parse_argument (argc, argv, "--icp_max_iter", icp_max_iter_);
  pcl::console::parse_argument (argc, argv, "--icp_corr_distance", icp_corr_distance_);
  pcl::console::parse_argument (argc, argv, "--hv_clutter_reg", hv_clutter_reg_);
  pcl::console::parse_argument (argc, argv, "--hv_inlier_th", hv_inlier_th_);
  pcl::console::parse_argument (argc, argv, "--hv_occlusion_th", hv_occlusion_th_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_clutter", hv_rad_clutter_);
  pcl::console::parse_argument (argc, argv, "--hv_regularizer", hv_regularizer_);
  pcl::console::parse_argument (argc, argv, "--hv_rad_normals", hv_rad_normals_);
  pcl::console::parse_argument (argc, argv, "--hv_detect_clutter", hv_detect_clutter_);
  
  pcl::console::parse_argument (argc, argv, "--min_scale", min_scale);
  pcl::console::parse_argument (argc, argv, "--nr_octaves", nr_octaves);
  pcl::console::parse_argument (argc, argv, "--nr_scales_per_octave", nr_scales_per_octave);
  pcl::console::parse_argument (argc, argv, "--min_contrast", min_contrast);
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

// Deprecated code due to inclusion of TabletopPerception
//~ void 
//~ cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud)
//~ { 
  //~ pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
  //~ 
  //~ pcl::fromROSMsg (*cloud, *temp_cloud);
  //~ 
  //~ std::string prefix_ = "scene";
  //~ 
  //~ if ((temp_cloud->width * temp_cloud->height) == 0)
        //~ return;
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
//~ 
      	// pcl::io::savePCDFile (ss.str (), *temp_cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
	//~ pcl::io::savePCDFile(ss.str(), *temp_cloud);
//~ }

int
main (int argc, char *argv[])
{
  ros::init (argc, argv, "correspondence_grouping");
  ros::NodeHandle nh("~");
  signal(SIGINT, sig_handler);
  
  if (show_issues)
  {
	  ROS_INFO ("Issues and Ideas: \n");
	  ROS_INFO ("1) Models that capture two sides of an object that are different from the two sides seen in the scene do not correspond (Example: milk_2.pcd and milk_3.pcd are two different perspectives of a milk carton but do not correspond\n");
	  ROS_INFO ("2) Still fixing global descriptors (may need to switch to two stages: training and testing - that means global descriptor would no longer be viable)\n");
	  ROS_INFO ("3) What if we took 8 viewpoints of the object starting from the front, front-right diagonal, right side, back-right diagonal, back, back-left diagonal, left side, front-left diagonal?\n");
	  ROS_INFO ("4) Should I combine UniformSampling and SIFT (after finding SIFT keypoints, use UniformSampling for depth keypoint extraction)?\n");
	  ROS_INFO ("5) I think global descriptors (OUR-CVFH) can be used: I have the necessary object data to train it now.\n");
	  ROS_INFO ("6) Implementing RIFT (Rotation-Invariant Feature Transform) since it builds on SIFT. Scratch that idea, RIFT requires the models and scenes to have textures, which I do not implement.\n");
	  ROS_INFO ("7) UniformSampling is a keypoint detector (actually it's a filter) and SHOT is the descriptor of that keypoint."); 
  } 
   
  std::string filename;
  nh.getParam("file", filename); // DO NOT PUT QUOTES FOR THE PRIVATE PARAMETER! EXAMPLE: _file:=milk 
 
  filename += ".pcd";
  
  // Change the topic if it no longer utilizes nav_kinect and replace it with the new camera (deprecated due to use of table_object_detection_node)
  //~ sub = nh.subscribe<sensor_msgs::PointCloud2> ("/nav_kinect/depth_registered/points", 1, cloud_cb);
  //~ sub = nh.subscribe<sensor_msgs::PointCloud2> ("/xtion_camera/depth_registered/points", 100, cloud_cb);
  
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
  
  //~ //
  //~ //  Saving scene.pcd Without Using the Subscriber
  //~ //
  //~ pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
  //~ 
  //~ pcl::fromROSMsg (table_scene.cloud_clusters[largest_pc_index], *temp_cloud);
  
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
	  
	  pcl::PointCloud<pcl::PointNormal>::Ptr model_point_normals (new pcl::PointCloud<pcl::PointNormal> ());
	  pcl::PointCloud<pcl::PointNormal>::Ptr scene_point_normals (new pcl::PointCloud<pcl::PointNormal> ());
	  pcl::PointCloud<pcl::PointWithScale> model_sift;
	  pcl::PointCloud<pcl::PointWithScale> scene_sift;
	  pcl::PointCloud<PointType>::Ptr model_sift_xyz (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr scene_sift_xyz (new pcl::PointCloud<PointType> ());
	  
	  // Implementation of Global Descriptors (WIP - uncommented to continue working on it)	
	  pcl::PointCloud<GlobalDescriptorType>::Ptr model_global_descriptors (new pcl::PointCloud<GlobalDescriptorType> ());
	  pcl::PointCloud<GlobalDescriptorType>::Ptr scene_global_descriptors (new pcl::PointCloud<GlobalDescriptorType> ());
	  
	  //
	  //  Saving scene.pcd Without Using the Subscriber
	  //
	  pcl::PointCloud<PointType>::Ptr temp_cloud(new pcl::PointCloud<PointType>);
	  
	  pcl::fromROSMsg (table_scene.cloud_clusters[num], *temp_cloud);
	  
	  pcl::copyPointCloud(*temp_cloud, *scene);
	  pcl::io::savePCDFile("test.pcd", *scene);
	  
	  //
	  //  Load clouds
	  //
	  if (pcl::io::loadPCDFile ("/home/totallybobdavis/catkin_ws/src/correspondence_grouping_obj-pcds/" + model_filenames[num], *model) < 0)
	  {
	    std::cout << "Error loading model cloud." << std::endl;
	    showHelp (argv[0]);
	    return (-1);
	  }
	  
	  // Uncomment if using custom scene.pcd
	  //~ if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
	  //~ {
	    //~ std::cout << "Error loading scene cloud." << std::endl;
	    //~ showHelp (argv[0]);
	    //~ return (-1);
	  //~ }
	  
	  // MLS Surface Reconstruction To Smooth and Resample Noisy Data
	  if (use_resampling_smoothing_)
	  {
		  std::vector<int> indices;
		  pcl::PointCloud<PointType>::Ptr scene1 (new pcl::PointCloud<PointType> ());
		  pcl::PointCloud<PointType>::Ptr model1 (new pcl::PointCloud<PointType> ());
		  pcl::removeNaNFromPointCloud(*scene, *scene1, indices);		  
		  pcl::removeNaNFromPointCloud(*model, *model1, indices);
		  
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
		  pcl::io::savePCDFile("test_1.pcd", *scene);
		  
		  pcl::copyPointCloud(*model1, *cloud);
		  mls.setInputCloud (cloud);
		  mls.setPolynomialFit (true);
		  mls.setSearchMethod (tree);
		  mls.setSearchRadius (0.03);
		  
		  mls.process (mls_points);
		  
		  pcl::copyPointCloud(mls_points, *model);
		  pcl::io::savePCDFile("test_2.pcd", *model);
	  }
	
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
	  //  Switch Model and Scene (if model keypoints > scene keypoints)
	  //	
	  if (flip_model_scene)
	  {
		  pcl::PointCloud<PointType>::Ptr copy_cloud (new pcl::PointCloud<PointType> ());
		  pcl::copyPointCloud(*scene, *copy_cloud);
		  pcl::copyPointCloud(*model, *scene);
		  pcl::copyPointCloud(*copy_cloud, *model);
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
	  
	  if (!use_uniform_sampling)
	  {
		  pcl::NormalEstimation<PointType, pcl::PointNormal> ne;
		  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
		  
		  ne.setInputCloud(model);
		  ne.setSearchMethod(tree_n);
		  ne.setRadiusSearch(0.2);
		  ne.compute(*model_point_normals);
		  
		  //~ ROS_INFO("%f", model_point_normals->points[1000].x);
		  //~ ROS_INFO("%f", model->points[1000].x);
		  
		  for (size_t i = 0; i < model_point_normals->points.size(); ++i)
		  {
			  model_point_normals->points[i].x = model->points[i].x;
			  model_point_normals->points[i].y = model->points[i].y;
			  model_point_normals->points[i].z = model->points[i].z;
		  }
		  
		  ne.setInputCloud(scene);
		  ne.compute (*scene_point_normals);
		  //~ ROS_INFO("%f", scene_point_normals->points[1000].x);
		  //~ ROS_INFO("%f", scene->points[1000].x);
		  
		  for (size_t i = 0; i < scene_point_normals->points.size(); ++i)
		  {
			  scene_point_normals->points[i].x = scene->points[i].x;
			  scene_point_normals->points[i].y = scene->points[i].y;
			  scene_point_normals->points[i].z = scene->points[i].z;
		  }
		  
		  //~ ROS_INFO("%f", model_point_normals->points[1000].x);
		  //~ ROS_INFO("%f", scene_point_normals->points[1000].x);
	  }
	  
	  //
	  //  Downsample Clouds to Extract keypoints (default = on is using SHOTEstimationOMP)
	  //
	  if (use_uniform_sampling)
	  {
		  pcl::UniformSampling<PointType> uniform_sampling;
		  uniform_sampling.setInputCloud (model);
		  uniform_sampling.setRadiusSearch (model_ss_);
		  uniform_sampling.filter (*model_keypoints); // Function no longer works in PCL version 1.7.0 - uncomment when fix is found (Fix found: Update to PCL version 1.8.0)
		  
		  /* Deprecated in PCL version 1.8.0 (uncomment if using PCL version 1.7.0)
		  //~ pcl::PointCloud<int> keypointIndices1;
		  //~ uniform_sampling.compute(keypointIndices1);
		  //~ pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints);
		  */
		  
		  std::cout << "Model total points: " << model->size () << "; Selected Keypoints (after filtering): " << model_keypoints->size () << std::endl;


		  uniform_sampling.setInputCloud (scene);
		  uniform_sampling.setRadiusSearch (scene_ss_);
		  uniform_sampling.filter (*scene_keypoints); // Function no longer works in PCL version 1.7.0 - uncomment when fix is found (Fix found: Update to PCL version 1.8.0)
		  
		  /* Deprecated in PCL version 1.8.0 (uncomment if using PCL version 1.7.0)
		  //~ pcl::PointCloud<int> keypointIndices2;
		  //~ uniform_sampling.compute(keypointIndices2);
		  //~ pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints);
		  */
		  
		  std::cout << "Scene total points: " << scene->size ()  << "; Selected Keypoints (after filtering): " << scene_keypoints->size () << std::endl; 
	  }
	  else
	  {
		  ROS_INFO ("%f, %d, %d, %f", min_scale, nr_octaves, nr_scales_per_octave, min_contrast);
		  
		  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
		  
		  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
		  sift.setSearchMethod (tree);
		  sift.setScales (min_scale, nr_octaves, nr_scales_per_octave);
		  sift.setMinimumContrast (min_contrast);
		  
		  sift.setInputCloud (model_point_normals);
		  //~ sift.setRadiusSearch(model_ss_);
		  sift.compute (model_sift);
		  
		  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_sift.points.size () << std::endl;
		  
		  sift.setInputCloud (scene_point_normals);
		  //~ sift.setRadiusSearch(scene_ss_);
		  sift.compute (scene_sift);
		  
		  std::cout << "Scene total points: " << scene->size ()  << "; Selected Keypoints: " << scene_sift.points.size () << std::endl; 
		  
		  pcl::copyPointCloud(model_sift, *model_sift_xyz);
		  pcl::copyPointCloud(scene_sift, *scene_sift_xyz);
		  
		  pcl::UniformSampling<PointType> uniform_sampling;
		  uniform_sampling.setInputCloud (model_sift_xyz);	
		  uniform_sampling.setRadiusSearch (model_ss_);
		  uniform_sampling.filter (*model_keypoints); // Function no longer works in PCL version 1.7.0 - uncomment when fix is found (Fix found: Update to PCL version 1.8.0)
		  
		  /* Deprecated in PCL version 1.8.0 (uncomment if using PCL version 1.7.0)
		  //~ pcl::PointCloud<int> keypointIndices1;
		  //~ uniform_sampling.compute(keypointIndices1);
		  //~ pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints);
		  */
		  
		  std::cout << "Model total points: " << model->size () << "; Selected Keypoints (after filtering): " << model_keypoints->size () << std::endl;


		  uniform_sampling.setInputCloud (scene_sift_xyz);
		  uniform_sampling.setRadiusSearch (scene_ss_);
		  uniform_sampling.filter (*scene_keypoints); // Function no longer works in PCL version 1.7.0 - uncomment when fix is found (Fix found: Update to PCL version 1.8.0)
		  
		  /* Deprecated in PCL version 1.8.0 (uncomment if using PCL version 1.7.0)
		  //~ pcl::PointCloud<int> keypointIndices2;
		  //~ uniform_sampling.compute(keypointIndices2);
		  //~ pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints);
		  */
		  
		  std::cout << "Scene total points: " << scene->size ()  << "; Selected Keypoints (after filtering): " << scene_keypoints->size () << std::endl; 
	  }
	  
	  if (obj_rec_RANSAC)
	  {
		  double pairWidth_d = 30.0;
		  float voxelSize_f = 4.0;
		  double normalRadius_d = 15.0;
		  double successProbability_d = 0.99;
		  
		  pcl::recognition::ObjRecRANSAC RANSAC_recognition (pairWidth_d, voxelSize_f);
		  std::list<pcl::recognition::ObjRecRANSAC::Output> matches;
		  //~ pcl::io::loadPCDFile("/home/totallybobdavis/catkin_ws/src/correspondence_grouping/pcd_files/milk_diag.pcd", *temp_cloud);
		  RANSAC_recognition.addModel(*model, *model_normals, "milk_diag");
		  
		  RANSAC_recognition.recognize (*scene, *scene_normals, matches, successProbability_d);
		  std::cout << "Models Recognized: " << matches.size() << std::endl;
		  
	  }
	  
	   //
	  //  Compute Descriptor for keypoints
	  //
	  //~ pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est; // default local descriptor
	  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	  pcl::OURCVFHEstimation<PointType, NormalType, GlobalDescriptorType> global_descr_est;
	  //~ pcl::FPFHEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	  
	  //
	  //  OURCVFHEstimation's Functions and Parameters
	  //
	  pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
	  global_descr_est.setInputCloud(model);
	  global_descr_est.setInputNormals(model_normals);
	  global_descr_est.setSearchMethod(kdtree);
	  // global_descr_est.setInputCloud(model_keypoints);
	  global_descr_est.setEPSAngleThreshold(5 / 180.0 * M_PI); // default = 5 degrees
	  global_descr_est.setCurvatureThreshold(1.0);
	  global_descr_est.setMinPoints (100);
	  global_descr_est.setNormalizeBins(true); // True = scale does not matter (I think?)
	  global_descr_est.setAxisRatio(0.8);
	  global_descr_est.compute(*model_global_descriptors);
	  
	  //~ pcl::visualization::PCLHistogramVisualizer show_model;
	  //~ show_model.addFeatureHistogram(*model_global_descriptors, 308);
	  //~ show_model.spinOnce();
	  
	  global_descr_est.setInputCloud(scene);
	  // global_descr_est.setInputCloud(scene_keypoints);
	  global_descr_est.setInputNormals(scene_normals);
	  global_descr_est.compute(*scene_global_descriptors);
	  
	  //~ pcl::visualization::PCLHistogramVisualizer show_scene;
	  //~ show_scene.addFeatureHistogram(*scene_global_descriptors, 308);
	  //~ show_scene.spinOnce();
	  
	  //
	  // DEFAULT - SHOTEstimationOMP's Functions and Parameters
	  //
	  descr_est.setNumberOfThreads(8);
	  descr_est.setRadiusSearch (descr_rad_);
	  
	  //~ if (!use_uniform_sampling)
	  //~ {
		  //~ pcl::copyPointCloud(model_sift, *model_keypoints);
		  //~ pcl::copyPointCloud(scene_sift, *scene_keypoints);
	  //~ }
	  
	  descr_est.setInputCloud (model_keypoints); // Default
	  //~ descr_est.setInputCloud (model);
	  descr_est.setInputNormals (model_normals);
	  descr_est.setSearchSurface (model);
	  descr_est.compute (*model_descriptors);

	  descr_est.setInputCloud (scene_keypoints); // Default
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
	
	  // Part of the new implementation of Global Hypothesis Verification
	  std::vector<int> model_good_keypoints_indices; 
	  std::vector<int> scene_good_keypoints_indices;
	
	  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	  for (size_t i = 0; i < scene_descriptors->size (); ++i)
	  {
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0]))  //skipping NaNs
		{
		  continue;
		}
		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1) //  add match only if the squared descriptor distance is less than 0.25 = default (SHOT descriptor distances are between 0 and 1 by design)
		{
		  if ( (use_uniform_sampling && neigh_sqr_dists[0] < 0.25) || (!use_uniform_sampling && neigh_sqr_dists[0] < 0.8) )
		  {
			  pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			  model_scene_corrs->push_back (corr);
			  model_good_keypoints_indices.push_back (corr.index_query);
			  scene_good_keypoints_indices.push_back (corr.index_match);
		  }  
		}
	  }
	  
	  pcl::PointCloud<PointType>::Ptr model_good_kp (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr scene_good_kp (new pcl::PointCloud<PointType> ());
	  pcl::copyPointCloud (*model_keypoints, model_good_keypoints_indices, *model_good_kp); // Default
	  pcl::copyPointCloud (*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp); // Default
	  //~ pcl::copyPointCloud (*model, model_good_keypoints_indices, *model_good_kp);
	  //~ pcl::copyPointCloud (*scene, scene_good_keypoints_indices, *scene_good_kp);
	  
	  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
	  
	  //
	  //  Find Global Model-Scene With Kdtree
	  //  
	  pcl::CorrespondencesPtr model_scene_global_corrs (new pcl::Correspondences ());
	  pcl::KdTreeFLANN<GlobalDescriptorType> global_match_search;
	  global_match_search.setInputCloud (model_global_descriptors);
	  
	  // Part of the new implementation of Global Hypothesis Verification
	  std::vector<int> global_model_good_keypoints_indices; 
	  std::vector<int> global_scene_good_keypoints_indices;
	
	  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
	  ROS_INFO("Number of Global Descriptors in Model: %lu", model_global_descriptors->size ());
	  ROS_INFO("Number of Global Descriptors in Scene: %lu", scene_global_descriptors->size ());
	  for (size_t i = 0; i < scene_global_descriptors->size (); ++i)
	  {
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_global_descriptors->at (i).histogram[0]))  //skipping NaNs
		{	
		  continue;
		}
		int found_neighs = global_match_search.nearestKSearch (scene_global_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1) 
		{
		  pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
		  model_scene_global_corrs->push_back (corr);
		  global_model_good_keypoints_indices.push_back (corr.index_query);
		  global_scene_good_keypoints_indices.push_back (corr.index_match);
		}
	  }
	  
	  std::cout << "Global Correspondences found: " << model_scene_global_corrs->size () << std::endl;
	  
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

	    rf_est.setInputCloud (model_keypoints); // Default
	    //~ rf_est.setInputCloud (model);
	    rf_est.setInputNormals (model_normals);
	    rf_est.setSearchSurface (model);
	    rf_est.compute (*model_rf);

	    rf_est.setInputCloud (scene_keypoints); // Default
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

	    clusterer.setInputCloud (model_keypoints); // Default
	    //~ clusterer.setInputCloud (model);
	    clusterer.setInputRf (model_rf);
	    clusterer.setSceneCloud (scene_keypoints); // Default
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
	  //  Output results (Correspondence Grouping)
	  //
	  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
	  for (size_t i = 0; i < rototranslations.size (); ++i)
	  {
	    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
	    std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;
	     
	    
	    //~ if(clustered_corrs[i].size() >= (model_keypoints->size() / 2)) // (model->size() / 2)) // (model_keypoints->size() / 2))
	    //~ {
		//~ ROS_INFO("Recognized model name: %s", model_filenames[num].c_str());
		//~ models_recognized.push_back(model_filenames[num].c_str());
	    //~ }	

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
	  //  Correspondence Grouping Visualization
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
	  
	  /**
	   * Stop if no instances
	   */
	  if (rototranslations.size () <= 0)
	  {
		cout << "*** No instances found! ***" << endl;
		return (0);
	  }
	  else
	  {
		cout << "Recognized Instances: " << rototranslations.size () << endl << endl;
	  }
	  
	  /**
	   * Generates clouds for each instances found 
	   */
	  std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;

	  for (size_t i = 0; i < rototranslations.size (); ++i)
	  {
		pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
		pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
		instances.push_back (rotated_model);
	  }

	  /**
	   * ICP
	   */
	  std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
	   if (true)
	  {
		cout << "--- ICP ---------" << endl;

		for (size_t i = 0; i < rototranslations.size (); ++i)
		{
		  pcl::IterativeClosestPoint<PointType, PointType> icp;
		  icp.setMaximumIterations (icp_max_iter_);
		  icp.setMaxCorrespondenceDistance (icp_corr_distance_);
		  icp.setInputTarget (scene);
		  icp.setInputSource (instances[i]);
		  pcl::PointCloud<PointType>::Ptr registered (new pcl::PointCloud<PointType>);
		  icp.align (*registered);
		  registered_instances.push_back (registered);
		  cout << "Instance " << i << " ";
		  if (icp.hasConverged ())
		  {
			cout << "Aligned! Fitness Score: " << icp.getFitnessScore() << endl;
		  }
		  else
		  {
			cout << "Not Aligned! Fitness Score: " << icp.getFitnessScore() << endl;
		  }
		}

		cout << "-----------------" << endl << endl;
	  }

	  /**
	   * Hypothesis Verification
	   */
	  cout << "--- Hypotheses Verification ---" << endl;
	  std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses

	  pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;

	  GoHv.setSceneCloud (scene);  // Scene Cloud
	  GoHv.addModels (registered_instances, true);  //Models to verify
	
	  GoHv.setInlierThreshold (hv_inlier_th_);
	  GoHv.setOcclusionThreshold (hv_occlusion_th_);
	  GoHv.setRegularizer (hv_regularizer_);
	  GoHv.setRadiusClutter (hv_rad_clutter_);
	  GoHv.setClutterRegularizer (hv_clutter_reg_);
	  GoHv.setDetectClutter (hv_detect_clutter_);
	  GoHv.setRadiusNormals (hv_rad_normals_);

	  GoHv.verify ();
	  GoHv.getMask (hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

	  for (int i = 0; i < hypotheses_mask.size (); i++)
	  {
		if (hypotheses_mask[i])
		{
		  cout << "Instance " << i << " is GOOD! <---" << endl;
		}
		else
		{
		  cout << "Instance " << i << " is bad!" << endl;
		}
	  }
	  cout << "-------------------------------" << endl;

	  /**
	   *  Global Hypothesis Verification Visualization
	   */
	  pcl::visualization::PCLVisualizer viewer ("Hypotheses Verification");
	  viewer.addPointCloud (scene, "scene_cloud");

	  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
	  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

	  pcl::PointCloud<PointType>::Ptr off_model_good_kp (new pcl::PointCloud<PointType> ());
	  pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
	  pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
	  pcl::transformPointCloud (*model_good_kp, *off_model_good_kp, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));

	  if (show_keypoints_)
	  {
		CloudStyle modelStyle = style_white;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, modelStyle.r, modelStyle.g, modelStyle.b);
		viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, modelStyle.size, "off_scene_model");
	  }

	  if (show_keypoints_)
	  {
		CloudStyle goodKeypointStyle = style_violet;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> model_good_keypoints_color_handler (off_model_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
																										goodKeypointStyle.b);
		viewer.addPointCloud (off_model_good_kp, model_good_keypoints_color_handler, "model_good_keypoints");
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "model_good_keypoints");

		pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_good_keypoints_color_handler (scene_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
																										goodKeypointStyle.b);
		viewer.addPointCloud (scene_good_kp, scene_good_keypoints_color_handler, "scene_good_keypoints");
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "scene_good_keypoints");
	  }

	  for (size_t i = 0; i < instances.size (); ++i)
	  {
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;
		
		// Shows the false positives of the model
		//~ CloudStyle clusterStyle = style_red;
		//~ pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler (instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
		//~ viewer.addPointCloud (instances[i], instance_color_handler, ss_instance.str ());
		//~ viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str ());

		CloudStyle registeredStyles = hypotheses_mask[i] ? style_green : style_cyan;
		ss_instance << "_registered" << endl;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler (registered_instances[i], registeredStyles.r,
																									   registeredStyles.g, registeredStyles.b);
		viewer.addPointCloud (registered_instances[i], registered_instance_color_handler, ss_instance.str ());
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, registeredStyles.size, ss_instance.str ());
	  }

	  while (!viewer.wasStopped ())
	  {
		viewer.spinOnce ();
	  }

	  return (0);

	  loop_rate.sleep();
  }
  
  return (0);
}
