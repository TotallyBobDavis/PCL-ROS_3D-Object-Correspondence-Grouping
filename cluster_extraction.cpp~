#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <vector>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl_conversions/pcl_conversions.h>  
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

// MAKE ADJUSTMENTS TO PARAMETERS TO BETTER EXTRACT CLUSTERS
int main (int argc, char** argv) 
 {
    ros::init (argc, argv, "color_segmentation");
    ros::NodeHandle nh_;
    tf::TransformListener listener_;
    typedef pcl::PointXYZRGB Point;
    std::string processing_frame_="/camera_link";
    pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new   pcl::search::KdTree<pcl::PointXYZRGB>);

  std::string topic = nh_.resolveName("/nav_kinect/depth_registered/points");
  ROS_INFO(" waiting for a point_cloud2 on topic %s", topic.c_str());
  sensor_msgs::PointCloud2::ConstPtr recent_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(topic);
  sensor_msgs::PointCloud old_cloud;
  sensor_msgs::convertPointCloud2ToPointCloud (*recent_cloud, old_cloud);

  sensor_msgs::PointCloud2 converted_cloud;
  sensor_msgs::convertPointCloudToPointCloud2 (old_cloud, converted_cloud);
  pcl::PointCloud<Point>::Ptr cloud (new pcl::PointCloud<Point>);
 // pcl::PointCloud <pcl::Point>::Ptr cloud (new pcl::PointCloud <pcl::Point>);
   try
     {
     pcl::fromROSMsg(converted_cloud, *cloud);
     }
    catch (...)
    {
      ROS_ERROR("Failure while converting the ROS Message to PCL point cloud.  segmentation now requires the input cloud to have color info, so you must input a cloud with XYZRGB points, not just XYZ.");
     throw;
  }
  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 2.0);
  pass.filter (*indices);

  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud (cloud);
  reg.setIndices (indices);
  reg.setSearchMethod (tree);
  reg.setDistanceThreshold (10);
  reg.setPointColorThreshold (6);
  reg.setRegionColorThreshold (5);
  reg.setMinClusterSize (1000);
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  ROS_INFO ("Number of clusters found matching the given constraints: %d.", (int)clusters.size ());

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud (colored_cloud);
  pcl::io::savePCDFileASCII ("test_pcd.pcd", *colored_cloud);

  while (!viewer.wasStopped ())
     {
         boost::this_thread::sleep (boost::posix_time::microseconds (100));
     }

 // Cluster extraction 
  int j = 0;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin (); it != clusters.end (); ++it)
  {
      cloud_cluster->clear();
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
      cloud_cluster->points.push_back (cloud->points[*pit]); //*
      cloud_cluster->width = cloud_cluster->points.size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
      std::ostringstream ss;
      ss << j;
      pcl::PCDWriter writer5;
     writer5.write<pcl::PointXYZRGB> (ss.str()+".pcd", *cloud_cluster, false);
     std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
     j++;
 }
 return (0); 
}
