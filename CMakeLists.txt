cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(correspondence_grouping)

find_package(PCL 1.5 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	actionlib
	actionlib_msgs
	move_base_msgs
	roscpp
	rospy
	tf
	message_generation
	std_msgs
 	geometry_msgs
  	turtlesim
  	message_generation
  	cv_bridge
  	image_transport
  	pcl_ros
  	pcl_conversions
  	sensor_msgs
)

catkin_package(

)

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (correspondence_grouping correspondence_grouping.cpp)
target_link_libraries (correspondence_grouping ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (cluster_extraction cluster_extraction.cpp)
target_link_libraries (cluster_extraction ${PCL_LIBRARIES})

add_executable (conditional_cluster_extraction conditional_cluster_extraction.cpp)
target_link_libraries (conditional_cluster_extraction ${PCL_LIBRARIES})
