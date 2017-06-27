cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(correspondence_grouping)

find_package(catkin REQUIRED COMPONENTS
	actionlib
	actionlib_msgs
	move_base_msgs
	roscpp
	rospy
	tf
	std_msgs
 	geometry_msgs
  	turtlesim
  	cv_bridge
  	image_transport
  	segbot_arm_perception
	segbot_arm_manipulation
  	pcl_ros
  	pcl_conversions
)

find_package(PCL 1.8 REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_package(
	CATKIN_DEPENDS 
	segbot_arm_perception 
	segbot_arm_manipulation
)

add_executable (correspondence_grouping src/correspondence_grouping.cpp)
target_link_libraries (correspondence_grouping ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(correspondence_grouping ${catkin_EXPORTED_TARGETS})

add_executable (cluster_extraction src/cluster_extraction.cpp)
target_link_libraries (cluster_extraction ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (conditional_cluster_extraction src/conditional_cluster_extraction.cpp)
target_link_libraries (conditional_cluster_extraction ${PCL_LIBRARIES})

add_executable (normal_distributions_transform src/normal_distributions_transform.cpp)
target_link_libraries (normal_distributions_transform ${PCL_LIBRARIES})

