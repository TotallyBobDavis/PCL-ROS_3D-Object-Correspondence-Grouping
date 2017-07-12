Download the catkin_pkg 'correspondence_grouping' and place it into a workspace. Compile that workspace with catkin_make and
make sure to source the proper devel/setup.bash file. Afterwards, navigate to the directory inside the package and run the
following command:

rosrun correspondence_grouping correspondence_grouping

This will run the executable that contains the PCL algorithm.

In addition, if you wish to add or remove PCD models to be searched for, edit the correspondence_grouping.cpp file by opening
it and navigating to the parseCommandLine function. In there, search for the following line:

model_filenames.push_back(...);

Push back the model PCD filename as a string (do not add the working directory).
Add additional push_backs in order to search for additional models.
Make sure the PCD files are in the catkin_pkg directory.

Run the following commands in the terminal in order to install the necessary PCL libraries

sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo apt-get install libpcl-all

Video demonstration:

https://drive.google.com/open?id=0B_CEqZ5sdfH-SVZUZE5XdmpXaDQ

GitHub:

https://github.com/jeffreyhuang23/PCL-ROS_3D-Object-Correspondence-Grouping

To run the Jeffrey branch with the current PCD files, use the following command:

rosrun correspondence_grouping  correspondence_grouping -s -f -v -u --model_ss 0.0001 --scene_ss 0.0001 --min_scale 0.0005 --nr_octaves 3 --nr_scale_per_octaves 4 --min_contrast 0.0 --algorithm Hough --cg_size 0.0055 --cg_thresh 6 --icp_max_iter 500 --hv_inlier_th 0.0075 _file:=milk_diag 
