#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Exercise-2 TODOs:

    ### Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    ### Voxel Grid Downsampling
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!
    LEAF_SIZE = 0.003

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    ### Statistical Outlier Removal Filter
    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()


    ### PassThrough Filter
    # Create a PassThrough filter object first across the Z axis
    passThroughZ = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passThroughZ.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passThroughZ.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passThroughZ.filter()

    ## Now, Create a PassThrough filter object across the Y axis
    passThroughY = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passThroughY.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passThroughY.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passThroughY.filter()

    ### RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    ### Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    ### Euclidean Clustering
    # Go from XYZRGB to RGB since to build the k-d tree we only needs spatial data
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    ### Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    # Refering to the minimum and maximum number of points that make up an object's cluster
    ec.set_MaxClusterSize(50000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    ### Create Cluster-Mask Point Cloud to visualize each cluster separately

    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO - set proper names    ### Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO   set names ### Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # Complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # labeled_features.append([feature, model_name])

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    print "Trying to match the pick list with the objects detected..."
    print "\n"
    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    object_list_param = rospy.get_param('/object_list')
    pick_list_objects = []
    for i in range(len(object_list_param)):
        pick_list_objects.append(object_list_param[i]['name'])

    print "\n"
    print "Pick List includes: "
    print pick_list_objects
    print "\n"
    pick_set_objects = set(pick_list_objects)
    detected_set_objects = set(detected_objects_labels)


    if detected_set_objects == pick_set_objects:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # Set which world we're in and our output file name. Make sure to also change the testworld# in pick_place_project.launch
    # at lines 13, 39
    test_scene_num = Int32()
    test_scene_num.data = 2
    outputFileName = "output_2.yaml"



    # Lets start by getting the pick list!
    # Retrieve the picklist from the parameter server (YAML files)
    object_list_param = rospy.get_param('/object_list')

    # Let's determine the location of the red and green boxes
    box_param = rospy.get_param('/dropbox')

    box_name = []
    box_group = []
    box_position = []
    # We'll loop through the two boxes
    for i in range(0, len(box_param)):
        box_name.append(box_param[i]['name'])
        box_group.append(box_param[i]['group'])
        box_position.append(box_param[i]['position'])



    # Get the PointCloud for a given object and obtain it's centroid
    # Compare the label with the pick list and provide the centroid


    # The way it will actually work within your code is that you'll iterate over each item in the pick list,
    # see whether you found it in your perception analysis, then if you did, populate the pick_pose message with the centroid.
    # Since you'll do this with each object one at a time, a convenient way to save the messages
    # for each object is in a list of dictionaries.


    # With each iteration over the pick list, you can create a dictionary with the above
    # function and then generate a list of dictionaries containing all your ROS service request messages.ie.e
    dict_list = []
    for i in range(0, len(object_list_param)):


        labels = []
        centroids = [] # to be list of tuples (x, y, z)
        for object in object_list:
            labels.append(object.label)
            points_arr = ros_to_pcl(object.cloud).to_array()
            temp = np.mean(points_arr, axis=0)[:3]
            centroids.append(temp)

        object_name = String()
        # This is getting the first object name and box from the picklist

        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        arm_name = String()
        place_pose = Pose()

        # arm_name - Right for Green Box, Left for Red Box
        if object_group == 'red':
            arm_name.data = 'left'
            place_pose.position.x = box_position[0][0]
            place_pose.position.y = box_position[0][1]
            place_pose.position.z = box_position[0][2]
        else:
            arm_name.data = 'right'
            place_pose.position.x = box_position[1][0]
            place_pose.position.y = box_position[1][1]
            place_pose.position.z = box_position[1][2]


        # pick_pose
        pick_pose = Pose()
        desired_object = object_list_param[i]['name']
        print "\n\nPicking up ", desired_object, "\n\n"
        print "\n\nPutting in ", object_group, "\n\n"


        # Match desired object with the centroid list/labels
        try:
            labelPosition = labels.index(desired_object)
            pick_pose.position.x = np.asscalar(centroids[labelPosition][0])
            pick_pose.position.y = np.asscalar(centroids[labelPosition][1])
            pick_pose.position.z = np.asscalar(centroids[labelPosition][2])
        except ValueError:
            continue

        # Populate various ROS messages
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    # Output your request parameters into output yaml file
    send_to_yaml(outputFileName, dict_list)



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscriber to bring in camera data (point cloud) from topic /pr2/world/points
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    # Publish to verify our above pcl_sub worked
    pcl_cluster_pub = rospy.Publisher("/pcl_world", PointCloud2, queue_size=1)
    # Publishers for the objects and the table
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    # Other publishers for the markers and labels
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model_worlds123_V2.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
