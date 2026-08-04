// KISS-ICP style map-based localization for Livox MID360 (LiDAR + IMU).
//
// - Prior map (PCD) -> static voxel hash map
// - Incoming scans: IMU-gyro deskew -> voxel downsample -> robust
//   point-to-point ICP against the map (adaptive correspondence threshold)
// - Between LiDAR frames the latest pose is propagated with gyro rotation +
//   constant body velocity and published at IMU rate.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "kiss_icp_localization/adaptive_threshold.hpp"
#include "kiss_icp_localization/bev_detector.hpp"
#include "kiss_icp_localization/registration.hpp"
#include "kiss_icp_localization/se3.hpp"
#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace kiss_loc {

struct ImuSample {
  double t;            // absolute time (s)
  Eigen::Vector3d w;   // bias-removed angular velocity, LiDAR frame (rad/s)
};

struct PendingScan {
  double t_begin = 0.0;  // header stamp (s)
  double t_end = 0.0;    // stamp of last point (s)
  double arrival_wall = 0.0;
  std::vector<Eigen::Vector3d> points;  // sensor frame
  std::vector<float> rel_time;          // per-point time since t_begin (s); empty if unknown
};

class LocalizationNode : public rclcpp::Node {
public:
  LocalizationNode() : Node("kiss_icp_localization") {
    declareParams();
    loadMap();
    initPoseFromParam();

    adaptive_ = std::make_unique<AdaptiveThreshold>(
        initial_threshold_, min_threshold_, min_motion_, adaptive_range_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 50);
    if (!loc_twist_topic_.empty())
      loc_twist_pub_ =
          create_publisher<geometry_msgs::msg::TwistStamped>(loc_twist_topic_, 50);
    aligned_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/kiss_loc/scan_aligned", 5);
    if (publish_2d_scan_)
      scan_2d_pub_ =
          create_publisher<sensor_msgs::msg::PointCloud2>(scan_2d_topic_, 5);
    if (publish_residual_cloud_)
      residual_pub_ =
          create_publisher<sensor_msgs::msg::PointCloud2>(residual_topic_, 5);
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/kiss_loc/map", rclcpp::QoS(1).transient_local());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_static_broadcaster_ =
        std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
    publishLidarStaticTf();  // static base_link -> lidar_frame (mount extrinsic)
    publishMapCloud();

    // Detection stage-2 off-track filter uses the track mask. A missing mask is
    // fatal because it's the only spatial filter (silent off-track false pos).
    if (detect_en_ && bev_params_.track_filter) {
      const std::string tp = resolveTrackMapPath();
      if (!track_mask_.Load(tp)) {
        RCLCPP_FATAL(get_logger(),
                     "track mask %s failed to load with detect_track_filter=true "
                     "— fix track_map_yaml or set detect_track_filter:=false",
                     tp.c_str());
        throw std::runtime_error("track mask load failed");
      }
      RCLCPP_INFO(get_logger(), "loaded detection track mask %s (dist_min %.2f m)",
                  tp.c_str(), bev_params_.track_dist_min);
    }

    if (detect_en_) {
      const TrackMask *track =
          (bev_params_.track_filter && track_mask_.Valid()) ? &track_mask_ : nullptr;
      detector_ = std::make_unique<BevDetector>(bev_params_, track);
      obstacle_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          obstacle_topic_, 5);
      det_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          detection_topic_, 5);
      obstacle_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
          obstacle_pose_topic_, 5);
      debug_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(debug_topic_, 5);
      RCLCPP_INFO(get_logger(),
                  "BEV detection enabled: res %.2f m, track_filter=%d "
                  "(dist_min %.2f), z-band [%.2f, %.2f] m (map frame)",
                  bev_params_.res, bev_params_.track_filter, bev_params_.track_dist_min,
                  bev_params_.z_min, bev_params_.z_max);
    }

    // Live tuning: rqt_reconfigure / `ros2 param set` on the detect_* (and
    // deskew) knobs takes effect on the next scan without a relaunch. The node
    // reads these into bev_params_ once at startup and copies them into the
    // detector, so without this callback a runtime set would be ignored.
    param_cb_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> &ps) {
          rcl_interfaces::msg::SetParametersResult r;
          r.successful = true;
          bool det = false;
          for (const auto &pm : ps) {
            const std::string &n = pm.get_name();
            if (n == "detect_z_min") { bev_params_.z_min = pm.as_double(); det = true; }
            else if (n == "detect_z_max") { bev_params_.z_max = pm.as_double(); det = true; }
            else if (n == "detect_track_dist_min") { bev_params_.track_dist_min = pm.as_double(); det = true; }
            else if (n == "detect_track_filter") { bev_params_.track_filter = pm.as_bool(); det = true; }
            else if (n == "detect_res") { bev_params_.res = pm.as_double(); det = true; }
            else if (n == "detect_eps") { bev_params_.eps = pm.as_double(); det = true; }
            else if (n == "detect_min_samples") { bev_params_.min_samples = pm.as_int(); det = true; }
            else if (n == "detect_min_cluster_cells") { bev_params_.min_cluster_cells = pm.as_int(); det = true; }
            else if (n == "detect_track_gate") { bev_params_.track_gate = pm.as_double(); det = true; }
            else if (n == "detect_moving_speed") { bev_params_.moving_speed = pm.as_double(); det = true; }
            else if (n == "detect_max_misses") { bev_params_.max_misses = pm.as_int(); det = true; }
            else if (n == "detect_self_filter") detect_self_filter_ = pm.as_bool();
            else if (n == "detect_self_x_min") detect_self_x_min_ = pm.as_double();
            else if (n == "detect_self_x_max") detect_self_x_max_ = pm.as_double();
            else if (n == "detect_self_y_min") detect_self_y_min_ = pm.as_double();
            else if (n == "detect_self_y_max") detect_self_y_max_ = pm.as_double();
            else if (n == "detect_deskew") detect_deskew_ = pm.as_bool();
            else if (n == "planar_prediction") planar_prediction_ = pm.as_bool();
            // Ground-band crop (2.5D / detection): live-tunable, config is the
            // source of truth (use_ground_yaml=false). crop_z_max is the 2.5D knob.
            else if (n == "crop_z_min") crop_z_min_ = pm.as_double();
            else if (n == "crop_z_max") crop_z_max_ = pm.as_double();
            else if (n == "crop_ground_offset") crop_h_ = pm.as_double();
            else if (n == "crop_ground_normal") {
              const auto v = pm.as_double_array();
              if (v.size() == 3) {
                crop_n_ = Eigen::Vector3d(v[0], v[1], v[2]);
                if (crop_n_.norm() > 1e-9) crop_n_.normalize();
                R_level_ =
                    Eigen::Quaterniond::FromTwoVectors(crop_n_, Eigen::Vector3d::UnitZ())
                        .toRotationMatrix();
              }
            }
            else if (n == "stamp_at_scan_end") stamp_at_scan_end_ = pm.as_bool();
          }
          if (det && detector_) detector_->SetParams(bev_params_);
          return r;
        });

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(200);
    pc2_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, sensor_qos,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { onPC2(msg); });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, sensor_qos,
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { onImu(msg); });
    if (use_odom_twist_) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          odom_twist_topic_, rclcpp::SensorDataQoS().keep_last(20),
          [this](const nav_msgs::msg::Odometry::SharedPtr msg) { onOdom(msg); });
    }
    if (use_initial_pose_topic_) {
      initpose_sub_ =
          create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
              "/initialpose", 5,
              [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr
                         msg) { onInitialPose(msg); });
    }

    if (band_2p5d_)
      RCLCPP_INFO(get_logger(),
                  "localization=2.5D: crop scan to height <= %.2f m above plane "
                  "n=(%.4f,%.4f,%.4f) off=%.4f (floor kept), then full voxel ICP "
                  "against the 3D map",
                  crop_z_max_, crop_n_.x(), crop_n_.y(), crop_n_.z(), crop_h_);
    else
      RCLCPP_INFO(get_logger(), "localization=3D: full scan into ICP (no band crop)");
    RCLCPP_INFO(get_logger(),
                "kiss_icp_localization ready: map %zu pts (voxel %.2f m), "
                "input PointCloud2 '%s', imu '%s' (imu_en=%d), odom_twist '%s'",
                map_.NumPoints(), map_voxel_size_, lidar_topic_.c_str(),
                imu_topic_.c_str(), imu_en_,
                use_odom_twist_ ? odom_twist_topic_.c_str() : "(none, CV)");
    if (use_initial_pose_topic_) {
      RCLCPP_INFO(get_logger(),
                  "waiting for IMU bias + a manual /initialpose (RViz 2D Pose "
                  "Estimate) before localization starts");
    }
  }

private:
  // ----------------------------- setup -----------------------------
  void declareParams() {
    // map_pcd_3d is the full-cloud voxel-ICP target (3D + 2.5D) and the dir
    // anchor for deriving the sibling track raster (map_track).
    map_pcd_3d_ = declare_parameter<std::string>("map_pcd_3d", "");
    map_voxel_size_ = declare_parameter<double>("map_voxel_size", 0.5);
    map_max_points_ = declare_parameter<int>("map_max_points_per_voxel", 30);
    // Added to every loaded map point's z. GLIM references the map z=0 to the
    // mapping/sensor height, so the ground (and thus base_link) sits at negative z
    // below the 2D /map grid (drawn at z=0). Set to |base_link z| to lift the map
    // ground to z=0 -> localization output lands on the grid. (x,y,yaw unaffected.)
    sensor_height_ = declare_parameter<double>("sensor_height", 0.0);
    // point-to-plane when the map PCD carries normals (fast_livo save_map does)
    use_normals_ = declare_parameter<bool>("use_normals", true);
    scan_voxel_size_ = declare_parameter<double>("scan_voxel_size", 0.35);
    min_range_ = declare_parameter<double>("min_range", 0.3);
    max_range_ = declare_parameter<double>("max_range", 60.0);
    point_filter_num_ = declare_parameter<int>("point_filter_num", 1);

    // ---- 2D / 3D localization + ground-band geometry ----
    // input_mode: "3d" = full scan into voxel ICP | "2.5d"/"2p5d" = keep only
    // points with height <= crop_z_max above the ground plane (floor kept), then
    // the same voxel ICP. Both match against the 3D map (map_pcd_3d).
    input_mode_ = declare_parameter<std::string>("input_mode", "3d");
    band_2p5d_ = (input_mode_ == "2.5d" || input_mode_ == "2p5d");
    detect_en_ = declare_parameter<bool>("detect_en", false);  // needed by ground-yaml check
    map_pcd_path_ = map_pcd_3d_;
    if (map_pcd_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "map_pcd_3d is empty — set it in the config");
      throw std::runtime_error("active map path not set");
    }
    auto cn = declare_parameter<std::vector<double>>("crop_ground_normal", {0.0, 0.0, 1.0});
    crop_n_ = (cn.size() == 3) ? Eigen::Vector3d(cn[0], cn[1], cn[2])
                               : Eigen::Vector3d(0.0, 0.0, 1.0);
    if (crop_n_.norm() > 1e-9) crop_n_.normalize();
    crop_h_ = declare_parameter<double>("crop_ground_offset", 0.0);
    crop_z_min_ = declare_parameter<double>("crop_z_min", 0.05);
    crop_z_max_ = declare_parameter<double>("crop_z_max", 0.30);
    // Ground-band crop geometry comes ONLY from the crop_* params above (this
    // config) and is live-tunable — no ground_lidar.yaml. `ground_yaml` is still
    // declared (harmless) so the launch-injected param doesn't error; it's unused.
    ground_yaml_ = declare_parameter<std::string>("ground_yaml", "");

    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/livox/lidar");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/livox/imu");
    // Wheel-odometry topic whose twist drives the TRANSLATION prediction (gyro
    // drives rotation). Empty -> translation falls back to the ICP-derived
    // constant-velocity estimate. Set "/vesc/odom" on the unicorn car.
    odom_twist_topic_ = declare_parameter<std::string>("odom_twist_topic", "");
    use_odom_twist_ = !odom_twist_topic_.empty();
    // livox driver with use_system_timestamp stamps the header with now() at
    // publish time, i.e. at the END of the 100 ms accumulation window
    stamp_at_scan_end_ = declare_parameter<bool>("stamp_at_scan_end", true);

    imu_en_ = declare_parameter<bool>("imu_en", true);
    deskew_en_ = declare_parameter<bool>("deskew_en", true);
    // The deskew translation ramps v linearly over the 100 ms frame (a = dv/dt)
    // and adds the 0.5*a*tau^2 term — fixes floor/cloud doubling under hard
    // accel/brake. Always applied (a no-op at constant v, clamped to max_accel);
    // no separate toggle.
    imu_rate_odom_ = declare_parameter<bool>("imu_rate_odom", true);
    // Constrain IMU/odom propagation to the map XY plane (+ yaw): z/roll/pitch are
    // held at the last LiDAR fix and set ONLY by the 3D ICP. Stops the high-speed
    // z-sink (pitched forward velocity leaking into world z). Live-tunable.
    planar_prediction_ = declare_parameter<bool>("planar_prediction", false);
    imu_init_samples_ = declare_parameter<int>("imu_init_samples", 100);
    auto r_il = declare_parameter<std::vector<double>>(
        "extrinsic_R_il", {1, 0, 0, 0, 1, 0, 0, 0, 1});
    if (r_il.size() == 9) {
      R_il_ = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r_il.data());
    } else {
      R_il_.setIdentity();
    }

    max_iterations_ = declare_parameter<int>("max_iterations", 50);
    convergence_eps_ = declare_parameter<double>("convergence_eps", 2e-3);
    initial_threshold_ = declare_parameter<double>("initial_threshold", 1.0);
    min_threshold_ = declare_parameter<double>("min_threshold", 0.1);
    min_motion_ = declare_parameter<double>("min_motion", 0.05);
    // characteristic range for the rotation term of the adaptive threshold.
    // Tightening this (e.g. to the true indoor scene scale ~15 m) measurably
    // LOSES map-lock during aggressive driving — the inflated threshold is
    // robustness margin, keep it at sensor max range like upstream KISS-ICP
    adaptive_range_ = declare_parameter<double>("adaptive_range", 60.0);
    vel_smoothing_ = declare_parameter<double>("vel_smoothing", 0.3);
    // Localization-derived body twist (linear v + angular w from consecutive
    // scan fixes), published on its own topic for downstream consumers (MPC).
    // Independent of the wheel-odom twist (use_odom_twist / v_body_) — this is
    // what the *pose estimate* says the car is doing. EMA-smoothed: per-fix
    // pose jitter of ~cm at 10 Hz otherwise turns into ~0.1 m/s twist noise.
    loc_twist_topic_ =
        declare_parameter<std::string>("loc_twist_topic", "/kiss_loc/twist");
    loc_twist_smoothing_ = declare_parameter<double>("loc_twist_smoothing", 0.5);
    // reject_trans/reject_rot_deg/reject_recover_count and max_velocity/max_accel
    // are hardcoded constants (see members) — robustness bounds that never needed
    // tuning; not exposed as params.

    initial_pose_ = declare_parameter<std::vector<double>>(
        "initial_pose", {0, 0, 0, 0, 0, 0});  // x y z roll pitch yaw
    use_initial_pose_topic_ = declare_parameter<bool>("use_initial_pose_topic", true);

    print_stats_ = declare_parameter<bool>("print_stats", false);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    // LiDAR position in base_link (rear axle). ICP gives map<-lidar; the published
    // map<-base_link applies this mount offset: translation from here, rotation =
    // R_level_ (the ground-normal mount tilt, yaw 0). Default matches the 2D laser.
    {
      const auto t_lb =
          declare_parameter<std::vector<double>>("lidar_xyz_in_base", {0.27, 0.0, 0.11});
      if (t_lb.size() == 3) lidar_in_base_ = Eigen::Vector3d(t_lb[0], t_lb[1], t_lb[2]);
    }
    // Frame the LiDAR publishes in; the node broadcasts a static base_link->this TF
    // (the mount extrinsic) so the raw cloud is placed in the map tree.
    lidar_frame_ = declare_parameter<std::string>("lidar_frame", "livox_frame");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/kiss_loc/odometry");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    publish_aligned_scan_ = declare_parameter<bool>("publish_aligned_scan", true);
    // 2D-flattened aligned scan (/kiss_loc/scan_2d): the ground-band slab of the
    // aligned scan flattened to z=0 in the map frame, a debug overlay on /map.
    publish_2d_scan_ = declare_parameter<bool>("publish_2d_scan", true);
    scan_2d_topic_ = declare_parameter<std::string>("scan_2d_topic", "/kiss_loc/scan_2d");

    // Per-point scan-to-map residual, thresholded: points the static prior map
    // can't explain (opponents, loose obstacles, unmapped clutter). Reuses the
    // correspondences AlignScanToMap already computes on its last ICP iteration
    // (see out_residuals in registration.hpp) — no extra map queries. A point
    // with no map neighbor within that iteration's adaptive search radius counts
    // as maximally novel (clamped to that radius) rather than being silently
    // dropped. residual_threshold is re-read from the parameter server every
    // scan, so it's rqt/`ros2 param set` tunable live with no extra plumbing.
    publish_residual_cloud_ = declare_parameter<bool>("publish_residual_cloud", false);
    residual_topic_ =
        declare_parameter<std::string>("residual_topic", "/kiss_loc/high_residual");
    declare_parameter<double>("residual_threshold", 0.15);

    // ---- BEV object detection (unmapped statics + opponents) ----
    // detect_en_ already declared above (ground-yaml fail-fast needs it).
    BevParams bp;
    bp.res = declare_parameter<double>("detect_res", 0.2);
    // false -> feed the raw (non-deskewed) scan to detection; see processScan.
    detect_deskew_ = declare_parameter<bool>("detect_deskew", true);
    // Ground/ceiling removal is a fixed map-frame z band (no RANSAC): keep
    // detect_z_min <= z <= detect_z_max. Below z_min is floor, above z_max is
    // ceiling/overhang. Independent of the localization input crop
    // (crop_z_min_/crop_z_max_) and of sensor_height. Defaults disabled.
    bp.z_min = declare_parameter<double>("detect_z_min", -1e9);
    bp.z_max = declare_parameter<double>("detect_z_max", 1e9);
    bp.eps = declare_parameter<double>("detect_eps", 0.2);
    bp.min_samples = declare_parameter<int>("detect_min_samples", 4);
    bp.min_cluster_cells = declare_parameter<int>("detect_min_cluster_cells", 2);
    bp.track_gate = declare_parameter<double>("detect_track_gate", 1.0);
    bp.moving_speed = declare_parameter<double>("detect_moving_speed", 0.5);
    bp.max_misses = declare_parameter<int>("detect_max_misses", 5);
    // stage-2 subtraction (erosion): reject points within detect_track_dist_min
    // of the nearest non-track (wall/off-track) cell (GLIM map_track). Empty
    // track_map_yaml -> <map dir>/map_track.yaml.
    bp.track_filter = declare_parameter<bool>("detect_track_filter", false);
    bp.track_dist_min = declare_parameter<double>("detect_track_dist_min", 0.2);
    // ego self-return box (sensor frame, -x = behind the car); see member decl.
    detect_self_filter_ = declare_parameter<bool>("detect_self_filter", false);
    detect_self_x_min_ = declare_parameter<double>("detect_self_x_min", -0.5);
    detect_self_x_max_ = declare_parameter<double>("detect_self_x_max", -0.1);
    detect_self_y_min_ = declare_parameter<double>("detect_self_y_min", -0.15);
    detect_self_y_max_ = declare_parameter<double>("detect_self_y_max", 0.15);
    track_map_path_ = declare_parameter<std::string>("track_map_yaml", "");
    bev_params_ = bp;
    arrow_scale_ = declare_parameter<double>("detect_arrow_scale", 0.5);
    // output topics (yaml-configurable): foreground/obstacle cloud + detection
    // markers. Defaults preserve the historical /kiss_loc/* names.
    obstacle_topic_ =
        declare_parameter<std::string>("detect_obstacle_topic", "/kiss_loc/obstacles");
    detection_topic_ =
        declare_parameter<std::string>("detect_marker_topic", "/kiss_loc/detections");
    // DBSCAN cluster centers as PoseArray (map frame) — e.g. feed MPCC's
    // /external_obstacles. Each pose.position = cluster bbox center.
    obstacle_pose_topic_ =
        declare_parameter<std::string>("detect_pose_topic", "/kiss_loc/obstacle_poses");
    // debug: every scan point tagged with which filter stage dropped it
    // (intensity-coded, see FilterStage) -- only published while subscribed.
    debug_topic_ =
        declare_parameter<std::string>("detect_debug_topic", "/kiss_loc/detect_debug");
  }

  // detection mask (= 2D track raster) yaml: explicit track_map_yaml, else
  // <active map dir>/map_track.yaml. Used by the BEV stage-2 off-track filter.
  std::string resolveTrackMapPath() const {
    if (!track_map_path_.empty()) return track_map_path_;
    if (map_pcd_path_.empty()) return "";
    const auto slash = map_pcd_path_.find_last_of('/');
    const std::string dir =
        (slash == std::string::npos) ? "" : map_pcd_path_.substr(0, slash + 1);
    return dir + "map_track.yaml";
  }

  void loadMap() {
    // map_pcd_path_ is guaranteed non-empty (constructor fatals otherwise).
    pcl::PointCloud<pcl::PointNormal> cloud;  // missing normal fields load as 0
    if (pcl::io::loadPCDFile<pcl::PointNormal>(map_pcd_path_, cloud) < 0) {
      RCLCPP_FATAL(get_logger(), "failed to load map PCD: %s", map_pcd_path_.c_str());
      throw std::runtime_error("failed to load map PCD");
    }
    std::vector<Eigen::Vector3d> pts, normals;
    pts.reserve(cloud.size());
    normals.reserve(cloud.size());
    size_t n_valid_normals = 0;
    for (const auto &p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      pts.emplace_back(p.x, p.y, p.z + sensor_height_);
      normals.emplace_back(p.normal_x, p.normal_y, p.normal_z);
      if (normals.back().allFinite() && normals.back().norm() > 0.5)
        ++n_valid_normals;
    }
    const bool want_normals = use_normals_;
    const bool normals_ok =
        want_normals && n_valid_normals > pts.size() / 2;
    map_ = VoxelHashMap(map_voxel_size_, map_max_points_);
    map_.Build(pts, normals_ok ? normals : std::vector<Eigen::Vector3d>{});
    if (want_normals && !map_.HasNormals()) {
      // fast_livo save_map writes zero normals — estimate per-voxel by PCA
      const auto t0 = std::chrono::steady_clock::now();
      map_.EstimateNormals();
      RCLCPP_INFO(get_logger(), "estimated map normals by voxel PCA (%.1f s)",
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count());
    }
    map_cloud_raw_ = std::move(pts);
    RCLCPP_INFO(get_logger(),
                "loaded map %s: %zu raw pts -> %zu in voxel map (%s)",
                map_pcd_path_.c_str(), map_cloud_raw_.size(), map_.NumPoints(),
                map_.HasNormals() ? "point-to-plane" : "point-to-point");
  }

  void initPoseFromParam() {
    T_ = Eigen::Isometry3d::Identity();
    if (initial_pose_.size() == 6) {
      T_.translation() =
          Eigen::Vector3d(initial_pose_[0], initial_pose_[1], initial_pose_[2]);
      T_.linear() =
          (Eigen::AngleAxisd(initial_pose_[5], Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(initial_pose_[4], Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(initial_pose_[3], Eigen::Vector3d::UnitX()))
              .toRotationMatrix();
    }
    // R_level_ levels the ground normal (crop_n_) onto the map vertical — the
    // LiDAR mount tilt used for the base<->lidar extrinsic (see publishOdom).
    R_level_ = Eigen::Quaterniond::FromTwoVectors(crop_n_, Eigen::Vector3d::UnitZ())
                   .toRotationMatrix();
    T_prop_ = T_;
  }

  void publishMapCloud() {
    // Same voxel size as the matching map (map_voxel_size) -- one knob instead
    // of two, and the viz cloud now matches what ICP actually sees.
    auto ds = VoxelDownsample(map_cloud_raw_, map_voxel_size_);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(ds.size());
    for (const auto &p : ds) cloud.emplace_back(p.x(), p.y(), p.z());
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = now();
    map_pub_->publish(msg);
  }

  // --------------------------- callbacks ---------------------------
  // scan.t_begin holds the header stamp on entry; place the scan window
  // around it according to where the driver anchors the stamp
  void finalizeScanTimes(PendingScan &scan, float max_rel) const {
    const double header_t = scan.t_begin;
    if (stamp_at_scan_end_ && max_rel > 0.0f) {
      scan.t_begin = header_t - max_rel;
      scan.t_end = header_t;
    } else {
      scan.t_end = header_t + max_rel;
    }
  }

  void onPC2(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    PendingScan scan;
    scan.t_begin = rclcpp::Time(msg->header.stamp).seconds();

    // optional per-point time field
    enum class TimeField { kNone, kOffsetU32Ns, kTimeF32, kTimestampF64 };
    TimeField tfield = TimeField::kNone;
    for (const auto &f : msg->fields) {
      if (f.name == "offset_time" &&
          f.datatype == sensor_msgs::msg::PointField::UINT32)
        tfield = TimeField::kOffsetU32Ns;
      else if (f.name == "time" && f.datatype == sensor_msgs::msg::PointField::FLOAT32)
        tfield = TimeField::kTimeF32;
      else if (f.name == "timestamp" &&
               f.datatype == sensor_msgs::msg::PointField::FLOAT64)
        tfield = TimeField::kTimestampF64;
    }

    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y"),
        iz(*msg, "z");
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<uint32_t>> it_off;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_time;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<double>> it_ts;
    if (tfield == TimeField::kOffsetU32Ns)
      it_off = std::make_unique<sensor_msgs::PointCloud2ConstIterator<uint32_t>>(
          *msg, "offset_time");
    else if (tfield == TimeField::kTimeF32)
      it_time =
          std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "time");
    else if (tfield == TimeField::kTimestampF64)
      it_ts = std::make_unique<sensor_msgs::PointCloud2ConstIterator<double>>(
          *msg, "timestamp");

    float max_rel = 0.0f;
    size_t i = 0;
    double ts0 = std::numeric_limits<double>::quiet_NaN();
    for (; ix != ix.end(); ++ix, ++iy, ++iz, ++i) {
      float rt = 0.0f;
      if (it_off) {
        rt = static_cast<float>(**it_off) * 1e-9f;
        ++(*it_off);
      } else if (it_time) {
        rt = **it_time;
        ++(*it_time);
      } else if (it_ts) {
        // livox driver: absolute device time (ns) — may differ from the
        // header stamp (system time), so take it relative to the first point
        double v = **it_ts;
        ++(*it_ts);
        if (v > 1e12) v *= 1e-9;  // ns -> s
        if (std::isnan(ts0)) ts0 = v;
        rt = static_cast<float>(v - ts0);
      }
      if (point_filter_num_ > 1 && static_cast<int>(i % point_filter_num_) != 0)
        continue;
      if (!keepPoint(*ix, *iy, *iz)) continue;
      scan.points.emplace_back(*ix, *iy, *iz);
      if (tfield != TimeField::kNone) {
        scan.rel_time.push_back(rt);
        max_rel = std::max(max_rel, rt);
      }
    }
    finalizeScanTimes(scan, max_rel);
    enqueueScan(std::move(scan));
  }

  // Wheel odometry: its body twist drives the TRANSLATION prediction (gyro keeps
  // rotation). Overwrites v_body_ directly; the ICP-derived velocity update in
  // processScan is skipped when use_odom_twist_ is set. Single-threaded executor
  // serialises this with processScan/onImu, so the shared v_body_ needs no lock.
  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    v_body_.x() = msg->twist.twist.linear.x;
    v_body_.y() = msg->twist.twist.linear.y;  // ~0 on a car
    v_body_.z() = 0.0;
  }

  bool keepPoint(float x, float y, float z) const {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const double r2 = double(x) * x + double(y) * y + double(z) * z;
    if (r2 <= min_range_ * min_range_ || r2 >= max_range_ * max_range_) return false;
    if (band_2p5d_) {
      // 2.5D: drop everything above crop_z_max (dynamic clutter / ceiling); KEEP
      // the floor (no z_min) so the ground constrains z/roll/pitch in the ICP.
      const double hgt = crop_n_.x() * x + crop_n_.y() * y + crop_n_.z() * z + crop_h_;
      if (hgt > crop_z_max_) return false;
    }
    return true;
  }

  void onImu(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    const Eigen::Vector3d w_raw(msg->angular_velocity.x, msg->angular_velocity.y,
                                msg->angular_velocity.z);

    if (!bias_ready_) {
      bias_acc_.push_back(w_raw);
      if (static_cast<int>(bias_acc_.size()) >= imu_init_samples_) {
        gyro_bias_.setZero();
        for (const auto &w : bias_acc_) gyro_bias_ += w;
        gyro_bias_ /= bias_acc_.size();
        bias_acc_.clear();
        bias_ready_ = true;
        RCLCPP_INFO(get_logger(), "gyro bias initialized: [%.5f %.5f %.5f] rad/s",
                    gyro_bias_.x(), gyro_bias_.y(), gyro_bias_.z());
      }
      return;
    }

    // angular velocity in LiDAR frame
    const Eigen::Vector3d w = R_il_.transpose() * (w_raw - gyro_bias_);
    if (!imu_buf_.empty() && t <= imu_buf_.back().t) return;  // out-of-order
    imu_buf_.push_back({t, w});
    while (!imu_buf_.empty() && imu_buf_.front().t < t - 10.0) imu_buf_.pop_front();

    // high-rate propagated odometry between LiDAR frames
    if (imu_rate_odom_ && have_first_fix_) {
      const double dt = t - t_prop_;
      if (t_prop_ > 0.0 && dt > 0.0 && dt < 0.1) {
        T_prop_.linear() = T_prop_.rotation() * So3Exp(w * dt);
        T_prop_.translation() += T_prop_.rotation() * (v_body_ * dt);
        // planar propagation: freeze z/roll/pitch to the last LiDAR fix (T_) so
        // the between-scan estimate advances only in the map XY plane + yaw.
        if (planar_prediction_) T_prop_ = holdZRP(T_prop_, T_);
        publishOdom(T_prop_, t);
      }
      t_prop_ = t;
    }

    processPending();
  }

  void onInitialPose(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    const auto &p = msg->pose.pose.position;
    const auto &q = msg->pose.pose.orientation;
    T.translation() = Eigen::Vector3d(p.x, p.y, p.z);
    T.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
    // /initialpose is a base_link pose (RViz), but the node tracks the LiDAR pose
    // T_. Convert: T_lidar = (map<-base) * (base<-lidar), the inverse of the mount
    // extrinsic publishOdom applies — so the published base lands where clicked.
    Eigen::Isometry3d T_base_lidar = Eigen::Isometry3d::Identity();
    T_base_lidar.linear() = R_level_;
    T_base_lidar.translation() = lidar_in_base_;
    T = T * T_base_lidar;
    // RViz 2D Pose Estimate has z = 0; keep current z to stay on the map floor
    if (have_first_fix_) T.translation().z() = T_.translation().z();
    T_ = T;
    T_prop_ = T;
    v_body_.setZero();
    adaptive_->Reset();
    const bool first_time = !have_manual_initial_pose_;
    have_manual_initial_pose_ = true;
    RCLCPP_WARN(get_logger(), "%s from /initialpose: [%.2f %.2f %.2f]",
                first_time ? "starting localization, anchored" : "re-anchored",
                T.translation().x(), T.translation().y(), T.translation().z());
    // re-publish the map cloud so RViz (and any late subscriber) refreshes
    publishMapCloud();
  }

  // ------------------------- scan processing -------------------------
  void enqueueScan(PendingScan &&scan) {
    if (scan.points.empty()) return;
    scan.arrival_wall = nowSec();
    pending_.push_back(std::move(scan));
    while (pending_.size() > 20) {
      pending_.pop_front();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "scan queue overflow, dropping oldest");
    }
    processPending();
  }

  double nowSec() { return get_clock()->now().seconds(); }

  void processPending() {
    if (imu_en_ && !bias_ready_) {
      // gyro bias still initializing — scans from this period are useless
      // (no deskew, no prediction) and the platform should be static anyway
      if (pending_.size() > 1) pending_.erase(pending_.begin(), pending_.end() - 1);
      return;
    }
    if (use_initial_pose_topic_ && !have_manual_initial_pose_) {
      // IMU is stable but no one has clicked "2D Pose Estimate" in RViz yet --
      // don't start locking against the (possibly wrong) initial_pose param.
      if (pending_.size() > 1) pending_.erase(pending_.begin(), pending_.end() - 1);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "IMU stable, waiting for /initialpose (RViz 2D Pose "
                           "Estimate) before starting localization");
      return;
    }
    while (!pending_.empty()) {
      const auto &front = pending_.front();
      bool imu_ok = !imu_en_ || (bias_ready_ && !imu_buf_.empty() &&
                                 imu_buf_.back().t >= front.t_end);
      if (!imu_ok) {
        // wait for IMU coverage, but never stall on a dead IMU stream
        if (nowSec() - front.arrival_wall < 0.3) return;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "processing scan without full IMU coverage");
      }
      PendingScan scan = std::move(pending_.front());
      pending_.pop_front();
      processScan(scan);
    }
  }

  // Integrate gyro over [ta, tb]; returns rotation of frame(tb) w.r.t. frame(ta).
  Eigen::Matrix3d integrateGyro(double ta, double tb) const {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (imu_buf_.empty() || tb <= ta) return R;
    double t_cur = ta;
    for (size_t i = 0; i < imu_buf_.size(); ++i) {
      const auto &s = imu_buf_[i];
      if (s.t <= t_cur) continue;
      const double t_next = std::min(s.t, tb);
      R = R * So3Exp(s.w * (t_next - t_cur));
      t_cur = t_next;
      if (t_cur >= tb) break;
    }
    if (t_cur < tb && !imu_buf_.empty())
      R = R * So3Exp(imu_buf_.back().w * (tb - t_cur));
    return R;
  }

  void processScan(const PendingScan &scan) {
    // wall-clock queueing delay: time this scan sat in pending_ before this
    // call started (grows first if the pipeline falls behind real time).
    const double t_start_wall = nowSec();
    const auto t_start = std::chrono::steady_clock::now();
    auto ms_since = [](const std::chrono::steady_clock::time_point &t0) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
          .count();
    };

    // 1) deskew to scan end using gyro rotation + constant body velocity
    const auto t_deskew = std::chrono::steady_clock::now();
    std::vector<Eigen::Vector3d> pts;
    const bool do_deskew = imu_en_ && deskew_en_ && bias_ready_ &&
                           !scan.rel_time.empty() && scan.t_end > scan.t_begin;
    if (do_deskew) {
      // Rotation timeline relative to scan begin, sampled at IMU timestamps.
      // ws[k] is the angular velocity active on segment (ts[k-1], ts[k]].
      std::vector<double> ts{scan.t_begin};
      std::vector<Eigen::Matrix3d> Rs{Eigen::Matrix3d::Identity()};
      std::vector<Eigen::Vector3d> ws{Eigen::Vector3d::Zero()};
      for (const auto &s : imu_buf_) {
        if (s.t <= scan.t_begin || s.t > scan.t_end + 0.01) continue;
        Rs.push_back(Rs.back() * So3Exp(s.w * (s.t - ts.back())));
        ts.push_back(s.t);
        ws.push_back(s.w);
      }
      const Eigen::Vector3d w_last = imu_buf_.empty()
                                         ? Eigen::Vector3d::Zero()
                                         : imu_buf_.back().w;
      auto rotAt = [&](double t) -> Eigen::Matrix3d {
        auto it = std::upper_bound(ts.begin(), ts.end(), t);
        const size_t idx = (it == ts.begin()) ? 0 : (it - ts.begin() - 1);
        const Eigen::Vector3d w = (idx + 1 < ts.size()) ? ws[idx + 1] : w_last;
        return Rs[idx] * So3Exp(w * std::max(0.0, t - ts[idx]));
      };
      const Eigen::Matrix3d R_end = rotAt(scan.t_end);
      // Body acceleration over this frame: (v_end - v_begin)/dt_frame, where
      // v_end = current v_body_ (latest odom ~at scan end), v_begin = previous
      // frame's v_body_ (~at scan begin). Clamped to +/-max_accel_ so an odom
      // glitch can't inject a huge quadratic term. Zero with no prior sample ->
      // reduces to the constant-v deskew (always applied; no toggle).
      Eigen::Vector3d a_body = Eigen::Vector3d::Zero();
      const double dt_frame = (last_scan_end_ > 0.0) ? scan.t_end - last_scan_end_ : 0.0;
      if (have_v_prev_ && dt_frame > 1e-3 && dt_frame < 0.5) {
        a_body = (v_body_ - v_body_prev_) / dt_frame;
        for (int k = 0; k < 3; ++k)
          a_body[k] = std::max(-max_accel_, std::min(max_accel_, a_body[k]));
      }
      pts.reserve(scan.points.size());
      for (size_t i = 0; i < scan.points.size(); ++i) {
        const double ti = scan.t_begin + scan.rel_time[i];
        const double tau = ti - scan.t_end;  // <= 0
        const Eigen::Matrix3d R_rel = R_end.transpose() * rotAt(ti);
        // constant-v term + acceleration (quadratic) correction
        Eigen::Vector3d trans = v_body_ * tau + 0.5 * a_body * (tau * tau);
        pts.push_back(R_rel * scan.points[i] + trans);
      }
    } else {
      pts = scan.points;
    }
    // remember this frame's body velocity for next frame's accel estimate
    v_body_prev_ = v_body_;
    have_v_prev_ = true;
    const double deskew_ms = ms_since(t_deskew);

    // 2) downsample
    const auto t_ds = std::chrono::steady_clock::now();
    const auto ds = VoxelDownsample(pts, scan_voxel_size_);
    const double ds_ms = ms_since(t_ds);
    if (ds.size() < 20) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "too few points after downsampling (%zu), skipping",
                           ds.size());
      return;
    }

    // 3) motion prediction
    const auto t_predict = std::chrono::steady_clock::now();
    Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
    const double dt = (last_scan_end_ > 0.0) ? scan.t_end - last_scan_end_ : 0.0;
    if (dt > 0.0 && dt < 0.5) {
      if (imu_en_ && bias_ready_)
        delta.linear() = integrateGyro(last_scan_end_, scan.t_end);
      delta.translation() = v_body_ * dt;
    }
    // planar propagation: the ICP initial guess advances only in map XY + yaw;
    // z/roll/pitch stay at the last fix so the 3D ICP alone sets them from LiDAR.
    const Eigen::Isometry3d T_pred =
        planar_prediction_ ? holdZRP(T_ * delta, T_) : T_ * delta;
    const double predict_ms = ms_since(t_predict);

    // 4) registration
    const double prep_ms = ms_since(t_start);  // deskew+downsample+predict total
    const auto t_icp = std::chrono::steady_clock::now();
    const double th = adaptive_->ComputeThreshold();
    // Per-point residuals from the ICP's own last-iteration correspondences
    // (see registration.hpp), filled by AlignScanToMap.
    std::vector<float> residuals;
    const bool want_residuals = publish_residual_cloud_ && residual_pub_ &&
                                residual_pub_->get_subscription_count() > 0;
    // full robust point-to-plane ICP against the voxel map (3D / 2.5D).
    auto result = AlignScanToMap(ds, map_, T_pred, th, th / 3.0, max_iterations_,
                                 convergence_eps_, use_normals_,
                                 want_residuals ? &residuals : nullptr);
    const double icp_ms = ms_since(t_icp);

    // 5) divergence gate — an isolated fix jumping away from the prediction
    // is treated as an ICP glitch and coasted over; if it persists, the
    // prediction is what's wrong, so re-anchor to the registration result
    const auto t_gate = std::chrono::steady_clock::now();
    const Eigen::Isometry3d dev = T_pred.inverse() * result.pose;
    const double dev_t = dev.translation().norm();
    const double dev_r = RotationAngle(dev.rotation());
    bool reanchored = false;
    if (dev_t > reject_trans_ || dev_r > reject_rot_deg_ * M_PI / 180.0) {
      ++consecutive_rejects_;
      if (consecutive_rejects_ < reject_recover_count_) {
        RCLCPP_WARN(get_logger(),
                    "registration rejected (dev %.2f m / %.1f deg, corr %d) — "
                    "coasting on prediction (%d consecutive)",
                    dev_t, dev_r * 180.0 / M_PI, result.num_correspondences,
                    consecutive_rejects_);
        T_ = T_pred;
        last_scan_end_ = scan.t_end;
        T_prop_ = T_;
        t_prop_ = scan.t_end;
        publishOdom(T_, scan.t_end);
        return;
      }
      RCLCPP_WARN(get_logger(),
                  "re-anchoring to registration result after %d rejects "
                  "(dev %.2f m / %.1f deg)",
                  consecutive_rejects_, dev_t, dev_r * 180.0 / M_PI);
      adaptive_->Reset();
      v_body_.setZero();
      reanchored = true;
    }
    consecutive_rejects_ = 0;
    if (!reanchored) adaptive_->UpdateModelDeviation(dev);

    // 6) state update — velocity from consecutive fixes, with physical
    // accel/speed limits: in corridor-degenerate stretches ICP can't observe
    // the along-track direction, and an unbounded velocity estimate feeds
    // back into the prediction and runs away
    if (!use_odom_twist_ && !reanchored && dt > 0.0 && dt < 0.5) {
      const Eigen::Vector3d v_new = result.pose.rotation().transpose() *
                                    (result.pose.translation() - T_.translation()) / dt;
      Eigen::Vector3d dv = (1.0 - vel_smoothing_) * (v_new - v_body_);
      const double dv_max = max_accel_ * dt;
      if (dv.norm() > dv_max) dv *= dv_max / dv.norm();
      v_body_ += dv;
      if (v_body_.norm() > max_velocity_)
        v_body_ *= max_velocity_ / v_body_.norm();
    }
    T_ = result.pose;
    last_scan_end_ = scan.t_end;
    T_prop_ = T_;
    t_prop_ = scan.t_end;
    have_first_fix_ = true;
    const double gate_ms = ms_since(t_gate);

    const auto t_publish = std::chrono::steady_clock::now();
    publishOdom(T_, scan.t_end);
    if (publish_aligned_scan_) publishAligned(ds, scan.t_end);
    if (publish_2d_scan_ && scan_2d_pub_->get_subscription_count() > 0)
      publishBand2D(ds, scan.t_end);
    if (want_residuals) publishHighResidual(ds, residuals, scan.t_end);
    const double publish_ms = ms_since(t_publish);

    // detection runs only on a confident, locked fix — a mislocalized pose
    // would paint the whole scan as foreground. The divergence-coast path
    // already returned above; here we additionally require convergence and
    // skip the frame we just re-anchored on.
    double detect_ms = 0.0;
    if (detect_en_ && detector_ && have_first_fix_ && !reanchored &&
        result.converged) {
      const auto t_detect = std::chrono::steady_clock::now();
      // detect_deskew:false feeds the RAW scan instead of the deskewed one.
      // When the deskew correction itself is wrong (stale/absent odom twist,
      // timestamp-anchor jitter under hard accel/turn), it splits the cloud
      // into two crisp sheets that DBSCAN picks up as phantom obstacles; the
      // raw scan's true motion blur smears continuously instead and mostly
      // dies in the track-erosion filter. Localization keeps its own deskew
      // either way (deskew_en).
      runDetection(detect_deskew_ ? pts : scan.points, scan.t_end);
      detect_ms = ms_since(t_detect);
    }

    if (print_stats_) {
      // STAT lines are machine-parseable (key=value) for offline analysis.
      // queue_ms: wait in pending_ before processing started (backlog symptom).
      // deskew/ds/predict/gate/publish_ms: processScan sub-stage breakdown
      // (prep_ms = deskew+ds+predict, kept for backward compat).
      // total_ms: full processScan compute; lat_ms: queue_ms+total_ms
      // (wall-clock arrival to STAT print) — a lat_ms >> total_ms gap means
      // the backlog (queue_ms), not this stage breakdown, is the bottleneck.
      const double total_ms = ms_since(t_start);
      RCLCPP_INFO(get_logger(),
                  "STAT t=%.3f raw=%zu ds=%zu queue_ms=%.1f deskew_ms=%.1f "
                  "ds_ms=%.1f predict_ms=%.1f prep_ms=%.1f icp_ms=%.1f "
                  "gate_ms=%.1f publish_ms=%.1f detect_ms=%.1f total_ms=%.1f "
                  "iters=%d corr=%d conv=%d th=%.3f dev_t=%.3f "
                  "dev_r=%.2f dt=%.3f v=%.2f q=%zu lat_ms=%.0f",
                  scan.t_end, scan.points.size(), ds.size(),
                  (t_start_wall - scan.arrival_wall) * 1e3, deskew_ms, ds_ms,
                  predict_ms, prep_ms, icp_ms, gate_ms, publish_ms, detect_ms,
                  total_ms, result.iterations, result.num_correspondences,
                  result.converged ? 1 : 0, th, dev_t, dev_r * 180.0 / M_PI,
                  dt, v_body_.norm(), pending_.size(),
                  (nowSec() - scan.arrival_wall) * 1e3);
    }
  }

  // --------------------------- publishing ---------------------------

  // Keep only the map-frame x, y and yaw of `moved`; restore z, roll and pitch
  // from `base`. Used when planar_prediction_ is on so IMU/odom propagation can
  // only advance the pose in the map XY plane (+ yaw) — z/roll/pitch then come
  // exclusively from the LiDAR ICP result (which owns `base` = the last fix).
  // This stops the high-speed z-sink: a pitched attitude otherwise leaks the
  // forward body velocity into a downward world-z increment, and free-running
  // gyro drifts roll/pitch between fixes. Yaw is taken about the map vertical.
  Eigen::Isometry3d holdZRP(const Eigen::Isometry3d &moved,
                            const Eigen::Isometry3d &base) const {
    const double yaw_moved =
        std::atan2(moved.linear()(1, 0), moved.linear()(0, 0));
    const double yaw_base =
        std::atan2(base.linear()(1, 0), base.linear()(0, 0));
    // base attitude with its yaw removed -> its roll/pitch only
    const Eigen::Matrix3d RP =
        Eigen::AngleAxisd(-yaw_base, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        base.linear();
    Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
    out.linear() =
        Eigen::AngleAxisd(yaw_moved, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        RP;
    out.translation() = Eigen::Vector3d(moved.translation().x(),
                                        moved.translation().y(),
                                        base.translation().z());
    return out;
  }

  // Static base_link -> lidar_frame TF = the mount extrinsic (rotation R_level_ =
  // ground-normal tilt, translation lidar_in_base_). Completes the tree
  // map -> base_link -> lidar_frame so the raw /livox cloud is placeable in map.
  void publishLidarStaticTf() {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = now();
    tf.header.frame_id = base_frame_;
    tf.child_frame_id = lidar_frame_;
    tf.transform.translation.x = lidar_in_base_.x();
    tf.transform.translation.y = lidar_in_base_.y();
    tf.transform.translation.z = lidar_in_base_.z();
    const Eigen::Quaterniond q(R_level_);
    tf.transform.rotation.w = q.w();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf_static_broadcaster_->sendTransform(tf);
  }

  void publishOdom(const Eigen::Isometry3d &T_in, double t) {
    const Eigen::Isometry3d T_lidar = T_in;
    // ICP gives map<-lidar; shift to map<-base_link (rear axle, level) via the mount
    // extrinsic base<-lidar = { R_level_ (ground-normal tilt, yaw 0), lidar_in_base_ }.
    // map<-base = (map<-lidar) * (base<-lidar)^-1.
    Eigen::Isometry3d T_base_lidar = Eigen::Isometry3d::Identity();
    T_base_lidar.linear() = R_level_;
    T_base_lidar.translation() = lidar_in_base_;
    const Eigen::Isometry3d T = T_lidar * T_base_lidar.inverse();
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = base_frame_;
    const Eigen::Quaterniond q(T.rotation());
    odom.pose.pose.position.x = T.translation().x();
    odom.pose.pose.position.y = T.translation().y();
    odom.pose.pose.position.z = T.translation().z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.twist.twist.linear.x = v_body_.x();
    odom.twist.twist.linear.y = v_body_.y();
    odom.twist.twist.linear.z = v_body_.z();
    odom_pub_->publish(odom);

    // Localization-derived body twist from consecutive base poses (finite
    // difference + EMA). Deliberately NOT v_body_: that is wheel-odom when
    // use_odom_twist is set. Physically impossible steps (re-anchor jumps,
    // stale dt) reset the differencing instead of spiking the estimate.
    if (loc_twist_pub_) {
      const double dt = t - loc_twist_prev_t_;
      if (loc_twist_prev_t_ > 0.0 && dt > 1e-4 && dt < 0.5) {
        const Eigen::Matrix3d R_prev = loc_twist_prev_T_.rotation();
        const Eigen::Vector3d v_new =
            R_prev.transpose() *
            (T.translation() - loc_twist_prev_T_.translation()) / dt;
        const Eigen::AngleAxisd dR(R_prev.transpose() * T.rotation());
        const Eigen::Vector3d w_new = dR.axis() * dR.angle() / dt;
        if (v_new.norm() <= max_velocity_) {
          const double a = loc_twist_smoothing_;
          loc_twist_v_ = a * loc_twist_v_ + (1.0 - a) * v_new;
          loc_twist_w_ = a * loc_twist_w_ + (1.0 - a) * w_new;
          geometry_msgs::msg::TwistStamped tw;
          tw.header.stamp = odom.header.stamp;
          tw.header.frame_id = base_frame_;
          tw.twist.linear.x = loc_twist_v_.x();
          tw.twist.linear.y = loc_twist_v_.y();
          tw.twist.linear.z = loc_twist_v_.z();
          tw.twist.angular.x = loc_twist_w_.x();
          tw.twist.angular.y = loc_twist_w_.y();
          tw.twist.angular.z = loc_twist_w_.z();
          loc_twist_pub_->publish(tw);
        }
      }
      loc_twist_prev_T_ = T;
      loc_twist_prev_t_ = t;
    }

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = T.translation().x();
      tf.transform.translation.y = T.translation().y();
      tf.transform.translation.z = T.translation().z();
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  void publishAligned(const std::vector<Eigen::Vector3d> &pts, double t) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(pts.size());
    for (const auto &p : pts) {
      const Eigen::Vector3d pw = T_ * p;
      cloud.emplace_back(pw.x(), pw.y(), pw.z());
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    aligned_pub_->publish(msg);
  }

  // 2D-flattened aligned scan: the ground-band slab of `pts` (sensor frame)
  // projected into the map frame and FLATTENED to z=0 — a planar (x,y) debug
  // overlay on the /map grid in RViz. Same z-band as detection (crop_z_min/max
  // above the ground plane).
  void publishBand2D(const std::vector<Eigen::Vector3d> &pts, double t) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(pts.size());
    // Band-crop against the MAP-vertical ground, not the static sensor-frame
    // mount tilt: the GLIM map is gravity-aligned, so the floor normal is (0,0,1)
    // regardless of the car's pitch/roll. Anchor the floor height to the 6-DoF
    // pose — the sensor sits crop_h_ above the ground — so a pitching car can't
    // tilt the band and sweep the floor in at range.
    const double ground_z = T_.translation().z() - crop_h_;
    for (const auto &p : pts) {
      const Eigen::Vector3d pw = T_ * p;
      const double hgt = pw.z() - ground_z;  // height above the map-vertical ground
      if (hgt < crop_z_min_ || hgt > crop_z_max_) continue;
      cloud.emplace_back(pw.x(), pw.y(), 0.0);  // flatten onto the map ground plane
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    scan_2d_pub_->publish(msg);
  }

  // Points whose registration residual against the prior map exceeds
  // residual_threshold — the static map can't explain them (opponent car, loose
  // obstacle, unmapped clutter). `residuals[i]` comes straight from
  // AlignScanToMap's last ICP iteration (registration.hpp) — no re-query here.
  // intensity = residual magnitude (m), for viz/tuning.
  void publishHighResidual(const std::vector<Eigen::Vector3d> &pts,
                           const std::vector<float> &residuals, double t) {
    const double thresh = get_parameter("residual_threshold").as_double();
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(pts.size() / 8);
    for (size_t i = 0; i < pts.size(); ++i) {
      if (residuals[i] <= thresh) continue;
      const Eigen::Vector3d pw = T_ * pts[i];
      pcl::PointXYZI pt;
      pt.x = pw.x();
      pt.y = pw.y();
      pt.z = pw.z();
      pt.intensity = residuals[i];
      cloud.push_back(pt);
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    residual_pub_->publish(msg);
  }

  // --------------------------- detection ---------------------------
  // `scan_sensor` is the deskewed scan in the sensor frame; transform to the
  // map frame with the just-solved pose and feed the BEV detector.
  void runDetection(const std::vector<Eigen::Vector3d> &scan_sensor, double t) {
    std::vector<Eigen::Vector3d> pts_map;
    pts_map.reserve(scan_sensor.size());
    for (const auto &p : scan_sensor) {
      // Reject the ego's own body return: a fixed box in the SENSOR frame (x
      // forward, so -x is behind the car). Applied before the map transform so
      // it stays glued to the car regardless of pose. Live-tunable (detect_self_*).
      if (detect_self_filter_ && p.x() >= detect_self_x_min_ &&
          p.x() <= detect_self_x_max_ && p.y() >= detect_self_y_min_ &&
          p.y() <= detect_self_y_max_)
        continue;
      pts_map.push_back(T_ * p);
    }

    const bool want_debug = debug_pub_->get_subscription_count() > 0;
    const BevResult res = detector_->Update(pts_map, t, want_debug);
    const auto stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));

    // foreground cloud: points after ground removal + track filter, i.e. the
    // BEV detector's input *before* DBSCAN clustering / tracking (on-track
    // obstacles + opponent kept, off-track / walls removed)
    if (obstacle_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<pcl::PointXYZ> oc;
      oc.reserve(res.obstacle_points.size());
      for (const auto &p : res.obstacle_points) oc.emplace_back(p.x(), p.y(), p.z());
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(oc, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = stamp;
      obstacle_pub_->publish(msg);
    }

    // debug cloud: EVERY input point, intensity = which filter stage dropped
    // it (see FilterStage) -- 0=survived as obstacle, 1=track, 2=z_max,
    // 3=ground plane. Lets you tell a real off-track filter miss apart from
    // an over-eager ground-plane fit while tuning.
    if (debug_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<pcl::PointXYZI> dc;
      dc.reserve(res.debug_points.size());
      for (const auto &dp : res.debug_points) {
        pcl::PointXYZI pt;
        pt.x = dp.p.x();
        pt.y = dp.p.y();
        pt.z = dp.p.z();
        pt.intensity = static_cast<float>(static_cast<int>(dp.stage));
        dc.push_back(pt);
      }
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(dc, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = stamp;
      debug_pub_->publish(msg);
    }

    // cluster centers as PoseArray (map frame) — published every cycle
    // (empty list included) so consumers (e.g. MPCC) use overwrite semantics.
    geometry_msgs::msg::PoseArray poses;
    poses.header.frame_id = map_frame_;
    poses.header.stamp = stamp;

    // detection markers: bbox cube + id/speed label per cluster
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id = map_frame_;
    clear.header.stamp = stamp;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    for (const auto &d : res.detections) {
      const double gz = 0.0;  // ground = map-frame z=0 (sensor_height convention)

      geometry_msgs::msg::Pose pose;
      pose.position.x = d.center.x();
      pose.position.y = d.center.y();
      pose.position.z = gz;  // ground level; consumers typically use x,y only
      pose.orientation.w = 1.0;
      poses.poses.push_back(pose);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = map_frame_;
      m.header.stamp = stamp;
      m.ns = "objects";
      m.id = d.id;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = d.center.x();
      m.pose.position.y = d.center.y();
      m.pose.position.z = gz + d.z_min + 0.5 * std::max(0.1, d.height);
      m.pose.orientation.w = 1.0;
      m.scale.x = std::max(0.1, d.size.x());
      m.scale.y = std::max(0.1, d.size.y());
      m.scale.z = std::max(0.1, d.height);
      m.color.a = 0.4f;
      // moving (opponent) = red, static (wall/obstacle) = gray
      m.color.r = d.moving ? 1.0f : 0.6f;
      m.color.g = d.moving ? 0.1f : 0.6f;
      m.color.b = d.moving ? 0.1f : 0.6f;
      m.lifetime = rclcpp::Duration::from_seconds(0.3);
      arr.markers.push_back(m);

      if (d.moving) {  // velocity arrow: direction = heading, length ~ speed
        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = map_frame_;
        arrow.header.stamp = stamp;
        arrow.ns = "velocity";
        arrow.id = d.id;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        const double az = gz + d.z_min + std::max(0.1, d.height) + 0.2;
        geometry_msgs::msg::Point p0, p1;
        p0.x = d.center.x();
        p0.y = d.center.y();
        p0.z = az;
        p1.x = d.center.x() + d.velocity.x() * arrow_scale_;
        p1.y = d.center.y() + d.velocity.y() * arrow_scale_;
        p1.z = az;
        arrow.points.push_back(p0);
        arrow.points.push_back(p1);
        arrow.scale.x = 0.05;  // shaft diameter
        arrow.scale.y = 0.12;  // head diameter
        arrow.scale.z = 0.15;  // head length
        arrow.color.a = 1.0f;
        arrow.color.r = 1.0f;
        arrow.color.g = 0.1f;
        arrow.color.b = 0.1f;
        arrow.lifetime = rclcpp::Duration::from_seconds(0.3);
        arr.markers.push_back(arrow);
      }
    }
    det_pub_->publish(arr);
    obstacle_pose_pub_->publish(poses);
  }

  // --------------------------- members ---------------------------
  // params
  std::string map_pcd_path_;  // the 3D map PCD (ICP target for 3D + 2.5D)
  std::string map_pcd_3d_, ground_yaml_;
  std::string lidar_topic_, imu_topic_, map_frame_, base_frame_;
  std::string lidar_frame_ = "livox_frame";  // LiDAR cloud frame_id (static TF child)
  // input_mode ("3d" | "2.5d") + wheel-odom translation source
  std::string input_mode_, odom_twist_topic_;
  bool use_odom_twist_ = false;
  double map_voxel_size_, scan_voxel_size_, min_range_, max_range_;
  double sensor_height_ = 0.0;  // sensor mount height; added to map z (ground -> grid z=0)
  int map_max_points_, point_filter_num_, max_iterations_, imu_init_samples_;
  double convergence_eps_, initial_threshold_, min_threshold_, min_motion_;
  double adaptive_range_, vel_smoothing_;
  // Hardcoded robustness bounds (not params): divergence gate + deskew/twist clamps.
  static constexpr double reject_trans_ = 2.0;       // fix jump reject [m]
  static constexpr double reject_rot_deg_ = 30.0;    // fix jump reject [deg]
  static constexpr int reject_recover_count_ = 3;    // rejects before re-anchor
  static constexpr double max_velocity_ = 15.0;      // twist-output outlier gate [m/s]
  static constexpr double max_accel_ = 10.0;         // deskew accel-term clamp [m/s^2]
  int consecutive_rejects_ = 0;
  bool stamp_at_scan_end_, imu_en_, deskew_en_, imu_rate_odom_,
      publish_tf_, publish_aligned_scan_, use_initial_pose_topic_, print_stats_,
      use_normals_;
  std::vector<double> initial_pose_;
  Eigen::Matrix3d R_il_ = Eigen::Matrix3d::Identity();
  // 2.5D mode: 3D PointCloud input + full voxel ICP (like 3D), but the scan is
  // cropped to keep only points at/below crop_z_max above the static GLIM ground
  // — floor INCLUDED, no z_min — so dynamic clutter / ceiling above the band is
  // dropped while the ground anchors z/roll/pitch. Matches the full 3D map.
  bool band_2p5d_ = false;
  Eigen::Vector3d crop_n_ = Eigen::Vector3d::UnitZ();
  double crop_h_ = 0.0, crop_z_min_ = 0.05, crop_z_max_ = 0.30;
  bool detect_en_ = false;
  bool detect_deskew_ = true;
  // Ego self-return rejection: drop scan points inside a fixed box in the SENSOR
  // frame (the LiDAR is forward-facing, aligned with base_link; -x is behind the
  // car). This removes the car's own chassis/antenna the LiDAR sees behind
  // itself, which would otherwise cluster as a phantom obstacle at the ego pose.
  // All live-tunable (detect_self_*) so the box can be dragged onto the self-hit.
  bool detect_self_filter_ = false;
  double detect_self_x_min_ = -0.5, detect_self_x_max_ = -0.1;
  double detect_self_y_min_ = -0.15, detect_self_y_max_ = 0.15;
  std::string odom_topic_, obstacle_topic_, detection_topic_, obstacle_pose_topic_,
      debug_topic_;
  // localization-derived twist publisher state (see publishOdom)
  std::string loc_twist_topic_;
  double loc_twist_smoothing_ = 0.5;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr loc_twist_pub_;
  Eigen::Isometry3d loc_twist_prev_T_ = Eigen::Isometry3d::Identity();
  double loc_twist_prev_t_ = -1.0;
  Eigen::Vector3d loc_twist_v_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d loc_twist_w_ = Eigen::Vector3d::Zero();
  BevParams bev_params_;
  std::string track_map_path_;
  TrackMask track_mask_;       // detection stage-2 off-track filter
  double arrow_scale_ = 0.5;

  // map & estimation state
  VoxelHashMap map_;
  std::vector<Eigen::Vector3d> map_cloud_raw_;
  std::unique_ptr<AdaptiveThreshold> adaptive_;
  Eigen::Isometry3d T_ = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_prop_ = Eigen::Isometry3d::Identity();
  Eigen::Vector3d v_body_ = Eigen::Vector3d::Zero();
  bool planar_prediction_ = false;                         // propagate only map xy+yaw
  Eigen::Vector3d v_body_prev_ = Eigen::Vector3d::Zero();  // last frame's v (for accel)
  bool have_v_prev_ = false;
  Eigen::Matrix3d R_level_ = Eigen::Matrix3d::Identity();  // levels crop_n_ -> map z (mount tilt)
  Eigen::Vector3d lidar_in_base_{0.27, 0.0, 0.11};  // LiDAR pos in base_link (mount offset)
  double last_scan_end_ = -1.0;
  double t_prop_ = -1.0;
  bool have_first_fix_ = false;
  // Startup gate: once IMU bias is ready, still hold off running ICP until a
  // manual /initialpose arrives from RViz (when use_initial_pose_topic_ is
  // on) -- the initial_pose param seeds T_ but is no longer trusted to start
  // locking on its own. Set true by onInitialPose(); checked in processPending().
  bool have_manual_initial_pose_ = false;

  // imu
  std::deque<ImuSample> imu_buf_;
  std::vector<Eigen::Vector3d> bias_acc_;
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  bool bias_ready_ = false;

  std::deque<PendingScan> pending_;

  // ros interfaces
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_, map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_2d_pub_;
  bool publish_2d_scan_ = true;
  std::string scan_2d_topic_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr residual_pub_;
  bool publish_residual_cloud_ = false;
  std::string residual_topic_;  // residual_threshold itself is read live, not cached
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initpose_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  // detection
  std::unique_ptr<BevDetector> detector_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr det_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obstacle_pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

}  // namespace kiss_loc

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<kiss_loc::LocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
