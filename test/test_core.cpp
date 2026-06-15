#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <random>
#include <vector>

#include "kiss_icp_localization/adaptive_threshold.hpp"
#include "kiss_icp_localization/registration.hpp"
#include "kiss_icp_localization/se3.hpp"
#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace {

// Synthetic room (floor + ceiling + 4 walls + a pillar breaking symmetry),
// sampled at random continuous surface positions. Regular-grid sampling is
// deliberately avoided: identical lattices in map and scan make point-to-point
// ICP lock onto the lattice — a pathology real sensor data does not have.
std::vector<Eigen::Vector3d> SampleRoom(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ux(-5, 5), uy(-4, 4), uz(0, 3),
      u01(0, 1), ua(0, 2 * M_PI);
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(n);
  for (int i = 0; i < n; ++i) {
    const double s = u01(rng);
    if (s < 0.30)
      pts.emplace_back(ux(rng), uy(rng), u01(rng) < 0.5 ? 0.0 : 3.0);
    else if (s < 0.55)
      pts.emplace_back(ux(rng), u01(rng) < 0.5 ? -4.0 : 4.0, uz(rng));
    else if (s < 0.80)
      pts.emplace_back(u01(rng) < 0.5 ? -5.0 : 5.0, uy(rng), uz(rng));
    else {
      const double a = ua(rng);
      pts.emplace_back(2.0 + 0.3 * std::cos(a), 1.0 + 0.3 * std::sin(a), uz(rng));
    }
  }
  return pts;
}

Eigen::Isometry3d MakePose(double x, double y, double z, double yaw) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(x, y, z);
  T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return T;
}

// Scan observed from T_gt: independently sampled surface points moved into
// the sensor frame, with measurement noise.
std::vector<Eigen::Vector3d> MakeScan(const Eigen::Isometry3d &T_gt, int n,
                                      unsigned seed, double noise_sigma) {
  std::mt19937 rng(seed + 1000);
  std::normal_distribution<double> noise(0.0, noise_sigma);
  std::vector<Eigen::Vector3d> scan;
  for (const auto &p : SampleRoom(n, seed)) {
    scan.push_back(T_gt.inverse() * p +
                   Eigen::Vector3d(noise(rng), noise(rng), noise(rng)));
  }
  return kiss_loc::VoxelDownsample(scan, 0.4);
}

}  // namespace

TEST(VoxelHashMap, ClosestNeighbor) {
  std::vector<Eigen::Vector3d> pts = {{0, 0, 0}, {1, 0, 0}, {0.4, 0.1, 0}};
  kiss_loc::VoxelHashMap map(0.5, 10);
  map.Build(pts);
  EXPECT_EQ(map.NumPoints(), 3u);

  kiss_loc::MapPoint closest;
  ASSERT_TRUE(map.GetClosestNeighbor({0.35, 0.0, 0.0}, 0.5, closest));
  EXPECT_NEAR((closest.p - Eigen::Vector3d(0.4, 0.1, 0)).norm(), 0.0, 1e-9);
  EXPECT_FALSE(map.GetClosestNeighbor({100, 100, 100}, 0.5, closest));
}

TEST(Registration, ConvergesFromLargeInitialError) {
  kiss_loc::VoxelHashMap map(0.5, 30);
  map.Build(SampleRoom(120000, 1));

  const Eigen::Isometry3d T_gt = MakePose(0.3, -0.2, 0.05, 5.0 * M_PI / 180.0);
  const auto scan = MakeScan(T_gt, 8000, 2, 0.005);

  // Coarse pass from identity (initial_threshold), then a tight refinement —
  // mirrors how the node's adaptive threshold shrinks over consecutive frames.
  const auto coarse = kiss_loc::AlignScanToMap(
      scan, map, Eigen::Isometry3d::Identity(), 1.0, 1.0 / 3.0, 100, 1e-6);
  EXPECT_GT(coarse.num_correspondences, 100);

  const auto result =
      kiss_loc::AlignScanToMap(scan, map, coarse.pose, 0.3, 0.1, 100, 1e-6);
  const Eigen::Isometry3d err = T_gt.inverse() * result.pose;
  EXPECT_LT(err.translation().norm(), 0.03);
  EXPECT_LT(kiss_loc::RotationAngle(err.rotation()), 0.5 * M_PI / 180.0);
}

TEST(Registration, TracksSmallMotionWithTightThreshold) {
  kiss_loc::VoxelHashMap map(0.5, 30);
  map.Build(SampleRoom(120000, 3));

  const Eigen::Isometry3d T_gt = MakePose(1.0, 0.5, 0.0, 0.3);
  const auto scan = MakeScan(T_gt, 8000, 4, 0.005);

  // prediction off by 5 cm / 1 deg — typical inter-frame error
  const Eigen::Isometry3d guess =
      T_gt * MakePose(0.05, -0.03, 0.01, 1.0 * M_PI / 180.0);
  const auto result =
      kiss_loc::AlignScanToMap(scan, map, guess, 0.3, 0.1, 100, 1e-6);

  const Eigen::Isometry3d err = T_gt.inverse() * result.pose;
  EXPECT_LT(err.translation().norm(), 0.02);
  EXPECT_LT(kiss_loc::RotationAngle(err.rotation()), 0.3 * M_PI / 180.0);
}

TEST(Registration, PointToPlaneConverges) {
  // room with analytic surface normals — guards the p2plane Jacobian sign
  const auto pts = SampleRoom(120000, 5);
  std::vector<Eigen::Vector3d> normals;
  normals.reserve(pts.size());
  for (const auto &p : pts) {
    if (std::abs(p.z()) < 1e-9 || std::abs(p.z() - 3.0) < 1e-9)
      normals.emplace_back(0, 0, 1);
    else if (std::abs(std::abs(p.y()) - 4.0) < 1e-9)
      normals.emplace_back(0, 1, 0);
    else if (std::abs(std::abs(p.x()) - 5.0) < 1e-9)
      normals.emplace_back(1, 0, 0);
    else  // pillar: radial
      normals.push_back(
          Eigen::Vector3d(p.x() - 2.0, p.y() - 1.0, 0).normalized());
  }
  kiss_loc::VoxelHashMap map(0.5, 30);
  map.Build(pts, normals);
  ASSERT_TRUE(map.HasNormals());

  const Eigen::Isometry3d T_gt = MakePose(0.3, -0.2, 0.05, 5.0 * M_PI / 180.0);
  const auto scan = MakeScan(T_gt, 8000, 6, 0.005);

  const auto coarse =
      kiss_loc::AlignScanToMap(scan, map, Eigen::Isometry3d::Identity(), 1.0,
                               1.0 / 3.0, 100, 1e-6, true);
  const auto result =
      kiss_loc::AlignScanToMap(scan, map, coarse.pose, 0.3, 0.1, 100, 1e-6, true);
  const Eigen::Isometry3d err = T_gt.inverse() * result.pose;
  EXPECT_LT(err.translation().norm(), 0.02);
  EXPECT_LT(kiss_loc::RotationAngle(err.rotation()), 0.3 * M_PI / 180.0);
}

TEST(AdaptiveThreshold, ShrinksWithGoodPredictions) {
  kiss_loc::AdaptiveThreshold th(1.0, 0.1, 0.05, 60.0);
  EXPECT_DOUBLE_EQ(th.ComputeThreshold(), 1.0);  // no samples yet
  Eigen::Isometry3d dev = Eigen::Isometry3d::Identity();
  dev.translation() = Eigen::Vector3d(0.2, 0, 0);
  for (int i = 0; i < 5; ++i) th.UpdateModelDeviation(dev);
  EXPECT_NEAR(th.ComputeThreshold(), 0.2, 1e-9);
  th.Reset();
  EXPECT_DOUBLE_EQ(th.ComputeThreshold(), 1.0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
