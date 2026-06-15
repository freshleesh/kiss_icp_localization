#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace kiss_loc {

struct RegistrationResult {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  int num_correspondences = 0;
  int iterations = 0;
  bool converged = false;
};

// Robust ICP (Gauss-Newton, Geman-McClure kernel) of a scan (sensor frame)
// against the static map, starting from initial_guess. Uses point-to-plane
// residuals where the map carries normals (and use_normals is set),
// point-to-point otherwise.
RegistrationResult AlignScanToMap(const std::vector<Eigen::Vector3d> &scan,
                                  const VoxelHashMap &map,
                                  const Eigen::Isometry3d &initial_guess,
                                  double max_corr_dist, double kernel_sigma,
                                  int max_iterations, double convergence_eps,
                                  bool use_normals = false);

}  // namespace kiss_loc
