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
//
// out_residuals, if non-null, is resized to scan.size() and filled with each
// point's |residual| (m) from the LAST iteration run — the same
// GetClosestNeighbor lookup the solver already does, so this is free (no
// extra map queries). A point with no map neighbor within max_corr_dist gets
// max_corr_dist itself (a lower bound on how far it actually is), so
// thresholding out_residuals uniformly flags both "far from its match" and
// "no match at all" points. Note this reflects the pose at the START of the
// last iteration, not the returned result.pose — off by at most one (usually
// tiny, near-converged) Gauss-Newton step.
RegistrationResult AlignScanToMap(const std::vector<Eigen::Vector3d> &scan,
                                  const VoxelHashMap &map,
                                  const Eigen::Isometry3d &initial_guess,
                                  double max_corr_dist, double kernel_sigma,
                                  int max_iterations, double convergence_eps,
                                  bool use_normals = false,
                                  std::vector<float> *out_residuals = nullptr);

}  // namespace kiss_loc
