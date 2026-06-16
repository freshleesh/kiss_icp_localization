#include "kiss_icp_localization/registration.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>

#include "kiss_icp_localization/se3.hpp"

namespace kiss_loc {

namespace {

// Accumulate one correspondence into the normal equations.
// Point-to-plane (1-D residual along the map normal) when the normal is
// known, point-to-point otherwise. Geman-McClure weight as in KISS-ICP.
inline void AddResidual(const Eigen::Vector3d &pw, const MapPoint &mp,
                        double kernel2, bool use_normals,
                        Eigen::Matrix<double, 6, 6> &JTJ,
                        Eigen::Matrix<double, 6, 1> &JTr) {
  const Eigen::Vector3d r = pw - mp.p;
  if (use_normals && mp.n.squaredNorm() > 0.25) {
    const double rn = mp.n.dot(r);
    const double w = kernel2 * kernel2 / ((kernel2 + rn * rn) * (kernel2 + rn * rn));
    Eigen::Matrix<double, 6, 1> Jt;
    Jt.head<3>() = mp.n;
    Jt.tail<3>() = pw.cross(mp.n);  // n·(dw×pw) = dw·(pw×n)
    JTJ.noalias() += Jt * (w * Jt.transpose());
    JTr.noalias() += Jt * (w * rn);
  } else {
    const double w = kernel2 * kernel2 /
                     ((kernel2 + r.squaredNorm()) * (kernel2 + r.squaredNorm()));
    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J.block<3, 3>(0, 3) = -Skew(pw);
    JTJ.noalias() += J.transpose() * w * J;
    JTr.noalias() += J.transpose() * (w * r);
  }
}

}  // namespace

RegistrationResult AlignScanToMap(const std::vector<Eigen::Vector3d> &scan,
                                  const VoxelHashMap &map,
                                  const Eigen::Isometry3d &initial_guess,
                                  double max_corr_dist, double kernel_sigma,
                                  int max_iterations, double convergence_eps,
                                  bool use_normals) {
  RegistrationResult result;
  result.pose = initial_guess;
  if (scan.empty() || map.Empty()) return result;

  use_normals = use_normals && map.HasNormals();
  const double kernel2 = kernel_sigma * kernel_sigma;
  Eigen::Isometry3d T = initial_guess;

  for (int iter = 0; iter < max_iterations; ++iter) {
    Eigen::Matrix<double, 6, 6> JTJ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> JTr = Eigen::Matrix<double, 6, 1>::Zero();
    int n = 0;

#ifdef _OPENMP
    // Cap threads: each iteration's parallel region only has a few thousand
    // points, so fork/join + reduction across many cores costs far more than
    // the work itself (24 threads measured ~15x slower than 8 on this scan
    // size). 8 saturates the useful parallelism here.
#pragma omp parallel num_threads(std::min(8, omp_get_max_threads()))
    {
      Eigen::Matrix<double, 6, 6> JTJ_p = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> JTr_p = Eigen::Matrix<double, 6, 1>::Zero();
      int n_p = 0;
#pragma omp for nowait
      for (size_t i = 0; i < scan.size(); ++i) {
        const Eigen::Vector3d pw = T * scan[i];
        MapPoint mp;
        if (!map.GetClosestNeighbor(pw, max_corr_dist, mp)) continue;
        AddResidual(pw, mp, kernel2, use_normals, JTJ_p, JTr_p);
        ++n_p;
      }
#pragma omp critical
      {
        JTJ += JTJ_p;
        JTr += JTr_p;
        n += n_p;
      }
    }
#else
    for (const auto &p : scan) {
      const Eigen::Vector3d pw = T * p;
      MapPoint mp;
      if (!map.GetClosestNeighbor(pw, max_corr_dist, mp)) continue;
      AddResidual(pw, mp, kernel2, use_normals, JTJ, JTr);
      ++n;
    }
#endif

    result.iterations = iter + 1;
    result.num_correspondences = n;
    if (n < 10) break;

    JTJ.diagonal().array() += 1e-6;  // ridge: keep near-degenerate frames bounded
    Eigen::Matrix<double, 6, 1> dx = JTJ.ldlt().solve(-JTr);
    if (!dx.allFinite()) break;
    // clamp runaway steps from ill-conditioned geometry
    const double step = dx.head<3>().norm();
    if (step > 1.0) dx *= 1.0 / step;

    // Left perturbation: [dt, dw]
    const Eigen::Matrix3d dR = So3Exp(dx.tail<3>());
    T.linear() = dR * T.rotation();
    T.translation() = dR * T.translation() + dx.head<3>();

    if (dx.norm() < convergence_eps) {
      result.converged = true;
      break;
    }
  }

  result.pose = T;
  return result;
}

}  // namespace kiss_loc
