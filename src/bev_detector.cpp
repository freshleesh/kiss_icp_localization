#include "kiss_icp_localization/bev_detector.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace kiss_loc {

namespace {
struct Obs {
  int n = 0;
  float rmin = std::numeric_limits<float>::max();
  float rmax = -std::numeric_limits<float>::max();
  int label = 0;  // DBSCAN: 0=undefined, -1=noise, >0=cluster id
};

// RANSAC-fit a near-horizontal plane to `pts` (3-point sampling, reject
// candidates whose normal isn't close to vertical, PCA-refit on the best
// inlier set). n.p + d = 0 on the plane, n.z() > 0. Returns false if there
// aren't enough points to try, or no iteration finds >= min_points inliers.
bool FitGroundPlane(const std::vector<Eigen::Vector3d> &pts, double dist_thresh,
                    int iters, double vertical_min, int min_points,
                    std::mt19937 &rng, Eigen::Vector3d &n_out, double &d_out) {
  const int n = static_cast<int>(pts.size());
  if (n < min_points) return false;
  std::uniform_int_distribution<int> pick(0, n - 1);

  int best_count = 0;
  Eigen::Vector3d best_n = Eigen::Vector3d::UnitZ();
  double best_d = 0.0;
  for (int it = 0; it < iters; ++it) {
    const int i0 = pick(rng), i1 = pick(rng), i2 = pick(rng);
    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
    const Eigen::Vector3d &p0 = pts[i0];
    Eigen::Vector3d cand = (pts[i1] - p0).cross(pts[i2] - p0);
    const double norm = cand.norm();
    if (norm < 1e-9) continue;
    cand /= norm;
    if (std::abs(cand.z()) < vertical_min) continue;  // reject wall-like planes
    if (cand.z() < 0) cand = -cand;
    const double d = -cand.dot(p0);
    int count = 0;
    for (const auto &p : pts) {
      if (std::abs(cand.dot(p) + d) < dist_thresh) ++count;
    }
    if (count > best_count) {
      best_count = count;
      best_n = cand;
      best_d = d;
    }
  }
  if (best_count < min_points) return false;

  // PCA refit on the best iteration's inliers for a cleaner fit.
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  int n_in = 0;
  for (const auto &p : pts) {
    if (std::abs(best_n.dot(p) + best_d) < dist_thresh) {
      mean += p;
      ++n_in;
    }
  }
  mean /= n_in;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto &p : pts) {
    if (std::abs(best_n.dot(p) + best_d) < dist_thresh) {
      const Eigen::Vector3d c = p - mean;
      cov.noalias() += c * c.transpose();
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  Eigen::Vector3d refined = es.eigenvectors().col(0);  // smallest-eigenvalue direction
  if (refined.z() < 0) refined = -refined;
  n_out = refined;
  d_out = -refined.dot(mean);
  return true;
}
}  // namespace

BevResult BevDetector::Update(const std::vector<Eigen::Vector3d> &points_map,
                              double stamp, bool want_debug) {
  BevResult out;
  const double inv_res = 1.0 / p_.res;

  const bool do_track = p_.track_filter && track_ && track_->Valid();

  // ---- map filter first: erosion by distance to the nearest non-track
  // (wall/off-track) cell. Off-track points are always 0 (always dropped);
  // on-track points need enough clearance from every wall to survive.
  // Points entirely off the loaded grid get +inf from DistToBlack (no data,
  // not "far from a wall") -- std::isfinite excludes those explicitly, since
  // `inf <= track_dist_min` is always false and would otherwise let anything
  // outside the map through unfiltered.
  std::vector<Eigen::Vector3d> candidates;
  candidates.reserve(points_map.size());
  if (want_debug) out.debug_points.reserve(points_map.size());
  for (const auto &p : points_map) {
    if (do_track) {
      const double db = track_->DistToBlack(p.x(), p.y());
      if (!(std::isfinite(db) && db > p_.track_dist_min)) {
        if (want_debug) out.debug_points.push_back({p, FilterStage::kTrack});
        continue;
      }
    }
    if (p.z() > p_.z_max) {
      if (want_debug) out.debug_points.push_back({p, FilterStage::kZMax});
      continue;
    }
    candidates.push_back(p);
  }

  // ---- RANSAC ground removal on the survivors, refit every frame ----
  Eigen::Vector3d gn;
  double gd = 0.0;
  const bool have_plane = FitGroundPlane(candidates, p_.ransac_dist_thresh,
                                         p_.ransac_iters, p_.ransac_vertical_min,
                                         p_.ransac_min_points, rng_, gn, gd);

  // ---- to 2D cells ----
  std::unordered_map<int64_t, Obs> cells;
  cells.reserve(candidates.size() / 4 + 16);
  out.obstacle_points.reserve(candidates.size() / 4 + 16);
  for (const auto &p : candidates) {
    // no plane found (too few survivors / nothing horizontal fit well) ->
    // pass everything through rather than guessing.
    if (have_plane && std::abs(gn.dot(p) + gd) <= p_.ransac_dist_thresh) {
      if (want_debug) out.debug_points.push_back({p, FilterStage::kGround});
      continue;  // ground inlier
    }
    const double r = p.z();
    const int ix = static_cast<int>(std::floor(p.x() * inv_res));
    const int iy = static_cast<int>(std::floor(p.y() * inv_res));
    Obs &o = cells[CellKey(ix, iy)];
    ++o.n;
    o.rmin = std::min(o.rmin, static_cast<float>(r));
    o.rmax = std::max(o.rmax, static_cast<float>(r));
    out.obstacle_points.push_back(p);
    if (want_debug) out.debug_points.push_back({p, FilterStage::kObstacle});
  }

  // ---- DBSCAN over occupied cells (density-based) ----
  // A cell is "core" if >= min_samples occupied cells (incl. itself) lie within
  // eps. Clusters grow only through core cells; cells reachable from a core are
  // border, the rest are noise and dropped. The res grid is the spatial index:
  // an eps-neighborhood is a +/-kr cell box (filtered to a true Euclidean disk).
  const int kr =
      std::max(1, std::min(20, static_cast<int>(std::ceil(p_.eps * inv_res))));
  const double eps2_cells = (p_.eps * inv_res) * (p_.eps * inv_res);
  auto decode = [](int64_t k, int &ix, int &iy) {
    ix = static_cast<int>(k >> 32);
    iy = static_cast<int>(static_cast<int32_t>(k & 0xffffffff));
  };
  auto range_query = [&](int64_t k, std::vector<int64_t> &out) {
    out.clear();
    int ix, iy;
    decode(k, ix, iy);
    for (int dx = -kr; dx <= kr; ++dx)
      for (int dy = -kr; dy <= kr; ++dy) {
        if (dx == 0 && dy == 0) continue;
        if (dx * dx + dy * dy > eps2_cells) continue;
        const int64_t nk = CellKey(ix + dx, iy + dy);
        if (cells.count(nk)) out.push_back(nk);
      }
  };

  int cid = 0;
  std::vector<int64_t> nb, seeds;
  for (auto &kv : cells) {
    if (kv.second.label != 0) continue;  // already visited
    range_query(kv.first, nb);
    if (static_cast<int>(nb.size()) + 1 < p_.min_samples) {
      kv.second.label = -1;  // provisional noise (may be claimed as border)
      continue;
    }
    ++cid;
    kv.second.label = cid;
    seeds = nb;
    for (size_t qi = 0; qi < seeds.size(); ++qi) {
      Obs &oq = cells.at(seeds[qi]);
      if (oq.label == -1) oq.label = cid;  // noise -> border of this cluster
      if (oq.label != 0) continue;         // already assigned/visited
      oq.label = cid;
      range_query(seeds[qi], nb);
      if (static_cast<int>(nb.size()) + 1 >= p_.min_samples)  // core -> expand
        for (const int64_t nk : nb) seeds.push_back(nk);
    }
  }

  // ---- aggregate clusters into detections ----
  struct Agg {
    int n = 0, cells = 0;
    int ixmin = INT32_MAX, ixmax = INT32_MIN, iymin = INT32_MAX, iymax = INT32_MIN;
    float hmin = std::numeric_limits<float>::max();
    float hmax = -std::numeric_limits<float>::max();
  };
  std::unordered_map<int, Agg> agg;
  for (const auto &kv : cells) {
    if (kv.second.label <= 0) continue;  // skip noise/undefined
    int ix, iy;
    decode(kv.first, ix, iy);
    Agg &a = agg[kv.second.label];
    a.n += kv.second.n;
    ++a.cells;
    a.ixmin = std::min(a.ixmin, ix);
    a.ixmax = std::max(a.ixmax, ix);
    a.iymin = std::min(a.iymin, iy);
    a.iymax = std::max(a.iymax, iy);
    a.hmin = std::min(a.hmin, kv.second.rmin);
    a.hmax = std::max(a.hmax, kv.second.rmax);
  }
  std::vector<Detection> raw;
  for (const auto &ak : agg) {
    const Agg &a = ak.second;
    if (a.cells < p_.min_cluster_cells) continue;
    Detection d;
    const double x0 = a.ixmin * p_.res, x1 = (a.ixmax + 1) * p_.res;
    const double y0 = a.iymin * p_.res, y1 = (a.iymax + 1) * p_.res;
    d.center = Eigen::Vector2d(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    d.size = Eigen::Vector2d(x1 - x0, y1 - y0);
    d.num_points = a.n;
    d.num_cells = a.cells;
    d.z_min = static_cast<double>(a.hmin);
    d.height = static_cast<double>(a.hmax - a.hmin);
    raw.push_back(d);
  }

  // ---- track association (greedy nearest bbox center) ----
  std::vector<bool> track_used(tracks_.size(), false);
  for (auto &d : raw) {
    int best = -1;
    double best_d2 = p_.track_gate * p_.track_gate;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (track_used[i]) continue;
      const double d2 = (tracks_[i].c - d.center).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(i);
      }
    }
    if (best >= 0) {
      Track &t = tracks_[best];
      const double dt = std::max(1e-3, stamp - t.last);
      const Eigen::Vector2d v_inst = (d.center - t.c) / dt;
      t.v = 0.6 * t.v + 0.4 * v_inst;  // EMA
      t.c = d.center;
      t.last = stamp;
      t.misses = 0;
      track_used[best] = true;
      d.id = t.id;
      d.velocity = t.v;
    } else {
      Track t;
      t.id = next_id_++;
      t.c = d.center;
      t.v.setZero();
      t.last = stamp;
      t.misses = 0;
      tracks_.push_back(t);
      track_used.push_back(true);
      d.id = t.id;
    }
    d.speed = d.velocity.norm();
    d.moving = d.speed > p_.moving_speed;
  }

  // age out unmatched tracks
  for (size_t i = 0; i < tracks_.size();) {
    if (!track_used[i]) ++tracks_[i].misses;
    if (tracks_[i].misses > p_.max_misses) {
      tracks_[i] = tracks_.back();
      track_used[i] = track_used.back();
      tracks_.pop_back();
      track_used.pop_back();
    } else {
      ++i;
    }
  }

  out.detections = std::move(raw);
  return out;
}

}  // namespace kiss_loc
