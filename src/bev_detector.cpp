#include "kiss_icp_localization/bev_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace kiss_loc {

namespace {
struct Obs {
  int n = 0;
  float rmin = std::numeric_limits<float>::max();
  float rmax = -std::numeric_limits<float>::max();
};
}  // namespace

BevResult BevDetector::Update(const std::vector<Eigen::Vector3d> &points_map,
                              double stamp) {
  BevResult out;
  const double inv_res = 1.0 / p_.res;

  const bool do_subtract = p_.subtract_map && map_ && !map_->Empty();

  // ---- ground removal + height crop (+ map subtraction), to 2D cells ----
  std::unordered_map<int64_t, Obs> cells;
  cells.reserve(points_map.size() / 4 + 16);
  out.obstacle_points.reserve(points_map.size() / 4 + 16);
  MapPoint mp;
  for (const auto &p : points_map) {
    const double r = p.z() - GroundZ(p.x(), p.y());
    if (r < p_.z_min || r > p_.z_max) continue;  // ground / overhead removed
    // drop points that belong to the prior map (walls / known structure)
    if (do_subtract && map_->GetClosestNeighbor(p, p_.map_dist, mp)) continue;
    const int ix = static_cast<int>(std::floor(p.x() * inv_res));
    const int iy = static_cast<int>(std::floor(p.y() * inv_res));
    Obs &o = cells[CellKey(ix, iy)];
    ++o.n;
    o.rmin = std::min(o.rmin, static_cast<float>(r));
    o.rmax = std::max(o.rmax, static_cast<float>(r));
    out.obstacle_points.push_back(p);
  }

  // ---- cluster occupied cells (connect within cluster_dist) ----
  // neighbor search radius in cells; decouples connectivity from cell size
  const int kr = std::max(
      1, std::min(20, static_cast<int>(std::lround(p_.cluster_dist * inv_res))));
  std::unordered_set<int64_t> visited;
  std::vector<int64_t> stack;
  std::vector<Detection> raw;
  for (const auto &seed_kv : cells) {
    const int64_t seed = seed_kv.first;
    if (visited.count(seed)) continue;
    stack.clear();
    stack.push_back(seed);
    visited.insert(seed);
    int n_pts = 0, n_cells = 0;
    int ixmin = INT32_MAX, ixmax = INT32_MIN, iymin = INT32_MAX, iymax = INT32_MIN;
    float hmin = std::numeric_limits<float>::max();
    float hmax = -std::numeric_limits<float>::max();
    while (!stack.empty()) {
      const int64_t k = stack.back();
      stack.pop_back();
      const Obs &o = cells.at(k);
      n_pts += o.n;
      ++n_cells;
      hmin = std::min(hmin, o.rmin);
      hmax = std::max(hmax, o.rmax);
      const int ix = static_cast<int>(k >> 32);
      const int iy = static_cast<int>(static_cast<int32_t>(k & 0xffffffff));
      ixmin = std::min(ixmin, ix);
      ixmax = std::max(ixmax, ix);
      iymin = std::min(iymin, iy);
      iymax = std::max(iymax, iy);
      for (int dx = -kr; dx <= kr; ++dx)
        for (int dy = -kr; dy <= kr; ++dy) {
          if (dx == 0 && dy == 0) continue;
          const int64_t nk = CellKey(ix + dx, iy + dy);
          if (cells.count(nk) && !visited.count(nk)) {
            visited.insert(nk);
            stack.push_back(nk);
          }
        }
    }
    if (n_cells < p_.min_cluster_cells) continue;
    Detection d;
    // bbox in world coords (cell index range, +1 cell to cover cell width)
    const double x0 = ixmin * p_.res, x1 = (ixmax + 1) * p_.res;
    const double y0 = iymin * p_.res, y1 = (iymax + 1) * p_.res;
    d.center = Eigen::Vector2d(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    d.size = Eigen::Vector2d(x1 - x0, y1 - y0);
    d.num_points = n_pts;
    d.num_cells = n_cells;
    d.height = static_cast<double>(hmax - hmin);
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
