#include "kiss_icp_localization/bev_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace kiss_loc {

namespace {
struct Obs {
  int n = 0;
  double sx = 0.0, sy = 0.0;
  float rmin = std::numeric_limits<float>::max();
  float rmax = -std::numeric_limits<float>::max();
};
}  // namespace

BevResult BevDetector::Update(const std::vector<Eigen::Vector3d> &points_map,
                              double stamp) {
  BevResult out;
  const double inv_res = 1.0 / p_.res;

  // ---- phase A: project band-filtered points into current-frame cells ----
  std::unordered_map<int64_t, Obs> cur;
  cur.reserve(points_map.size() / 4 + 16);
  for (const auto &p : points_map) {
    const double r = p.z() - GroundZ(p.x(), p.y());
    if (r < p_.z_min || r > p_.z_max) continue;
    const int ix = static_cast<int>(std::floor(p.x() * inv_res));
    const int iy = static_cast<int>(std::floor(p.y() * inv_res));
    Obs &o = cur[CellKey(ix, iy)];
    ++o.n;
    o.sx += p.x();
    o.sy += p.y();
    o.rmin = std::min(o.rmin, static_cast<float>(r));
    o.rmax = std::max(o.rmax, static_cast<float>(r));
  }

  // ---- phase B/C: classify against background, then update background ----
  std::unordered_set<int64_t> fg_cells;
  fg_cells.reserve(cur.size());
  for (const auto &kv : cur) {
    const int64_t key = kv.first;
    auto it = bg_.find(key);
    double score = 0.0;
    if (it != bg_.end()) {
      const double dt = std::max(0.0, stamp - it->second.last);
      score = it->second.score * std::exp(-dt / p_.tau);
    }
    if (score < p_.persist) fg_cells.insert(key);
    // reinforce the cell with this observation
    BgCell &c = bg_[key];
    c.score = std::min(score + 1.0, p_.score_cap);
    c.last = stamp;
  }

  ++frame_;

  // periodically drop background cells that have fully decayed
  if (p_.prune_every > 0 && frame_ % p_.prune_every == 0) {
    for (auto it = bg_.begin(); it != bg_.end();) {
      const double dt = std::max(0.0, stamp - it->second.last);
      if (it->second.score * std::exp(-dt / p_.tau) < 0.05)
        it = bg_.erase(it);
      else
        ++it;
    }
  }

  if (!WarmedUp()) return out;  // background still settling

  // ---- foreground point cloud (second cheap pass) ----
  out.foreground.reserve(cur.size());
  for (const auto &p : points_map) {
    const double r = p.z() - GroundZ(p.x(), p.y());
    if (r < p_.z_min || r > p_.z_max) continue;
    const int ix = static_cast<int>(std::floor(p.x() * inv_res));
    const int iy = static_cast<int>(std::floor(p.y() * inv_res));
    if (fg_cells.count(CellKey(ix, iy))) out.foreground.push_back(p);
  }

  // ---- cluster foreground cells (8-connected components) ----
  std::unordered_set<int64_t> visited;
  std::vector<int64_t> stack;
  std::vector<Detection> raw;
  for (const int64_t seed : fg_cells) {
    if (visited.count(seed)) continue;
    stack.clear();
    stack.push_back(seed);
    visited.insert(seed);
    int n_pts = 0, n_cells = 0;
    double sx = 0.0, sy = 0.0;
    float hmin = std::numeric_limits<float>::max();
    float hmax = -std::numeric_limits<float>::max();
    while (!stack.empty()) {
      const int64_t k = stack.back();
      stack.pop_back();
      const Obs &o = cur.at(k);
      n_pts += o.n;
      ++n_cells;
      sx += o.sx;
      sy += o.sy;
      hmin = std::min(hmin, o.rmin);
      hmax = std::max(hmax, o.rmax);
      const int ix = static_cast<int>(k >> 32);
      const int iy = static_cast<int>(static_cast<int32_t>(k & 0xffffffff));
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) continue;
          const int64_t nk = CellKey(ix + dx, iy + dy);
          if (fg_cells.count(nk) && !visited.count(nk)) {
            visited.insert(nk);
            stack.push_back(nk);
          }
        }
    }
    if (n_cells < p_.min_cluster_cells) continue;
    Detection d;
    d.centroid = Eigen::Vector2d(sx / n_pts, sy / n_pts);
    d.num_points = n_pts;
    d.num_cells = n_cells;
    d.height = static_cast<double>(hmax - hmin);
    double ext = 0.0;  // estimate radius from cell footprint
    ext = 0.5 * p_.res * std::sqrt(static_cast<double>(n_cells));
    d.extent = std::max(ext, p_.res);
    raw.push_back(d);
  }

  // ---- track association (greedy nearest centroid) ----
  std::vector<bool> track_used(tracks_.size(), false);
  for (auto &d : raw) {
    int best = -1;
    double best_d2 = p_.track_gate * p_.track_gate;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (track_used[i]) continue;
      const double d2 = (tracks_[i].c - d.centroid).squaredNorm();
      if (d2 < best_d2) {
        best_d2 = d2;
        best = static_cast<int>(i);
      }
    }
    if (best >= 0) {
      Track &t = tracks_[best];
      const double dt = std::max(1e-3, stamp - t.last);
      const Eigen::Vector2d v_inst = (d.centroid - t.c) / dt;
      t.v = 0.6 * t.v + 0.4 * v_inst;  // EMA
      t.c = d.centroid;
      t.last = stamp;
      t.misses = 0;
      track_used[best] = true;
      d.id = t.id;
      d.velocity = t.v;
    } else {
      Track t;
      t.id = next_id_++;
      t.c = d.centroid;
      t.v.setZero();
      t.last = stamp;
      t.misses = 0;
      tracks_.push_back(t);
      track_used.push_back(true);
      d.id = t.id;
      d.velocity.setZero();
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
