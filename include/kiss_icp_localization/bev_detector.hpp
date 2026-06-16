#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kiss_loc {

// 2D bird's-eye-view detector for unmapped objects (static obstacles +
// opponents). It maintains a rolling per-cell persistence grid in the map
// frame as a *self-healing background*: points whose ground-relative height
// falls in a band are projected to 2D; a cell that has been observed for
// several consecutive frames becomes static background, so anything newly
// appearing (or a moved obstacle, until it settles) is foreground. Moving vs
// static is then resolved by tracking cluster centroids over time.
//
// Localization (scan-to-prior-map ICP) is unaffected — this only consumes the
// already-deskewed, already-localized scan, so it adds no extra registration.
struct BevParams {
  double res = 0.2;          // BEV cell size [m]
  // ground plane in map frame: z_ground = ga*x + gb*y + gc
  double ga = 0.0, gb = 0.0, gc = 0.0;
  double z_min = 0.2;        // keep points this far above ground [m]
  double z_max = 1.5;        // ... and below this (drop overhead structure)
  double tau = 1.0;          // background persistence time constant [s]
  double persist = 3.0;      // decayed score >= persist  => static background
  double score_cap = 20.0;   // saturate per-cell score
  int min_cluster_cells = 3; // discard clusters smaller than this
  double track_gate = 1.0;   // max centroid step to associate a track [m]
  double moving_speed = 0.5; // |v| above this  => moving (opponent) [m/s]
  int max_misses = 5;        // drop a track after this many unmatched frames
  int warmup_frames = 5;     // emit detections only after background warms up
  int prune_every = 50;      // sweep stale background cells every N frames
};

struct Detection {
  int id = -1;
  Eigen::Vector2d centroid{0.0, 0.0};  // map frame [m]
  Eigen::Vector2d velocity{0.0, 0.0};  // map frame [m/s]
  double speed = 0.0;
  bool moving = false;
  int num_points = 0;
  int num_cells = 0;
  double extent = 0.0;  // horizontal radius [m]
  double height = 0.0;  // cluster height above ground [m]
};

struct BevResult {
  std::vector<Detection> detections;
  std::vector<Eigen::Vector3d> foreground;  // map-frame foreground points
};

class BevDetector {
public:
  explicit BevDetector(const BevParams &p) : p_(p) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for decay and track velocity.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp);

  bool WarmedUp() const { return frame_ >= p_.warmup_frames; }
  std::size_t BackgroundCells() const { return bg_.size(); }

private:
  double GroundZ(double x, double y) const { return p_.ga * x + p_.gb * y + p_.gc; }
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  struct BgCell {
    double score = 0.0;
    double last = 0.0;
  };

  BevParams p_;
  std::unordered_map<int64_t, BgCell> bg_;

  struct Track {
    int id;
    Eigen::Vector2d c;
    Eigen::Vector2d v;
    double last;
    int misses;
  };
  std::vector<Track> tracks_;
  int next_id_ = 0;
  long frame_ = 0;
};

}  // namespace kiss_loc
