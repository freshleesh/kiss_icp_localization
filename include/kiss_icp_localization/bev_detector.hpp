#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kiss_icp_localization/track_mask.hpp"

namespace kiss_loc {

// Per-frame BEV obstacle detector. Using the localized (map-frame) scan it
// removes the ground via the known map ground plane, crops to a vehicle-height
// band, and drops points outside the track mask (drivable area). The survivors
// are obstacles / the opponent on the track; they are projected to a 2D grid,
// clustered, and tracked across frames so each gets a velocity / moving flag
// (moving => opponent).
//
// Stateless per frame (no rolling background): nothing to warm up, no
// leading-edge false positives as the platform drives into new area, and
// insensitive to small pose jitter. Consumes the already-deskewed, already-
// localized scan, so it adds no registration. A moved/new static object on the
// track is reported as an obstacle — the desired behavior.
struct BevParams {
  double res = 0.2;          // BEV cell size [m]
  // Plain map-frame z band: a point counts if p.z is in [z_min, z_max].
  // No ground-plane normal/rotation involved.
  double z_min = 0.05;       // keep points with map-frame z at least this [m]
  double z_max = 0.30;       // ... and at most this
  // Track filter (erosion): drop points by unsigned distance to the nearest
  // non-track (black/wall/off-track) cell (TrackMask::DistToBlack). A point is
  // dropped when DistToBlack(x,y) <= track_dist_min. Points off the track are
  // always 0 (always dropped); points inside grow with distance from the
  // nearest wall, so only returns solidly inside the track -- at least
  // track_dist_min from any wall/off-track cell -- count as obstacle
  // candidates. This is the in-plane companion to the normal-direction z-band
  // crop above.
  bool track_filter = false;    // require points to lie away from the track mask boundary
  double track_dist_min = 0.2;  // min distance from black/wall [m] to count as a candidate
  // DBSCAN over occupied cells, independent of cell size `res`: eps is the
  // neighborhood radius and a cell needs >= min_samples occupied cells (incl.
  // itself) within eps to be a core. Sparse cells become noise and are dropped,
  // which rejects spurious returns. Cost grows as (eps/res)^2.
  double eps = 0.2;          // DBSCAN neighborhood radius [m]
  int min_samples = 4;       // min cells within eps (incl. self) for a core cell
  int min_cluster_cells = 2; // final: discard clusters smaller than this (cells)
  double track_gate = 1.0;   // max bbox-center step to associate a track [m]
  double moving_speed = 0.5; // |v| above this  => moving (opponent) [m/s]
  int max_misses = 5;        // drop a track after this many unmatched frames
};

struct Detection {
  int id = -1;
  Eigen::Vector2d center{0.0, 0.0};    // bbox center, map frame [m]
  Eigen::Vector2d size{0.0, 0.0};      // bbox extent x,y [m]
  Eigen::Vector2d velocity{0.0, 0.0};  // map frame [m/s]
  double speed = 0.0;
  bool moving = false;
  int num_points = 0;
  int num_cells = 0;
  double height = 0.0;  // cluster height above ground [m]
};

struct BevResult {
  std::vector<Detection> detections;
  std::vector<Eigen::Vector3d> obstacle_points;  // map-frame band-cropped points
};

class BevDetector {
public:
  // track: 2D track mask (may be null when track_filter off, or invalid if the
  // mask failed to load — filtering is then skipped).
  explicit BevDetector(const BevParams &p, const TrackMask *track = nullptr)
      : p_(p), track_(track) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for track velocity.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp);

private:
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  BevParams p_;
  const TrackMask *track_ = nullptr;

  struct Track {
    int id;
    Eigen::Vector2d c;
    Eigen::Vector2d v;
    double last;
    int misses;
  };
  std::vector<Track> tracks_;
  int next_id_ = 0;
};

}  // namespace kiss_loc
