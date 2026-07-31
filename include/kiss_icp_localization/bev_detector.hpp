#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "kiss_icp_localization/track_mask.hpp"

namespace kiss_loc {

// Per-frame BEV obstacle detector (v2). Using the localized (map-frame) scan,
// it first drops points by the track mask (map filter -- see track_dist_min),
// then RANSAC-fits a ground plane to the SURVIVORS of that filter each frame
// and removes its inliers. What's left are obstacles / the opponent on the
// track; they are projected to a 2D grid, clustered, and tracked across
// frames so each gets a velocity / moving flag (moving => opponent).
//
// Per-frame RANSAC (vs. a fixed z-band or a static calibrated ground plane)
// adapts to whatever the actual ground looks like in that scan -- no
// map_z_offset/ground_lidar.yaml dependency -- at the cost of needing enough
// map-filter survivors for a plane fit to be well-posed every frame.
//
// Stateless per frame (no rolling background): nothing to warm up, no
// leading-edge false positives as the platform drives into new area, and
// insensitive to small pose jitter. Consumes the already-deskewed, already-
// localized scan, so it adds no registration. A moved/new static object on the
// track is reported as an obstacle — the desired behavior.
struct BevParams {
  double res = 0.2;          // BEV cell size [m]
  // Two filters only: (1) a map-frame z band, (2) the track mask.
  // ---- (2) Track filter: drop points by unsigned distance to the nearest
  // non-track (black/wall/off-track) cell (TrackMask::DistToBlack). A point is
  // dropped when DistToBlack(x,y) <= track_dist_min. Off-track points are 0
  // (always dropped); on-track points grow with distance from the wall, so only
  // returns at least track_dist_min from any wall/off-track cell survive.
  bool track_filter = false;    // require points to lie away from the track mask boundary
  double track_dist_min = 0.2;  // min distance from black/wall [m] to count as a candidate
  // ---- (1) Map-frame z band: keep only z_min <= z <= z_max. Below z_min is
  // ground/floor, above z_max is ceiling/overhang. Replaces the old RANSAC
  // ground fit with a fixed band (localization gives a reliable map-frame z, so
  // the ground sits at a known height). Defaults are effectively disabled
  // (keep everything); set both per map/car.
  double z_min = -1e9;   // drop points below this map-frame z (ground)
  double z_max = 1e9;    // drop points above this map-frame z (ceiling/overhang)
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
  double z_min = 0.0;   // cluster's lowest point, map-frame z [m]
  double height = 0.0;  // cluster's map-frame z span (max - min) [m]
};

// Which stage dropped a point (or 0 if it survived as an obstacle candidate),
// for debug visualization -- see BevResult::debug_points.
enum class FilterStage : int {
  kObstacle = 0,  // survived the z band + track filter
  kTrack = 1,     // dropped by the track/wall erosion filter
  kZMax = 2,      // dropped for z > z_max (ceiling/overhang)
  kGround = 3,    // dropped for z < z_min (floor/ground)
};

struct DebugPoint {
  Eigen::Vector3d p;
  FilterStage stage;
};

struct BevResult {
  std::vector<Detection> detections;
  std::vector<Eigen::Vector3d> obstacle_points;  // map-frame, track+ground filtered
  // every input point tagged with the filter stage that dropped it (or
  // kObstacle if it survived) -- intensity-coded for debug viz.
  std::vector<DebugPoint> debug_points;
};

class BevDetector {
public:
  // track: 2D track mask (may be null when track_filter off, or invalid if the
  // mask failed to load — filtering is then skipped).
  explicit BevDetector(const BevParams &p, const TrackMask *track = nullptr)
      : p_(p), track_(track) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for track velocity.
  // want_debug: fill BevResult::debug_points (every input point tagged with
  // its filter stage) -- skipped by default since nobody's usually watching.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp,
                   bool want_debug = false);

  // Live-update the tuning params (keeps track state); for rqt / ros2 param set.
  void SetParams(const BevParams &p) { p_ = p; }

private:
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  BevParams p_;
  const TrackMask *track_ = nullptr;
  std::mt19937 rng_{42};  // fixed seed: reproducible RANSAC across runs

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
