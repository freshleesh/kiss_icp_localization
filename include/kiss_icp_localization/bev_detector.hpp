#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kiss_icp_localization/voxel_hash_map.hpp"

namespace kiss_loc {

// Per-frame BEV obstacle detector. Using the localized (map-frame) scan it
// removes the ground via the known map ground plane and crops to a vehicle-
// height band. With map subtraction on (default), points close to the prior
// map (walls / known structure) are dropped so only *unmapped* objects remain
// — obstacles and the opponent; off, every cluster (walls included) is kept.
// Survivors are projected to a 2D grid, clustered, and tracked across frames
// so each gets a velocity / moving flag (moving => opponent).
//
// Stateless per frame (no rolling background): nothing to warm up, no
// leading-edge false positives as the platform drives into new area, and
// insensitive to small pose jitter. Consumes the already-deskewed, already-
// localized scan and the prior map, so it adds no registration. Map
// subtraction has complete coverage (no leading-edge problem) and treats a
// moved/new static object as an obstacle — which is the desired behavior.
struct BevParams {
  double res = 0.2;          // BEV cell size [m]
  // ground plane in map frame: z_ground = ga*x + gb*y + gc
  double ga = 0.0, gb = 0.0, gc = 0.0;
  double z_min = 0.15;       // keep points this far above ground [m] (ground removal)
  double z_max = 1.5;        // ... and below this (vehicle height / drop overhead)
  bool subtract_map = true;  // drop points near the prior map (keep unmapped only)
  double map_dist = 0.3;     // a point within this of the prior map = mapped [m]
  // clustering connects occupied cells within this distance, independent of the
  // cell size `res`: keep res fine for precise bbox while bridging point gaps so
  // sparse/far objects don't fragment. >= res; cost grows as (cluster_dist/res)^2.
  double cluster_dist = 0.2; // max gap bridged when clustering [m]
  int min_cluster_cells = 2; // discard clusters smaller than this (in `res` cells)
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
  // map: prior map for subtraction (may be null when subtract_map is false).
  explicit BevDetector(const BevParams &p, const VoxelHashMap *map = nullptr)
      : p_(p), map_(map) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for track velocity.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp);

private:
  double GroundZ(double x, double y) const { return p_.ga * x + p_.gb * y + p_.gc; }
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  BevParams p_;
  const VoxelHashMap *map_ = nullptr;

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
