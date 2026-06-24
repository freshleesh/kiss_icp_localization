#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kiss_icp_localization/track_mask.hpp"
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
  // Ground plane comes from ground_lidar.yaml (GLIM): the same sensor-frame
  // plane used to crop the localization input scan. It is passed to Update()
  // already transformed into the map frame for the just-solved pose, so the
  // height of a point p is ground_normal.p + ground_offset. The z-band below
  // matches crop_z_min/crop_z_max from that yaml.
  double z_min = 0.05;       // keep points this far above ground [m] (ground removal)
  double z_max = 0.30;       // ... and below this (drop overhead)
  bool subtract_map = true;  // drop points near the prior map (keep unmapped only)
  double map_dist = 0.3;     // a point within this of the prior map = mapped [m]
  // Stage-2 subtraction: drop points that fall outside the track. The track is a
  // 2D mask (TrackMask, GLIM map_track) of the drivable area; a point is dropped
  // when its horizontal (in-plane) distance to the nearest track cell exceeds
  // track_margin. This is the in-plane companion to the normal-direction z-band
  // crop above, and removes objects sensed beyond the track where the prior map
  // carries no information (the leading source of off-track false positives).
  bool track_filter = false; // require points to lie on/near the track mask
  double track_margin = 0.3; // max horizontal distance outside the track [m]
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
  // map: prior map for stage-1 subtraction (may be null when subtract_map off).
  // track: 2D track mask for stage-2 subtraction (may be null when track_filter
  // off, or invalid if the mask failed to load — filtering is then skipped).
  explicit BevDetector(const BevParams &p, const VoxelHashMap *map = nullptr,
                       const TrackMask *track = nullptr)
      : p_(p), map_(map), track_(track) {}

  // points_map: deskewed scan already transformed into the map frame.
  // stamp: scan time [s] (monotonic), used for track velocity.
  // ground_normal / ground_offset: ground plane in the map frame (the GLIM
  // sensor-frame plane from ground_lidar.yaml rotated by the current pose);
  // height above ground of a point p is ground_normal.dot(p) + ground_offset.
  BevResult Update(const std::vector<Eigen::Vector3d> &points_map, double stamp,
                   const Eigen::Vector3d &ground_normal, double ground_offset);

private:
  static int64_t CellKey(int ix, int iy) {
    return (static_cast<int64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint32_t>(iy);
  }

  BevParams p_;
  const VoxelHashMap *map_ = nullptr;
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
