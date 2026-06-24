#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace kiss_loc {

// 2D track mask (drivable area) loaded from a ROS-style occupancy yaml+pgm pair,
// e.g. the GLIM pipeline's map_track.{yaml,pgm} where track pixels = 255 (the
// flood-fill free region, sealed by walls). On load it precomputes a Euclidean
// distance transform so DistOutsideTrack(x, y) returns, for a map-frame point,
// 0 if the cell is inside the track and the horizontal distance [m] to the
// nearest track cell otherwise (+inf if the point is off the grid). The BEV
// detector uses it to reject objects that fall outside the track, where the
// prior map carries no information.
class TrackMask {
public:
  // Load from a map_server-style yaml (image / resolution / origin). Returns
  // false (and leaves the mask invalid -> filtering disabled) on any error.
  bool Load(const std::string &yaml_path) {
    std::ifstream y(yaml_path);
    if (!y) return false;
    std::string image, line;
    double res = 0, ox = 0, oy = 0;
    while (std::getline(y, line)) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string key = line.substr(0, colon);
      std::string val = line.substr(colon + 1);
      key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
      if (key == "image") {
        std::stringstream ss(val);
        ss >> image;  // strips surrounding whitespace
      } else if (key == "resolution") {
        res = std::atof(val.c_str());
      } else if (key == "origin") {
        // origin: [x, y, yaw]
        for (char &c : val)
          if (c == '[' || c == ']' || c == ',') c = ' ';
        std::stringstream ss(val);
        ss >> ox >> oy;
      }
    }
    if (image.empty() || res <= 0) return false;

    // resolve pgm path relative to the yaml's directory
    std::string dir;
    const auto slash = yaml_path.find_last_of('/');
    if (slash != std::string::npos) dir = yaml_path.substr(0, slash + 1);
    std::vector<uint8_t> px;
    int w = 0, h = 0;
    if (!ReadPGM(dir + image, px, w, h)) return false;

    res_ = res;
    ox_ = ox;
    oy_ = oy;
    W_ = w;
    H_ = h;
    BuildDistance(px);
    return Valid();
  }

  bool Valid() const { return W_ > 0 && H_ > 0 && !dist_.empty(); }

  // x, y in the map frame [m]; horizontal distance by which the point lies
  // outside the track [m] (0 if inside; +inf if off the grid). With no mask
  // loaded returns 0 so the caller's threshold keeps everything.
  double DistOutsideTrack(double x, double y) const {
    if (!Valid()) return 0.0;
    const int c = static_cast<int>(std::floor((x - ox_) / res_));
    const int r = static_cast<int>(std::floor((y - oy_) / res_));
    if (c < 0 || c >= W_ || r < 0 || r >= H_)
      return std::numeric_limits<double>::infinity();
    return dist_[static_cast<size_t>(r) * W_ + c];
  }

private:
  // P5 (binary) PGM reader. Stores rows top-to-bottom as written on disk.
  static bool ReadPGM(const std::string &path, std::vector<uint8_t> &px, int &w,
                      int &h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P5") return false;
    int maxval = 0;
    auto skip_ws_comments = [&]() {
      int ch;
      while ((ch = f.peek()) != EOF) {
        if (std::isspace(ch)) {
          f.get();
        } else if (ch == '#') {
          std::string comment;
          std::getline(f, comment);
        } else {
          break;
        }
      }
    };
    skip_ws_comments();
    f >> w;
    skip_ws_comments();
    f >> h;
    skip_ws_comments();
    f >> maxval;
    f.get();  // single whitespace after maxval before binary data
    if (w <= 0 || h <= 0 || maxval <= 0) return false;
    px.resize(static_cast<size_t>(w) * h);
    f.read(reinterpret_cast<char *>(px.data()), px.size());
    return static_cast<size_t>(f.gcount()) == px.size();
  }

  // 1D squared-distance transform (Felzenszwalb & Huttenlocher).
  static void DT1D(const std::vector<float> &f, std::vector<float> &d) {
    const int n = static_cast<int>(f.size());
    std::vector<int> v(n);
    std::vector<float> z(n + 1);
    const float INF = 1e20f;
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (int q = 1; q < n; ++q) {
      float s = ((f[q] + 1.0f * q * q) - (f[v[k]] + 1.0f * v[k] * v[k])) /
                (2.0f * (q - v[k]));
      while (s <= z[k]) {
        --k;
        s = ((f[q] + 1.0f * q * q) - (f[v[k]] + 1.0f * v[k] * v[k])) /
            (2.0f * (q - v[k]));
      }
      ++k;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    d.resize(n);
    k = 0;
    for (int q = 0; q < n; ++q) {
      while (z[k + 1] < q) ++k;
      const float dx = q - v[k];
      d[q] = dx * dx + f[v[k]];
    }
  }

  // Build dist_ (meters to nearest track cell) from the PGM pixels. dist_ is
  // indexed [r*W + c] with r = 0 at the LOW world-y row (the pgm is stored
  // top-to-bottom = high world-y first, so we flip on read-in).
  void BuildDistance(const std::vector<uint8_t> &px) {
    const float INF = 1e20f;
    std::vector<float> g(static_cast<size_t>(W_) * H_);
    for (int r = 0; r < H_; ++r) {
      const int src_row = H_ - 1 - r;  // flip to world y-up
      for (int c = 0; c < W_; ++c) {
        const bool track = px[static_cast<size_t>(src_row) * W_ + c] > 127;
        g[static_cast<size_t>(r) * W_ + c] = track ? 0.0f : INF;
      }
    }
    // transform along columns, then rows
    std::vector<float> col(H_), out, row(W_);
    for (int c = 0; c < W_; ++c) {
      for (int r = 0; r < H_; ++r) col[r] = g[static_cast<size_t>(r) * W_ + c];
      DT1D(col, out);
      for (int r = 0; r < H_; ++r) g[static_cast<size_t>(r) * W_ + c] = out[r];
    }
    for (int r = 0; r < H_; ++r) {
      for (int c = 0; c < W_; ++c) row[c] = g[static_cast<size_t>(r) * W_ + c];
      DT1D(row, out);
      for (int c = 0; c < W_; ++c) g[static_cast<size_t>(r) * W_ + c] = out[c];
    }
    dist_.resize(g.size());
    for (size_t i = 0; i < g.size(); ++i)
      dist_[i] = std::sqrt(g[i]) * static_cast<float>(res_);
  }

  int W_ = 0, H_ = 0;
  double res_ = 0.05, ox_ = 0.0, oy_ = 0.0;
  std::vector<float> dist_;  // [r*W + c], r=0 at low world-y, meters to track
};

}  // namespace kiss_loc
