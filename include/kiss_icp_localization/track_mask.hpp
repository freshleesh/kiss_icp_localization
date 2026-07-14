#pragma once

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace kiss_loc {

// 2D track mask (drivable area) loaded from a ROS-style occupancy yaml+pgm pair,
// e.g. the GLIM pipeline's map_track.{yaml,pgm} where track pixels = 255 (the
// flood-fill free region, sealed by walls). On load it precomputes a SIGNED
// Euclidean distance field so SignedOutside(x, y) returns, for a map-frame point,
// the horizontal distance [m] to the track boundary: negative inside the track
// (magnitude = distance to the nearest wall/black cell), positive outside
// (distance to the nearest track cell), +inf off the grid. The BEV detector
// rejects a point when SignedOutside > track_margin, so a single signed margin
// tunes the boundary: margin>0 dilates (keeps an outer ring, tolerant),
// margin<0 erodes (drops a near-wall band inside the track, aggressive).
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
    if (!ReadImage(dir + image, px, w, h)) {
      // Unreadable (unsupported PNG variant, corrupt file, etc.) -- fall back
      // to a sibling file with the same basename and a .pgm extension, in case
      // one was generated alongside (e.g. by glim_map_pipeline.py).
      const auto dot = image.find_last_of('.');
      const std::string pgm =
          (dot == std::string::npos ? image : image.substr(0, dot)) + ".pgm";
      if (pgm == image || !ReadImage(dir + pgm, px, w, h)) return false;
    }

    res_ = res;
    ox_ = ox;
    oy_ = oy;
    W_ = w;
    H_ = h;
    BuildDistance(px);
    return Valid();
  }

  bool Valid() const { return W_ > 0 && H_ > 0 && !sdf_.empty(); }

  // x, y in the map frame [m]; signed horizontal distance to the track boundary:
  // negative inside the track (|.| = distance to nearest wall), positive outside
  // (distance to nearest track cell), +inf off the grid. With no mask loaded
  // returns -inf so the caller's threshold keeps everything.
  double SignedOutside(double x, double y) const {
    if (!Valid()) return -std::numeric_limits<double>::infinity();
    const int c = static_cast<int>(std::floor((x - ox_) / res_));
    const int r = static_cast<int>(std::floor((y - oy_) / res_));
    if (c < 0 || c >= W_ || r < 0 || r >= H_)
      return std::numeric_limits<double>::infinity();
    return sdf_[static_cast<size_t>(r) * W_ + c];
  }

  // x, y in the map frame [m]; unsigned horizontal distance to the nearest
  // track (white) cell: 0 when (x,y) is itself on/inside the track, >0 when
  // outside (how far to the nearest drivable cell). +inf off the grid or with
  // no mask loaded (so a max-distance threshold keeps nothing, not everything --
  // callers should check Valid() first if they want fail-open behavior).
  double DistToTrack(double x, double y) const {
    if (!Valid()) return std::numeric_limits<double>::infinity();
    const int c = static_cast<int>(std::floor((x - ox_) / res_));
    const int r = static_cast<int>(std::floor((y - oy_) / res_));
    if (c < 0 || c >= W_ || r < 0 || r >= H_)
      return std::numeric_limits<double>::infinity();
    return dist_to_track_[static_cast<size_t>(r) * W_ + c];
  }

  // x, y in the map frame [m]; unsigned horizontal distance to the nearest
  // non-track (black/wall/unmapped) cell: 0 when (x,y) is itself off the
  // track, >0 when on/inside it (how far to the nearest wall). This is what
  // erosion needs: require DistToBlack > some minimum to count a point as an
  // obstacle candidate, so returns close to a wall are dropped regardless of
  // still being technically inside the track polygon. +inf off the grid or
  // with no mask loaded.
  double DistToBlack(double x, double y) const {
    if (!Valid()) return std::numeric_limits<double>::infinity();
    const int c = static_cast<int>(std::floor((x - ox_) / res_));
    const int r = static_cast<int>(std::floor((y - oy_) / res_));
    if (c < 0 || c >= W_ || r < 0 || r >= H_)
      return std::numeric_limits<double>::infinity();
    return dist_to_black_[static_cast<size_t>(r) * W_ + c];
  }

  // Bilinearly-interpolated signed distance `val` [m] and its in-plane gradient
  // (gx, gy) = dSDF/d(x,y) [m/m, dimensionless] at map-frame (x, y). The stored
  // sdf_ samples are treated as cell *centers* (world x = ox_ + (c+0.5)*res_).
  // Returns false when the 2x2 interpolation stencil falls off the grid (so the
  // caller drops that point from the 2D scan-to-SDF fit). (gx, gy) is the true
  // gradient of the signed field: it points toward INCREASING signed distance,
  // i.e. outward across the nearest wall (from deep-inside, through the wall, to
  // outside). Magnitude ~1 near a wall.
  bool ValueGrad(double x, double y, double &val, double &gx, double &gy) const {
    if (!Valid()) return false;
    const double fx = (x - ox_) / res_ - 0.5;
    const double fy = (y - oy_) / res_ - 0.5;
    const int c0 = static_cast<int>(std::floor(fx));
    const int r0 = static_cast<int>(std::floor(fy));
    if (c0 < 0 || c0 + 1 >= W_ || r0 < 0 || r0 + 1 >= H_) return false;
    const double tx = fx - c0;
    const double ty = fy - r0;
    const double v00 = sdf_[static_cast<size_t>(r0) * W_ + c0];
    const double v10 = sdf_[static_cast<size_t>(r0) * W_ + (c0 + 1)];
    const double v01 = sdf_[static_cast<size_t>(r0 + 1) * W_ + c0];
    const double v11 = sdf_[static_cast<size_t>(r0 + 1) * W_ + (c0 + 1)];
    val = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 +
          (1 - tx) * ty * v01 + tx * ty * v11;
    // gradient of the bilinear patch (chain rule through the cell->world scale)
    gx = ((1 - ty) * (v10 - v00) + ty * (v11 - v01)) / res_;
    gy = ((1 - tx) * (v01 - v00) + tx * (v11 - v10)) / res_;
    return true;
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

  // Dispatches by file signature (not extension -- the yaml's `image:` field
  // isn't trustworthy) to ReadPGM or ReadPNG. Output px is always single-byte
  // grayscale, row-major, same convention as ReadPGM.
  static bool ReadImage(const std::string &path, std::vector<uint8_t> &px,
                        int &w, int &h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char sig[8];
    f.read(reinterpret_cast<char *>(sig), 8);
    if (f.gcount() != 8) return false;
    static const unsigned char kPngSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    f.close();
    if (std::memcmp(sig, kPngSig, 8) == 0) return ReadPNG(path, px, w, h);
    return ReadPGM(path, px, w, h);
  }

  // Non-interlaced 8-bit PNG reader (grayscale/RGB/palette/gray+alpha/RGBA --
  // covers what matplotlib's imsave and map_server tooling actually produce).
  // Uses zlib directly for IDAT inflate; no other image-codec dependency.
  // Output is single-channel: the R (or gray) channel of each pixel, which is
  // what grayscale-colormap track/occupancy images actually carry. Returns
  // false for anything else (16-bit depth, interlaced, corrupt) so the caller
  // can fall back to a sibling .pgm.
  static bool ReadPNG(const std::string &path, std::vector<uint8_t> &px, int &w,
                      int &h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char sig[8];
    f.read(reinterpret_cast<char *>(sig), 8);
    if (f.gcount() != 8) return false;

    int bitdepth = 0, colortype = -1, interlace = 0;
    std::vector<unsigned char> palette;       // RGB triplets
    std::vector<unsigned char> idat;          // concatenated compressed data
    bool have_ihdr = false;

    while (f) {
      unsigned char lenb[4], typeb[4];
      f.read(reinterpret_cast<char *>(lenb), 4);
      if (f.gcount() != 4) break;
      f.read(reinterpret_cast<char *>(typeb), 4);
      if (f.gcount() != 4) break;
      const uint32_t len = (static_cast<uint32_t>(lenb[0]) << 24) |
                            (static_cast<uint32_t>(lenb[1]) << 16) |
                            (static_cast<uint32_t>(lenb[2]) << 8) | lenb[3];
      const std::string type(reinterpret_cast<char *>(typeb), 4);
      std::vector<unsigned char> data(len);
      if (len > 0) {
        f.read(reinterpret_cast<char *>(data.data()), len);
        if (static_cast<uint32_t>(f.gcount()) != len) return false;
      }
      unsigned char crc[4];
      f.read(reinterpret_cast<char *>(crc), 4);  // not verified

      if (type == "IHDR") {
        if (len != 13) return false;
        w = (static_cast<int>(data[0]) << 24) | (static_cast<int>(data[1]) << 16) |
            (static_cast<int>(data[2]) << 8) | data[3];
        h = (static_cast<int>(data[4]) << 24) | (static_cast<int>(data[5]) << 16) |
            (static_cast<int>(data[6]) << 8) | data[7];
        bitdepth = data[8];
        colortype = data[9];
        interlace = data[12];
        have_ihdr = true;
        if (w <= 0 || h <= 0 || bitdepth != 8 || interlace != 0) return false;
      } else if (type == "PLTE") {
        palette = data;
      } else if (type == "IDAT") {
        idat.insert(idat.end(), data.begin(), data.end());
      } else if (type == "IEND") {
        break;
      }
    }
    if (!have_ihdr || idat.empty()) return false;

    int channels = 0;
    switch (colortype) {
      case 0: channels = 1; break;  // gray
      case 2: channels = 3; break;  // RGB
      case 3: channels = 1; break;  // palette index
      case 4: channels = 2; break;  // gray+alpha
      case 6: channels = 4; break;  // RGBA
      default: return false;
    }
    if (colortype == 3 && palette.size() < 3) return false;

    const size_t row_bytes = static_cast<size_t>(w) * channels;
    const size_t raw_size = static_cast<size_t>(h) * (row_bytes + 1);  // +1 filter byte/row
    std::vector<unsigned char> raw(raw_size);
    uLongf dest_len = static_cast<uLongf>(raw_size);
    if (uncompress(raw.data(), &dest_len, idat.data(),
                   static_cast<uLong>(idat.size())) != Z_OK ||
        dest_len != raw_size)
      return false;

    // un-filter each scanline in place (PNG filter types 0-4)
    std::vector<unsigned char> prev(row_bytes, 0), cur(row_bytes);
    px.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
      const unsigned char *row = raw.data() + static_cast<size_t>(y) * (row_bytes + 1);
      const unsigned char filt = row[0];
      const unsigned char *in = row + 1;
      for (size_t x = 0; x < row_bytes; ++x) {
        const int a = (x >= static_cast<size_t>(channels)) ? cur[x - channels] : 0;
        const int b = prev[x];
        const int c = (x >= static_cast<size_t>(channels)) ? prev[x - channels] : 0;
        int recon = in[x];
        switch (filt) {
          case 0: break;                                        // None
          case 1: recon += a; break;                             // Sub
          case 2: recon += b; break;                              // Up
          case 3: recon += (a + b) / 2; break;                    // Average
          case 4: {                                               // Paeth
            const int p = a + b - c;
            const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
            recon += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
            break;
          }
          default: return false;
        }
        cur[x] = static_cast<unsigned char>(recon);
      }
      // extract the gray/R channel into px
      for (int x = 0; x < w; ++x) {
        unsigned char v;
        if (colortype == 3) {
          const size_t pi = static_cast<size_t>(cur[x]) * 3;
          v = (pi + 0 < palette.size()) ? palette[pi] : 0;
        } else {
          v = cur[static_cast<size_t>(x) * channels];  // gray or R channel
        }
        px[static_cast<size_t>(y) * w + x] = v;
      }
      prev.swap(cur);
    }
    return true;
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

  // Build sdf_ (signed meters to the track boundary: <0 inside, >0 outside) from
  // the PGM pixels. Indexed [r*W + c] with r = 0 at the LOW world-y row (the pgm
  // is stored top-to-bottom = high world-y first, so we flip on read-in).
  void BuildDistance(const std::vector<uint8_t> &px) {
    const float INF = 1e20f;
    // world-y-up track mask
    std::vector<uint8_t> track(static_cast<size_t>(W_) * H_);
    for (int r = 0; r < H_; ++r) {
      const int src_row = H_ - 1 - r;
      for (int c = 0; c < W_; ++c)
        track[static_cast<size_t>(r) * W_ + c] =
            px[static_cast<size_t>(src_row) * W_ + c] > 127 ? 1 : 0;
    }
    // Euclidean distance [m] from every cell to the nearest seed cell.
    auto edt = [&](bool seed_is_track) {
      std::vector<float> g(static_cast<size_t>(W_) * H_);
      for (size_t i = 0; i < g.size(); ++i) {
        const bool seed = seed_is_track ? track[i] : !track[i];
        g[i] = seed ? 0.0f : INF;
      }
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
      for (size_t i = 0; i < g.size(); ++i)
        g[i] = std::sqrt(g[i]) * static_cast<float>(res_);
      return g;
    };
    const std::vector<float> d_track = edt(true);   // 0 inside track, >0 outside
    const std::vector<float> d_black = edt(false);  // 0 outside track, >0 inside
    sdf_.resize(static_cast<size_t>(W_) * H_);
    for (size_t i = 0; i < sdf_.size(); ++i)
      sdf_[i] = d_track[i] - d_black[i];  // <0 inside (= -dist to wall), >0 outside
    dist_to_track_ = d_track;  // unsigned distance to nearest track cell (DistToTrack)
    dist_to_black_ = d_black;  // unsigned distance to nearest non-track cell (DistToBlack)
  }

  int W_ = 0, H_ = 0;
  double res_ = 0.05, ox_ = 0.0, oy_ = 0.0;
  std::vector<float> sdf_;  // [r*W + c], r=0 at low world-y, signed m to boundary
  std::vector<float> dist_to_track_;  // [r*W + c], unsigned m to nearest track cell
  std::vector<float> dist_to_black_;  // [r*W + c], unsigned m to nearest non-track cell
};

}  // namespace kiss_loc
