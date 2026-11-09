// Binary trace format (see docs/trace_format.md). Used for raw traces and LLC traces.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ac {

#pragma pack(push, 1)
struct TraceRec {
  uint64_t addr;    // byte address (raw trace) or line-aligned address (LLC trace)
  uint64_t pc;      // instruction address that issued the access
  uint32_t icount;  // instructions executed since the previous record, including this one
  uint8_t size;     // access size in bytes (raw) or line size (LLC trace)
  uint8_t flags;    // bit0 = store
  uint16_t pad;
};
struct TraceHeader {
  char magic[8];          // "ACTRACE\0"
  uint32_t version;       // 1
  uint32_t rec_size;      // 24
  uint64_t num_records;   // 0 = unknown (streamed)
  uint64_t total_icount;  // 0 = unknown (streamed)
  uint8_t reserved[32];
};
#pragma pack(pop)
static_assert(sizeof(TraceRec) == 24, "TraceRec must be 24 bytes");
static_assert(sizeof(TraceHeader) == 64, "TraceHeader must be 64 bytes");
constexpr uint8_t FLAG_WRITE = 1;

class TraceReader {
 public:
  TraceReader() : buf_(1 << 16) {}
  ~TraceReader() { close(); }
  bool open(const std::string& path) {
    f_ = (path == "-") ? stdin : fopen(path.c_str(), "rb");
    if (!f_) return false;
    setvbuf(f_, nullptr, _IOFBF, 1 << 20);
    if (fread(&h_, sizeof h_, 1, f_) != 1) return false;
    return memcmp(h_.magic, "ACTRACE", 7) == 0 && h_.version == 1 && h_.rec_size == sizeof(TraceRec);
  }
  bool next(TraceRec& r) {
    if (pos_ == n_) {
      n_ = fread(buf_.data(), sizeof(TraceRec), buf_.size(), f_);
      pos_ = 0;
      if (n_ == 0) return false;
    }
    r = buf_[pos_++];
    return true;
  }
  const TraceHeader& header() const { return h_; }
  void close() {
    if (f_ && f_ != stdin) fclose(f_);
    f_ = nullptr;
  }

 private:
  FILE* f_ = nullptr;
  TraceHeader h_{};
  std::vector<TraceRec> buf_;
  size_t pos_ = 0, n_ = 0;
};

class TraceWriter {
 public:
  TraceWriter() { buf_.reserve(1 << 16); }
  ~TraceWriter() { if (f_) close(0); }
  bool open(const std::string& path) {
    f_ = (path == "-") ? stdout : fopen(path.c_str(), "wb");
    if (!f_) return false;
    setvbuf(f_, nullptr, _IOFBF, 1 << 20);
    TraceHeader h{};
    memcpy(h.magic, "ACTRACE", 8);
    h.version = 1;
    h.rec_size = sizeof(TraceRec);
    return fwrite(&h, sizeof h, 1, f_) == 1;
  }
  void write(const TraceRec& r) {
    buf_.push_back(r);
    ++n_;
    if (buf_.size() == (1u << 16)) flush();
  }
  // Patches the header (counts) when the output is a seekable file. On a pipe the counts stay 0.
  bool close(uint64_t total_icount) {
    if (!f_) return true;
    flush();
    bool ok = true;
    if (f_ != stdout) {
      TraceHeader h{};
      memcpy(h.magic, "ACTRACE", 8);
      h.version = 1;
      h.rec_size = sizeof(TraceRec);
      h.num_records = n_;
      h.total_icount = total_icount;
      ok = fseek(f_, 0, SEEK_SET) == 0 && fwrite(&h, sizeof h, 1, f_) == 1;
      ok = (fclose(f_) == 0) && ok;
    } else {
      ok = fflush(f_) == 0;
    }
    f_ = nullptr;
    return ok;
  }
  uint64_t count() const { return n_; }

 private:
  void flush() {
    if (!buf_.empty()) fwrite(buf_.data(), sizeof(TraceRec), buf_.size(), f_);
    buf_.clear();
  }
  FILE* f_ = nullptr;
  std::vector<TraceRec> buf_;
  uint64_t n_ = 0;
};

}  // namespace ac
