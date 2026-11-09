// Learned eviction policies running inside the simulator: MLP, LSTM+attention and LightGBM (GBDT) models, hand-written inference (no ONNX).
// Features are defined in training/features.py and MUST stay identical. Model formats: training/modelio.py.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "policies.h"

namespace ac {

constexpr int LP_NUM = 9;                 // numeric features per candidate
constexpr uint32_t LP_NONE = 0xFFFFFFFFu;

// Debug sink used only by the sample-window exporter (tools/winexport): when `on`, every pick() stores the candidate features, the model
// scores and the PC ids of the decision it just made. Off by default; costs one branch per decision.
struct DbgSink {
  bool on = false;
  uint32_t ways = 0, cur = 0;
  bool has[32];
  float f[32][LP_NUM];
  double s[32];
  uint32_t lpc[32];
};
inline DbgSink& dbgSink() { static DbgSink d; return d; }

// ---------------------------------------------------------------- MLP
struct MlpModel {
  uint32_t n_num = 0, emb = 0, n_ids = 0;
  std::vector<uint32_t> dims;
  std::vector<uint64_t> vocab;
  std::vector<float> mean, stdv, emb_last, emb_cur;
  std::vector<std::vector<float>> W, b;
  uint64_t params = 0, bytes = 0;

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model file: " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    bytes = raw.size();
    size_t off = 0;
    auto need = [&](size_t n) { if (off + n > raw.size()) throw std::runtime_error("truncated model file: " + path); };
    need(8);
    if (memcmp(raw.data(), "ACMLP001", 8) != 0) throw std::runtime_error("not an ACMLP001 file: " + path);
    off = 8;
    auto u32 = [&]() { need(4); uint32_t v; memcpy(&v, &raw[off], 4); off += 4; return v; };
    n_num = u32(); emb = u32(); n_ids = u32(); uint32_t nl = u32();
    if (n_num != (uint32_t)LP_NUM) throw std::runtime_error("model expects a different number of numeric features");
    dims.resize(nl + 1);
    for (auto& d : dims) d = u32();
    vocab.resize(n_ids - 1);
    need(8 * vocab.size()); memcpy(vocab.data(), &raw[off], 8 * vocab.size()); off += 8 * vocab.size();
    auto fl = [&](std::vector<float>& v, size_t n) { v.resize(n); need(4 * n); memcpy(v.data(), &raw[off], 4 * n); off += 4 * n; };
    fl(mean, n_num); fl(stdv, n_num); fl(emb_last, (size_t)n_ids * emb); fl(emb_cur, (size_t)n_ids * emb);
    W.resize(nl); b.resize(nl);
    for (uint32_t l = 0; l < nl; l++) { fl(W[l], (size_t)dims[l + 1] * dims[l]); fl(b[l], dims[l + 1]); }
    if (off != raw.size()) throw std::runtime_error("trailing bytes in model file: " + path);
    if (dims[0] != n_num + 2 * emb || dims.back() != 1) throw std::runtime_error("bad layer dims in " + path);
    params = 2 * n_num + 2ull * n_ids * emb;
    for (uint32_t l = 0; l < nl; l++) params += (uint64_t)dims[l] * dims[l + 1] + dims[l + 1];
  }
  uint64_t opsPerCandidate() const {  // multiply-adds counted as 2 ops
    uint64_t o = 2ull * n_num;
    for (size_t l = 0; l + 1 < dims.size(); l++) o += 2ull * dims[l] * dims[l + 1] + dims[l + 1];
    return o;
  }
  // x: LP_NUM raw numeric features; pc ids as in the vocabulary (0 = unknown)
  float score(const float* x, uint32_t pcid, uint32_t cur, std::vector<float>& h0, std::vector<float>& h1) const {
    h0.assign(dims[0], 0.f);
    for (uint32_t j = 0; j < n_num; j++) h0[j] = (x[j] - mean[j]) / stdv[j];
    for (uint32_t j = 0; j < emb; j++) { h0[n_num + j] = emb_last[(size_t)pcid * emb + j]; h0[n_num + emb + j] = emb_cur[(size_t)cur * emb + j]; }
    std::vector<float>* in = &h0;
    std::vector<float>* outv = &h1;
    for (size_t l = 0; l < W.size(); l++) {
      const uint32_t ni = dims[l], no = dims[l + 1];
      outv->assign(no, 0.f);
      for (uint32_t o = 0; o < no; o++) {
        float s = b[l][o];
        const float* w = &W[l][(size_t)o * ni];
        for (uint32_t i = 0; i < ni; i++) s += w[i] * (*in)[i];
        (*outv)[o] = (l + 1 < W.size() && s < 0.f) ? 0.f : s;
      }
      std::swap(in, outv);
    }
    return (*in)[0];
  }
};

// ---------------------------------------------------------------- LSTM + attention (small PARROT-style model)
// Per history step: [pc embedding (emb); sign(delta line); log2(1+|delta|)/20; is_write; same_set_as_current; valid]  (S = emb + 5)
// The history is the last `hist` LLC accesses ending with the current one; the LSTM starts from zero state at the oldest step.
// Candidate: c = relu(A1 [x_norm; emb[last pc]; emb[cur pc]]); query u = Wq c; keys = [h_i ; pos_i]; attention a = softmax(u.k / sqrt(K));
// context g = sum a_i h_i; score = v . relu(B [c; g]).  Format: training/modelio.py (ACLSTM01).
constexpr int LP_HIST_FEATS = 5;

struct LstmModel {
  uint32_t n_num = 0, emb = 0, n_ids = 0, hist = 0, hidden = 0, dpos = 0, dc = 0, dh = 0, S = 0, F = 0, K = 0;
  std::vector<uint64_t> vocab;
  std::vector<float> mean, stdv, e, wih, whh, b, pos, a1w, a1b, qw, qb, bw, bb, vw;
  float vb = 0;
  uint64_t params = 0, bytes = 0;
  struct Scratch { std::vector<float> h, c, z, Hs, xin, cv, u, att, g, zz; };

  void load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model file: " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    bytes = raw.size();
    size_t off = 0;
    auto need = [&](size_t n) { if (off + n > raw.size()) throw std::runtime_error("truncated model file: " + path); };
    need(8);
    if (memcmp(raw.data(), "ACLSTM01", 8) != 0) throw std::runtime_error("not an ACLSTM01 file: " + path);
    off = 8;
    auto u32 = [&]() { need(4); uint32_t v; memcpy(&v, &raw[off], 4); off += 4; return v; };
    n_num = u32(); emb = u32(); n_ids = u32(); hist = u32(); hidden = u32(); dpos = u32(); dc = u32(); dh = u32();
    if (n_num != (uint32_t)LP_NUM) throw std::runtime_error("model expects a different number of numeric features");
    if (hist == 0 || hidden == 0 || dc == 0 || dh == 0 || n_ids == 0) throw std::runtime_error("bad LSTM dimensions in " + path);
    S = emb + LP_HIST_FEATS; F = n_num + 2 * emb; K = hidden + dpos;
    vocab.resize(n_ids - 1);
    need(8 * vocab.size()); memcpy(vocab.data(), &raw[off], 8 * vocab.size()); off += 8 * vocab.size();
    params = 0;
    auto fl = [&](std::vector<float>& v, size_t n) { v.resize(n); need(4 * n); memcpy(v.data(), &raw[off], 4 * n); off += 4 * n; params += n; };
    fl(mean, n_num); fl(stdv, n_num); fl(e, (size_t)n_ids * emb);
    fl(wih, (size_t)4 * hidden * S); fl(whh, (size_t)4 * hidden * hidden); fl(b, (size_t)4 * hidden);
    fl(pos, (size_t)hist * dpos);
    fl(a1w, (size_t)dc * F); fl(a1b, dc); fl(qw, (size_t)K * dc); fl(qb, K);
    fl(bw, (size_t)dh * (dc + hidden)); fl(bb, dh); fl(vw, dh);
    std::vector<float> t; fl(t, 1); vb = t[0];
    if (off != raw.size()) throw std::runtime_error("trailing bytes in model file: " + path);
  }
  uint64_t opsPerDecision(uint32_t ways) const {  // multiply-adds counted as 2 ops; activations not counted
    uint64_t lstm = (uint64_t)hist * (8ull * hidden * (S + hidden) + 4ull * hidden);
    uint64_t cand = 2ull * F * dc + 2ull * dc * K + 2ull * hist * K + 2ull * hist * hidden + 2ull * (dc + hidden) * dh + 2ull * dh;
    return lstm + (uint64_t)ways * cand;
  }
  // x: hist*S step inputs (oldest first). Fills s.Hs with the hist hidden states.
  void encode(const float* x, Scratch& s) const {
    s.h.assign(hidden, 0.f); s.c.assign(hidden, 0.f); s.z.assign(4 * (size_t)hidden, 0.f); s.Hs.assign((size_t)hist * hidden, 0.f);
    for (uint32_t t = 0; t < hist; t++) {
      const float* xt = x + (size_t)t * S;
      for (uint32_t r = 0; r < 4 * hidden; r++) {
        float acc = b[r];
        const float* wi = &wih[(size_t)r * S];
        for (uint32_t j = 0; j < S; j++) acc += wi[j] * xt[j];
        const float* wh = &whh[(size_t)r * hidden];
        for (uint32_t j = 0; j < hidden; j++) acc += wh[j] * s.h[j];
        s.z[r] = acc;
      }
      for (uint32_t k = 0; k < hidden; k++) {
        const float ig = 1.f / (1.f + std::exp(-s.z[k])), fg = 1.f / (1.f + std::exp(-s.z[hidden + k]));
        const float gg = std::tanh(s.z[2 * hidden + k]), og = 1.f / (1.f + std::exp(-s.z[3 * hidden + k]));
        s.c[k] = fg * s.c[k] + ig * gg;
        s.h[k] = og * std::tanh(s.c[k]);
        s.Hs[(size_t)t * hidden + k] = s.h[k];
      }
    }
  }
  // x: LP_NUM raw numeric features of one candidate; pcid = id of the line's last-access PC; cur = id of the current PC. Needs encode() first.
  float score(const float* x, uint32_t pcid, uint32_t cur, Scratch& s) const {
    s.xin.assign(F, 0.f);
    for (uint32_t j = 0; j < n_num; j++) s.xin[j] = (x[j] - mean[j]) / stdv[j];
    for (uint32_t j = 0; j < emb; j++) { s.xin[n_num + j] = e[(size_t)pcid * emb + j]; s.xin[n_num + emb + j] = e[(size_t)cur * emb + j]; }
    s.cv.assign(dc, 0.f);
    for (uint32_t o = 0; o < dc; o++) {
      float acc = a1b[o]; const float* w = &a1w[(size_t)o * F];
      for (uint32_t j = 0; j < F; j++) acc += w[j] * s.xin[j];
      s.cv[o] = acc < 0.f ? 0.f : acc;
    }
    s.u.assign(K, 0.f);
    for (uint32_t o = 0; o < K; o++) {
      float acc = qb[o]; const float* w = &qw[(size_t)o * dc];
      for (uint32_t j = 0; j < dc; j++) acc += w[j] * s.cv[j];
      s.u[o] = acc;
    }
    const float inv = (float)(1.0 / std::sqrt((double)K));
    s.att.assign(hist, 0.f);
    float mx = -3.4e38f;
    for (uint32_t i = 0; i < hist; i++) {
      float acc = 0.f;
      const float* hi = &s.Hs[(size_t)i * hidden]; const float* pi = &pos[(size_t)i * dpos];
      for (uint32_t j = 0; j < hidden; j++) acc += s.u[j] * hi[j];
      for (uint32_t j = 0; j < dpos; j++) acc += s.u[hidden + j] * pi[j];
      s.att[i] = acc * inv;
      if (s.att[i] > mx) mx = s.att[i];
    }
    float sum = 0.f;
    for (uint32_t i = 0; i < hist; i++) { s.att[i] = std::exp(s.att[i] - mx); sum += s.att[i]; }
    s.g.assign(hidden, 0.f);
    for (uint32_t i = 0; i < hist; i++) {
      const float a = s.att[i] / sum; const float* hi = &s.Hs[(size_t)i * hidden];
      for (uint32_t j = 0; j < hidden; j++) s.g[j] += a * hi[j];
    }
    float out = vb;
    for (uint32_t o = 0; o < dh; o++) {
      float acc = bb[o]; const float* w = &bw[(size_t)o * (dc + hidden)];
      for (uint32_t j = 0; j < dc; j++) acc += w[j] * s.cv[j];
      for (uint32_t j = 0; j < hidden; j++) acc += w[dc + j] * s.g[j];
      if (acc > 0.f) out += vw[o] * acc;
    }
    return out;
  }
};

// ---------------------------------------------------------------- GBDT (LightGBM text model, numerical splits only)
struct GbdtTree {
  int num_leaves = 1;
  std::vector<int> feat, left, right;
  std::vector<double> thr, leaf;
  std::vector<uint8_t> dtype;
};
struct GbdtModel {
  std::vector<uint64_t> vocab;
  std::vector<GbdtTree> trees;
  uint64_t bytes = 0, nodes = 0;
  mutable uint64_t visits = 0;   // node comparisons performed (for ops per decision)

  static std::vector<double> nums(const std::string& v) {
    std::vector<double> r; std::stringstream ss(v); std::string t;
    while (ss >> t) r.push_back(strtod(t.c_str(), nullptr));
    return r;
  }
  void loadText(const std::string& txt) {
    bytes = txt.size();
    size_t sep = txt.find("\n---\n");
    if (txt.rfind("ACGBDT 1", 0) != 0 || sep == std::string::npos) throw std::runtime_error("not an ACGBDT file");
    std::stringstream hs(txt.substr(0, sep));
    std::string line; std::getline(hs, line); std::getline(hs, line);
    std::stringstream vs(line); std::string tag; size_t k = 0; vs >> tag >> k;
    if (tag != "vocab") throw std::runtime_error("bad ACGBDT header");
    vocab.clear();
    for (size_t i = 0; i < k; i++) { unsigned long long v; vs >> v; vocab.push_back(v); }
    std::stringstream ms(txt.substr(sep + 5));
    GbdtTree* cur = nullptr;
    while (std::getline(ms, line)) {
      if (line.rfind("Tree=", 0) == 0) { trees.emplace_back(); cur = &trees.back(); continue; }
      if (!cur) continue;
      size_t eq = line.find('=');
      if (eq == std::string::npos) continue;
      std::string key = line.substr(0, eq), val = line.substr(eq + 1);
      if (key == "num_leaves") cur->num_leaves = atoi(val.c_str());
      else if (key == "num_cat" && atoi(val.c_str()) != 0) throw std::runtime_error("categorical splits are not supported");
      else if (key == "split_feature") { for (double x : nums(val)) cur->feat.push_back((int)x); }
      else if (key == "threshold") cur->thr = nums(val);
      else if (key == "decision_type") { for (double x : nums(val)) cur->dtype.push_back((uint8_t)(int)x); }
      else if (key == "left_child") { for (double x : nums(val)) cur->left.push_back((int)x); }
      else if (key == "right_child") { for (double x : nums(val)) cur->right.push_back((int)x); }
      else if (key == "leaf_value") cur->leaf = nums(val);
    }
    if (trees.empty()) throw std::runtime_error("no trees in model");
    for (auto& t : trees) {
      if ((int)t.leaf.size() != t.num_leaves) throw std::runtime_error("bad leaf count in tree");
      if (t.num_leaves > 1 && (t.feat.size() != t.thr.size() || t.feat.size() != t.left.size() || t.feat.size() != t.right.size() || t.dtype.size() != t.feat.size()))
        throw std::runtime_error("inconsistent tree arrays");
      nodes += t.num_leaves + t.feat.size();
    }
  }
  void load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open model file: " + path);
    std::stringstream ss; ss << f.rdbuf();
    loadText(ss.str());
  }
  double score(const float* x) const {
    double s = 0;
    for (const auto& t : trees) {
      if (t.num_leaves == 1) { s += t.leaf[0]; continue; }
      int n = 0;
      for (;;) {
        visits++;
        double v = (double)x[t.feat[n]];
        const int dt = t.dtype[n], mt = (dt >> 2) & 3;
        bool left;
        if ((mt == 2 && std::isnan(v)) || (mt == 1 && std::fabs(v) <= 1e-35)) left = (dt >> 1) & 1;
        else left = v <= t.thr[n];
        n = left ? t.left[n] : t.right[n];
        if (n < 0) { s += t.leaf[~n]; break; }
      }
    }
    return s;
  }
};

// ---------------------------------------------------------------- the policy
class LearnedPolicy : public Policy {
 public:
  enum Kind { MLP, GBDT, LSTM };
  LearnedPolicy(Kind k, const std::string& path) : kind_(k) {
    if (k == MLP) { mlp_.load(path); vocab_ = mlp_.vocab; }
    else if (k == LSTM) { lstm_.load(path); vocab_ = lstm_.vocab; }
    else { gb_.load(path); vocab_ = gb_.vocab; }
    for (size_t i = 0; i < vocab_.size(); i++) pcid_[vocab_[i]] = (uint32_t)i + 1;
    std::string base = path;
    size_t sl = base.find_last_of('/'); if (sl != std::string::npos) base = base.substr(sl + 1);
    size_t dot = base.find_last_of('.'); if (dot != std::string::npos) base = base.substr(0, dot);
    name_ = std::string(k == MLP ? "mlp:" : (k == LSTM ? "lstm:" : "gbdt:")) + base;
  }
  const char* name() const override { return name_.c_str(); }
  void reset(uint32_t sets, uint32_t ways) override {
    ways_ = ways; size_t n = (size_t)sets * ways;
    last_.assign(n, 0); ins_.assign(n, 0); cnt_.assign(n, 0); gap_.assign(n, LP_NONE); lpc_.assign(n, 0); lwr_.assign(n, 0);
    lat_.clear(); decisions_ = 0; gb_.visits = 0;
    if (kind_ == LSTM) { ring_line_.assign(lstm_.hist, 0); ring_pc_.assign(lstm_.hist, 0); ring_wr_.assign(lstm_.hist, 0); set_mask_ = (uint64_t)sets - 1; }
  }
  void push(const AccessCtx& a) {   // every LLC access (hit or fill) enters the history ring exactly once, in trace order
    if (kind_ != LSTM) return;
    const size_t s = (size_t)(a.seq % lstm_.hist);
    ring_line_[s] = a.line; ring_pc_[s] = id(a.pc); ring_wr_[s] = a.is_write ? 1 : 0;
  }
  void onHit(uint32_t set, uint32_t way, const AccessCtx& a) override {
    size_t i = (size_t)set * ways_ + way;
    gap_[i] = (uint32_t)a.seq - last_[i]; last_[i] = (uint32_t)a.seq; cnt_[i]++; lpc_[i] = id(a.pc); lwr_[i] = a.is_write;
    push(a);
  }
  void onFill(uint32_t set, uint32_t way, const AccessCtx& a) override {
    size_t i = (size_t)set * ways_ + way;
    last_[i] = ins_[i] = (uint32_t)a.seq; cnt_[i] = 1; gap_[i] = LP_NONE; lpc_[i] = id(a.pc); lwr_[i] = a.is_write;
    push(a);
  }
  uint32_t victim(uint32_t set, const AccessCtx& a, const uint64_t*) override { return pick(set, a, nullptr); }
  // Restricted choice: scores are computed exactly as in victim() (features use the whole set), but only allowed ways can win.
  uint32_t victimAllowed(uint32_t set, const AccessCtx& a, const uint64_t*, const uint8_t* allowed, uint32_t) override { return pick(set, a, allowed); }
  uint32_t pick(uint32_t set, const AccessCtx& a, const uint8_t* allowed) {
    auto t0 = std::chrono::steady_clock::now();
    const size_t b = (size_t)set * ways_;
    const uint32_t cur = id(a.pc);
    DbgSink& dbg = dbgSink();
    if (dbg.on) { dbg.ways = ways_; dbg.cur = cur; for (uint32_t w = 0; w < 32; w++) dbg.has[w] = false; }
    std::vector<double>& x0 = x0_; x0.resize(ways_);
    double mean0 = 0;
    for (uint32_t w = 0; w < ways_; w++) { x0[w] = std::log2(1.0 + (double)((uint64_t)a.seq - last_[b + w])); mean0 += x0[w]; }
    mean0 /= ways_;
    if (kind_ == LSTM) {
      const int64_t H = (int64_t)lstm_.hist, E = (int64_t)lstm_.emb, S = (int64_t)lstm_.S, seq = (int64_t)a.seq;
      xs_.assign((size_t)(H * S), 0.f);
      for (int64_t j = 0; j < H; j++) {
        const int64_t pos = seq - (H - 1) + j;
        float* x = &xs_[(size_t)(j * S)];
        if (pos < 0) { for (int64_t q = 0; q < E; q++) x[q] = lstm_.e[q]; continue; }   // padding step: pc id 0, all features 0 (valid = 0)
        const bool isCur = pos == seq;
        const uint64_t line = isCur ? a.line : ring_line_[(size_t)(pos % H)];
        const uint32_t pid = isCur ? cur : ring_pc_[(size_t)(pos % H)];
        const bool wr = isCur ? a.is_write : ring_wr_[(size_t)(pos % H)] != 0;
        int64_t delta = 0;
        if (pos >= 1) delta = (int64_t)line - (int64_t)ring_line_[(size_t)((pos - 1) % H)];
        for (int64_t q = 0; q < E; q++) x[q] = lstm_.e[(size_t)pid * E + q];
        x[E + 0] = delta > 0 ? 1.f : (delta < 0 ? -1.f : 0.f);
        x[E + 1] = (float)(std::log2(1.0 + (double)(delta < 0 ? -delta : delta)) / 20.0);
        x[E + 2] = wr ? 1.f : 0.f;
        x[E + 3] = ((line & set_mask_) == (a.line & set_mask_)) ? 1.f : 0.f;
        x[E + 4] = 1.f;
      }
      lstm_.encode(xs_.data(), ls_);
    }
    uint32_t best = 0; double bs = 0; bool have = false;
    float f[LP_NUM];
    float g[LP_NUM + 2];
    for (uint32_t w = 0; w < ways_; w++) {
      if (allowed && !allowed[w]) continue;
      uint32_t rank = 0;
      for (uint32_t o = 0; o < ways_; o++) if (last_[b + o] > last_[b + w]) rank++;
      const bool hp = cnt_[b + w] >= 2;
      f[0] = (float)x0[w];
      f[1] = (float)std::log2(1.0 + (double)((uint64_t)a.seq - ins_[b + w]));
      f[2] = (float)std::log2(1.0 + (double)cnt_[b + w]);
      f[3] = hp ? 1.f : 0.f;
      f[4] = hp ? (float)std::log2(1.0 + (double)gap_[b + w]) : 0.f;
      f[5] = ways_ > 1 ? (float)((double)rank / (double)(ways_ - 1)) : 0.f;
      f[6] = (float)(x0[w] - mean0);
      f[7] = lwr_[b + w] ? 1.f : 0.f;
      f[8] = a.is_write ? 1.f : 0.f;
      double s;
      if (kind_ == MLP) s = mlp_.score(f, lpc_[b + w], cur, h0_, h1_);
      else if (kind_ == LSTM) s = lstm_.score(f, lpc_[b + w], cur, ls_);
      else { memcpy(g, f, sizeof f); g[LP_NUM] = (float)lpc_[b + w]; g[LP_NUM + 1] = (float)cur; s = gb_.score(g); }
      if (dbg.on && w < 32) { memcpy(dbg.f[w], f, sizeof f); dbg.s[w] = s; dbg.lpc[w] = lpc_[b + w]; dbg.has[w] = true; }
      if (!have || s > bs) { bs = s; best = w; have = true; }
    }
    decisions_++;
    uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    if (lat_.size() < (1u << 24)) lat_.push_back((uint32_t)std::min<uint64_t>(ns, 0xFFFFFFFFull));
    return best;
  }
  std::string statsJson() const override {
    std::vector<uint32_t> l = lat_;
    double mean = 0, p99 = 0, p50 = 0;
    if (!l.empty()) {
      for (uint32_t v : l) mean += v;
      mean /= l.size();
      std::sort(l.begin(), l.end());
      p99 = l[(size_t)((l.size() - 1) * 0.99)]; p50 = l[(l.size() - 1) / 2];
    }
    double ops = 0;
    uint64_t params = 0, bytes = 0;
    if (kind_ == MLP) { ops = (double)ways_ * (double)mlp_.opsPerCandidate(); params = mlp_.params; bytes = mlp_.bytes; }
    else if (kind_ == LSTM) { ops = (double)lstm_.opsPerDecision(ways_); params = lstm_.params; bytes = lstm_.bytes; }
    else { ops = decisions_ ? (double)gb_.visits / (double)decisions_ : 0.0; params = gb_.nodes; bytes = gb_.bytes; }
    std::ostringstream o;
    o << "{\"kind\":\"" << (kind_ == MLP ? "mlp" : (kind_ == LSTM ? "lstm" : "gbdt")) << "\",\"params\":" << params << ",\"model_bytes\":" << bytes
      << ",\"ops_per_decision\":" << ops << ",\"decisions\":" << decisions_ << ",\"lat_mean_ns\":" << mean
      << ",\"lat_p50_ns\":" << p50 << ",\"lat_p99_ns\":" << p99 << "}";
    return o.str();
  }

 private:
  uint32_t id(uint64_t pc) const { auto it = pcid_.find(pc); return it == pcid_.end() ? 0 : it->second; }
  Kind kind_;
  MlpModel mlp_;
  LstmModel lstm_;
  LstmModel::Scratch ls_;
  std::vector<float> xs_;
  std::vector<uint64_t> ring_line_;
  std::vector<uint32_t> ring_pc_;
  std::vector<uint8_t> ring_wr_;
  uint64_t set_mask_ = 0;
  GbdtModel gb_;
  std::vector<uint64_t> vocab_;
  std::unordered_map<uint64_t, uint32_t> pcid_;
  std::string name_;
  uint32_t ways_ = 0;
  std::vector<uint32_t> last_, ins_, cnt_, gap_, lpc_;
  std::vector<uint8_t> lwr_;
  std::vector<double> x0_;
  std::vector<float> h0_, h1_;
  std::vector<uint32_t> lat_;
  uint64_t decisions_ = 0;
};

}  // namespace ac
