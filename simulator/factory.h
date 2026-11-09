// Policy factory: baseline names ("lru", "srrip", ...) and learned models ("mlp:<file>", "gbdt:<file>", "lstm:<file>").
#pragma once
#include "composite.h"
#include "learned.h"
#include "policies.h"

namespace ac {

inline std::unique_ptr<Policy> makePolicy(const std::string& n, const Config& c);

// "req=P0+P1+...", "own=P0+P1+...", "gen=P": composite driver for merged multi-application traces (simulator/composite.h).
inline std::unique_ptr<Policy> makeComposite(const std::string& n, const Config& c) {
  const CompositePolicy::Mode m = n[0] == 'r' ? CompositePolicy::REQ : n[0] == 'o' ? CompositePolicy::OWN : CompositePolicy::GEN;
  std::vector<std::unique_ptr<Policy>> ps;
  std::string rest = n.substr(4), item;
  size_t pos = 0;
  while (pos <= rest.size()) {
    size_t e = rest.find('+', pos);
    if (e == std::string::npos) e = rest.size();
    item = rest.substr(pos, e - pos);
    auto p = item.empty() ? nullptr : makePolicy(item, c);
    if (!p) throw std::runtime_error("unknown policy inside " + n.substr(0, 4) + ": '" + item + "'");
    ps.push_back(std::move(p));
    pos = e + 1;
  }
  return std::unique_ptr<Policy>(new CompositePolicy(m, std::move(ps), n));
}

inline std::unique_ptr<Policy> makePolicy(const std::string& n, const Config& c) {
  if (n.rfind("req=", 0) == 0 || n.rfind("own=", 0) == 0 || n.rfind("gen=", 0) == 0) return makeComposite(n, c);
  if (n.rfind("mlp:", 0) == 0) return std::unique_ptr<Policy>(new LearnedPolicy(LearnedPolicy::MLP, n.substr(4)));
  if (n.rfind("lstm:", 0) == 0) return std::unique_ptr<Policy>(new LearnedPolicy(LearnedPolicy::LSTM, n.substr(5)));
  if (n.rfind("gbdt:", 0) == 0) return std::unique_ptr<Policy>(new LearnedPolicy(LearnedPolicy::GBDT, n.substr(5)));
  if (n == "lru") return std::unique_ptr<Policy>(new LruPolicy());
  if (n == "mru") return std::unique_ptr<Policy>(new MruPolicy());
  if (n == "fifo") return std::unique_ptr<Policy>(new FifoPolicy());
  if (n == "random") return std::unique_ptr<Policy>(new RandomPolicy(c.seed));
  if (n == "srrip") return std::unique_ptr<Policy>(new SrripPolicy(c.srrip_bits));
  if (n == "belady") return std::unique_ptr<Policy>(new BeladyPolicy());
  if (n == "bip") return std::unique_ptr<Policy>(new BipPolicy(c.seed, 32));
  if (n == "brrip") return std::unique_ptr<Policy>(new RripDuelPolicy(c.srrip_bits, c.seed, 32, false));
  if (n == "drrip") return std::unique_ptr<Policy>(new RripDuelPolicy(c.srrip_bits, c.seed, 32, true));
  return nullptr;
}

}  // namespace ac
