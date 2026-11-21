#include "ops/AneFfnCalibration.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::ops::ane_ffn {
namespace {

// The split's predicted gain below which the GPU runs the FFN alone.
constexpr double kMinimumGain = 0.05;
// How much longer than the GPU's alone a chunk's split layer may take and
// the chunk still split: six timings of each function's split layer of
// Qwen3.8-27B on an M5 Max spread over ±3%, and the FFN takes about half of
// a prefill, so a chunk that splits within this prefills at most about 1%
// slower.
constexpr double kChunkNoise = 0.02;
// The gain on a full chunk's split layer at which refine() moves a unit, and
// the most units it moves.
constexpr double kRefineGain = 0.02;
constexpr uint32_t kRefineSteps = 3;
// The units choosePair() climbs at a time, which refine() covers either way,
// and the gain on a full command at which a pair's split serves beside the
// single chunks': its memory comes out of the KV cache.
constexpr uint32_t kPairStride = kRefineSteps;
constexpr double kPairGain = 0.05;
// The time of a command no split runs.
constexpr double kUnrun = std::numeric_limits<double>::max();
// The first word of a Calibration's text, which a change to its fields
// renumbers.
constexpr std::string_view kCalibrationTag = "ane-ffn-calibration-2";
// The pair's units in a Calibration's text until a start has timed them.
constexpr std::string_view kUntimed = "-";

// The GPU alone's milliseconds of a chunk's FFN layer by its rows: the line
// through its timings, which the GPU follows within 1% between them.
Line gpuAlone(const std::array<ChunkTimings::Gpu, 2> &gpu) {
  const auto &[low, high] = gpu;
  const double slope = (high.milliseconds - low.milliseconds) / (high.rows - low.rows);
  return {low.milliseconds - slope * low.rows, slope};
}

} // namespace

double usable(double milliseconds) {
  if (!(milliseconds > 0.0 && std::isfinite(milliseconds)))
    throw std::runtime_error("calibration timings are not usable");
  return milliseconds;
}

uint32_t nearestUnits(double share, uint32_t units) {
  if (!(share > 0.0 && share < 1.0) || units < 2)
    throw std::invalid_argument("ANE FFN share must lie in (0, 1) of two units or more");
  return static_cast<uint32_t>(std::clamp<long>(std::lround(share * units), 1, units - 1));
}

// Not a number where either part is not.
double Model::operator()(double share) const noexcept {
  const double a = ane(share), g = gpu(share) + u * a;
  return a > g ? a : g;
}

// The model, from the split's layers as they run between stand-ins for the
// mixer (Measurement). The GPU's part alone gives G at both shares. At the
// high share, near 0.8, the ANE's part takes the longer on every Mac measured
// (Qwen3.8-27B's fastest split gives the ANE 8 to 27 of its 34 units): the
// split layer there is A, which scales with the ANE's channels. Where the
// GPU's part takes the longer at the low share, the split layer there exceeds
// it by the bandwidth the ANE's part takes beside it, which gives u (a tenth
// to a third on an M6). Where the ANE's part takes the longer there, as on an
// M5 Max, the split layer's excess over A is the ANE's own. Where the ANE's
// part does not take the longer at the high share, A comes out longer than it
// is, and the choice smaller.
Model fit(const Timing &low, const Timing &high) {
  if (!(low.share < high.share)) throw std::invalid_argument("ANE FFN calibration needs a low and a high share");
  const double slope = (high.gpu - low.gpu) / (high.share - low.share);
  Model model{{low.gpu - slope * low.share, slope}, {0.0, high.split / high.share}};
  const double gpu = model.gpu(low.share), ane = model.ane(low.share);
  if (gpu > ane) model.u = std::clamp((low.split - gpu) / ane, 0.0, 1.0);
  return model;
}

// The units whose T, or T a unit either way, is least at its longest: the
// timings can put T's least a unit off, and a unit past where the ANE's part
// takes the longer costs the ANE's time per unit, a unit short of it the
// GPU's. Where the ANE's is the longer (M5 Max: 1.7 ms a unit of Qwen3.8-27B's
// full-chunk layer against the GPU's 0.6), the choice keeps below T's least;
// where the GPU's is (M6), it need not. Chunks of fewer rows, which the ANE
// pads up to its functions' rows, move T's least lower still; minimumRows
// keeps the split from the chunks it loses. The choice runs only if T there
// gains kMinimumGain on the GPU alone.
std::optional<uint32_t> choose(const Model &model, double gpuAlone, uint32_t units, uint32_t maxAneUnits) {
  if (!maxAneUnits || maxAneUnits >= units) throw std::invalid_argument("ANE FFN choice of no feasible units");
  std::vector<double> at(units);
  for (uint32_t aneUnits = 1; aneUnits < units; ++aneUnits) {
    at[aneUnits] = model(static_cast<double>(aneUnits) / units);
    if (!(at[aneUnits] > 0.0 && std::isfinite(at[aneUnits]))) return std::nullopt;
  }
  const auto longest = [&](uint32_t aneUnits) {
    return std::max({at[aneUnits], at[std::max(aneUnits - 1, 1u)], at[std::min(aneUnits + 1, units - 1)]});
  };
  uint32_t chosen = 1;
  for (uint32_t aneUnits = 2; aneUnits <= maxAneUnits; ++aneUnits)
    if (longest(aneUnits) < longest(chosen)) chosen = aneUnits;
  if (!(at[chosen] <= (1.0 - kMinimumGain) * gpuAlone)) return std::nullopt;
  return chosen;
}

// T falls steeply on either side of where the parts meet: on an M1 Max with
// Qwen3.8-27B a 14K prefill runs 202 tok/s at 17 of 34 units and 193 at 15
// and at 19, so a choice two units off costs 5%. The fit can put it there on
// a Mac whose parts are not the lines it assumes, so the units it chooses are
// timed against their neighbours: a unit below first, then a unit above
// unless below gained, moving on while each unit takes kRefineGain less, for
// at most kRefineSteps units. A unit of the M1 Max's gains about 2.5% near
// the least, and timings that spread over a few percent move at most those
// steps.
uint32_t refine(uint32_t chosen, uint32_t maxAneUnits, const std::function<double(uint32_t aneUnits)> &timed) {
  if (!chosen || chosen > maxAneUnits) throw std::invalid_argument("ANE FFN refinement of no feasible units");
  uint32_t best = chosen;
  double least = usable(timed(chosen));
  for (const int direction : {-1, 1}) {
    for (uint32_t step = 0; step < kRefineSteps; ++step) {
      const long next = static_cast<long>(best) + direction;
      if (next < 1 || next > static_cast<long>(maxAneUnits)) break;
      const double milliseconds = usable(timed(static_cast<uint32_t>(next)));
      if (!(milliseconds <= (1.0 - kRefineGain) * least)) break;
      best = static_cast<uint32_t>(next);
      least = milliseconds;
    }
    if (best != chosen) break;
  }
  return best;
}

// A chunk splits while its function's split layer takes at most kChunkNoise
// longer than the GPU's layer alone of the chunk's rows (gpuAlone). Every
// chunk a function runs takes about as long as the function's own rows: the
// ANE's part runs them and takes the longer. The functions do not cost the
// ANE in proportion to their rows (on an M5 Max 640 rows took 93% of 768's,
// 512 rows 73%), so each is timed, and each row count it runs checked: a
// batch's chunk sums its lanes' rows.
std::optional<uint32_t> minimumRows(const ChunkTimings &timings) {
  const auto &[low, high] = timings.gpu;
  const std::vector<ChunkTimings::Function> &functions = timings.functions;
  bool ordered = !functions.empty() && low.rows < high.rows;
  for (size_t index = 0; ordered && index < functions.size(); ++index)
    ordered = functions[index].least && functions[index].least <= functions[index].rows &&
              (!index || functions[index].rows + 1 == functions[index - 1].least);
  if (!ordered) throw std::invalid_argument("ANE FFN chunks are not those of its functions");
  const Line gpu = gpuAlone(timings.gpu);
  std::optional<uint32_t> minimum;
  for (const ChunkTimings::Function &function : functions)
    for (uint32_t rows = function.rows; rows >= function.least; --rows) {
      if (!(function.split <= (1.0 + kChunkNoise) * gpu(rows))) return minimum;
      minimum = rows;
    }
  return minimum;
}

// A pair's Neural Engine works beside the other chunk's mixer, so its
// fastest units lie above the single chunks', by what the mixer takes of a
// layer, which no FFN layer timed alone shows: 25 of Qwen3.8-27B's 34 on an
// M1 Max against 17. The units climb from the single chunks' kPairStride at a
// time while each command takes kRefineGain less, then move to a neighbour's
// as refine() does. Units no split runs at gain nothing, which ends a climb
// where the ANE compiler refuses more channels.
std::optional<PairTiming> choosePair(uint32_t singleUnits, uint32_t mostUnits,
                                     const std::function<std::optional<double>(uint32_t pairUnits)> &timed) {
  if (!singleUnits || singleUnits > mostUnits) throw std::invalid_argument("ANE FFN pair of no feasible units");
  std::map<uint32_t, double> times;
  const auto time = [&](uint32_t units) {
    if (const auto found = times.find(units); found != times.end()) return found->second;
    const std::optional<double> took = timed(units);
    return times[units] = took ? usable(*took) : kUnrun;
  };
  uint32_t chosen = singleUnits;
  for (double least = kUnrun; chosen < mostUnits;) {
    const uint32_t next = std::min(chosen + kPairStride, mostUnits);
    const double took = time(next);
    if (took == kUnrun || !(took <= (1.0 - kRefineGain) * least)) break;
    chosen = next;
    least = took;
  }
  chosen = refine(chosen, mostUnits, time);
  if (time(chosen) == kUnrun) return std::nullopt;
  return PairTiming{chosen, time(chosen)};
}

bool pairGains(double single, double pair) { return usable(pair) <= (1.0 - kPairGain) * usable(single); }

std::string Calibration::text() const {
  std::ostringstream text;
  text << std::setprecision(9) << kCalibrationTag << ' ' << aneUnits << ' ' << minimumRows;
  for (const ChunkTimings::Gpu &point : gpu) text << ' ' << point.rows << ' ' << point.milliseconds;
  if (pairUnits)
    text << ' ' << *pairUnits;
  else
    text << ' ' << kUntimed;
  return text.str();
}

// A record of units holds least rows and the GPU alone at ascending rows of
// usable milliseconds; one of none holds neither, and no pair's units.
std::optional<Calibration> Calibration::parse(std::string_view text) {
  std::istringstream in{std::string(text)};
  std::string tag, pair;
  Calibration calibration;
  in >> tag >> calibration.aneUnits >> calibration.minimumRows;
  for (ChunkTimings::Gpu &point : calibration.gpu) in >> point.rows >> point.milliseconds;
  in >> pair;
  std::string rest;
  if (!in || tag != kCalibrationTag || (in >> rest)) return std::nullopt;
  if (pair != kUntimed) {
    uint32_t units = 0;
    const auto [end, error] = std::from_chars(pair.data(), pair.data() + pair.size(), units);
    if (error != std::errc{} || end != pair.data() + pair.size()) return std::nullopt;
    calibration.pairUnits = units;
  }
  const auto &[low, high] = calibration.gpu;
  const auto usableMilliseconds = [](double value) { return value > 0.0 && std::isfinite(value); };
  if (!calibration.aneUnits)
    return calibration.minimumRows || calibration.pairUnits ? std::nullopt : std::optional(Calibration{});
  if (!calibration.minimumRows || !(low.rows < high.rows) || !usableMilliseconds(low.milliseconds) ||
      !usableMilliseconds(high.milliseconds))
    return std::nullopt;
  return calibration;
}

// Each layer the GPU waits for the Neural Engine's evaluation before it joins
// the ANE's part to its own, so evaluations that take as long as the GPU's
// whole layers alone make the split slower than the GPU alone, whatever the
// rest of the layer takes: the breaker trips there, and never short of it. An
// evaluation counts against the GPU alone's layer of its function's rows,
// which the chunk pads up to, so the GPU alone would take no longer over the
// chunk; the reason scales the window's ratio to a layer of the most rows.
// At the units calibration chooses, at or just below where the parts meet
// (choose), an evaluation takes at most about as long as the GPU's part
// (Qwen3.8-27B on an M5 Max: 13.2 ms against the GPU alone's 19.0), and
// another process keeping the ANE busy costs the split 2-8%: far short of
// the trip point. Judging kWindow commands at once, it weighs one slow
// evaluation, as a function's first after its program loads again, among all
// of theirs.
Breaker::Breaker(const std::array<ChunkTimings::Gpu, 2> &gpu) {
  const auto &[low, high] = gpu;
  if (!(low.rows < high.rows)) throw std::invalid_argument("ANE FFN breaker needs the GPU alone at two row counts");
  static_cast<void>(usable(low.milliseconds));
  static_cast<void>(usable(high.milliseconds));
  gpu_ = gpuAlone(gpu);
  mostRows_ = high.rows;
}

std::string Breaker::add(uint32_t rows, uint32_t evaluations, double milliseconds) {
  if (!gpu_) return {};
  ane_ += milliseconds;
  alone_ += evaluations * (*gpu_)(rows);
  if (++commands_ < kWindow) return {};
  const double ratio = ane_ / alone_, most = (*gpu_)(mostRows_);
  commands_ = 0;
  ane_ = alone_ = 0.0;
  if (!(ratio >= 1.0)) return {};
  std::ostringstream reason;
  reason << std::fixed << std::setprecision(1) << "losing to the GPU alone (" << ratio * most
         << " ms on the Neural Engine per " << mostRows_ << "-row layer against " << most << " ms on the GPU alone)";
  return reason.str();
}

} // namespace splash::ops::ane_ffn
