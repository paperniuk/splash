#include "engine/AneFfnStartup.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "ane/Program.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::engine {
namespace {

namespace ane_ffn = ops::ane_ffn;
using Calibration = ane_ffn::Calibration;
using Kind = AneFfnOutcome::Kind;

// `tokens` with its thousands grouped, as the server prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

uint32_t contextOf(const EngineMemoryPlanResult &result) {
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// Why the split does not run on `model` as `setting` allows, whatever the
// memory plan holds: off as given, or not taken by the model or this Mac; none
// when it may run.
std::optional<std::pair<Kind, std::string>> notRunning(const AneFfnModel &model, const AneFfnSetting &setting) {
  if (!model.dense) return std::pair{Kind::Unsupported, std::string("the target's FFN layers are not all dense")};
  if (!setting.enabled) return std::pair{Kind::Off, std::string("as given")};
  if (!model.unsupported.empty()) return std::pair{Kind::Unsupported, model.unsupported};
  if (std::optional<std::string> reason = model.unavailable()) return std::pair{Kind::Unsupported, std::move(*reason)};
  return std::nullopt;
}

// The units of the least split `setting` lets `model` take: the given share's,
// else one.
uint32_t leastUnits(const AneFfnModel &model, const AneFfnSetting &setting) {
  return setting.given ? ane_ffn::nearestUnits(setting.given->share, model.units) : 1;
}

// Logs `outcome` of a start of `model` as `setting` allows: a model whose FFN
// layers are not all dense and one --disable-ane leaves alone without its
// layers said, and a refusal of a context the GPU alone does not hold either,
// which the start reports on its own (ServingRuntime::start), log nothing. A
// refusal of a relaunch's context is a plain line, without the way out the
// warning of a refused --max-context gives. An automatic context below the GPU
// alone's is named beside it.
void logOutcome(const AneFfnOutcome &outcome, const AneFfnModel &model, const AneFfnSetting &setting,
                RequestedContext requested) {
  const std::string context = !requested.tokens && outcome.context != outcome.contextWithout
                                  ? "; context " + grouped(outcome.context) + " tokens (" +
                                        grouped(outcome.contextWithout) + " with --disable-ane)"
                                  : "";
  switch (outcome.kind) {
  case Kind::Off:
    if (model.dense && model.unsupported.empty()) logLine("The GPU runs the prefill FFN alone, ", outcome.reason, ".");
    return;
  case Kind::Unsupported:
    if (model.dense) logLine("The GPU runs the prefill FFN alone: ", outcome.reason, ".");
    return;
  case Kind::NoGain:
    logLine("The GPU runs the prefill FFN alone: ", outcome.reason, context, ".");
    return;
  case Kind::Refused:
    if (requested.tokens > outcome.contextWithout) return;
    if (requested.relaunch)
      logLine("Neural Engine FFN split off: ", outcome.reason, ".");
    else if (requested.tokens && outcome.context)
      logWarning("Neural Engine FFN split off: ", outcome.reason, ", and ",
                 setting.given ? "the given share" : "any split", " leaves at most ", grouped(outcome.context),
                 " tokens; pass --max-context ", outcome.context,
                 " or less to run it, or --disable-ane, which also silences this line.");
    else
      logWarning("Neural Engine FFN split off: ", outcome.reason, "; --disable-ane silences this line.");
    return;
  case Kind::Unavailable:
    logWarning("Neural Engine FFN split unavailable (", outcome.reason, "); the GPU runs the prefill FFN alone",
               context, ". --disable-ane silences this line.");
    return;
  case Kind::Split:
    logLine("Neural Engine FFN split ", outcome.reason, context, ".");
    return;
  }
}

// The split of `pairUnits` that takes the commands a calibrated split runs as
// pairs of chunks.
std::unique_ptr<ops::AneFfn> pairSplit(const AneFfnModel &model, uint32_t pairUnits) {
  std::unique_ptr<ops::AneFfn> pair = model.prepare(pairUnits, false, ops::AneFfn::Mode::Pair).split;
  if (pair) pair->setPairMinimumRows(AneFfnSetting::kPairMinimumRows);
  return pair;
}

// AneFfnStart::timePairs of a split calibrated as `calibration`, which
// `maxAneUnits` remembers, with at most `mostUnits` for the pair's split
// (ops::ane_ffn::choosePair). The split that serves holds every function, of
// which the ANE compiler refuses more channels than of the one the commands
// are timed on (0.94 of Qwen3.8-27B's on an M1 Pro, where 0.88 compiles):
// units that do not build give way to fewer, and the next start loads what
// this one compiled.
void timePairs(const AneFfnModel &model, ane_ffn::Calibration calibration, uint32_t maxAneUnits, uint32_t mostUnits,
               const std::function<bool()> &cancelled, const AneFfnPairPrefill &prefill) {
  logLine("Timing the Neural Engine FFN split's pairs of chunks on whole prefills.");
  const auto started = AwakeClock::now();
  const auto fixed = [](double value, int digits) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(digits) << value;
    return text.str();
  };
  const auto share = [&](uint32_t pairUnits) { return fixed(static_cast<double>(pairUnits) / model.units, 2); };
  // Whether `build` made a split of `pairUnits` that runs.
  const auto runs = [&](uint32_t pairUnits, const std::function<void()> &build) {
    try {
      build();
      return true;
    } catch (const ane::Interrupted &) {
      throw;
    } catch (const std::exception &error) {
      if ((cancelled && cancelled()) || !model.healthy()) throw;
      logLine("Neural Engine FFN split runs no pair of chunks at share ", share(pairUnits), " (", error.what(), ").");
      return false;
    }
  };
  try {
    // One slow command of single chunks would show a gain where there is
    // none: the lesser of two.
    const double single = std::min(prefill(0, nullptr), prefill(0, nullptr));
    const std::optional<ane_ffn::PairTiming> pair =
        ane_ffn::choosePair(calibration.aneUnits, mostUnits, [&](uint32_t pairUnits) -> std::optional<double> {
          double seconds = 0.0;
          const bool ran = runs(pairUnits, [&] {
            const std::unique_ptr<ops::AneFfn> split = model.timingPair(pairUnits);
            seconds = prefill(pairUnits, split.get());
            if (split && split->retired()) throw std::runtime_error(split->reason());
          });
          return ran ? std::optional(seconds) : std::nullopt;
        });
    uint32_t serves = pair && ane_ffn::pairGains(single, pair->time) ? pair->units : 0;
    while (serves >= calibration.aneUnits && !runs(serves, [&] { static_cast<void>(pairSplit(model, serves)); }))
      --serves;
    if (serves < calibration.aneUnits) serves = 0;
    calibration.pairUnits = serves;
    model.remember(maxAneUnits, calibration);
    std::ostringstream line;
    line << "Neural Engine FFN split runs ";
    if (serves)
      line << "commands of " << AneFfnSetting::kPairMinimumRows << " rows or more as pairs of chunks at share "
           << share(serves) << " from the next start";
    else
      line << "no pairs of chunks";
    if (pair)
      line << ": " << (serves ? "" : "at share " + share(pair->units) + " ") << "a " << ops::AneFfn::kMaximumRows
           << "-row prefill takes " << fixed(pair->time, 2) << " s against " << fixed(single, 2)
           << " as a single chunk";
    line << " (timed in " << fixed(std::chrono::duration<double>(AwakeClock::now() - started).count(), 1) << " s).";
    logLine(line.str());
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    logWarning("Neural Engine FFN split's pairs of chunks were not timed (", error.what(), ").");
  }
}

AneFfnStart decide(const AneFfnModel &model, const AneFfnSetting &setting, RequestedContext requested,
                   const std::function<EngineMemoryPlanResult(uint64_t)> &planMemory,
                   const std::function<bool()> &cancelled) {
  const uint32_t without = contextOf(planMemory(0));
  const auto gpuAlone = [&](Kind kind, std::string reason, uint32_t context) {
    return AneFfnStart{{}, {}, {kind, std::move(reason), context, without}};
  };
  if (std::optional<std::pair<Kind, std::string>> off = notRunning(model, setting))
    return gpuAlone(off->first, std::move(off->second), without);
  const std::optional<uint32_t> given =
      setting.given ? std::optional(ane_ffn::nearestUnits(setting.given->share, model.units)) : std::nullopt;
  if (setting.given && (setting.given->minimumRows < ops::AneFfn::kMinimumRows ||
                        setting.given->minimumRows > ops::AneFfn::kMaximumRows))
    throw std::invalid_argument("ANE FFN split has no function of " + std::to_string(setting.given->minimumRows) +
                                " rows");

  // Feasibility. The split's memory comes out of the KV cache, more of it
  // with each unit the ANE takes: the plan must still hold the context asked
  // for, or with none asked some context, which becomes the automatic one.
  // A given split may run pairs of chunks, which take their own memory. A
  // calibrated split's pairs run on a second split, of `pairUnits`, whose
  // memory is set aside beside the first's.
  const bool pairs = setting.given.has_value();
  const auto planPaired = [&](uint32_t aneUnits, uint32_t pairUnits) {
    return planMemory(model.plannedBytes(aneUnits, pairs) + (pairUnits ? model.plannedBytes(pairUnits, true) : 0));
  };
  const auto planWith = [&](uint32_t aneUnits) { return planPaired(aneUnits, 0); };
  const auto contextWith = [&](uint32_t aneUnits) { return contextOf(planWith(aneUnits)); };
  // The most units from `low`, which `holds`, to `high`, that it holds.
  const auto most = [&](uint32_t low, uint32_t high, const std::function<bool(uint32_t)> &holds) {
    while (low < high) {
      const uint32_t middle = low + (high - low + 1) / 2;
      if (holds(middle))
        low = middle;
      else
        high = middle - 1;
    }
    return low;
  };
  const uint32_t least = leastUnits(model, setting);
  const uint32_t context = std::max<uint32_t>(requested.tokens, 1);
  if (const uint32_t leaves = contextWith(least); leaves < context) {
    if (!requested.tokens) return gpuAlone(Kind::Refused, "it leaves no memory for context", without);
    const std::string asked = requested.relaunch
                                  ? "the first engine's context of " + grouped(requested.tokens) + " tokens"
                                  : "--max-context " + std::to_string(requested.tokens);
    return gpuAlone(Kind::Refused,
                    requested.tokens > without ? "the GPU alone does not hold " + asked + " either"
                                               : asked + " cannot be held with it",
                    leaves);
  }
  // The most units whose plan holds the context asked for, or with none
  // asked the most whose plan holds any.
  const uint32_t maxAneUnits =
      given ? least : most(least, model.units - 1, [&](uint32_t aneUnits) { return contextWith(aneUnits) >= context; });
  // Whether the plan holds the context with a pair's split of `pairUnits`
  // beside a split of `aneUnits`.
  const auto holdsPair = [&](uint32_t aneUnits, uint32_t pairUnits) {
    return contextOf(planPaired(aneUnits, pairUnits)) >= context;
  };
  // The units of the pair's split the plan sets aside beside a split
  // calibrated as `calibration`: those the calibration holds, where the plan
  // holds the context with them; 0 for none.
  const auto pairUnitsOf = [&](const ane_ffn::Calibration &calibration) {
    const uint32_t pairUnits = calibration.pairUnits.value_or(0);
    return pairUnits && holdsPair(calibration.aneUnits, pairUnits) ? pairUnits : 0;
  };
  // `aneUnits` as a share of the model's units, in a log's words.
  const auto share = [&](uint32_t aneUnits) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2) << static_cast<double>(aneUnits) / model.units;
    return text.str();
  };
  // The automatic context of a start whose split takes `aneUnits` and its
  // pair's `pairUnits`, the GPU alone's for none: what the plan holds with
  // them.
  const auto automaticWith = [&](uint32_t aneUnits, uint32_t pairUnits) {
    return aneUnits ? contextOf(planPaired(aneUnits, pairUnits)) : without;
  };
  // The units the automatic context assumes: this Mac's calibration once
  // remembered, so that every start that takes it serves the same context,
  // whatever it makes of the split, and until then the most.
  uint32_t assumed = maxAneUnits, assumedPair = 0;
  const auto gpuAloneServing = [&](Kind kind, std::string reason) {
    return gpuAlone(kind, std::move(reason), automaticWith(assumed, assumedPair));
  };

  logLine("Setting up the Neural Engine FFN split (splash serve --disable-ane keeps the FFN on the GPU).");
  const auto started = AwakeClock::now();
  const auto seconds = [&] {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << std::chrono::duration<double>(AwakeClock::now() - started).count()
         << " s";
    return text.str();
  };
  const std::string layer = " ms per " + std::to_string(ops::AneFfn::kMaximumRows) + "-row FFN layer";
  const auto fixed = [](double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << value;
    return text.str();
  };
  // The split that serves, with what the start found of it: its share, its
  // least chunk, the timings that chose them if it calibrated, the error
  // verify() found and how it was set up; with the memory of a pair's split
  // of `pairUnits` set aside, and what `paired` says of it.
  const auto serve = [&](AneFfnPrepared prepared, uint32_t aneUnits, uint32_t minimumRows, const std::string &timed,
                         const char *how, uint32_t pairUnits = 0, const std::string &paired = "") {
    EngineMemoryPlanResult plan = planPaired(aneUnits, pairUnits);
    if (!plan.plan) throw std::logic_error("the memory plan does not hold the split it chose");
    std::ostringstream reason;
    reason << std::fixed << std::setprecision(2) << "at " << (given ? "the given share " : "share ")
           << static_cast<double>(aneUnits) / model.units << " for chunks of " << minimumRows << " rows or more"
           << paired << timed << ", " << fixed(100.0 * prepared.error)
           << "% from the GPU alone on the Neural Engine's part (" << how << " in " << seconds() << ")";
    return AneFfnStart{std::move(prepared.split), std::move(plan.plan),
                       {Kind::Split, reason.str(), automaticWith(aneUnits, pairUnits), without}};
  };
  // A calibrated split that serves, and its pair's as `calibration` holds
  // it. A pair's split that cannot be built leaves the commands single
  // chunks, in the same memory plan. A calibration that holds no pair's
  // units yet has them timed within those the plan would hold.
  const auto servePaired = [&](AneFfnPrepared prepared, const ane_ffn::Calibration &calibration,
                               const std::string &timed, const char *how) {
    const uint32_t pairUnits = pairUnitsOf(calibration), aneUnits = calibration.aneUnits;
    std::unique_ptr<ops::AneFfn> pair;
    std::string paired;
    if (pairUnits) {
      try {
        pair = pairSplit(model, pairUnits);
        paired = ", and at " + share(pairUnits) + " for commands of " +
                 std::to_string(AneFfnSetting::kPairMinimumRows) + " rows or more as pairs of chunks";
      } catch (const ane::Interrupted &) {
        throw;
      } catch (const std::exception &error) {
        if ((cancelled && cancelled()) || !model.healthy()) throw;
        logWarning("Neural Engine FFN split cannot run commands as pairs of chunks (", error.what(),
                   "); they run as single chunks.");
      }
    }
    AneFfnStart start = serve(std::move(prepared), aneUnits, calibration.minimumRows, timed, how, pairUnits, paired);
    start.pair = std::move(pair);
    if (!calibration.pairUnits && holdsPair(aneUnits, aneUnits)) {
      const uint32_t mostUnits = most(aneUnits, model.units - 1,
                                      [&](uint32_t pairUnits) { return holdsPair(aneUnits, pairUnits); });
      start.timePairs = [calibration, maxAneUnits, mostUnits, cancelled](const AneFfnModel &model,
                                                                         const AneFfnPairPrefill &prefill) {
        timePairs(model, calibration, maxAneUnits, mostUnits, cancelled, prefill);
      };
    }
    return start;
  };
  // This Mac's calibration as recalled; what the next start does once this one tried to remember `record`.
  std::optional<Calibration> calibration;
  const auto keep = [&](const Calibration &record) {
    return Calibration::nextStart(model.remember(maxAneUnits, record) ? record : calibration);
  };
  try {
    if (given) {
      // A split that cannot give a pair its slots still serves single chunks.
      AneFfnPrepared prepared;
      try {
        prepared = model.prepare(*given, false, ops::AneFfn::Mode::Pair);
      } catch (const ane::Interrupted &) {
        throw;
      } catch (const std::exception &error) {
        if ((cancelled && cancelled()) || !model.healthy()) throw;
        logWarning("Neural Engine FFN split cannot run a command as a pair of chunks (", error.what(),
                   "); its chunks run one at a time.");
        prepared = model.prepare(*given, false, ops::AneFfn::Mode::Fallback);
      }
      if (prepared.split) prepared.split->setMinimumRows(setting.given->minimumRows);
      return serve(std::move(prepared), *given, setting.given->minimumRows, "", "set up");
    }
    // The split takes the calibration's least chunk, and judges itself
    // against the GPU alone as calibrated, forgetting the calibration once it
    // loses.
    const auto take = [&](AneFfnPrepared &prepared, const ane_ffn::Calibration &calibration) {
      if (!prepared.split) return;
      prepared.split->setMinimumRows(calibration.minimumRows);
      prepared.split->setBreaker(ane_ffn::Breaker(calibration.gpu),
                                 [forget = model.forget, maxAneUnits] { forget(maxAneUnits); });
    };
    // No split gains enough: the start remembers so, and the GPU alone serves
    // its own context.
    const auto noGain = [&](std::string reason) {
      model.remember(maxAneUnits, {});
      assumed = 0;
      return gpuAloneServing(Kind::NoGain, std::move(reason));
    };

    // Why `step` failed on a backend that serves and in a start that goes on,
    // or none, and whether the Neural Engine refused the split.
    bool refusal = false;
    const auto failure = [&](const auto &step) -> std::optional<std::string> {
      try {
        step();
        return std::nullopt;
      } catch (const ane::Interrupted &) {
        throw;
      } catch (const std::exception &error) {
        if ((cancelled && cancelled()) || !model.healthy()) throw;
        refusal = dynamic_cast<const ane::Refused *>(&error);
        return error.what();
      }
    };
    calibration = model.recall(maxAneUnits);
    if (calibration && calibration->aneUnits &&
        (calibration->aneUnits > maxAneUnits || calibration->minimumRows < ops::AneFfn::kMinimumRows ||
         calibration->minimumRows > ops::AneFfn::kMaximumRows))
      calibration.reset();
    if (calibration && (calibration->aneUnits || calibration->refusals != 1)) {
      assumed = calibration->aneUnits;
      if (!calibration->aneUnits)
        return gpuAloneServing(calibration->refusals ? Kind::Unavailable : Kind::NoGain, calibration->withoutSplit());
      assumedPair = pairUnitsOf(*calibration);
      // The Neural Engine's refusal counts in the split's place; another failure keeps it.
      AneFfnPrepared prepared;
      if (const auto reason = failure([&] { prepared = model.prepare(calibration->aneUnits, false, ops::AneFfn::Mode::Single); }))
        return gpuAloneServing(Kind::Unavailable, "its remembered share's split was not prepared: " + *reason + "; " +
                                                      keep(refusal ? ane_ffn::refused(calibration) : *calibration));
      take(prepared, *calibration);
      return servePaired(std::move(prepared), *calibration, "", "set up as calibrated before");
    }

    const ane_ffn::Timings timings = model.time();
    const auto choice = [&](const ane_ffn::Model &fitted) {
      return ane_ffn::choose(fitted, timings.gpuAlone, model.units, maxAneUnits);
    };
    const ane_ffn::Model fitted = ane_ffn::fit(timings.low, timings.high);
    const std::optional<uint32_t> first = choice(fitted);
    if (!first)
      return noGain("no Neural Engine split beats its " + fixed(timings.gpuAlone) + layer +
                    " by enough (calibrated in " + seconds() + ")");
    const auto timedSplit = [&](uint32_t aneUnits) {
      AneFfnPrepared split = model.prepare(aneUnits, true, ops::AneFfn::Mode::Single);
      if (split.chunks.functions.empty()) throw std::logic_error("the split's chunks were not timed");
      return split;
    };
    // The split's full chunk at the first units, timed nearest T's least as
    // the fit's were and on their layers, before their split is prepared, so
    // no two splits are held at once, fits u again (ane_ffn::refit); the
    // units that model chooses serve where their split is prepared and takes
    // no chunk longer (ane_ffn::slowerNowhere), else the first ones, whose
    // split prepared: a refit whose split cannot be timed or prepared, on a
    // backend that serves and in a start that goes on, says nothing against
    // them, and a first split that does not prepare again leaves this start
    // the GPU alone and the next its calibration. Either ended the start with
    // the GPU alone, nothing remembered, so that every start calibrated again
    // and ended so.
    const auto refitting = [&](const char *failed, const auto &step) {
      if (const std::optional<std::string> reason = failure(step))
        logLine("The Neural Engine FFN split's refitted share was not ", failed, " (", *reason,
                "); its first share serves.");
    };
    std::optional<uint32_t> refitted;
    refitting("found", [&] {
      refitted = choice(ane_ffn::refit(fitted, static_cast<double>(*first) / model.units,
                                       model.timeAt(*first, timings.gpuAlone).split));
    });
    AneFfnPrepared prepared = timedSplit(*first);
    uint32_t chosen = *first;
    std::optional<std::string> unprepared;
    if (refitted && *refitted != *first) {
      const ane_ffn::ChunkTimings firstChunks = std::move(prepared.chunks);
      prepared.split.reset();
      std::optional<AneFfnPrepared> candidate;
      refitting("prepared", [&] { candidate = timedSplit(*refitted); });
      if (candidate && ane_ffn::slowerNowhere(candidate->chunks, firstChunks)) {
        prepared = std::move(*candidate);
        chosen = *refitted;
      } else {
        candidate.reset();
        unprepared = failure([&] { prepared = model.prepare(*first, false, ops::AneFfn::Mode::Single); });
        prepared.chunks = firstChunks;
      }
    }
    const ane_ffn::ChunkTimings &chunks = prepared.chunks;
    const std::string timed =
        fixed(chunks.functions.front().split) + layer + " against " + fixed(chunks.gpu[1].milliseconds);
    const std::optional<uint32_t> minimumRows = ane_ffn::minimumRows(chunks);
    if (!minimumRows)
      return noGain("no chunk of the Neural Engine split beats it by enough, " + timed + " (calibrated in " +
                    seconds() + ")");
    const Calibration made{chosen, *minimumRows, chunks.gpu, calibration ? calibration->refusals : 0};
    if (unprepared) {
      assumed = chosen;
      return gpuAloneServing(Kind::Unavailable, "its calibrated share's split was not prepared again: " + *unprepared +
                                                    "; " + keep(refusal ? ane_ffn::refused(calibration, made) : made));
    }
    take(prepared, made);
    AneFfnStart start =
        servePaired(std::move(prepared), made, ": " + timed + " on the GPU alone", "calibrated");
    model.remember(maxAneUnits, made);
    return start;
  } catch (const ane::Interrupted &) {
    throw;
  } catch (const std::exception &error) {
    if ((cancelled && cancelled()) || !model.healthy()) throw;
    if (given || !dynamic_cast<const ane::Refused *>(&error)) return gpuAloneServing(Kind::Unavailable, error.what());
    return gpuAloneServing(Kind::Unavailable, error.what() + ("; " + keep(ane_ffn::refused(calibration))));
  }
}

} // namespace

std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept {
  switch (kind) {
  case Kind::Off:
    return "off";
  case Kind::Unsupported:
    return "unsupported";
  case Kind::Refused:
    return "refused";
  case Kind::NoGain:
    return "no_gain";
  case Kind::Unavailable:
    return "unavailable";
  case Kind::Split:
    return "split";
  }
  return "unknown";
}

AneFfnSetting AneFfnSetting::fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows) {
  if (!share) {
    if (minimumRows) throw std::invalid_argument("ANE FFN least chunk rows given without a share");
    return {};
  }
  if (*share == 0.0) return {.enabled = false};
  return {.given = Given{*share, minimumRows.value_or(kGivenMinimumRows)}};
}

AneFfnStart startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, RequestedContext requested,
                        const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
                        const std::function<bool()> &cancelled) {
  AneFfnStart start = decide(model, setting, requested, planMemory, cancelled);
  logOutcome(start.outcome, model, setting, requested);
  return start;
}

bool aneFfnRefused(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
                   const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory) {
  return !notRunning(model, setting) &&
         contextOf(planMemory(model.plannedBytes(leastUnits(model, setting), setting.given.has_value()))) <
             std::max<uint32_t>(requestedContextTokens, 1);
}

} // namespace splash::engine
