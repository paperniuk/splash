#import "ane/Handoff.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace splash::ane {
namespace {

// Where a job of the current command stands: queued until the event reaches
// its `ready`, started from then until it reports, then succeeded or failed.
enum class Phase : uint8_t { Queued, Started, Succeeded, Failed };
// A job of the current command: its phase, when the handoff learned that the
// event reached its `ready`, and once it succeeded, how long after that it
// reported.
struct Job final {
  Phase phase = Phase::Queued;
  AwakeClock::time_point started{};
  AwakeClock::duration ran{};
};

std::string seconds(AwakeClock::duration duration) {
  char text[32];
  std::snprintf(text, sizeof text, "%g", std::chrono::duration<double>(duration).count());
  return text;
}

} // namespace

struct Handoff::State final {
  State(std::vector<metal::SharedEvent> events, AwakeClock::duration bound)
      : events(std::move(events)), bound(bound), values(this->events.size()) {}

  // Retires with `why` unless already retired, and raises each event to the
  // highest value allocated of it. Under `mutex`.
  void retire(std::string why) {
    if (!retired) {
      retired = true;
      reason = std::move(why);
    }
    for (size_t lane = 0; lane < events.size(); ++lane) events[lane].signal(values[lane]);
    changed.notify_all();
  }
  // Fails the job of `phase` with `why`. Under `mutex`.
  void fail(Phase &phase, std::string why) {
    phase = Phase::Failed;
    retire(std::move(why));
  }

  // The event reached the `ready` of job `id`: its bound starts.
  static void reached(std::shared_ptr<State> state, uint64_t id) noexcept {
    {
      std::lock_guard lock(state->mutex);
      const auto job = state->jobs.find(id);
      if (job == state->jobs.end() || job->second.phase != Phase::Queued) return;
      job->second.phase = Phase::Started;
      job->second.started = AwakeClock::now();
    }
    // dispatch_time(DISPATCH_TIME_NOW, ...) counts mach absolute time, which
    // stops while the Mac sleeps, as AwakeClock's CLOCK_UPTIME_RAW does.
    const auto bound = std::chrono::duration_cast<std::chrono::nanoseconds>(state->bound).count();
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, bound), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^{
                     expired(state, id);
                   });
  }
  // Job `id`'s bound passed.
  static void expired(const std::shared_ptr<State> &state, uint64_t id) noexcept {
    try {
      std::lock_guard lock(state->mutex);
      const auto job = state->jobs.find(id);
      if (job != state->jobs.end() && job->second.phase == Phase::Started)
        state->fail(job->second.phase, "an evaluation did not complete within " + seconds(state->bound) + " s");
    } catch (...) {
    }
  }
  // The agent's report of job `id`; only its first counts.
  static void reported(const std::shared_ptr<State> &state, uint64_t id, bool success) noexcept {
    try {
      std::lock_guard lock(state->mutex);
      if (!state->outstanding.erase(id)) return;
      state->changed.notify_all();
      const auto found = state->jobs.find(id);
      if (found == state->jobs.end()) return;
      Job &job = found->second;
      if (job.phase == Phase::Succeeded || job.phase == Phase::Failed) return;
      if (!success) {
        state->fail(job.phase, "an evaluation failed");
        return;
      }
      if (job.phase == Phase::Started) job.ran = AwakeClock::now() - job.started;
      job.phase = Phase::Succeeded;
    } catch (...) {
    }
  }

  std::mutex mutex;
  std::condition_variable changed;
  const std::vector<metal::SharedEvent> events;
  const AwakeClock::duration bound;
  // The highest value allocated of each event.
  std::vector<uint64_t> values;
  // The jobs of the command queued last, by id, until finish() or cancel().
  std::unordered_map<uint64_t, Job> jobs;
  // The jobs of any command the agent started and has not reported.
  std::unordered_set<uint64_t> outstanding;
  uint64_t lastJob = 0;
  bool retired = false;
  std::string reason;
};

Handoff::Handoff(metal::MetalBackend &backend, AwakeClock::duration bound, uint32_t lanes) {
  std::vector<metal::SharedEvent> events;
  for (uint32_t lane = 0; lane < std::max<uint32_t>(lanes, 1); ++lane) events.push_back(backend.newSharedEvent());
  state_ = std::make_shared<State>(std::move(events), bound);
}

Handoff::~Handoff() {
  try {
    cancel("the handoff ended");
  } catch (...) {
  }
}

const metal::SharedEvent &Handoff::event(uint32_t lane) const { return state_->events.at(lane); }

uint64_t Handoff::next(uint32_t lane) {
  std::lock_guard lock(state_->mutex);
  uint64_t &value = state_->values.at(lane);
  ++value;
  if (state_->retired) state_->events[lane].signal(value);
  return value;
}

void Handoff::queue(uint64_t ready, uint64_t done, const Start &start, uint32_t lane) {
  const metal::SharedEvent &event = state_->events.at(lane);
  uint64_t id = 0;
  {
    std::lock_guard lock(state_->mutex);
    id = ++state_->lastJob;
    if (state_->retired) {
      state_->jobs.emplace(id, Job{Phase::Failed});
      return;
    }
    state_->jobs.emplace(id, Job{});
    state_->outstanding.insert(id);
  }
  std::string failure;
  try {
    start(event, ready, done, [state = state_, id](bool success) { State::reported(state, id, success); });
    event.notify(ready, [state = state_, id] { State::reached(state, id); });
    return;
  } catch (const std::exception &error) {
    failure = std::string("an evaluation could not start: ") + error.what();
  } catch (...) {
    failure = "an evaluation could not start";
  }
  std::lock_guard lock(state_->mutex);
  state_->outstanding.erase(id);
  state_->fail(state_->jobs.at(id).phase, std::move(failure));
}

std::optional<AwakeClock::duration> Handoff::finish() {
  State &state = *state_;
  std::unique_lock lock(state.mutex);
  const auto settled = [&] {
    bool succeeded = true;
    for (const auto &[id, job] : state.jobs) {
      if (job.phase == Phase::Failed) return true;
      succeeded &= job.phase == Phase::Succeeded;
    }
    return succeeded;
  };
  if (!state.changed.wait_until(lock, AwakeClock::now() + state.bound, settled))
    for (auto &[id, job] : state.jobs)
      if (job.phase != Phase::Succeeded)
        state.fail(job.phase, "an evaluation did not report within " + seconds(state.bound) + " s of its command");
  // The time some job ran: lanes' jobs overlap.
  std::vector<std::pair<AwakeClock::time_point, AwakeClock::time_point>> spans;
  std::optional<AwakeClock::duration> ran = AwakeClock::duration{};
  for (const auto &[id, job] : state.jobs) {
    if (job.phase != Phase::Succeeded) {
      ran.reset();
      break;
    }
    if (job.ran > AwakeClock::duration{}) spans.emplace_back(job.started, job.started + job.ran);
  }
  std::ranges::sort(spans);
  AwakeClock::time_point covered{};
  for (const auto &[begin, end] : spans) {
    if (!ran) break;
    if (end > covered) *ran += end - std::max(begin, covered);
    covered = std::max(covered, end);
  }
  state.jobs.clear();
  return ran;
}

void Handoff::cancel(std::string reason) {
  State &state = *state_;
  std::unique_lock lock(state.mutex);
  state.retire(std::move(reason));
  state.jobs.clear();
  state.changed.wait_until(lock, AwakeClock::now() + state.bound, [&] { return state.outstanding.empty(); });
}

void Handoff::retire(std::string reason) {
  std::lock_guard lock(state_->mutex);
  state_->retire(std::move(reason));
}

bool Handoff::retired() const {
  std::lock_guard lock(state_->mutex);
  return state_->retired;
}

bool Handoff::idle() const {
  std::lock_guard lock(state_->mutex);
  return state_->outstanding.empty();
}

std::string Handoff::reason() const {
  std::lock_guard lock(state_->mutex);
  return state_->reason;
}

} // namespace splash::ane
