#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/AneFfnCalibration.hpp"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::ops {

namespace ane_ffn {
class Measurement;
} // namespace ane_ffn

// The dense FFN of a prefill chunk split by intermediate channel between the
// GPU and the Neural Engine. The GPU runs the leading channels with its
// prefill kernels over views of the projections' planes;
// the ANE runs the rest as one W8A8 program over the chunk's rows, with int8
// weights the GPU requantizes from those planes layer by layer into its
// surfaces, and the GPU adds the ANE's partial down projection to its own. A
// shared event orders each ANE evaluation between the GPU's packing of its
// inputs and that join, within the one prefill command (ane::Handoff). A
// command of two chunks runs them layer by layer, each in a slot of its own
// surfaces and event, so that the ANE runs one chunk's FFN beside the other's
// mixer on the GPU. Any failure of the ANE's work stops the split for the
// life of the process: finish() then reports the command's results unusable,
// and the caller runs the chunk again on the GPU alone, which every later
// chunk runs too. A split that loses to the GPU alone stops the same way,
// after a command whose results are usable (setBreaker).
class AneFfn final {
public:
  // A chunk runs on the smallest of the program's functions that holds it,
  // one every kProgramStep rows from kMinimumRows and one of the prefill
  // budget, since an evaluation costs the ANE its function's rows. This fork
  // starts at 128 rows: on M1 and M2 the GPU alone is slow enough that smaller
  // chunks gain (calibration on an M1 Max splits the 27B's chunks from 287
  // rows; on a 16-core M1 Pro chunks of 130-250 rows run 13-27% faster split),
  // and calibration still drops the functions that lose to it. The ANE
  // compiler refuses a program of more than kMaximumFunctions functions, so
  // the step's functions just below the budget go: their chunks run on the
  // budget's function. The chunks of a pair, at most half the budget, run on
  // the same functions.
  static constexpr uint32_t kMinimumRows = 128, kProgramStep = 128, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static constexpr size_t kMaximumFunctions = 15;
  static_assert(kMaximumRows >= kMinimumRows && (kMaximumRows - kMinimumRows) % kProgramStep == 0,
                "the functions cover the chunks the split takes");
  // The chunks of a command: each has its slot.
  static constexpr uint32_t kSlots = 2;
  // The GPU part's down projection is a view of leading inputs, which only
  // the prefill residual tiles (GgufPrefill, GgufPrefillInt8) run: no chunk
  // the split takes stages.
  static_assert(kMinimumRows > kMaximumStagedPrefillRows, "the split's chunks run the prefill residual tiles");

  // The ANE takes `aneUnits` of the units() of intermediate channels, and
  // the GPU the rest through `linear`'s prefill plans. The program is
  // compiled and loaded here, and the wait for the ANE's service ends with
  // ane::Interrupted once `interrupted` returns true. The split takes every
  // chunk its functions hold until setMinimumRows(). `mode` is how it runs
  // chunks and what memory it takes (plannedBytes()): Single, one chunk at a
  // time, each layer's weights staged during the layer before into one of two
  // sets, which hides the staging behind the GPU's part; Pair, a slot for each
  // chunk of a command of two (pairs()) over one set, staged once every
  // evaluation of the layer before is done; Fallback, one chunk at a time over
  // that one set, for a split that cannot be built as a Pair, in the memory a
  // Pair takes.
  enum class Mode : uint8_t { Single, Pair, Fallback };
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         uint32_t aneUnits, std::function<bool()> interrupted, Mode mode);
  // Ends what the event's notifications share with the split (commit()).
  ~AneFfn();
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // Why the split does not take `layers`, or null when it does: unrotated
  // quantized projections of one shape (AneFfn.cpp).
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units the split moves intermediate channels in, of `layers`, which
  // it takes: the ANE takes from one to all but one of them.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);
  // The Metal memory of the split of `layers` with the ANE taking `aneUnits`,
  // of a Pair or Fallback when `pairs`, a Single otherwise.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits,
                                             bool pairs);
  // What tells this Mac's calibrations of a split of `layers` apart for one
  // macOS version, engine build and most units its plan holds: the layers'
  // shapes and formats, and the Int8Activations of `linear`, whose plans run
  // the GPU's part and so move the units the split takes (Qwen3.8-27B MLX
  // 4-bit's takes 10 of 34 with them on, 15 off, on a 20-core M5 Pro).
  [[nodiscard]] static std::string calibrationKey(std::span<const SwiGluProjections> layers, const Linear &linear);

  // The fraction of intermediate channels the ANE takes.
  [[nodiscard]] double share() const noexcept;
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  // The least rows of a chunk the split takes (ane_ffn::minimumRows), from
  // kMinimumRows to kMaximumRows.
  [[nodiscard]] uint32_t minimumRows() const noexcept { return minimumRows_; }
  void setMinimumRows(uint32_t rows);
  // Stops the split once it loses to the GPU alone: `breaker` judges each
  // command finish() finds split, and `lost` runs once when it trips. Until
  // this is called, none does.
  void setBreaker(ane_ffn::Breaker breaker, std::function<void()> lost) noexcept {
    breaker_ = std::move(breaker);
    lost_ = std::move(lost);
  }

  // The commands finish() found split since the split was built, their
  // evaluations, and the Neural Engine's milliseconds over them
  // (ane::Handoff::finish).
  struct Served final {
    uint64_t commands = 0, evaluations = 0;
    double milliseconds = 0.0;
  };
  [[nodiscard]] const Served &served() const noexcept { return served_; }

  // Checks the split against the GPU alone on `layers`, those it was built
  // from, with chunks in `ffn` and the residual and output of alternate
  // layers in `hidden`, all of which it overwrites: each function in each
  // slot at its own rows over the first two layers, which stage the ANE's
  // weights twice, with the ANE's output filled with NaN before so that rows it
  // leaves unwritten show; each through begin(), the layers encoded as add()
  // encodes them, commit() and its completion check; each against the
  // leading rows of a full chunk on the GPU, which computes every row alike
  // whatever the chunk's rows. Returns the largest error of what the ANE
  // adds to the GPU's part (AneFfn.cpp). Throws if an output is not finite,
  // that error exceeds its bound or the ANE's work fails, which stops the
  // split.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden);

  // Whether the split takes a chunk of `rows` rows: from minimumRows() to
  // kMaximumRows, until it stops.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // Whether the split takes a chunk of `rows` rows as one of a command's two:
  // from kMinimumRows to half of kMaximumRows, where it was built with pairs,
  // until it stops.
  [[nodiscard]] bool pairs(uint32_t rows) const;
  // The rows of the first of the two chunks a command of `rows` rows runs as,
  // the second taking the rest; none where the split takes no such pair. The
  // first chunk fills its function. Of the pairs the split takes, the one
  // whose larger function is the smallest, an evaluation taking its
  // function's rows whatever the chunk's; of those the one whose functions
  // hold the fewest rows; and of those the one whose first chunk is the
  // smaller, the second's evaluation having the next layer's weight staging
  // to run beside as well. On an M1 Max, 577 rows of the 27B as 256 and 321
  // took 3.4 s, as 384 and 193 4.3 s, and 1058 rows as 192 and 866 6.1 s
  // against 4.9 s as 640 and 418. The caller decides whether the command
  // splits at all (splits()).
  [[nodiscard]] std::optional<uint32_t> pairRows(uint32_t rows) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts encoding a command; false once the split stopped.
  bool begin();
  // Layer `layer`'s FFN of a chunk of `rows` rows, from kMinimumRows to
  // kMaximumRows: output = residual + FFN of ffn.normalized. A command adds
  // its layers in order from layer 0 after begin(), and a layer once per
  // chunk, of at most kSlots. With an `owner`, which names the chunk to
  // settle(), `output` holds the layer's rows only once the chunk's join is
  // encoded: by settle(), by the chunk's add() of the next layer, or by the
  // next layer's first add(), which replaces the weights the evaluation
  // reads. Until then the Neural Engine works beside whatever the GPU is
  // given: with two chunks a layer, one chunk's FFN beside the other's mixer.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows, const void *owner = nullptr);
  // Encodes the joins add() left pending, `owner`'s or every chunk's for
  // null: before the command reads that chunk's output, and before commit().
  void settle(metal::CommandGraph &graph, const void *owner = nullptr);
  // Starts the ANE's evaluations of the command encoded since begin(), then
  // submits it (MetalBackend::submitCommandAsync). A submission that throws
  // stops the split before the exception goes on.
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion);
  // After the command committed last has completed: whether its outputs hold
  // the split's FFN. False when its ANE work failed, which stops the split
  // and logs why, once; true otherwise, also when the breaker stops the
  // split after it.
  [[nodiscard]] bool finish();

  // The engine's idle release (engine::ReleasableMemory). Between commands:
  // unloads the program, whose memory the ANE service gives back, and keeps
  // the split's own (allocatedBytes()), which the backend's residency
  // keep-alive unwires: the surfaces and scratch every command writes again,
  // and the row scales and signs computed once. True once it unloaded the
  // program; false, doing nothing, when it is unloaded, or the split stopped
  // while evaluations it gave up on may still use it. A stopped split unloads
  // it once they have all reported, and keeps it unloaded. A program that does
  // not unload stops the split. Throws std::logic_error while a command it
  // encoded is not committed or not finished.
  [[nodiscard]] bool release();
  // Loads the program release() unloaded again, unless the split stopped;
  // nothing otherwise. Never throws: a program that does not load stops the
  // split, and the GPU runs every later chunk alone.
  void restore() noexcept;

private:
  // Times the split's layers and their GPU part against the GPU's alone.
  friend class ane_ffn::Measurement;

  // The split's channels: hidden and intermediate, the layers', the GPU's and
  // the ANE's, and the segments of the hidden inputs and of down's inputs the
  // ANE takes.
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
    std::vector<uint32_t> inputSegments, downSegments;
  };
  // The ANE's int8 weights of a layer: gate's and up's
  // rows over each segment of the hidden inputs, down's rows over each
  // segment of its ANE inputs, and each one's per-row scales.
  struct Weights final {
    std::vector<ane::Surface> gate, up, down;
    ane::Surface gateScale, upScale, downScale;
  };
  // A slot's chunk: its rotated input rows in int8 segments, their per-token
  // scales and the ANE's partial down projection of them. They hold
  // kMaximumRows rows, whose row stride each function declares.
  struct Io final {
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
  };
  // Everything the split allocates (allocate() in AneFfn.cpp). A join that
  // reads a partial value or token scale that is not finite sets the status
  // word (ane_ffn_join).
  struct Memory final {
    metal::MetalBuffer signs, rowScales, rotated, status;
    std::array<Io, kSlots> io;
    // One set, or two with a Single's.
    std::array<Weights, 2> sets;
  };
  // A projection's weight planes as the ane_ffn kernels bind them: an image
  // tensor's plane0, plane1 and meta, in groups of 32 inputs, of their
  // `groups` per row, and its `format`.
  struct Planes final {
    // Throws unless the planes hold all of the projection's weights.
    explicit Planes(const Projection &projection);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0;
  };
  // A matrix of a layer: its index in Layer::planes, and the part of the
  // layer's per-row int8 scales rowScales() views.
  enum class Matrix : uint8_t { Gate, Up, Down };
  // A layer, held by value: the planes of its gate, up and down, from which
  // the GPU stages the ANE's int8 weights, and the GPU's share of it: gate
  // and up over views of their leading rows, down over the leading inputs of
  // its rows.
  struct Layer final {
    std::array<Planes, 3> planes;
    Projection gate, up, down;
  };
  // An input of the ANE program's functions: its name, the surface each
  // slot binds to it, and the rows and width the functions read of it, a
  // width of 0 reading the chunk's rows.
  struct Input final {
    std::string name;
    std::array<const ane::Surface *, kSlots> surfaces;
    uint32_t rows = 0, width = 0;
  };
  // What encode() adds of a split layer: the GPU's work alone (its channels,
  // the staging of the layer's int8 weights, and the packing and join of the
  // ANE's channels), or with the ANE's evaluation, which the slot's event
  // orders between the packing and the join.
  enum class Parts : uint8_t { Gpu, Both };
  // The program's function of `rows` rows over each slot's surfaces, by slot.
  struct Evaluation final {
    uint32_t rows = 0;
    std::vector<ane::Program::Binding> bindings;
  };
  // An evaluation of slot `slot` over its function's binding `binding`: it
  // starts once the slot's event reaches `ready`, and raises it to `done`.
  struct Job final {
    uint32_t evaluation, slot, binding;
    uint64_t ready, done;
  };

  // The split with a function of each of `functionRows` rows, ascending.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         uint32_t aneUnits, std::function<bool()> interrupted, std::span<const uint32_t> functionRows, Mode mode);
  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  // Every allocation of the split, which `buffer` and `surface` make:
  // plannedBytes() counts them and the constructor allocates them.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, uint32_t slots, uint32_t sets, MakeBuffer &&buffer,
                                       MakeSurface &&surface);
  // The program's inputs; the program over them, with a function of each of
  // `rows` rows; and its function of `rows` rows.
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory, uint32_t slots,
                                                        uint32_t sets);
  [[nodiscard]] static std::string program(const Shape &shape, std::span<const Input> inputs,
                                           const ane::Surface &output, std::span<const uint32_t> rows);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows);
  // Fills the CPU-visible bf16 rows `normalized` with the same values of a
  // normalized row's magnitude each time.
  static void fillNormalized(const metal::MetalBuffer &normalized);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts, const void *owner = nullptr);
  // Encodes slot `slot`'s pending join.
  void settleSlot(metal::CommandGraph &graph, uint32_t slot);
  // Layer `layer`'s int8 weights and row scales into set `set`, once no
  // evaluation reads it.
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  // Starts `job` on the ANE through the handoff.
  void queue(const Job &job);
  // finish() but for its log line and judgment: if every evaluation of the
  // command committed last succeeded and its joins read finite values, the
  // Neural Engine's time over them (ane::Handoff::finish); none otherwise,
  // which stops the split.
  [[nodiscard]] std::optional<AwakeClock::duration> completed();
  // Logs, once, that the split stopped.
  void warnStopped();

  metal::MetalBackend &backend_;
  const Linear linear_;
  Shape shape_;
  uint32_t minimumRows_ = kMinimumRows;
  // The slots and weight sets the split was built with (Mode).
  uint32_t slots_ = 1, sets_ = 2;
  std::vector<Layer> layers_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  // By rows, ascending.
  std::vector<Evaluation> evaluations_;
  // The jobs of the command being encoded, which commit() starts; the layer
  // it adds next, and the chunks added of the layer before it, which take
  // the slots from firstSlot_ on in turn (verify() checks each slot as the
  // first).
  std::vector<Job> jobs_;
  uint32_t nextLayer_ = 0, layerChunks_ = 0, firstSlot_ = 0;
  // A slot's join of the evaluation added last on it, until it is encoded:
  // the event value its evaluation raises (none for the GPU's part alone),
  // and the chunk's output rows.
  struct Pending final {
    const void *owner;
    std::optional<uint64_t> done;
    metal::MetalBuffer output;
    uint32_t rows;
  };
  std::array<std::optional<Pending>, kSlots> pending_;
  // The evaluations of the command committed last, its function's rows, and
  // whether it ran two chunks.
  uint32_t committedEvaluations_ = 0, committedRows_ = 0;
  bool committedPair_ = false;
  ane_ffn::Breaker breaker_;
  std::function<void()> lost_;
  Served served_;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  // From a release() that unloaded the program until restore().
  bool released_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  // The split as the event's notifications reach it to queue a command's
  // later jobs (commit()): null once it ends.
  struct Lifeline final {
    explicit Lifeline(AneFfn *split) : split(split) {}
    std::mutex mutex;
    AneFfn *split;
  };
  std::shared_ptr<Lifeline> lifeline_ = std::make_shared<Lifeline>(this);
  // Destroyed first: it waits for the evaluations the ANE has not reported.
  ane::Handoff handoff_;
};

} // namespace splash::ops
