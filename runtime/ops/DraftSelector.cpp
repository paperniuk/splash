#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary) : vocabulary_(vocabulary) {
  if (!vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * SPLASH_DRAFT_CANDIDATES;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kShards * sizeof(uint32_t),
          candidates * (kShards + SPLASH_DRAFT_CANDIDATES) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float)};
}

void DraftSelector::add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks, DraftVocabulary vocabulary,
                        std::span<const DraftTreeLane> lanes) const {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  if (vocabulary.leading > vocabulary.trailing || vocabulary.trailing > vocabulary_ || !vocabulary.columns(vocabulary_))
    throw std::invalid_argument("a draft's tokens lie in its vocabulary");
  const uint32_t count = static_cast<uint32_t>(lanes.size());
  const uint32_t columns = vocabulary.columns(vocabulary_);
  SelectorBatchParams params{};
  params.lanes = count;
  params.vocabulary = vocabulary_;
  params.columns = columns;
  params.leading = vocabulary.leading;
  params.trailing = vocabulary.trailing;
  DraftTreeParams tree{};
  tree.lanes = count;
  for (uint32_t lane = 0; lane < count; ++lane) {
    const DraftTreeLane &l = lanes[lane];
    // A tree's rows hold its anchor and every row's path below it, a verify
    // block's rows.
    if (l.depth >= SPLASH_TARGET_VERIFY_ROWS || l.chain > SPLASH_TREE_CHAIN_TOKENS ||
        l.children > SPLASH_TREE_CHILDREN)
      throw std::invalid_argument("invalid draft tree lane");
    if (!std::isfinite(l.calibration.slope) ||
        !std::ranges::all_of(l.calibration.intercepts, [](float value) { return std::isfinite(value); }))
      throw std::invalid_argument("invalid draft tree calibration");
    if (!(l.temperature >= 0.0F && std::isfinite(l.temperature)))
      throw std::invalid_argument("invalid draft tree temperature");
    params.anchor[lane] = tree.anchor[lane] = l.anchor;
    tree.depth[lane] = l.depth;
    tree.chain[lane] = l.chain;
    tree.children[lane] = l.children;
    tree.slope[lane] = l.calibration.slope;
    std::ranges::copy(l.calibration.intercepts, tree.intercept[lane]);
    tree.temperature[lane] = l.temperature;
    tree.noise[lane][0] = static_cast<uint32_t>(l.noise);
    tree.noise[lane][1] = static_cast<uint32_t>(l.noise >> 32);
  }
  // A lane's proposal positions take rows 1-7 of its eight query rows of the
  // logits and the selector hidden rows; its tree takes a verify block's rows.
  const DraftSelectorWorkspace workspace = DraftSelector::workspace(count * kPositions);
  const uint64_t queryRows = uint64_t{count} * SPLASH_DRAFT_QUERY_ROWS;
  const uint64_t codebook = uint64_t{vocabulary_} * SPLASH_DRAFT_SELECTOR_RANK * 2;
  const uint64_t treeRows = uint64_t{count} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.logits, queryRows * columns * sizeof(float), "draft logits");
  requireBytes(buffers.partialIds, workspace.partialIdsBytes, "draft selector partial id");
  requireBytes(buffers.partialValues, workspace.partialValuesBytes, "draft selector partial value");
  requireBytes(buffers.candidates, workspace.candidatesBytes, "draft candidate");
  requireBytes(buffers.unary, workspace.unaryBytes, "draft candidate score");
  requireBytes(buffers.selectorHidden, queryRows * SPLASH_DRAFT_SELECTOR_RANK * 2, "draft selector hidden");
  requireBytes(codebooks.predecessor, codebook, "draft predecessor codebook");
  requireBytes(codebooks.successor, codebook, "draft successor codebook");
  requireBytes(buffers.chainTokens, uint64_t{count} * SPLASH_TREE_CHAIN_TOKENS * sizeof(uint32_t),
               "draft tree chain token");
  requireBytes(buffers.chainRates,
               uint64_t{count} * SPLASH_TREE_CHAIN_TOKENS * SPLASH_TREE_AGREEMENTS * sizeof(float),
               "draft tree chain rate");
  requireBytes(buffers.treeTokens, treeRows * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, treeRows * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, treeRows * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, uint64_t{count} * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.treeDraws, treeRows * SPLASH_TREE_CHILDREN, "draft tree draw");
  requireBytes(buffers.treeNext, treeRows * SPLASH_DRAFT_CANDIDATES * sizeof(float), "draft tree next");
  requireBytes(buffers.treeCaps, treeRows * sizeof(float), "draft tree cap");
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            params, {uint64_t{count} * kPositions * kShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{count} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  // A group's threads take the lattice's rows, one a position and
  // predecessor, before its first thread grows the tree.
  graph.add("draft_select_tree",
            {buffers.candidates, buffers.unary, buffers.partialValues, buffers.chainTokens, buffers.chainRates,
             buffers.treeTokens, buffers.trees, buffers.treeScores, buffers.treeSizes, buffers.treeDraws,
             buffers.treeNext, buffers.treeCaps},
            tree, {count, 1, 1}, {kPositions * SPLASH_DRAFT_CANDIDATES, 1, 1});
}

void DraftSelector::addVerifyRows(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
                                  const StepRows &step, float floor) {
  requireStepRows(step);
  if (!(floor >= 0.0F && std::isfinite(floor)))
    throw std::invalid_argument("invalid verify rows floor");
  const VerifyRowsParams params{.lanes = step.lanes, .rows = step.rows, .lane_cap = step.laneCap, .floor = floor};
  // It reads each lane's tree and writes the step's rows.
  const uint64_t treeRows = uint64_t{step.lanes} * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(buffers.treeTokens, treeRows * sizeof(uint32_t), "draft tree token");
  requireBytes(buffers.trees, treeRows * sizeof(TreeRow), "draft tree row");
  requireBytes(buffers.treeScores, treeRows * 2 * sizeof(float), "draft tree score");
  requireBytes(buffers.treeSizes, uint64_t{step.lanes} * sizeof(uint32_t), "draft tree size");
  requireBytes(buffers.inputTokens, uint64_t{step.rows} * sizeof(uint32_t), "verify input token");
  requireBytes(buffers.treeRows, uint64_t{step.rows} * sizeof(TreeRow), "verify tree row");
  requireBytes(buffers.laneRows, uint64_t{step.lanes} * sizeof(LaneRows), "verify lane rows");
  requireBytes(buffers.verifySources, uint64_t{step.rows} * sizeof(uint32_t), "verify source");
  graph.add("draft_verify_rows",
            {buffers.treeTokens, buffers.trees, buffers.treeScores, buffers.treeSizes, buffers.inputTokens,
             buffers.treeRows, buffers.laneRows, buffers.verifySources},
            params, {1, 1, 1}, {1, 1, 1});
}

void DraftSelector::addDSparkContext(metal::CommandGraph &graph, const DSparkMarkovBuffers &b, uint32_t rows) {
  if (!rows || rows > 7) throw std::invalid_argument("invalid DSpark recurrent context rows");
  requireBytes(b.hidden, 8 * 2304 * 2, "DSpark Markov hidden");
  requireBytes(b.feature, 2304 * 2, "DSpark Markov feature");
  requireBytes(b.recurrentWeights, uint64_t{768} * 5120 * 2, "DSpark recurrent weights");
  requireBytes(b.recurrentBias, 768 * 2, "DSpark recurrent bias");
  requireBytes(b.context, 7 * 768 * 4, "DSpark recurrent context");
  requireBytes(b.state, 256 * 2, "DSpark recurrent state");
  graph.add("dspark_markov_init", {b.state}, {1, 1, 1});
  graph.add("dspark_markov_context", {b.hidden, b.feature, b.recurrentWeights, b.recurrentBias, b.context},
            {uint64_t{rows} * 768, 1, 1});
}

void DraftSelector::addDSparkStep(metal::CommandGraph &graph, const DSparkMarkovBuffers &b,
                                  uint32_t position, uint32_t anchor) {
  if (position >= 7 || anchor >= 201024) throw std::invalid_argument("invalid DSpark Markov step");
  requireBytes(b.embedding, uint64_t{201024} * 256 * 2, "DSpark Markov embedding");
  requireBytes(b.recurrentWeights, uint64_t{768} * 5120 * 2, "DSpark recurrent weights");
  requireBytes(b.context, 7 * 768 * 4, "DSpark recurrent context");
  requireBytes(b.state, 256 * 2, "DSpark recurrent state");
  requireBytes(b.raw, 768 * 2, "DSpark recurrent raw");
  requireBytes(b.latent, 256 * 2, "DSpark Markov latent");
  requireBytes(b.tokens, 7 * 4, "DSpark tokens");
  graph.add("dspark_markov_linear", {b.state, b.embedding, b.recurrentWeights, b.context, b.tokens, b.raw},
            DSparkMarkovParams{position, anchor}, {768, 1, 1});
  graph.add("dspark_markov_update", {b.raw, b.state, b.latent}, {1, 1, 1});
}

void DraftSelector::addDSparkSelect(metal::CommandGraph &graph, const DSparkMarkovBuffers &b, uint32_t position,
                                    uint32_t vocabulary, uint32_t legal) {
  if (position >= 7 || !legal || legal > vocabulary ||
      (legal + SPLASH_DSPARK_GROUP_ROWS - 1) / SPLASH_DSPARK_GROUP_ROWS > SPLASH_DSPARK_MAX_GROUPS)
    throw std::invalid_argument("invalid DSpark proposal position");
  const DSparkPartialBytes partial;
  requireBytes(b.foldWeight, uint64_t{legal} * SPLASH_DSPARK_RANK * 2, "DSpark fold weight");
  requireBytes(b.foldBias, uint64_t{legal} * sizeof(float), "DSpark fold bias");
  requireBytes(b.latent, uint64_t{SPLASH_DSPARK_RANK} * 2, "DSpark Markov latent");
  requireBytes(b.baseLogits, (uint64_t{position} * vocabulary + legal) * sizeof(float), "DSpark base logits");
  requireBytes(b.partialValues, partial.values, "DSpark partial value");
  requireBytes(b.partialIds, partial.ids, "DSpark partial id");
  requireBytes(b.tokens, 7 * 4, "DSpark tokens");
  requireBytes(b.probabilities, 7 * 4, "DSpark probabilities");
  const DSparkFoldScanParams params{position, legal, vocabulary};
  graph.add("dspark_fold_scan",
            {b.foldWeight, b.foldBias, b.latent, b.baseLogits, b.partialValues, b.partialIds}, params,
            {(legal + SPLASH_DSPARK_GROUP_ROWS - 1) / SPLASH_DSPARK_GROUP_ROWS, 1, 1},
            {SPLASH_DSPARK_GROUP_THREADS, 1, 1});
  graph.add("dspark_fold_reduce", {b.partialValues, b.partialIds, b.tokens, b.probabilities}, params, {1, 1, 1},
            {SPLASH_DSPARK_GROUP_THREADS, 1, 1});
}

void DraftSelector::addDSparkFoldInputs(metal::CommandGraph &graph, metal::MetalBuffer hiddenOut,
                                        metal::MetalBuffer hiddenOutBias, metal::MetalBuffer inputs) {
  constexpr uint64_t rows = SPLASH_DSPARK_RANK + 1, hidden = 2304;
  requireBytes(hiddenOut, hidden * SPLASH_DSPARK_RANK * 2, "DSpark hidden output");
  requireBytes(hiddenOutBias, hidden * 2, "DSpark hidden output bias");
  requireBytes(inputs, rows * hidden * 2, "DSpark fold inputs");
  graph.add("dspark_fold_inputs", {std::move(hiddenOut), std::move(hiddenOutBias), std::move(inputs)},
            {rows * hidden / 256, 1, 1});
}

void DraftSelector::addDSparkFoldStore(metal::CommandGraph &graph, metal::MetalBuffer product,
                                       metal::MetalBuffer weight, metal::MetalBuffer bias, uint32_t start,
                                       uint32_t rows, uint32_t vocabulary) {
  if (!rows || !vocabulary || uint64_t{start} + rows > SPLASH_DSPARK_RANK + 1)
    throw std::invalid_argument("invalid DSpark fold rows");
  requireBytes(product, uint64_t{rows} * vocabulary * sizeof(float), "DSpark fold product");
  requireBytes(weight, uint64_t{vocabulary} * SPLASH_DSPARK_RANK * 2, "DSpark fold weight");
  requireBytes(bias, uint64_t{vocabulary} * sizeof(float), "DSpark fold bias");
  graph.add("dspark_fold_store", {std::move(product), std::move(weight), std::move(bias)},
            DSparkFoldParams{start, rows, vocabulary}, {(vocabulary + 255) / 256, 1, 1});
}

void DraftSelector::addDSparkChain(metal::CommandGraph &graph, const DraftSelectorBuffers &b,
                                    metal::MetalBuffer probabilities, std::span<const DraftTreeLane> lanes) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid DSpark chain batch");
  DSparkChainParams params{};
  for (uint32_t i = 0; i < lanes.size(); ++i) {
    if (lanes[i].depth > 7) throw std::invalid_argument("DSpark proposes at most seven tokens");
    params.anchor[i] = lanes[i].anchor;
    params.length[i] = lanes[i].depth;
    params.forced[i] = lanes[i].children == 0;
    params.slope[i] = lanes[i].calibration.slope;
    std::ranges::copy(lanes[i].calibration.intercepts, params.intercept[i]);
  }
  const uint64_t count = lanes.size(), treeRows = count * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(probabilities, count * SPLASH_DRAFT_PROPOSAL_TOKENS * sizeof(float), "DSpark probabilities");
  requireBytes(b.chainTokens, count * SPLASH_TREE_CHAIN_TOKENS * 4, "DSpark chain tokens");
  requireBytes(b.treeTokens, treeRows * 4, "DSpark tree tokens");
  requireBytes(b.trees, treeRows * sizeof(TreeRow), "DSpark tree rows");
  requireBytes(b.treeScores, treeRows * 2 * sizeof(float), "DSpark tree scores");
  requireBytes(b.treeSizes, count * 4, "DSpark tree sizes");
  requireBytes(b.treeDraws, treeRows * SPLASH_TREE_CHILDREN, "DSpark tree draws");
  requireBytes(b.treeNext, treeRows * SPLASH_DRAFT_CANDIDATES * sizeof(float), "DSpark tree next");
  requireBytes(b.treeCaps, treeRows * sizeof(float), "DSpark tree caps");
  graph.add("dspark_chain_tree", {b.chainTokens, b.treeTokens, b.trees, b.treeScores, b.treeSizes,
                                  b.treeDraws, b.treeNext, b.treeCaps, probabilities}, params, {count, 1, 1});
}

} // namespace splash::ops
