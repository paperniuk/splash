#include "ops/Inkling.hpp"
#include "metal/abi/Inkling.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace splash::ops {
namespace {
constexpr uint32_t threads = 256;
uint32_t groups(uint32_t n) { return (n + threads - 1) / threads; }
InklingRowParams rowParams(InklingRows r) { return {r.stride, r.offset}; }
void requireRows(InklingRows r, uint32_t width, uint32_t rows, uint32_t bytes) {
  if (!width || r.width < width || r.stride < r.width) throw std::invalid_argument("invalid Inkling row shape");
  requireBytes(r.buffer, uint64_t{r.offset} * bytes + rowBytes(rows, r.stride, width, bytes), "Inkling rows");
}
void requireHistory(const metal::MetalBuffer &b, uint32_t offset, uint32_t stride, uint32_t channels, uint32_t taps) {
  if (taps < 2 || stride < channels) throw std::invalid_argument("invalid Inkling history shape");
  requireBytes(b, (uint64_t{offset} + uint64_t{taps - 2} * stride + channels) * 2, "Inkling history");
}
void shape(uint32_t channels, uint32_t taps) {
  if (!channels || taps != 4) throw std::invalid_argument("invalid Inkling convolution shape");
}
}
void Inkling::addSum(metal::CommandGraph &graph, InklingRows a, bool aFloat, InklingRows b, bool bFloat,
                     InklingRows output, bool outputFloat, uint32_t width, uint32_t rows) {
  if (!rows) throw std::invalid_argument("invalid Inkling sum rows");
  requireRows(a, width, rows, aFloat ? 4 : 2);
  if (b.buffer) requireRows(b, width, rows, bFloat ? 4 : 2);
  requireRows(output, width, rows, outputFloat ? 4 : 2);
  InklingParams p{};
  p.hidden = width; p.rows = rows; p.source[0] = rowParams(a); p.source[1] = rowParams(b);
  p.output = rowParams(output); p.count = bool(b.buffer); p.partialFloat = aFloat;
  p.dynamic = bFloat; p.writeHistory = outputFloat;
  graph.add("inkling_sum", {a.buffer, b.buffer ? b.buffer : a.buffer, output.buffer}, p,
            {groups(width), rows, 1}, {threads, 1, 1});
}
// A chunk's sums: each state's squares (6) and each query's dots (4 x 6).
constexpr uint32_t kDepthChunk = 256, kDepthSums = 32;
uint64_t Inkling::depthSumsBytes(uint32_t hidden, uint32_t rows) {
  return uint64_t{rows} * ((hidden + kDepthChunk - 1) / kDepthChunk) * kDepthSums * sizeof(float);
}
void Inkling::addDepthMix(metal::CommandGraph &graph, std::span<const InklingRows> states, InklingRows partial,
                          bool partialFloat, metal::MetalBuffer queries, uint32_t queryCount, InklingRows output,
                          metal::MetalBuffer sums, uint32_t hidden, uint32_t rows) {
  if (states.size() > 5 || !rows || !queryCount || queryCount > 4) throw std::invalid_argument("invalid Inkling depth mix");
  InklingParams p{}; p.hidden=hidden; p.rows=rows; p.count=static_cast<uint32_t>(states.size()+1); p.queries=queryCount; p.partialFloat=partialFloat;
  std::vector<metal::MetalBuffer> bindings;
  for (size_t i=0;i<states.size();++i) { requireRows(states[i],hidden,rows,2); p.source[i]=rowParams(states[i]); bindings.push_back(states[i].buffer); }
  requireRows(partial,hidden,rows,partialFloat?4:2); p.source[states.size()]=rowParams(partial); bindings.push_back(partial.buffer);
  while (bindings.size()<6) bindings.push_back(partial.buffer);
  requireBytes(queries,uint64_t{queryCount}*hidden*4,"Inkling depth queries");
  requireRows(output,hidden*queryCount,rows,4); p.output=rowParams(output);
  requireBytes(sums,depthSumsBytes(hidden,rows),"Inkling depth sums");
  bindings.push_back(queries); bindings.push_back(output.buffer); bindings.push_back(std::move(sums));
  const uint32_t chunks = (hidden + kDepthChunk - 1) / kDepthChunk;
  graph.add("inkling_depth_sums",bindings,p,{chunks,rows,1},{kDepthChunk,1,1});
  graph.add("inkling_depth_mix",std::move(bindings),p,{chunks,rows,1},{kDepthChunk,1,1});
}
void Inkling::addMessageGate(metal::CommandGraph &graph, InklingRows x, InklingRows message,
                             const InklingGateWeights &weights, uint32_t rank, InklingRows low, InklingRows output,
                             uint32_t hidden, uint32_t rows) {
  addMessageGates(graph, x, message, std::span(&weights, 1), rank, low, output, {}, hidden, rows);
}
void Inkling::addMessageGates(metal::CommandGraph &graph, InklingRows x, InklingRows message,
                              std::span<const InklingGateWeights> weights, uint32_t rank, InklingRows low,
                              InklingRows output, const InklingPartSteps &steps, uint32_t hidden, uint32_t rows) {
  // The down kernel's threadgroups take one output each; both kernels read
  // weights 8 bf16 a load, and down reads x and m 4 floats a load.
  const uint32_t parts = static_cast<uint32_t>(weights.size());
  if (!rows || !rank || rank > 64 || rank % 8 || hidden % 8 || !parts || parts > 3)
    throw std::invalid_argument("invalid Inkling gate shape");
  const auto later = [&](InklingRows r, uint32_t step) { return InklingRows{r.buffer, r.width, r.stride, r.offset + (parts - 1) * step}; };
  for (const auto &[rowsOf, step, size] : {std::tuple{x, steps.x, 4u}, {message, steps.message, 4u},
                                            {output, steps.output, 4u}, {low, steps.low, 4u}}) {
    requireRows(rowsOf, rowsOf.width, rows, size);
    requireRows(later(rowsOf, step), rowsOf.width, rows, size);
  }
  if (x.width != hidden || message.width != hidden || output.width != hidden || low.width != rank)
    throw std::invalid_argument("invalid Inkling gate rows");
  if ((x.offset | x.stride | message.offset | message.stride | steps.x | steps.message) % 4)
    throw std::invalid_argument("Inkling gate rows are 16-byte aligned");
  for (const InklingGateWeights &w : weights) {
    requireBytes(w.down,uint64_t{rank}*hidden*4,"Inkling gate down");
    requireBytes(w.forget,uint64_t{rank}*hidden*2,"Inkling gate forget");
    requireBytes(w.gain,uint64_t{rank}*hidden*2,"Inkling gate gain");
  }
  const auto part = [&](uint32_t i) { return weights[std::min(i, parts - 1)]; };
  constexpr uint32_t kTileRows = 8;
  const uint32_t tiles = (rows + kTileRows - 1) / kTileRows;
  InklingParams p{}; p.hidden=hidden; p.rows=rows; p.rank=rank; p.source[0]=rowParams(x); p.source[1]=rowParams(message);
  p.partSteps[0]=steps.x; p.partSteps[1]=steps.message; p.partSteps[2]=steps.low; p.partSteps[3]=steps.output;
  p.output=rowParams(low);
  // Steps past 64 tiles run down as a matrix multiply, whose grid is a
  // threadgroup a 32-row tile and part: below it too few to fill the GPU.
  // A 59-row prompt's prefill took 241.8 ms on its multiplies and 205.0 on
  // per-output threadgroups on an M1 Max, a 285-row one 601 and 588; at 633
  // rows they matched, at 7234 the per-output ones took 3% longer.
  constexpr uint32_t kMatrixTiles = 64;
  if (tiles > kMatrixTiles)
    graph.add("inkling_gate_down_mma",{x.buffer,message.buffer,part(0).down,part(1).down,part(2).down,low.buffer},p,
              {(rows + 31) / 32,1,parts},{threads,1,1});
  else
    graph.add("inkling_gate_down",{x.buffer,message.buffer,part(0).down,part(1).down,part(2).down,low.buffer},p,
              {tiles,rank/4,parts},{threads,1,1});
  p.dynamicRows=rowParams(low); p.output=rowParams(output);
  graph.add("inkling_gate_up",{x.buffer,message.buffer,low.buffer,part(0).forget,part(1).forget,part(2).forget,
                               part(0).gain,part(1).gain,part(2).gain,output.buffer},p,
            {tiles,(hidden+63)/64,parts},{64,1,1});
}
void Inkling::addNorm(metal::CommandGraph &graph, InklingRows input, const NormWeights &norm, InklingRows output, uint32_t hidden, uint32_t rows) {
  addNorms(graph, input, std::span(&norm, 1), output, {}, hidden, rows);
}
void Inkling::addNorms(metal::CommandGraph &graph, InklingRows input, std::span<const NormWeights> norms,
                       InklingRows output, const InklingPartSteps &steps, uint32_t hidden, uint32_t rows) {
  const uint32_t parts = static_cast<uint32_t>(norms.size());
  if (!rows || !parts || parts > 3) throw std::invalid_argument("invalid Inkling norm rows");
  requireRows(input,hidden,rows,4); requireRows(output,hidden,rows,2);
  requireRows({input.buffer,input.width,input.stride,input.offset+(parts-1)*steps.x},hidden,rows,4);
  requireRows({output.buffer,output.width,output.stride,output.offset+(parts-1)*steps.output},hidden,rows,2);
  for (const NormWeights &n : norms)
    if (n.float32 != norms[0].float32) throw std::invalid_argument("Inkling norms of one dispatch share a type");
  const auto part = [&](uint32_t i) { return norms[std::min(i, parts - 1)].buffer; };
  InklingParams p{}; p.hidden=hidden; p.rows=rows; p.source[0]=rowParams(input); p.output=rowParams(output);
  p.partSteps[0]=steps.x; p.partSteps[3]=steps.output;
  graph.add(normKernel("inkling_norm",norms[0],hidden),{input.buffer,part(0),part(1),part(2),output.buffer},p,{rows,parts,1},{threads,1,1});
}
void Inkling::addConvolution(metal::CommandGraph &graph, InklingRows input, std::span<const metal::MetalBuffer> historyIn,
 std::span<const metal::MetalBuffer> historyOut, uint32_t historyOffset, uint32_t historyStride, const InklingConvolution &convolution,
 InklingRows dynamicTaps, InklingRows output, metal::MetalBuffer layout, const LayoutRows &rows, metal::MetalBuffer tree, InklingRows tape, const InklingResidual &residual) {
  shape(convolution.channels,convolution.taps); bool hasTree=bool(tree), write=!historyOut.empty();
  requireLayoutRows(rows,hasTree?32:std::numeric_limits<uint32_t>::max());
  if (historyIn.size()!=rows.sequences || (write && historyOut.size()!=rows.sequences) || (hasTree && write) ||
      (convolution.dynamic && (!convolution.heads || convolution.channels%convolution.heads))) throw std::invalid_argument("invalid Inkling convolution");
  requireRows(input,convolution.channels,rows.rows,2); requireRows(output,convolution.channels,rows.rows,residual.outputFloat?4:2);
  requireBytes(convolution.weights,uint64_t{convolution.channels}*convolution.taps*2,"Inkling convolution weights");
  requireBytes(layout,uint64_t{rows.sequences}*sizeof(SplashLaneRows),"Inkling convolution layout");
  if(hasTree) requireBytes(tree,uint64_t{rows.rows}*sizeof(SplashTreeRow),"Inkling convolution tree");
  if(convolution.dynamic) requireRows(dynamicTaps,convolution.heads*convolution.taps,rows.rows,4);
  for(const auto &h:historyIn) requireHistory(h,historyOffset,historyStride,convolution.channels,convolution.taps);
  for(const auto &h:historyOut) requireHistory(h,historyOffset,historyStride,convolution.channels,convolution.taps);
  auto in=laneBindings(historyIn), out=laneBindings(write?historyOut:historyIn);
  InklingParams p{}; p.hidden=convolution.channels; p.rows=rows.rows; p.taps=convolution.taps; p.heads=convolution.heads; p.dynamic=convolution.dynamic; p.sequences=rows.sequences; p.tree=hasTree; p.writeHistory=write; p.historyOffset=historyOffset; p.historyStride=historyStride; p.source[0]=rowParams(input); p.output=rowParams(output); p.dynamicRows=rowParams(dynamicTaps);
  if(tape.buffer){ requireRows(tape,convolution.channels,rows.rows,2); p.count=1; p.source[1]=rowParams(tape); }
  p.outputFloat=residual.outputFloat;
  if(residual.addend.buffer){ requireRows(residual.addend,convolution.channels,rows.rows,residual.addendFloat?4:2); p.addend=1; p.addendFloat=residual.addendFloat; p.source[2]=rowParams(residual.addend); }
  graph.add("inkling_conv",{input.buffer,in[0],in[1],in[2],in[3],out[0],out[1],out[2],out[3],convolution.weights,convolution.dynamic?dynamicTaps.buffer:convolution.weights,output.buffer,layout,hasTree?tree:convolution.weights,tape.buffer?tape.buffer:output.buffer,residual.addend.buffer?residual.addend.buffer:output.buffer},p,{rows.bound+(write?convolution.taps-1:0),groups(convolution.channels),rows.sequences},{threads,1,1});
}
void Inkling::addConvolutionHistory(metal::CommandGraph &graph, InklingRows input, metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, uint32_t historyOffset, uint32_t historyStride, uint32_t channels, uint32_t taps, uint32_t rows) {
  shape(channels,taps); requireRows(input,channels,rows,2); requireHistory(historyIn,historyOffset,historyStride,channels,taps); requireHistory(historyOut,historyOffset,historyStride,channels,taps);
  InklingParams p{}; p.hidden=channels; p.rows=rows; p.taps=taps; p.historyOffset=historyOffset; p.historyStride=historyStride; p.source[0]=rowParams(input);
  graph.add("inkling_history",{input.buffer,historyIn,historyOut},p,{taps-1,groups(channels),1},{threads,1,1});
}
void Inkling::addConvolutionCommit(metal::CommandGraph &graph, InklingRows inputs, uint32_t blocks, uint32_t blockStride, uint32_t channels, std::span<const metal::MetalBuffer> historyIn, std::span<const metal::MetalBuffer> historyOut, uint32_t historyBlockStride, uint32_t historyStride, uint32_t taps, metal::MetalBuffer paths, metal::MetalBuffer layout, const LayoutRows &rows, uint32_t pathRows) {
  shape(channels,taps); requireLayoutRows(rows,32);
  if(!blocks || blockStride<channels || historyIn.size()!=rows.sequences || historyOut.size()!=rows.sequences || uint64_t{blocks}*channels>std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("invalid Inkling commit");
  uint64_t inputWidth=uint64_t{blocks-1}*blockStride+channels;
  if(inputWidth>std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("invalid Inkling commit width");
  requireRows(inputs,static_cast<uint32_t>(inputWidth),rows.rows,2);
  if(historyStride<channels) throw std::invalid_argument("invalid Inkling history stride");
  uint64_t bytes=(uint64_t{blocks-1}*historyBlockStride+uint64_t{taps-2}*historyStride+channels)*2;
  for(const auto &h:historyIn) requireBytes(h,bytes,"Inkling commit history");
  for(const auto &h:historyOut) requireBytes(h,bytes,"Inkling commit next history");
  requireBytes(paths,uint64_t{rows.sequences}*4,"Inkling commit paths"); requireBytes(layout,uint64_t{rows.sequences}*sizeof(SplashLaneRows),"Inkling commit layout");
  auto in=laneBindings(historyIn),out=laneBindings(historyOut); InklingParams p{}; p.hidden=channels; p.rows=rows.rows; p.taps=taps; p.sequences=rows.sequences; p.blocks=blocks; p.blockStride=blockStride; p.historyBlockStride=historyBlockStride; p.historyStride=historyStride; p.count=pathRows; p.source[0]=rowParams(inputs);
  graph.add("inkling_commit",{inputs.buffer,in[0],in[1],in[2],in[3],out[0],out[1],out[2],out[3],paths,layout},p,{taps-1,groups(blocks*channels),rows.sequences},{threads,1,1});
}
void Inkling::addLogitCap(metal::CommandGraph &graph, metal::MetalBuffer logits, float softcap, uint32_t vocabulary, uint32_t width, uint32_t rows) {
  if(!rows || !width || vocabulary>width || !(softcap>0) || uint64_t{rows}*width>std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("invalid Inkling logit cap");
  requireBytes(logits,uint64_t{rows}*width*4,"Inkling logits"); InklingParams p{}; p.hidden=width; p.rows=rows; p.count=vocabulary; p.softcap=softcap;
  graph.add("inkling_cap",{logits},p,{groups(rows*width),1,1},{threads,1,1});
}
} // namespace splash::ops
