#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/ModelDescriptor.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main() {
  using namespace splash;
  try {
    const test::TemporaryDirectory dir("inkling-config");
    const auto path = dir.path() / "config.json";
    const std::string config = R"({
      "model_type":"inkling2_loop","eos_token_id":200006,
      "text_config":{
        "model_type":"inkling2_loop_text","model_max_length":1048576,"hidden_size":2304,
        "num_hidden_layers":24,"vocab_size":201024,"num_attention_heads":24,"num_key_value_heads":8,
        "head_dim":128,"d_rel":16,"rel_extent":1024,"q_bias":false,"o_bias":false,
        "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"rms_norm_eps":0.000001,
        "use_value_norm_no_gamma":true,"qk_l2_clip_threshold":22.627416997969522,
        "rel_l2_clip_threshold":8.0,"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
        "sconv_kernel_size":4,"unpadded_vocab_size":200058,"final_logit_softcapping":30.0,
        "swa_head_dim":128,"swa_num_attention_heads":24,"swa_num_key_value_heads":8,
        "sliding_window_size":1024,"n_layers_per_attn_res_block":6,"attn_res_normalize_values":true,
        "n_routed_experts":128,"num_experts_per_tok":8,"n_shared_experts":0,"intermediate_size":1152,
        "mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,"use_gate_bias":true,
        "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}},
      "quantization":{"group_size":64,"bits":4,"mode":"affine"}})";
    const auto inspect = [&](const std::string &data) {
      test::writeFile(path, data);
      return model::inspectSourceConfiguration("mlx-affine", "none", path, std::nullopt, std::nullopt);
    };
    test::require(inspect(config) == model::InklingLayout{}.family, "Inkling config was not recognized");
    const auto changed = [&](std::string_view from, std::string_view to) {
      auto result = config;
      const auto at = result.find(from);
      test::require(at != std::string::npos, "test config field missing");
      result.replace(at, from.size(), to);
      return result;
    };
    test::rejects([&] { inspect(changed("\"num_hidden_layers\":24", "\"num_hidden_layers\":48")); },
                  "num_hidden_layers", "visit count accepted as stored layer count");
    test::rejects([&] { inspect(changed("\"num_passes\":3", "\"num_passes\":2")); },
                  "num_passes", "invalid pass count accepted");
    test::rejects([&] { inspect(changed("\"group_size\":64", "\"group_size\":32")); },
                  "group", "wrong quantization group accepted");
    test::rejects([&] { inspect(changed("\"mode\":\"affine\"", "\"mode\":\"mxfp4\"")); },
                  "affine", "wrong quantization mode accepted");
    test::rejects([&] { inspect(changed("\"eos_token_id\":200006", "\"eos_token_id\":200007")); },
                  "eos_token_id", "wrong stop token accepted");
    test::rejects([&] { inspect(changed("\"mode\":\"affine\"}",
                                        "\"mode\":\"affine\",\"language_model.model.layers.0.self_attn.q_proj\":false}")); },
                  "unquantized", "an unquantized attention projection was accepted");
    std::cout << "Inkling configuration PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
