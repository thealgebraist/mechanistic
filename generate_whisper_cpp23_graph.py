#!/usr/bin/env python3
"""Generate a constexpr C++23 form of the complete 74-node Whisper graph."""
import json
from pathlib import Path
g=json.loads(Path("outputs/whisper_tiny_en_probabilistic_graph.json").read_text());config=json.loads(Path("work/whisper_tiny_en/config.json").read_text());generation=json.loads(Path("work/whisper_tiny_en/generation_config.json").read_text());trace=json.loads(Path("outputs/whisper_tiny_en_trace.json").read_text())
ops=sorted({x["opcode"] for x in g["ops"]});enum={x:"".join(y.title() for y in x.lower().split("_")) for x in ops}
ports=[];weights=[];nodes=[]
def esc(s):return s.replace("\\","\\\\").replace('"','\\"')
for x in g["ops"]:
    ib=len(ports);ports+=x["inputs"];ic=len(x["inputs"]);ob=len(ports);ports+=x["outputs"];oc=len(x["outputs"]);wb=len(weights);weights+=x["weights"];wc=len(x["weights"])
    nodes.append(f'  Node{{{x["index"]},Opcode::{enum[x["opcode"]]},"{esc(x["stage"])}",{ib},{ic},{ob},{oc},{wb},{wc},"{esc(x["semantics"])}"}}')
tensors=sorted(g["tensor_metadata"])
text='''#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace generated_whisper {
enum class Opcode : std::uint8_t {'''+','.join(enum[x] for x in ops)+'''};
struct Node{std::uint16_t index;Opcode opcode;std::string_view stage;std::uint16_t input_begin,input_count,output_begin,output_count,weight_begin,weight_count;std::string_view semantics;};
inline constexpr std::array<std::string_view,'''+str(len(ports))+'> ports={\n  '+',\n  '.join(f'"{esc(x)}"' for x in ports)+'''\n};
inline constexpr std::array<std::string_view,'''+str(len(weights))+'> weight_refs={\n  '+',\n  '.join(f'"{esc(x)}"' for x in weights)+'''\n};
inline constexpr std::array<std::string_view,'''+str(len(tensors))+'> tensor_names={\n  '+',\n  '.join(f'"{esc(x)}"' for x in tensors)+'''\n};
inline constexpr std::array<Node,'''+str(len(nodes))+'> nodes={\n'+',\n'.join(nodes)+'''\n};
inline constexpr std::array<std::int32_t,'''+str(1+len(generation.get("forced_decoder_ids",[])))+'> forced_prefix={'+','.join(map(str,[config["decoder_start_token_id"]]+[token for _,token in generation.get("forced_decoder_ids",[])]))+'''};
inline constexpr std::array<std::int32_t,'''+str(len(generation["suppress_tokens"]))+'> suppress_tokens={'+','.join(map(str,generation["suppress_tokens"]))+'''};
inline constexpr std::array<std::int32_t,'''+str(len(generation["begin_suppress_tokens"]))+'> begin_suppress_tokens={'+','.join(map(str,generation["begin_suppress_tokens"]))+'''};
inline constexpr std::array<std::int32_t,'''+str(len(trace["generated_token_ids"]))+'> expected_sample_tokens={'+','.join(map(str,trace["generated_token_ids"]))+'''};
inline constexpr std::array<float,'''+str(len(trace["token_steps"]))+'> expected_selected_mass={'+','.join(f'{x["target_probability"]:.9g}f' for x in trace["token_steps"])+'''};
inline constexpr std::int32_t eos_token='''+str(generation["eos_token_id"])+''';
}
'''
Path("generated_whisper_graph.hpp").write_text(text)
print(json.dumps({"certificate":"GENERATED_WHISPER_CPP23_GRAPH_OK","nodes":len(nodes),"tensor_names":len(tensors),"weight_references":len(weights),"ports":len(ports)},indent=2))
