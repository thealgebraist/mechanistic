#!/usr/bin/env python3
"""Generate a concrete Lean program certificate from flan_full_graph.json."""
import hashlib, json
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
names = sorted({op["op"] for op in graph["ops"]})
ctor = {name: "op" + "".join(part.title() for part in name.lower().split("_")) for name in names}
tags = ",\n  ".join(ctor[op["op"]] for op in graph["ops"])
constructors = "\n  | ".join(ctor[name] for name in names)
sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
source = f'''import Std
import ProgramComposition
import ApproximateProgramComposition
import AffineProgramComposition

/-! Generated from outputs/flan_full_graph.json. Do not edit by hand. -/

namespace GeneratedFlanProgram

def fullGraphSha256 : String := "{sha}"
def checkpointSha256 : String := "{graph['checkpoint_sha256']}"

inductive OpTag where
  | {constructors}
  deriving Repr, DecidableEq

open OpTag

def programTags : List OpTag := [
  {tags}
]

theorem programTags_length : programTags.length = 129 := by rfl

/- `primitive` is the fixed ordered semantics for each generated tag.  Source
and register execution are instantiated with that same semantics. -/
def paired {{State : Type}} (primitive : OpTag → State → State) (tag : OpTag) :
    ProgramComposition.PairedOpcode State State Eq :=
  {{ sourceEval := primitive tag
    targetEval := primitive tag
    preserves := by
      intro s t h
      cases h
      rfl }}

def concreteProgram {{State : Type}} (primitive : OpTag → State → State) :
    List (ProgramComposition.PairedOpcode State State Eq) :=
  programTags.map (paired primitive)

theorem concrete_129_opcode_program_preserves
    {{State : Type}} (primitive : OpTag → State → State) :
    ∀ {{sourceState registerState}}, sourceState = registerState →
      ProgramComposition.runSource (concreteProgram primitive) sourceState =
      ProgramComposition.runTarget (concreteProgram primitive) registerState := by
  intro sourceState registerState h
  exact ProgramComposition.program_preserves_relation (concreteProgram primitive) h

/- `opcode` supplies one local backend-refinement obligation per tag.  Mapping
over the generated tag list binds those obligations to all 129 occurrences. -/
def approximateProgram {{Source Target : Type}} {{distance : Source → Target → Nat}}
    (opcode : OpTag → ApproximateProgramComposition.PairedOpcode Source Target distance) :
    List (ApproximateProgramComposition.PairedOpcode Source Target distance) :=
  programTags.map opcode

theorem concrete_129_opcode_error_adds
    {{Source Target : Type}} {{distance : Source → Target → Nat}}
    (opcode : OpTag → ApproximateProgramComposition.PairedOpcode Source Target distance) :
    ∀ sourceState registerState,
      distance
          (ApproximateProgramComposition.runSource (approximateProgram opcode) sourceState)
          (ApproximateProgramComposition.runTarget (approximateProgram opcode) registerState) ≤
        distance sourceState registerState +
          ApproximateProgramComposition.totalBudget (approximateProgram opcode) := by
  exact ApproximateProgramComposition.program_error_adds (approximateProgram opcode)

def affineProgram {{Source Target : Type}} {{distance : Source → Target → Nat}}
    (opcode : OpTag → AffineProgramComposition.PairedOpcode Source Target distance) :
    List (AffineProgramComposition.PairedOpcode Source Target distance) :=
  programTags.map opcode

theorem concrete_129_opcode_error_affine
    {{Source Target : Type}} {{distance : Source → Target → Nat}}
    (opcode : OpTag → AffineProgramComposition.PairedOpcode Source Target distance) :
    ∀ sourceState registerState,
      distance
          (AffineProgramComposition.runSource (affineProgram opcode) sourceState)
          (AffineProgramComposition.runTarget (affineProgram opcode) registerState) ≤
        AffineProgramComposition.totalGain (affineProgram opcode) *
          distance sourceState registerState +
        AffineProgramComposition.totalBias (affineProgram opcode) := by
  exact AffineProgramComposition.program_error_affine (affineProgram opcode)

end GeneratedFlanProgram
'''
Path("GeneratedFlanProgram.lean").write_text(source)
print(json.dumps({"artifact": "GeneratedFlanProgram.lean", "opcodes": len(graph["ops"]),
                  "graph_sha256": sha}, indent=2))
