import Std

/-! Exact lowering of token-indexed checkpoint lookup; no arithmetic occurs. -/

namespace EmbeddingLowering

def sourceLookup (table : Tok → Vec) (token : Tok) : Vec := table token
def registerLookup (table : Tok → Vec) (token : Tok) : Vec := table token

theorem lookup_exact_for_every_token
    (table : Tok → Vec) : ∀ token,
      registerLookup table token = sourceLookup table token := by
  intro token
  rfl

theorem lookup_sequence_exact
    (table : Tok → Vec) : ∀ (tokens : List Tok),
      tokens.map (registerLookup table) = tokens.map (sourceLookup table) := by
  intro tokens
  rfl

end EmbeddingLowering
