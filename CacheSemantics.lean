import Std

/-! Universal structural semantics of autoregressive token and KV-cache append. -/

namespace CacheSemantics

structure DecoderRegisters (Token Key Value : Type) where
  tokens : List Token
  keys : List Key
  values : List Value

def pushUpdate (r : DecoderRegisters Token Key Value)
    (token : Token) (key : Key) (value : Value) :
    DecoderRegisters Token Key Value :=
  { tokens := r.tokens ++ [token]
    keys := r.keys ++ [key]
    values := r.values ++ [value] }

def pushMany : DecoderRegisters Token Key Value →
    List (Token × Key × Value) → DecoderRegisters Token Key Value
  | r, [] => r
  | r, (token, key, value) :: rest => pushMany (pushUpdate r token key value) rest

def newTokens : List (Token × Key × Value) → List Token
  | [] => []
  | (token, _, _) :: rest => token :: newTokens rest

def newKeys : List (Token × Key × Value) → List Key
  | [] => []
  | (_, key, _) :: rest => key :: newKeys rest

def newValues : List (Token × Key × Value) → List Value
  | [] => []
  | (_, _, value) :: rest => value :: newValues rest

theorem newTokens_length : ∀ rows : List (Token × Key × Value),
    (newTokens rows).length = rows.length := by
  intro rows; induction rows with
  | nil => rfl
  | cons row rest ih => rcases row with ⟨token, key, value⟩; simp [newTokens, ih]

theorem newKeys_length : ∀ rows : List (Token × Key × Value),
    (newKeys rows).length = rows.length := by
  intro rows; induction rows with
  | nil => rfl
  | cons row rest ih => rcases row with ⟨token, key, value⟩; simp [newKeys, ih]

theorem newValues_length : ∀ rows : List (Token × Key × Value),
    (newValues rows).length = rows.length := by
  intro rows; induction rows with
  | nil => rfl
  | cons row rest ih => rcases row with ⟨token, key, value⟩; simp [newValues, ih]

theorem pushMany_tokens (r : DecoderRegisters Token Key Value) :
    ∀ rows, (pushMany r rows).tokens = r.tokens ++ newTokens rows := by
  intro rows
  induction rows generalizing r with
  | nil => simp [pushMany, newTokens]
  | cons row rest ih =>
      rcases row with ⟨token, key, value⟩
      simp only [pushMany, newTokens]
      rw [ih]
      simp [pushUpdate, List.append_assoc]

theorem pushMany_keys (r : DecoderRegisters Token Key Value) :
    ∀ rows, (pushMany r rows).keys = r.keys ++ newKeys rows := by
  intro rows
  induction rows generalizing r with
  | nil => simp [pushMany, newKeys]
  | cons row rest ih =>
      rcases row with ⟨token, key, value⟩
      simp only [pushMany, newKeys]
      rw [ih]
      simp [pushUpdate, List.append_assoc]

theorem pushMany_values (r : DecoderRegisters Token Key Value) :
    ∀ rows, (pushMany r rows).values = r.values ++ newValues rows := by
  intro rows
  induction rows generalizing r with
  | nil => simp [pushMany, newValues]
  | cons row rest ih =>
      rcases row with ⟨token, key, value⟩
      simp only [pushMany, newValues]
      rw [ih]
      simp [pushUpdate, List.append_assoc]

theorem cache_lengths_after_pushMany (r : DecoderRegisters Token Key Value) :
    ∀ rows,
      (pushMany r rows).tokens.length = r.tokens.length + rows.length ∧
      (pushMany r rows).keys.length = r.keys.length + rows.length ∧
      (pushMany r rows).values.length = r.values.length + rows.length := by
  intro rows
  rw [pushMany_tokens, pushMany_keys, pushMany_values]
  constructor
  · simp [newTokens_length]
  constructor
  · simp [newKeys_length]
  · simp [newValues_length]

end CacheSemantics
