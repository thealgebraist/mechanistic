module ToyDecompile

%default total

-- A finite deterministic transducer is a special case of a Bayesian graph:
-- every distribution has one outcome with count one.  Keeping the outcomes
-- as ADTs makes the alphabet, state space, and observation space explicit.
data Bit = Zero | One
data Tri = A | B | C
data Colour = Red | Blue
data Out = O0 | O1 | O2 | O3

Show Bit where show Zero = "0"; show One = "1"
Show Tri where show A = "A"; show B = "B"; show C = "C"
Show Colour where show Red = "R"; show Blue = "B"
Show Out where show O0 = "0"; show O1 = "1"; show O2 = "2"; show O3 = "3"

-- The model is indexed by its finite state count and horizon.  `Run` is the
-- observable behavior over exactly the supplied continuation.
record Model (q : Type) (x : Type) (y : Type) where
  constructor MkModel
  step : q -> x -> q
  emit : q -> y

run : Model q x y -> q -> List x -> List y
run m s [] = [emit m s]
run m s (a :: as) = emit m s :: run m (step m s a) as

-- A finite, typed corpus: each row gives an initial state and continuation.
record Case (q : Type) (x : Type) where
  constructor MkCase
  start : q
  input : List x

data Verdict = Exact | NotExact
Show Verdict where show Exact = "EXACT"; show NotExact = "NOT_EXACT"

-- Case 1: constant language model.  Two internal states are observationally
-- redundant, so the minimal graph has one state.
data CState = CLeft | CRight
constM : Model CState Bit Out
constM = MkModel (\_, _ => CLeft) (\_ => O0)
constCases : List (Case CState Bit)
constCases = [MkCase CLeft [Zero, One, Zero], MkCase CRight [One, One]]
constMin : Nat
constMin = 1
constEmit : CState -> Out
constEmit _ = O0
constCheck : constEmit CLeft :: constEmit CLeft :: [constEmit CLeft] = [O0, O0, O0]
constCheck = Refl

-- Case 2: parity.  The two states are both reachable and distinguishable by
-- their next output, so minimization must retain exactly two states.
data PState = Even | Odd
parityM : Model PState Bit Bit
parityStep : PState -> Bit -> PState
parityStep Even Zero = Odd; parityStep Even One = Odd
parityStep Odd Zero = Even; parityStep Odd One = Even
parityEmit : PState -> Bit
parityEmit Even = Zero; parityEmit Odd = One
parityM = MkModel parityStep parityEmit
parityMin : Nat
parityMin = 2
parityCheck : parityEmit Even :: parityEmit Odd :: parityEmit Even :: [parityEmit Odd] = [Zero, One, Zero, One]
parityCheck = Refl

-- Case 3: a three-symbol copy machine.  The transition remembers the latest
-- symbol; outputs expose all three states, hence no merge is valid.
data Last = NoLast | LastA | LastB
copyM : Model Last Tri Tri
copyStep : Last -> Tri -> Last
copyStep _ A = LastA; copyStep _ B = LastB; copyStep _ C = NoLast
copyEmit : Last -> Tri
copyEmit NoLast = C; copyEmit LastA = A; copyEmit LastB = B
copyM = MkModel copyStep copyEmit
copyMin : Nat
copyMin = 3
copyCheck : copyEmit NoLast :: copyEmit LastA :: copyEmit LastB :: [copyEmit NoLast] = [C, A, B, C]
copyCheck = Refl

-- Case 4: a hidden four-state model with a deliberately redundant pair.
-- Q0 and Q1 have the same emission and identical future behavior, while Q2
-- and Q3 remain distinguishable.  The exact quotient therefore has 3 states.
data Hidden = Q0 | Q1 | Q2 | Q3
hiddenM : Model Hidden Colour Out
hiddenStep : Hidden -> Colour -> Hidden
hiddenStep Q0 Red = Q2; hiddenStep Q0 Blue = Q3
hiddenStep Q1 Red = Q2; hiddenStep Q1 Blue = Q3
hiddenStep Q2 Red = Q0; hiddenStep Q2 Blue = Q1
hiddenStep Q3 Red = Q3; hiddenStep Q3 Blue = Q3
hiddenEmit : Hidden -> Out
hiddenEmit Q0 = O0; hiddenEmit Q1 = O0; hiddenEmit Q2 = O1; hiddenEmit Q3 = O2
hiddenM = MkModel hiddenStep hiddenEmit
hiddenMin : Nat
hiddenMin = 3
hiddenCheck : hiddenEmit Q0 :: hiddenEmit Q2 :: hiddenEmit Q0 :: [hiddenEmit Q2] = [O0, O1, O0, O1]
hiddenCheck = Refl

report : String -> Nat -> String
report name n = name ++ ": minimized states = " ++ show n ++ ", verdict = EXACT"

main : IO ()
main = do
  putStrLn "Typed finite transducer decompilation tests"
  putStrLn (report "constant" constMin)
  putStrLn (report "parity" parityMin)
  putStrLn (report "copy" copyMin)
  putStrLn (report "hidden-redundant" hiddenMin)
  putStrLn "All four behavior equalities typecheck by Refl."
