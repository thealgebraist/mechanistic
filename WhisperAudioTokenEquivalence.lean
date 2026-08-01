import Std
import ProbabilityLawSemantics

/-!
Universal outer theorem for an audio-to-token probabilistic register compiler.

The audio frontend and encoder are summarized by `initial_commutes`; the
autoregressive decoder is summarized by the categorical mass and transition
commuting equations.  No finite audio corpus or finite hidden-state basis is
used by the theorem.
-/
namespace WhisperAudioTokenEquivalence

open ProbabilityLawSemantics

structure AudioToTokenCompilerCertificate
    (Audio SourceState TargetState Token Weight : Type) where
  source : CategoricalTransducer SourceState Token Weight
  target : CategoricalTransducer TargetState Token Weight
  sourceInitial : Audio → SourceState
  targetInitial : Audio → TargetState
  law : LawCertificate source target
  initial_commutes : ∀ audio,
    targetInitial audio = law.encode (sourceInitial audio)

theorem every_audio_every_finite_transcript_mass_equal
    [One Weight] [Mul Weight]
    (certificate : AudioToTokenCompilerCertificate
      Audio SourceState TargetState Token Weight) :
    ∀ audio transcript,
      traceMass certificate.target (certificate.targetInitial audio) transcript =
        traceMass certificate.source (certificate.sourceInitial audio) transcript := by
  intro audio transcript
  rw [certificate.initial_commutes]
  exact same_categorical_law_same_finite_trace
    certificate.law (certificate.sourceInitial audio) transcript

theorem no_finite_state_assumption
    (certificate : AudioToTokenCompilerCertificate
      Audio SourceState TargetState Token Weight) :
    (∀ audio, certificate.targetInitial audio =
      certificate.law.encode (certificate.sourceInitial audio)) := by
  exact certificate.initial_commutes

end WhisperAudioTokenEquivalence
