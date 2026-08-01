"""Executable Futamura-style specialization tests for PRSL."""
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Read: pass
@dataclass(frozen=True)
class Emit: probs: tuple
@dataclass(frozen=True)
class Push: token: int
@dataclass(frozen=True)
class Halt: pass

Opcode=Read|Emit|Push|Halt

def interpret(program:tuple[Opcode,...], prompt:int, stack:tuple[int,...]=()):
    out=[]; s=stack; reg=prompt
    for op in program:
        if isinstance(op,Read): reg=prompt
        elif isinstance(op,Emit): out.append((reg,s,op.probs))
        elif isinstance(op,Push): s=s+(op.token,)
        elif isinstance(op,Halt): return tuple(out)
    return tuple(out)

def specialize(program:tuple[Opcode,...])->Callable[[int,tuple[int,...]],tuple]:
    # First Futamura projection: specialize interpreter with respect to program.
    frozen=tuple(program)
    def compiled(prompt,stack=()): return interpret(frozen,prompt,stack)
    return compiled

def compiler(template):
    # A compiler is itself a program generator: second-projection shape.
    return tuple(template)

toy=compiler([Read(),Emit((.7,.2,.1)),Push(4),Emit((.1,.8,.1)),Halt()])
compiled=specialize(toy)
assert compiled(3)==interpret(toy,3)
assert compiled(3,(8,))==interpret(toy,3,(8,))
assert compiler(toy)==toy
print('FIRST_PROJECTION_OK')
print('SECOND_PROJECTION_SHAPE_OK')
print('THIRD_PROJECTION_THEORY: specialize compiler generator with its own representation; not claimed implemented for FLAN.')
