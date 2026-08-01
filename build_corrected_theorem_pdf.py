from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak

out=Path('output/pdf'); out.mkdir(parents=True,exist_ok=True)
pdf=out/'neural_network_turing_machine_size_theorem.pdf'
s=getSampleStyleSheet()
s.add(ParagraphStyle(name='T',parent=s['Title'],alignment=TA_CENTER,fontSize=20,leading=24,textColor=colors.HexColor('#17365d')))
s.add(ParagraphStyle(name='C',parent=s['Code'],fontSize=8.5,leading=11,backColor=colors.HexColor('#f3f6f9'),borderPadding=6))
story=[]
def H(x): story.append(Paragraph(x,s['Heading1']))
def P(x): story.append(Paragraph(x,s['BodyText']))
def C(x): story.append(Preformatted(x,s['C']))
story += [Paragraph('Neural Networks, Turing Machines, and Exponential State Graphs',s['T']),Paragraph('A corrected matrix-based analysis of simulation size, explicit-state blowup, and probabilistic semantics',s['Heading2']),Spacer(1,12)]
H('Abstract')
P('The universal claim that every complex neural network must require exponentially many instructions when translated into a bounded linear-size Turing machine is false. A finite-precision neural network can be simulated by a deterministic register machine or Turing machine with polynomially many arithmetic operations. Exponential growth appears under a different requirement: explicitly enumerating a finite-state graph that preserves all distinguishable histories. Probabilistic execution is required only when the source model samples, or when an abstraction deliberately discards information and represents the discarded state distribution by a Bayesian kernel.')
H('1. Formal model')
P('Consider a feed-forward network with input dimension d0, layer widths d1 through dL, and q-bit rational parameters. Each layer is:')
C('h_0 = x\nz_l = W_l h_{l-1} + b_l\nh_l = sigma_l(z_l)\ny = R h_L')
P('The matrices W_l and vectors b_l are finite binary strings. Assume each arithmetic operation is rounded to q bits, and each activation sigma_l is computable in polynomial time in q. This includes fixed-point ReLU networks and quantized transformer blocks with a declared rounding rule.')
H('2. Polynomial Turing-machine simulation theorem')
P('Theorem. Let N be a q-bit finite-precision network with parameter count P and layer dimensions d0,...,dL. There exists a deterministic multi-tape Turing machine T_N that computes exactly the rounded output of N using a number of transition steps polynomial in P, q, and the input bit length.')
P('Proof sketch. Store each activation coordinate as a q-bit signed integer. To compute one matrix entry, the machine repeatedly reads a q-bit weight and activation, multiplies them, adds the result to an accumulator, and rounds. Schoolbook multiplication costs O(q^2) bit operations and addition costs O(q). The number of scalar products is:')
C('M = sum_{l=1..L} d_l d_{l-1}')
P('Therefore affine layers cost O(M q^2) bit steps up to polynomial tape-management factors. Applying each sigma_l costs d_l times a polynomial in q. The final readout has the same form. Since M is at most the number of stored matrix entries P, total work is polynomial in P and q. The machine is deterministic because every rounded operation is deterministic. QED.')
H('3. Why the exponential claim fails')
P('A Turing machine has reusable tape and registers. It does not need one instruction or one control state for every possible activation vector. Matrix multiplication is a loop over shared instructions. Thus a network with P parameters can be represented by a program of size O(P) plus a generic arithmetic interpreter, even though the set of possible activation vectors may have size 2^(q d_l).')
C('Large configuration space != large instruction set\nReusable arithmetic loop != explicit state enumeration')
P('If the target is an explicit finite automaton whose states are complete neural configurations, the number of states can be exponential. That is a lower bound on explicit-state representation, not on Turing-machine instruction count.')
H('4. An explicit-state exponential lower bound')
P('Let the input be an m-bit string x. A small linear circuit can copy x: its output vector is y = I_m x, where I_m is the m by m identity matrix. The circuit has O(m) wires and O(m) output operations. Now require a finite-state transducer to read x once and emit the m bits later, one bit at a time, without an auxiliary random-access tape.')
P('After reading all m input bits, the transducer must distinguish every pair x != x-prime, because their required future output sequences differ. By the standard distinguishability argument, every x must reach a different state. Hence at least 2^m states are required.')
C('|States| >= 2^m\nNeural/circuit description size = O(m)')
P('This proves exponential blowup for a restricted explicit finite-state target. It does not prove exponential Turing-machine instructions, because a Turing machine can store x on its tape and scan it during output.')
H('5. Deterministic versus probabilistic targets')
P('A deterministic neural network with deterministic rounding has a deterministic transition function:')
C('s --input token--> T(s,input)\ny = E(s)')
P('No Bayesian or probabilistic machine is required for semantic equivalence. A probabilistic Turing machine is appropriate when the model samples from a distribution, for example:')
C('P(next token = y | s) = softmax(logits(s))_y')
P('A Bayesian graph also appears after abstraction. If many concrete states map to one abstract state a, then the induced transition is a mixture:')
C('K(a-prime | a,u) = Integral 1[alpha(T(s,u))=a-prime] rho_a(ds)')
P('The probability here is induced by abstraction; it does not imply that the original deterministic neural computation was probabilistic.')
H('6. FLAN-T5 specialization consequences')
P('For FLAN-T5, an exact finite-precision simulator can be built as a deterministic register machine containing weights, encoder states, decoder KV cache, and arithmetic subroutines. Sampling the decoder turns this simulator into a probabilistic Turing machine. The PRSL-STACK-1 artifact is a bounded specialization: it freezes a prompt domain, prefix horizon, and top-k quantized output distributions.')
P('Its current certificate covers eight prompts, top-2 branch expansion, horizon 3, and 56 states. At 2,850 bytes it has maximum measured first-token total-variation error 0.3650418194413511. This is a finite-domain approximation certificate, not a lower-bound proof about all FLAN implementations.')
H('7. Corrected theorem')
P('The strongest defensible statement is:')
C('A finite-precision neural network has a polynomial-size deterministic\nTuring-machine simulator in the number of stored parameters and bit precision.\n\nAn explicit finite-state graph may require exponentially many states in the\ninput/history length. If the source uses sampling, or if abstraction discards\nstate information, the target semantics is naturally probabilistic.')
P('This separates three quantities that must not be conflated: program/instruction size, workspace/configuration count, and explicit graph-state count.')
H('8. Verification and limits')
P('The matrix simulation proof assumes finite precision, computable activations, and a declared input representation. Exact real-valued neural networks are not finite binary objects until an encoding and arithmetic semantics are chosen. Lower bounds depend on the target model and what memory it may use. Therefore no unconditional exponential instruction lower bound follows merely from neural-network complexity.')
P('Sources and reproducibility: the FLAN-T5 checkpoint and tokenizer are the local google/flan-t5-small artifacts; the bounded PRSL compiler and checker are flan_stack_compile.py and check_flan_stack.py. The Futamura specialization test is futamura_prsl.py.')
def footer(c,doc):
 c.saveState(); c.setFont('Helvetica',8); c.setFillColor(colors.grey); c.drawString(.65*inch,.4*inch,'Corrected neural-network simulation theorem'); c.drawRightString(7.85*inch,.4*inch,f'page {doc.page}'); c.restoreState()
SimpleDocTemplate(str(pdf),pagesize=letter,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.6*inch,bottomMargin=.65*inch,title='Neural Networks and Turing Machines').build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
