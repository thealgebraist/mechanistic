"""Emit the certified finite PRSL quotient as a standalone C++23 program."""
import json
from pathlib import Path
q=json.loads(Path("outputs/flan_domain32_quotient.json").read_text())
o=[]; a=o.append
a('#include <array>'); a('#include <iomanip>'); a('#include <iostream>'); a('#include <map>'); a('#include <random>'); a('#include <string>'); a('#include <vector>'); a('using namespace std;')
a('struct Edge{int token; int next;}; struct State{int depth; array<int,8> token; array<int,8> mass; int other; int edge_count; array<Edge,8> edge;};')
a(f'constexpr int H={q["horizon"]}; constexpr int N={len(q["states"])}; constexpr int D={len(q["prompts"])};')
a('const State states[N] = {')
for s in q['states']:
 em=s['emit'][:2]; toks=[x[0] for x in em]+[0]*6; mass=[x[1] for x in em]+[0]*6; edges=s['next'][:2]; ee=[f'Edge{{{x[0]},{x[1]}}}' for x in edges]+['Edge{0,0}']*(8-len(edges)); other=65535-sum(x[1] for x in em)
 a('{%d,{{%s}},{{%s}},%d,%d,{{%s}}},'%(s['depth'],','.join(map(str,toks)),','.join(map(str,mass)),other,len(edges),','.join(ee)))
a('};')
a('const int roots[D] = {'+','.join(str(q['roots'][str(i)]) for i in range(len(q['prompts'])))+'};')
a('const char* prompts[D] = {'+','.join(json.dumps(x) for x in q['prompts'])+'};')
a('void replay(int sid,string prefix,double p,map<string,double>& out){ const auto&s=states[sid]; if(s.depth==H){out[prefix]+=p;return;} double z=65535.0; for(int i=0;i<8;++i) if(s.mass[i]){double pi=p*s.mass[i]/z; if(s.depth==H-1) out[prefix+to_string(s.token[i])+" "]+=pi; else {int nx=-1;for(int j=0;j<s.edge_count;++j)if(s.edge[j].token==s.token[i])nx=s.edge[j].next;if(nx>=0)replay(nx,prefix+to_string(s.token[i])+" ",pi,out);}} out[prefix+"OTHER"]+=p*s.other/z; }')
a('void sample(int pid,unsigned seed){mt19937 g(seed);int sid=roots[pid];cout<<"prompt="<<pid<<" text="<<prompts[pid]<<"\\npath=";for(int step=0;step<H;++step){const auto&s=states[sid];uniform_int_distribution<int> u(1,65535);int draw=u(g),chosen=-1;for(int i=0;i<2;++i){if(draw<=s.mass[i]){chosen=i;break;}draw-=s.mass[i];}if(chosen<0){cout<<"OTHER\\n";return;}cout<<s.token[chosen]<<" ";if(step==H-1){cout<<"\\n";return;}int nx=-1;for(int j=0;j<s.edge_count;++j)if(s.edge[j].token==s.token[chosen])nx=s.edge[j].next;if(nx<0){cout<<"OTHER\\n";return;}sid=nx;}cout<<"\\n";}')
a('int main(int argc,char**argv){cout<<"PRSL-STACK-1-CXX23\\n"<<"states="<<N<<" prompts="<<D<<" horizon="<<H<<"\\n";if(argc==4&&string(argv[1])=="--sample"){sample(stoi(argv[2]),stoul(argv[3]));return 0;}cout<<fixed<<setprecision(12);for(int d=0;d<D;++d){map<string,double> law;replay(roots[d],"",1.0,law);double mass=0;for(auto&x:law)mass+=x.second;cout<<"prompt="<<d<<" outcomes="<<law.size()<<" mass="<<mass<<"\\n";}}')
Path('flan_prsl_cxx23.cpp').write_text('\n'.join(o)+'\n')
print({'source':'outputs/flan_domain32_quotient.json','output':'flan_prsl_cxx23.cpp','states':len(q['states']),'prompts':len(q['prompts'])})
