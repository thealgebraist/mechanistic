#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Model {
  std::uint16_t states{}, tokens{}, denominator{}, initial{};
  std::vector<std::string> token_name, state_name;
  std::vector<std::vector<std::uint16_t>> emission, successor;
};
struct Row { int state{}, block{}; std::string name; std::vector<int> emission, successor_block; };

static std::uint16_t get16(std::ifstream& in){int a=in.get(),b=in.get();if(a<0||b<0)throw std::runtime_error("truncated model");return static_cast<std::uint16_t>(a|(b<<8));}
static std::string get_string(std::ifstream& in){int n=in.get();if(n<0)throw std::runtime_error("truncated label");std::string s(n,'\0');in.read(s.data(),n);if(!in)throw std::runtime_error("truncated label bytes");return s;}
static Model read_model(const std::string& path){std::ifstream in(path,std::ios::binary);char magic[4]{};in.read(magic,4);if(std::string(magic,4)!="PTM1"||get16(in)!=1)throw std::runtime_error("bad model header");Model m;m.states=get16(in);m.tokens=get16(in);m.denominator=get16(in);m.initial=get16(in);for(int i=0;i<m.tokens;++i)m.token_name.push_back(get_string(in));for(int i=0;i<m.states;++i)m.state_name.push_back(get_string(in));m.emission.assign(m.states,std::vector<std::uint16_t>(m.tokens));m.successor.assign(m.states,std::vector<std::uint16_t>(m.tokens));for(auto&r:m.emission)for(auto&x:r)x=get16(in);for(auto&r:m.successor)for(auto&x:r)x=get16(in);return m;}
static std::vector<std::string> split(const std::string&s,char delimiter){std::vector<std::string>v;std::istringstream in(s);for(std::string x;std::getline(in,x,delimiter);)v.push_back(x);return v;}
static std::vector<int> ints(const std::string&s){std::vector<int>v;for(const auto&x:split(s,','))v.push_back(std::stoi(x));return v;}

static std::vector<Row> read_certificate(const std::string& path,int& declared_blocks){
  std::ifstream in(path);std::string line;declared_blocks=-1;std::vector<Row>rows;
  while(std::getline(in,line)){if(line.rfind("#quotient_states\t",0)==0)declared_blocks=std::stoi(line.substr(line.find('\t')+1));if(line.empty()||line[0]=='#'||line.rfind("state\t",0)==0)continue;auto c=split(line,'\t');if(c.size()!=5)throw std::runtime_error("bad certificate row");rows.push_back({std::stoi(c[0]),std::stoi(c[1]),c[2],ints(c[3]),ints(c[4])});}
  if(declared_blocks<1)throw std::runtime_error("missing quotient count");return rows;
}

int main(int argc,char**argv)try{
  if(argc!=3)throw std::runtime_error("usage: verifier model.ptm certificate.tsv");
  const Model m=read_model(argv[1]);int blocks=0;const auto rows=read_certificate(argv[2],blocks);
  if(rows.size()!=m.states)throw std::runtime_error("certificate does not cover source states");
  std::vector<int>projection(m.states,-1),representative(blocks,-1);
  for(const auto&r:rows){if(r.state<0||r.state>=m.states||projection[r.state]>=0||r.block<0||r.block>=blocks)throw std::runtime_error("invalid projection row");if(r.name!=m.state_name[r.state]||r.emission.size()!=m.tokens||r.successor_block.size()!=m.tokens)throw std::runtime_error("row shape/name mismatch");projection[r.state]=r.block;if(representative[r.block]<0)representative[r.block]=r.state;for(int a=0;a<m.tokens;++a){if(r.emission[a]!=m.emission[r.state][a])throw std::runtime_error("emission does not match binary");}}
  if(std::find(representative.begin(),representative.end(),-1)!=representative.end())throw std::runtime_error("projection is not surjective");

  // Matrix equation lambda Q = O and transition equations delta_a Q = Q T_a.
  for(const auto&r:rows){int rep=representative[r.block];if(m.emission[r.state]!=m.emission[rep])throw std::runtime_error("lambda Q != O");for(int a=0;a<m.tokens;++a){int expected=projection[m.successor[r.state][a]];if(r.successor_block[a]!=expected||expected!=projection[m.successor[rep][a]])throw std::runtime_error("delta_a Q != Q T_a");}}

  // Every concrete state must be reachable by a positive-mass generated word.
  std::vector<bool>reachable(m.states);std::queue<int>todo;reachable[m.initial]=true;todo.push(m.initial);
  while(!todo.empty()){int q=todo.front();todo.pop();for(int a=0;a<m.tokens;++a)if(m.emission[q][a]){int d=m.successor[q][a];if(!reachable[d]){reachable[d]=true;todo.push(d);}}}
  if(std::find(reachable.begin(),reachable.end(),false)!=reachable.end())throw std::runtime_error("unreachable source state");

  // Independent coarsest-partition check.  Start from equal observations and
  // refine by token-indexed successor blocks to a fixed point.
  std::vector<int>independent(m.states);
  for(;;){std::vector<std::vector<int>>signatures;std::vector<int>next(m.states);for(int q=0;q<m.states;++q){std::vector<int>sig(m.emission[q].begin(),m.emission[q].end());for(auto d:m.successor[q])sig.push_back(independent[d]);auto it=std::find(signatures.begin(),signatures.end(),sig);if(it==signatures.end()){signatures.push_back(sig);next[q]=static_cast<int>(signatures.size()-1);}else next[q]=static_cast<int>(it-signatures.begin());}if(next==independent)break;independent=std::move(next);}
  for(int a=0;a<m.states;++a)for(int b=0;b<m.states;++b)if((projection[a]==projection[b])!=(independent[a]==independent[b]))throw std::runtime_error("quotient is not the coarsest stable partition");

  // Exact concrete sentence law: keys are found .
  const std::vector<int>sentence{1,3,4,6};std::uint64_t numerator=1,denominator_power=1;int q=m.initial;
  for(int token:sentence){numerator*=m.emission[q][token];denominator_power*=m.denominator;q=m.successor[q][token];}
  if(numerator*128!=denominator_power*35)throw std::runtime_error("unexpected sentence probability");
  std::cout<<"TOY_PROBABILISTIC_GRAPH_CPP23_OK source_states="<<m.states<<" quotient_states="<<blocks
           <<" reachable="<<m.states<<" matrix_equations=EXACT minimal=EXACT all_finite_traces=BY_INDUCTION sentence_probability=35/128\n";
  return 0;
}catch(const std::exception&e){std::cerr<<"TOY_PROBABILISTIC_GRAPH_CPP23_FAIL "<<e.what()<<'\n';return 1;}
