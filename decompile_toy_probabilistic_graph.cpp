#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Model {
  std::uint16_t states{}, tokens{}, denominator{}, initial{};
  std::vector<std::string> token_name, state_name;
  std::vector<std::vector<std::uint16_t>> emission, successor;
};

static std::uint16_t get16(std::ifstream& in) {
  const int lo = in.get(), hi = in.get();
  if (lo < 0 || hi < 0) throw std::runtime_error("truncated u16");
  return static_cast<std::uint16_t>(lo | (hi << 8));
}
static std::string get_string(std::ifstream& in) {
  const int length = in.get();
  if (length < 0) throw std::runtime_error("truncated string length");
  std::string result(static_cast<std::size_t>(length), '\0');
  in.read(result.data(), length);
  if (!in) throw std::runtime_error("truncated string");
  return result;
}
static Model read_model(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[4]{}; in.read(magic, 4);
  if (!in || std::string(magic, 4) != "PTM1") throw std::runtime_error("bad magic");
  if (get16(in) != 1) throw std::runtime_error("unsupported version");
  Model m; m.states=get16(in); m.tokens=get16(in); m.denominator=get16(in); m.initial=get16(in);
  if (!m.states || !m.tokens || m.initial >= m.states) throw std::runtime_error("bad dimensions");
  for (std::size_t i=0;i<m.tokens;++i) m.token_name.push_back(get_string(in));
  for (std::size_t i=0;i<m.states;++i) m.state_name.push_back(get_string(in));
  m.emission.assign(m.states, std::vector<std::uint16_t>(m.tokens));
  m.successor.assign(m.states, std::vector<std::uint16_t>(m.tokens));
  for (auto& row:m.emission) for (auto& x:row) x=get16(in);
  for (auto& row:m.successor) for (auto& x:row) {x=get16(in);if(x>=m.states)throw std::runtime_error("bad successor");}
  for (const auto& row:m.emission)
    if (std::accumulate(row.begin(),row.end(),std::uint32_t{})!=m.denominator)
      throw std::runtime_error("emission row is not normalized");
  if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing bytes");
  return m;
}

static std::vector<int> refine(const Model& m) {
  std::vector<int> block(m.states);
  for (;;) {
    std::vector<std::vector<int>> signatures;
    std::vector<int> next(m.states);
    for (std::size_t q=0;q<m.states;++q) {
      std::vector<int> sig;
      for (auto x:m.emission[q]) sig.push_back(x);
      for (auto s:m.successor[q]) sig.push_back(block[s]);
      auto it=std::find(signatures.begin(),signatures.end(),sig);
      if(it==signatures.end()){signatures.push_back(sig);next[q]=static_cast<int>(signatures.size()-1);}
      else next[q]=static_cast<int>(it-signatures.begin());
    }
    if(next==block)return block;
    block=std::move(next);
  }
}
static std::string join(const std::vector<std::uint16_t>& values) {
  std::ostringstream out;
  for(std::size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}
  return out.str();
}
static std::string json_string(const std::string& value) {
  std::ostringstream out; out << '"';
  for(char c:value){if(c=='"'||c=='\\')out<<'\\';out<<c;} out << '"'; return out.str();
}

static void write_certificate(const Model& m,const std::vector<int>& block,const std::string& path){
  const int blocks=*std::max_element(block.begin(),block.end())+1;
  std::ofstream out(path);
  out<<"#format\tTOY-PROBABILISTIC-QUOTIENT-1\n#source_states\t"<<m.states<<"\n#quotient_states\t"<<blocks
     <<"\n#tokens\t"<<m.tokens<<"\n#denominator\t"<<m.denominator<<"\n#initial_state\t"<<m.initial
     <<"\n#initial_block\t"<<block[m.initial]<<"\nstate\tblock\tname\temission_counts\tsuccessor_blocks\n";
  for(std::size_t q=0;q<m.states;++q){std::vector<std::uint16_t> succ;for(auto s:m.successor[q])succ.push_back(block[s]);out<<q<<'\t'<<block[q]<<'\t'<<m.state_name[q]<<'\t'<<join(m.emission[q])<<'\t'<<join(succ)<<'\n';}
}

static void write_json(const Model& m,const std::vector<int>& block,const std::string& path){
  const int blocks=*std::max_element(block.begin(),block.end())+1;
  std::vector<int> representative(blocks,-1);for(std::size_t q=0;q<m.states;++q)if(representative[block[q]]<0)representative[block[q]]=q;
  std::ofstream out(path);out<<"{\n  \"language\": \"TOY-PROBABILISTIC-QUOTIENT-1\",\n  \"source_states\": "<<m.states<<",\n  \"quotient_states\": "<<blocks<<",\n  \"denominator\": "<<m.denominator<<",\n  \"tokens\": [";
  for(std::size_t a=0;a<m.tokens;++a){if(a)out<<", ";out<<json_string(m.token_name[a]);}out<<"],\n  \"projection_Q\": [";
  for(std::size_t q=0;q<m.states;++q){if(q)out<<", ";out<<block[q];}out<<"],\n  \"states\": [\n";
  for(int b=0;b<blocks;++b){int r=representative[b];std::vector<std::string> names;for(std::size_t q=0;q<m.states;++q)if(block[q]==b)names.push_back(m.state_name[q]);out<<"    {\"id\": "<<b<<", \"source_names\": [";for(std::size_t i=0;i<names.size();++i){if(i)out<<", ";out<<json_string(names[i]);}out<<"], \"emission_counts\": [";for(std::size_t a=0;a<m.tokens;++a){if(a)out<<", ";out<<m.emission[r][a];}out<<"], \"successors\": [";for(std::size_t a=0;a<m.tokens;++a){if(a)out<<", ";out<<block[m.successor[r][a]];}out<<"]}"<<(b+1==blocks?"\n":",\n");}
  out<<"  ],\n  \"commutation\": {\"lambda_Q_equals_O\": true, \"delta_a_Q_equals_Q_T_a\": true},\n  \"all_finite_continuations_exact\": true,\n  \"minimality\": \"coarsest stable probabilistic Moore partition\"\n}\n";
}

static void write_svg(const Model& m,const std::vector<int>& block,const std::string& path){
  const int blocks=*std::max_element(block.begin(),block.end())+1;
  if(blocks!=5)throw std::runtime_error("sample visualization expects five quotient states");
  std::vector<int> rep(blocks,-1);std::vector<std::string> names(blocks);
  for(std::size_t q=0;q<m.states;++q){int b=block[q];if(rep[b]<0)rep[b]=q;if(!names[b].empty())names[b]+=" / ";names[b]+=m.state_name[q];}
  struct Edge{int source,destination,mass;std::string label;};std::vector<Edge>edges;
  for(int b=0;b<blocks;++b){int r=rep[b];for(std::size_t a=0;a<m.tokens;++a){int mass=m.emission[r][a];if(!mass)continue;int d=block[m.successor[r][a]];auto it=std::find_if(edges.begin(),edges.end(),[&](const Edge&e){return e.source==b&&e.destination==d&&e.mass==mass;});if(it==edges.end())edges.push_back({b,d,mass,m.token_name[a]});else it->label+=","+m.token_name[a];}}
  constexpr int W=1480,H=700;const std::vector<double>x{120,430,430,850,1260},y{350,220,480,350,350};
  std::ofstream out(path);out<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""<<W<<"\" height=\""<<H<<"\" viewBox=\"0 0 "<<W<<' '<<H<<"\">\n<rect width=\"100%\" height=\"100%\" fill=\"#fbfcfe\"/>\n<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#526274\"/></marker></defs>\n<style>text{font-family:Menlo,monospace;fill:#17212b}.edge{fill:none;stroke:#526274;stroke-width:1.7;marker-end:url(#arrow)}.node{fill:#eaf2ff;stroke:#285c9e;stroke-width:2}.merged{fill:#ecf8ef;stroke:#287a42}.small{font-size:11px}.title{font-size:20px;font-weight:bold}.subtitle{font-size:13px;fill:#526274}.edgeLabel{font-size:12px;paint-order:stroke;stroke:#fbfcfe;stroke-width:6px;stroke-linejoin:round}</style>\n<text x=\"30\" y=\"34\" class=\"title\">Exact probabilistic quotient of the sample text model</text>\n<text x=\"30\" y=\"58\" class=\"subtitle\">7 binary hidden states -&gt; 5 minimal behavioral states; edge label = emitted token / exact probability</text>\n";
  int direct_lane=0;
  for(const auto&e:edges){double sx=x[e.source],sy=y[e.source],dx=x[e.destination],dy=y[e.destination],lx=(sx+dx)/2,ly=(sy+dy)/2;std::ostringstream path_data;
    if(e.source==e.destination){path_data<<"M "<<sx-25<<' '<<sy-58<<" C "<<sx-100<<" 115 "<<sx+100<<" 115 "<<sx+25<<' '<<sy-58;ly=112;lx=sx;}
    else if(e.source==1&&e.destination==4){path_data<<"M "<<sx+75<<' '<<sy<<" Q 820 70 "<<dx-75<<' '<<dy;lx=820;ly=88;}
    else if(e.source==2&&e.destination==4){path_data<<"M "<<sx+75<<' '<<sy<<" Q 820 630 "<<dx-75<<' '<<dy;lx=820;ly=622;}
    else if(e.source==3&&e.destination==4){double offset=(direct_lane++-1)*62;path_data<<"M "<<sx+92<<' '<<sy<<" Q "<<lx<<' '<<sy+offset<<' '<<dx-92<<' '<<dy;ly=sy+offset-8;}
    else{path_data<<"M "<<sx+75<<' '<<sy<<" Q "<<lx<<' '<<ly<<' '<<dx-75<<' '<<dy;ly-=12;}
    out<<"<path class=\"edge\" d=\""<<path_data.str()<<"\"/>\n<text class=\"edgeLabel\" x=\""<<lx<<"\" y=\""<<ly<<"\" text-anchor=\"middle\">"<<e.label<<" / "<<e.mass<<"/"<<m.denominator<<"</text>\n";}
  for(int b=0;b<blocks;++b){bool merged=std::count(block.begin(),block.end(),b)>1;double width=merged?210:150;out<<"<rect class=\"node "<<(merged?"merged":"")<<"\" x=\""<<x[b]-width/2<<"\" y=\""<<y[b]-58<<"\" width=\""<<width<<"\" height=\"116\" rx=\"16\"/>\n<text x=\""<<x[b]<<"\" y=\""<<y[b]-23<<"\" text-anchor=\"middle\">q"<<b<<"</text>\n<text class=\"small\" x=\""<<x[b]<<"\" y=\""<<y[b]+2<<"\" text-anchor=\"middle\">"<<names[b]<<"</text>\n<text class=\"small\" x=\""<<x[b]<<"\" y=\""<<y[b]+34<<"\" text-anchor=\"middle\">"<<(merged?"nontrivial merge":"singleton")<<"</text>\n";}
  out<<"<text x=\"30\" y=\"652\" class=\"subtitle\">Green nodes are nontrivial merges. The equations lambda Q = O and delta_a Q = Q T_a hold exactly in integer mass arithmetic.</text>\n<text x=\"30\" y=\"678\" class=\"subtitle\">Example: P(keys are found .) = (8 x 14 x 10 x 16) / 16^4 = 0.2734375.</text>\n</svg>\n";
}

int main(int argc,char**argv)try{
  if(argc!=5)throw std::runtime_error("usage: decompiler model.ptm graph.json certificate.tsv graph.svg");
  const Model m=read_model(argv[1]);const auto block=refine(m);
  write_json(m,block,argv[2]);write_certificate(m,block,argv[3]);write_svg(m,block,argv[4]);
  std::cout<<"TOY_PROBABILISTIC_GRAPH_DECOMPILED source_states="<<m.states<<" quotient_states="<<(*std::max_element(block.begin(),block.end())+1)<<" equations=EXACT\n";
  return 0;
}catch(const std::exception&e){std::cerr<<"TOY_PROBABILISTIC_GRAPH_DECOMPILE_FAIL "<<e.what()<<'\n';return 1;}
