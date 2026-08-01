#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;
struct Edge { uint16_t token, next; };
struct State { uint8_t depth; uint16_t other; array<uint16_t,2> token{}, mass{}; array<Edge,2> edge{}; };
template<class T> T read(ifstream& f){ T x{}; f.read(reinterpret_cast<char*>(&x),sizeof x); return x; }
void law(uint16_t id,const vector<State>& s,uint8_t H,string pre,double p,map<string,double>& out){
 const auto& x=s[id]; if(x.depth==H){out[pre]+=p;return;} const double z=65535.0;
 for(int i=0;i<2;++i) if(x.mass[i]){double pi=p*x.mass[i]/z; if(x.depth==H-1) out[pre+to_string(x.token[i])+" "]+=pi; else law(x.edge[i].next,s,H,pre+to_string(x.token[i])+" ",pi,out);}
 out[pre+"OTHER"]+=p*x.other/z;
}
int main(int argc,char**argv){
 string path=argc>1?argv[1]:"outputs/flan_domain32.prslb"; ifstream f(path,ios::binary); f.seekg(0,ios::end); auto file_bytes=f.tellg(); f.seekg(0); char magic[6]; f.read(magic,6); if(string(magic,6)!=string("PRSL1\0",6)){cerr<<"bad magic\n";return 2;}
 uint16_t n=read<uint16_t>(f), d=read<uint16_t>(f); uint8_t H=read<uint8_t>(f), branches=read<uint8_t>(f); if(branches!=2)return 3; vector<uint16_t> roots(d);for(auto&x:roots)x=read<uint16_t>(f);vector<State>s(n);
 for(auto&x:s){x.depth=read<uint8_t>(f);x.other=read<uint16_t>(f);for(int i=0;i<2;++i){x.token[i]=read<uint16_t>(f);x.mass[i]=read<uint16_t>(f);}for(auto&e:x.edge){e.token=read<uint16_t>(f);e.next=read<uint16_t>(f);}}
 bool ok=true; for(int i=0;i<d;++i) ok &= roots[i]<n && s[roots[i]].depth==0;
 for(uint16_t i=0;i<n;++i){const auto&x=s[i];ok &= uint32_t(x.mass[0])+x.mass[1]+x.other==65535; if(x.depth<H-1){for(int j=0;j<2;++j)ok &= x.edge[j].next<n && x.edge[j].token==x.token[j] && s[x.edge[j].next].depth==x.depth+1;}else{for(int j=0;j<2;++j)ok &= x.edge[j].token==0 && x.edge[j].next==0;}}
 auto consumed=f.tellg(); ok &= consumed==file_bytes; cout<<"PRSL-BINARY-CXX23 states="<<n<<" prompts="<<d<<" horizon="<<int(H)<<" bytes="<<file_bytes<<" structural="<<(ok?"OK":"FAIL")<<"\n"; if(!ok)return 4;for(int i=0;i<d;++i){map<string,double>o;law(roots[i],s,H,"",1.,o);double z=0;for(auto&x:o)z+=x.second;cout<<fixed<<setprecision(12)<<"prompt="<<i<<" outcomes="<<o.size()<<" mass="<<z<<"\n";if(abs(z-1.)>1e-12)return 5;} }
