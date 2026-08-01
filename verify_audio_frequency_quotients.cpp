#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Node { int id; double lo,hi; std::string kind; std::vector<int> members; };

int main(int argc,char**argv){
  assert(argc==2);
  std::ifstream in(argv[1]); assert(in);
  std::string line; std::getline(in,line);
  std::map<std::string,std::vector<Node>> methods;
  while(std::getline(in,line)){
    std::stringstream ss(line); std::string method,id,lo,hi,kind,members;
    std::getline(ss,method,'\t');std::getline(ss,id,'\t');std::getline(ss,lo,'\t');
    std::getline(ss,hi,'\t');std::getline(ss,kind,'\t');std::getline(ss,members);
    Node n{std::stoi(id),std::stod(lo),std::stod(hi),kind,{}};
    std::stringstream ms(members);std::string x;
    while(std::getline(ms,x,','))if(!x.empty())n.members.push_back(std::stoi(x));
    assert(n.lo>=0 && n.hi<=8000.000001 && n.lo<=n.hi && !n.members.empty());
    methods[method].push_back(std::move(n));
  }
  assert(methods.size()==5);
  for(auto&[name,nodes]:methods){
    assert(nodes.size()==80);
    std::sort(nodes.begin(),nodes.end(),[](auto&a,auto&b){return a.id<b.id;});
    for(int i=0;i<80;++i)assert(nodes[i].id==i);
  }
  auto partition=[&](const std::string&name,int count){
    std::vector<int> seen(count);
    for(const auto&n:methods.at(name))for(int x:n.members){assert(x>=0&&x<count);++seen[x];}
    for(int n:seen)assert(n==1);
  };
  partition("linear-subband",201);
  partition("wavelet-packet",128);
  partition("carfac-cochlea",81);
  for(const auto&n:methods.at("goertzel-resonator"))assert(n.members.size()==1);
  const auto& cochlea=methods.at("carfac-cochlea");
  assert(std::abs(cochlea.front().lo)<1e-8);
  assert(std::abs(cochlea.back().hi-8000)<1e-8);
  for(int i=1;i<80;++i)assert(std::abs(cochlea[i-1].hi-cochlea[i].lo)<1e-6);
  std::cout<<"AUDIO_FREQUENCY_QUOTIENT_PARTITIONS_OK methods=5 nodes=400\n";
}
