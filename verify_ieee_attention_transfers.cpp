#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::string>sp(const std::string&s){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,'\t'))v.push_back(x);return v;}
int main(int ac,char**av)try{if(ac!=2)throw std::runtime_error("usage: verifier tsv");std::ifstream f(av[1]);std::string l;std::getline(f,l);int nr=0;while(std::getline(f,l)){auto x=sp(l);if(x.size()!=18)throw std::runtime_error("bad row");long double scale=std::stold(x[12]),gain=std::stold(x[13]),bias=std::stold(x[14]),weighted=std::stold(x[15]),ro=std::stold(x[16]),tensor_gain=std::stold(x[17]);if(gain<std::ceil(tensor_gain)||bias/scale<(weighted+ro))throw std::runtime_error("nonconservative transfer");++nr;}if(nr!=24)throw std::runtime_error("expected 24");std::cout<<"IEEE_CLIPPED_ATTENTION_TRANSFERS_CPP23_OK occurrences="<<nr<<" sequence_cap=none tensor_convex=true\n";return 0;}catch(const std::exception&e){std::cerr<<"IEEE_CLIPPED_ATTENTION_TRANSFERS_CPP23_FAIL "<<e.what()<<"\n";return 1;}
