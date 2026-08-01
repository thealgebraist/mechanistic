#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,'\t'))v.push_back(x);return v;}
int main(int ac,char**av)try{
 if(ac!=2)throw std::runtime_error("usage: verifier tsv");std::ifstream f(av[1]);std::string l;std::getline(f,l);int nr=0;const long double u=std::ldexp(1.L,-24),eta=std::ldexp(1.L,-150),Lg=2;
 while(std::getline(f,l)){auto x=split(l);if(x.size()!=10)throw std::runtime_error("bad row");long double B=std::stold(x[1]),A=std::stold(x[2]),C=std::stold(x[3]),O=std::stold(x[4]),U=std::stold(x[5]),V=std::stold(x[6]),scale=std::stold(x[7]),gain=std::stold(x[8]),bias=std::stold(x[9]);auto gam=[&](long double k){return k*u/(1-k*u);};long double G=O*A*C*B*(Lg+1),r0=gam(1024)*U+1024*eta,r1=gam(1024)*V+1024*eta,ge=Lg*r0+u*U+eta,pe=ge*V+(U+ge)*r1+u*(U+ge)*(V+r1)+eta,oe=O*pe+gam(2048)*O*(U*V+pe)+2048*eta;if(gain<std::ceil(G)||bias/scale<oe)throw std::runtime_error("nonconservative transfer");++nr;}
 if(nr!=16)throw std::runtime_error("expected 16");std::cout<<"IEEE_CLIPPED_MLP_TRANSFERS_CPP23_OK occurrences="<<nr<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"IEEE_CLIPPED_MLP_TRANSFERS_CPP23_FAIL "<<e.what()<<"\n";return 1;}
