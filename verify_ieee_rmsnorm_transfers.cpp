#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,'\t'))v.push_back(x);return v;}
int main(int argc,char**argv)try{if(argc!=2)throw std::runtime_error("usage: verifier tsv");std::ifstream f(argv[1]);std::string l;std::getline(f,l);int count=0;const long double u=std::ldexp(1.0L,-24),eta=std::ldexp(1.0L,-150);
 while(std::getline(f,l)){auto x=split(l);if(x.size()!=8)throw std::runtime_error("bad row");long double B=std::stold(x[1]),w=std::stold(x[2]),eps=std::stold(x[3]),d=std::stold(x[4]),scale=std::stold(x[5]),gain=std::stold(x[6]),bias=std::stold(x[7]);(void)B;long double se=std::sqrt(eps),k=2*d+1,gamma=k*u/(1-k*u);long double L=w*(1+std::sqrt(d))/se;long double alpha=gamma+u*(1+gamma),beta0=k*eta,beta=beta0+u*(eps+beta0)+eta,rho=alpha+beta/eps,om=1-rho;if(!(rho<1))throw std::runtime_error("rho must be below one");long double rt=rho/(2*std::pow(om,1.5L))+u/std::sqrt(om),nb=std::sqrt(d),en=nb*rt+u*nb*(1+u)/std::sqrt(om)+eta,eo=w*en+u*w*(nb+en)+eta;
  if(gain<std::ceil(L))throw std::runtime_error("gain not conservative");if(static_cast<long double>(bias)/scale<eo)throw std::runtime_error("bias not conservative");++count;}
 if(count!=42)throw std::runtime_error("expected 42");std::cout<<"IEEE_RMSNORM_TRANSFERS_CPP23_OK occurrences="<<count<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"IEEE_RMSNORM_TRANSFERS_CPP23_FAIL "<<e.what()<<"\n";return 1;}
