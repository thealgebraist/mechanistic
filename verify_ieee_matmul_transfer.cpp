#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream q(s);std::string x;while(std::getline(q,x,'\t'))v.push_back(x);return v;}
int main(int argc,char**argv)try{
 if(argc!=2)throw std::runtime_error("usage: verifier transfer.tsv");std::ifstream f(argv[1]);std::string l;std::getline(f,l);std::getline(f,l);auto x=split(l);if(x.size()!=8)throw std::runtime_error("bad row");
 const long double n=std::stold(x[1]),W=std::stold(x[2]),product=std::stold(x[4]),scale=std::stold(x[5]);
 const auto gain=std::stoull(x[6]),bias=std::stoull(x[7]);const long double u=std::ldexp(1.0L,-24),k=2*n,gamma=k*u/(1-k*u),eta2=std::ldexp(1.0L,-150);
 if(gain<std::ceil(W))throw std::runtime_error("gain not conservative");if(static_cast<long double>(bias)/scale<gamma*product+k*eta2)throw std::runtime_error("bias not conservative");
 std::cout<<"IEEE_MATMUL_TRANSFER_CPP23_OK dot_length="<<static_cast<int>(n)<<" gain="<<gain<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"IEEE_MATMUL_TRANSFER_CPP23_FAIL "<<e.what()<<"\n";return 1;}
