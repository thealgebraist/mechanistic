#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
int main(int ac,char**av)try{if(ac!=2)throw std::runtime_error("usage: verifier tsv");std::ifstream f(av[1]);std::string l;std::getline(f,l);std::unordered_set<std::string> kinds;int expected_pc=0,last_macro=-1;while(std::getline(f,l)){std::stringstream s(l);std::string a,b,k,r;std::getline(s,a,'\t');std::getline(s,b,'\t');std::getline(s,k,'\t');std::getline(s,r,'\t');int pc=std::stoi(a),m=std::stoi(b);if(pc!=expected_pc++)throw std::runtime_error("noncontiguous pc");if(m<last_macro||m>last_macro+1)throw std::runtime_error("noncontiguous macro spans");last_macro=m;kinds.insert(k);if(k.rfind("F32_",0)==0&&r!="binary32_roundTiesToEven_after_each_scalar_result")throw std::runtime_error("missing f32 rounding");}if(last_macro!=128)throw std::runtime_error("expected 129 macros");if(!kinds.count("CATEGORICAL_INVERSE_CDF")||!kinds.count("F32_RSQRT")||!kinds.count("F32_EXP"))throw std::runtime_error("missing probabilistic/nonlinear primitive");std::cout<<"FLAN_BINARY32_MICROCODE_CPP23_OK micro_ops="<<expected_pc<<" kinds="<<kinds.size()<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"FLAN_BINARY32_MICROCODE_CPP23_FAIL "<<e.what()<<"\n";return 1;}
