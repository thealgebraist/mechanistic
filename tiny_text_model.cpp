#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Vec=std::vector<double>;
struct Example { std::vector<int> x; int y; };

static Vec softmax(const Vec& z){ double m=*std::max_element(z.begin(),z.end()); Vec p(z.size()); double s=0; for(size_t i=0;i<z.size();++i)s+=p[i]=std::exp(z[i]-m); for(double&v:p)v/=s; return p; }
static std::vector<std::string> split(const std::string& s){std::istringstream in(s);std::vector<std::string> v;for(std::string w;in>>w;)v.push_back(w);return v;}

struct Model {
  int V,H; std::vector<std::string> tok; std::map<std::string,int> id; std::vector<Vec> E,W2; Vec b1,b2; std::mt19937 rng{7};
  Model(std::vector<std::string> t,int h):V((int)t.size()),H(h),tok(std::move(t)),E(V,Vec(H)),W2(V,Vec(H)),b1(H),b2(V){for(int i=0;i<V;++i)id[tok[i]]=i; std::normal_distribution<double>d(0,.08);for(auto&a:E)for(auto&v:a)v=d(rng);for(auto&a:W2)for(auto&v:a)v=d(rng);}
  std::pair<Vec,Vec> forward(const std::vector<int>& x) const { Vec h(H); for(int i:x)for(int j=0;j<H;++j)h[j]+=E[i][j]; for(double&v:h)v=std::tanh(v); Vec z=b2; for(int y=0;y<V;++y)for(int j=0;j<H;++j)z[y]+=W2[y][j]*h[j]; return {h,softmax(z)}; }
  void train(const std::vector<Example>& ds,int epochs){double lr=.035;for(int ep=0;ep<epochs;++ep){for(auto&e:ds){auto [h,p]=forward(e.x);Vec dz=p;dz[e.y]-=1;Vec dh(H);for(int j=0;j<H;++j)for(int y=0;y<V;++y)dh[j]+=W2[y][j]*dz[y];for(int y=0;y<V;++y){b2[y]-=lr*dz[y];for(int j=0;j<H;++j)W2[y][j]-=lr*dz[y]*h[j];}for(int j=0;j<H;++j){double q=lr*dh[j]*(1-h[j]*h[j]);b1[j]-=q;for(int i:e.x)E[i][j]-=q;}}}}
  void trace(const std::string& prompt,const std::string& out) const {auto w=split(prompt);std::vector<int>x;for(auto&s:w)x.push_back(id.at(s));auto [h,p]=forward(x);std::vector<int> order(V);std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int a,int b){return p[a]>p[b];});std::ofstream f(out);f<<"prompt\t"<<prompt<<"\n";f<<"token\tprob\tlogit\n";for(int k=0;k<6&&k<V;++k){int y=order[k];double logit=std::log(p[y]);f<<tok[y]<<"\t"<<std::setprecision(8)<<p[y]<<"\t"<<logit<<"\n";}f<<"\nhidden_unit\tactivation\tcontribution_to_top\n";int top=order[0];for(int j=0;j<H;++j)f<<j<<"\t"<<h[j]<<"\t"<<W2[top][j]*h[j]<<"\n";}
  void analyze(const std::string& prompt,const std::vector<std::string>& candidates) const {std::vector<int>x;for(auto&s:split(prompt))x.push_back(id.at(s));auto base=forward(x);int top=std::max_element(base.second.begin(),base.second.end())-base.second.begin();std::cout<<"\nPROMPT: "<<prompt<<"\nbase top: "<<tok[top]<<" p="<<base.second[top]<<"\n\nCANDIDATE LOGITS / hidden delta\n";for(auto&s:candidates){auto xx=x;xx.push_back(id.at(s));auto q=forward(xx);int t=std::max_element(q.second.begin(),q.second.end())-q.second.begin();double d=std::sqrt(std::inner_product(base.first.begin(),base.first.end(),q.first.begin(),0.,std::plus<>(),[](double a,double b){return (a-b)*(a-b);}));std::cout<<"append "<<s<<" -> "<<tok[t]<<" p="<<q.second[t]<<" | hidden_L2="<<d<<"\n";}}
};

int main(){
 std::vector<std::string> T={"the","key","keys","to","cabinet","cabinet,","are","is","on","table","tables","missing","found","bright","old","and","theory","facts","."}; Model m(T,12);
 std::vector<Example> d; auto add=[&](const std::string&s,const std::string&y){std::vector<int>x;for(auto&w:split(s))x.push_back(m.id[w]);d.push_back({x,m.id[y]});};
 for(int i=0;i<160;++i){add("the key to the cabinet","is");add("the keys to the cabinet","are");add("the key is on the","table");add("the keys are on the","tables");add("the bright key is","found");add("the old keys are","missing");add("theory and","facts");}
 m.train(d,90); m.trace("the keys to the cabinet","outputs/tiny_trace.tsv"); m.analyze("the keys to the cabinet",{"are","is","missing"});
 // Mechanistic probes: zero one hidden unit and measure the selected logit shift.
 std::vector<int>x;for(auto&s:split("the keys to the cabinet"))x.push_back(m.id[s]);auto base=m.forward(x);int are=m.id["are"];std::cout<<"\nUNIT ABLATION (change in logit for 'are')\n";for(int j=0;j<m.H;++j){double c=m.W2[are][j]*base.first[j];std::cout<<"unit "<<j<<" contribution="<<c<<"\n";}
 std::cout<<"\nWrote outputs/tiny_trace.tsv\n";
}
