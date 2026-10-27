// Quartet evidence and optimized candidate-edge scoring.
// See LICENSE in this directory.
#pragma once
#define CAMUS_QCOUNT_LIBRARY
#include "count.cpp"
#undef CAMUS_QCOUNT_LIBRARY
#include <charconv>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace camus {

constexpr uint64_t taxa_mask = (uint64_t(1)<<60)-1;
constexpr std::array<int,3> topologies{6,10,12};
using Branch = std::array<int,2>;

inline int taxon(uint64_t q,int i) {return (q>>(15*i))&32767;}
inline int topology(uint64_t q) {return q>>60;}
inline uint64_t canonical(std::array<int,4> taxa) {
    int topo=3;
    for(int i=0;i<3;++i) for(int j=i+1;j<4;++j) if(taxa[i]>taxa[j]) {
        if(((topo>>i)&1)!=((topo>>j)&1)) topo^=(1<<i)|(1<<j);
        std::swap(taxa[i],taxa[j]);
    }
    if(std::adjacent_find(taxa.begin(),taxa.end())!=taxa.end()) throw std::runtime_error("Quartet taxa must be distinct");
    if(topo&1) topo^=15;
    uint64_t q=uint64_t(topo)<<60;
    for(int i=0;i<4;++i) q|=uint64_t(taxa[i])<<(15*i);
    return q;
}
inline int channel(uint64_t q) {
    int t=topology(q);
    if(t==6)return 0;
    if(t==10)return 1;
    if(t==12)return 2;
    throw std::runtime_error("Invalid quartet topology");
}

struct Tree {
    std::vector<Node> nodes;
    std::vector<int> parent,depth,tip_node,postorder;
    std::vector<std::string> names;
    std::unordered_map<std::string_view,int> tip_id;
    std::vector<uint8_t> below;
    std::vector<int> lcas;
    int n=0,m=0;

    explicit Tree(const std::string& path) {
        std::ifstream file(path);if(!file)throw std::runtime_error("Cannot read constraint tree");
        std::string text((std::istreambuf_iterator<char>(file)),{});
        auto parsed=Parser{text}.parse();
        std::function<int(int)> copy=[&](int old) {
            while(old!=0 && parsed[old].children.size()==1) old=parsed[old].children[0];
            int id=nodes.size();nodes.emplace_back();nodes[id].name=parsed[old].name;
            for(int c:parsed[old].children) {int child=copy(c);nodes[id].children.push_back(child);}
            return id;
        };
        copy(0);m=nodes.size();
        for(const auto& node:nodes) {
            if(node.children.empty()) names.push_back(node.name);
            else if(node.children.size()!=2) throw std::runtime_error("Constraint tree must be rooted and binary");
        }
        if(nodes[0].children.size()!=2)throw std::runtime_error("Constraint tree must be rooted and binary");
        std::sort(names.begin(),names.end());n=names.size();
        if(n<4||n>=32768)throw std::runtime_error("Require 4 <= tips < 32768");
        tip_id.reserve(n);
        for(int i=0;i<n;++i)tip_id.emplace(names[i],i);
        parent.assign(m,-1);depth.assign(m,0);tip_node.resize(n);
        below.assign(size_t(m)*n,0);
        std::function<void(int)> visit=[&](int u) {
            if(nodes[u].children.empty()) {
                int t=tip_id.at(nodes[u].name);tip_node[t]=u;below[size_t(u)*n+t]=1;
            }
            for(int v:nodes[u].children) {
                parent[v]=u;depth[v]=depth[u]+1;visit(v);
                for(int t=0;t<n;++t)below[size_t(u)*n+t]|=below[size_t(v)*n+t];
            }
            postorder.push_back(u);
        };
        visit(0);lcas.resize(size_t(m)*m);
        for(int a=0;a<m;++a)for(int b=0;b<m;++b) {
            int u=a,v=b;
            while(depth[u]>depth[v])u=parent[u];
            while(depth[v]>depth[u])v=parent[v];
            while(u!=v){u=parent[u];v=parent[v];}
            lcas[size_t(a)*m+b]=u;
        }
    }
    int lca(int u,int v)const{return lcas[size_t(u)*m+v];}
    bool under(int u,int v)const{return u!=v&&lca(u,v)==u;}
    bool contains(int u,int t)const{return below[size_t(u)*n+t];}
    int sibling(int u)const{auto& c=nodes[parent[u]].children;return c[0]==u?c[1]:c[0];}
    int cycle_length(int u,int w)const{int v=lca(u,w);return depth[u]+depth[w]-2*depth[v]+1+(u==v);}
    bool should_score(int u,int w)const{return u!=0&&w!=0&&!under(w,u)&&cycle_length(u,w)>3;}
    int distance(int a,int b)const{int u=tip_node[a],v=tip_node[b];return depth[u]+depth[v]-2*depth[lca(u,v)];}
    int displayed(uint64_t q)const{
        int a=taxon(q,0),b=taxon(q,1),c=taxon(q,2),d=taxon(q,3);
        int x=distance(a,b)+distance(c,d),y=distance(a,c)+distance(b,d),z=distance(a,d)+distance(b,c);
        if(x<y&&x<z)return 12;
        if(y<x&&y<z)return 10;
        if(z<x&&z<y)return 6;
        throw std::runtime_error("Binary constraint tree has an unresolved quartet");
    }
    void preorder(int u,const std::function<void(int)>& f)const{f(u);for(int v:nodes[u].children)preorder(v,f);}
    void across_postorder(int v,const std::function<void(int,int)>& f)const{
        auto& children=nodes[v].children;
        std::function<void(int,int)> visit=[&](int u,int other){for(int c:nodes[u].children)visit(c,other);f(u,other);};
        visit(children[0],children[1]);visit(children[1],children[0]);
    }
};

struct CSV {
    std::ifstream input;
    std::string line;
    std::vector<std::string> decoded;
    std::vector<std::string_view> fields;
    std::map<std::string,int> columns;
    size_t line_number=1;
    explicit CSV(const std::string& path):input(path){
        if(!input||!std::getline(input,line))throw std::runtime_error("Cannot read CSV: "+path);
        auto header=csv_fields(line);
        for(size_t i=0;i<header.size();++i)if(!columns.emplace(header[i],i).second)throw std::runtime_error("Duplicate CSV column");
        fields.reserve(header.size());
    }
    int column(const std::string& name)const{auto at=columns.find(name);if(at==columns.end())throw std::runtime_error("Missing CSV column "+name);return at->second;}
    bool next(){
        while(std::getline(input,line)){
            ++line_number;if(line.empty()||line=="\r")continue;
            fields.clear();
            if(line.find('"')!=std::string::npos){decoded=csv_fields(line);for(auto& s:decoded)fields.push_back(s);}
            else{
                size_t start=0,end=0;
                if(!line.empty()&&line.back()=='\r')line.pop_back();
                while((end=line.find(',',start))!=std::string::npos){fields.emplace_back(line.data()+start,end-start);start=end+1;}
                fields.emplace_back(line.data()+start,line.size()-start);
            }
            if(fields.size()!=columns.size())throw std::runtime_error("Malformed CSV row "+std::to_string(line_number));
            return true;
        }
        if(!input.eof())throw std::runtime_error("CSV read failed");
        return false;
    }
};
inline uint32_t parse_count(std::string_view text){
    uint32_t value=0;auto result=std::from_chars(text.data(),text.data()+text.size(),value);
    if(result.ec!=std::errc()||result.ptr!=text.data()+text.size())throw std::runtime_error("Counts must fit uint32");
    return value;
}
inline double parse_weight(std::string_view text){
    while(!text.empty()&&std::isspace(static_cast<unsigned char>(text.front())))text.remove_prefix(1);
    while(!text.empty()&&std::isspace(static_cast<unsigned char>(text.back())))text.remove_suffix(1);
    if(!text.empty()&&text.front()=='+')text.remove_prefix(1);
    double value=0;auto result=std::from_chars(text.data(),text.data()+text.size(),value);
    if(result.ec!=std::errc()||result.ptr!=text.data()+text.size()||!std::isfinite(value))throw std::runtime_error("Weights must be finite");
    return value;
}

template<class Value>struct Entry{uint64_t q;Value value;};
template<class Value>struct Evidence{
    std::array<std::vector<Entry<Value>>,3> records;
    std::array<uint64_t,4> offsets{};
    std::array<bool,3> ordered{true,true,true};
    uint32_t count_total=0;
    double weight_total=0;
    double guide_weight_total=0;
    void add(uint64_t q,Value value){
        if(value==0)return;
        int c=channel(q);auto& entries=records[c];
        if(!entries.empty()&&entries.back().q>q)ordered[c]=false;
        entries.push_back({q,value});
    }
    void coalesce(){
        double total=0;
        for(int c=0;c<3;++c){
            auto& entries=records[c];
            if(!ordered[c])std::stable_sort(entries.begin(),entries.end(),[](auto& a,auto& b){return a.q<b.q;});
            size_t write=0;
            for(size_t at=0;at<entries.size();){
                uint64_t q=entries[at].q;Value value=0;
                do{
                    if constexpr(std::is_same_v<Value,uint32_t>){
                        uint64_t next=uint64_t(value)+entries[at].value;
                        if(next>UINT32_MAX)throw std::runtime_error("Accumulated count exceeds uint32");
                        value=uint32_t(next);
                    }else{
                        value+=entries[at].value;
                        if(!std::isfinite(value))throw std::runtime_error("Accumulated weight is not finite");
                    }
                    ++at;
                }while(at<entries.size()&&entries[at].q==q);
                if(value!=0){entries[write++]={q,value};total+=double(value);}
            }
            entries.resize(write);
        }
        if constexpr(std::is_same_v<Value,double>)if(std::isinf(total))throw std::runtime_error("Total quartet weight is not finite");
    }
    void filter(const Tree& tree,bool score_guide=false){
        double guide_sum=0,guide_correction=0;
        for(int c=0;c<3;++c){
            auto& entries=records[c];size_t write=0;
            for(auto entry:entries){
                if(tree.displayed(entry.q)!=topology(entry.q))entries[write++]=entry;
                else if constexpr(std::is_same_v<Value,double>){
                    if(score_guide){
                        // Neumaier summation retains small signed contributions
                        // while the existing guide-filter pass reads each row.
                        double next=guide_sum+entry.value;
                        guide_correction+=std::abs(guide_sum)>=std::abs(entry.value)?
                            (guide_sum-next)+entry.value:(entry.value-next)+guide_sum;
                        guide_sum=next;
                    }
                }
            }
            entries.resize(write);
            if(entries.size()*4<entries.capacity()*3)entries.shrink_to_fit();
            offsets[c+1]=offsets[c]+entries.size();
            for(auto entry:entries){
                if constexpr(std::is_same_v<Value,uint32_t>)count_total+=entry.value;
                else weight_total+=entry.value;
            }
        }
        guide_weight_total=guide_sum+guide_correction;
        if(!std::isfinite(guide_weight_total))throw std::runtime_error("Guide score is not finite");
        if(offsets[3]>UINT32_MAX)throw std::runtime_error("Compact quartet index exceeds uint32 capacity");
    }
    const Entry<Value>& at(uint32_t id)const{
        int c=id<offsets[1]?0:id<offsets[2]?1:2;return records[c][id-offsets[c]];
    }
    uint64_t size()const{return offsets[3];}
};

template<class Value>Evidence<Value> read_evidence(const std::string& path,const Tree& tree,bool grouped,bool signed_weights){
    CSV csv(path);Evidence<Value> evidence;
    std::array<int,4> taxa_columns;
    for(int i=0;i<4;++i)taxa_columns[i]=csv.column(grouped?"t"+std::to_string(i):std::string(1,char('a'+i)));
    std::array<int,3> values{};
    if(grouped)for(int i=0;i<3;++i)values[i]=csv.column("c"+std::to_string(i));
    else values[0]=csv.column(std::is_same_v<Value,uint32_t>?"count":"weight");
    std::array<int,3> keep{-1,-1,-1};
    if(grouped&&csv.columns.count("keep0"))for(int i=0;i<3;++i)keep[i]=csv.column("keep"+std::to_string(i));
    while(csv.next()){
        std::array<int,4> taxa;
        for(int i=0;i<4;++i){
            auto found=tree.tip_id.find(csv.fields[taxa_columns[i]]);
            if(found==tree.tip_id.end())throw std::runtime_error("Unknown quartet taxon");
            taxa[i]=found->second;
        }
        if(grouped){
            std::array<std::array<int,4>,3> groups{taxa,{taxa[0],taxa[2],taxa[1],taxa[3]},{taxa[0],taxa[3],taxa[1],taxa[2]}};
            for(int i=0;i<3;++i){
                auto value=parse_count(csv.fields[values[i]]);
                uint64_t q=canonical(groups[i]);
                if(keep[i]>=0){auto flag=csv.fields[keep[i]];if(flag!="0"&&flag!="1")throw std::runtime_error("Keep flags must be 0 or 1");if(flag=="0")continue;}
                evidence.add(q,Value(value));
            }
        }else{
            Value value;
            if constexpr(std::is_same_v<Value,uint32_t>)value=parse_count(csv.fields[values[0]]);
            else{value=parse_weight(csv.fields[values[0]]);if(!signed_weights&&value<0)throw std::runtime_error("Negative quartet weight requires signed mode");}
            evidence.add(canonical(taxa),value);
        }
    }
    evidence.coalesce();return evidence;
}

template<class Value>using Score = std::conditional_t<std::is_same_v<Value,uint32_t>,uint64_t,double>;

// Cutting the pendant branch of p in the unrooted three-tip guide (p,x,y)
// identifies every donor edge that displays the new bottom-p | x-y quartet.
// Even codes denote a subtree side; odd codes its complement plus boundary.
inline std::array<uint32_t,3> triplet_components(const Tree& tree,int a,int b,int c){
    std::array<int,3> taxa{a,b,c};
    int median=tree.lca(tree.tip_node[a],tree.tip_node[b]);
    for(int node:{tree.lca(tree.tip_node[a],tree.tip_node[c]),tree.lca(tree.tip_node[b],tree.tip_node[c])})
        if(tree.depth[node]>tree.depth[median])median=node;
    std::array<uint32_t,3> result;
    for(int i=0;i<3;++i){
        if(!tree.contains(median,taxa[i]))result[i]=2*median+1;
        else{
            const auto& children=tree.nodes[median].children;
            int child=tree.contains(children[0],taxa[i])?children[0]:children[1];
            result[i]=2*child;
        }
    }
    return result;
}

template<class Value>std::vector<Score<Value>> edge_scores(const Tree& tree,const Evidence<Value>& evidence,
                                                          int threads){
    std::vector<Score<Value>> scores(size_t(tree.m)*tree.m,0);omp_set_num_threads(threads);
    if(evidence.size()==0)return scores;
    std::vector<uint64_t> triples(tree.n),pairs(tree.n);
    for(int i=0;i<tree.n;++i){triples[i]=choose(i,3);pairs[i]=choose(i,2);}
    // The cache is O(n^3), bounded at 64 MiB. Sparse large-tip inputs can query
    // components directly instead of constructing a dense triplet table.
    uint64_t triple_count=choose(tree.n,3);
    std::vector<std::array<uint32_t,3>> components;
    if(triple_count<=64*1024*1024/sizeof(components[0])){
        components.resize(triple_count);
        for(int c=2;c<tree.n;++c)for(int b=1;b<c;++b)for(int a=0;a<b;++a)
            components[triples[c]+pairs[b]+a]=triplet_components(tree,a,b,c);
    }
    #pragma omp parallel for schedule(dynamic,1)
    for(int w=1;w<tree.m;++w){
        const uint8_t* in_w=&tree.below[size_t(w)*tree.n];
        if(tree.n-std::accumulate(in_w,in_w+tree.n,0)<3)continue;
        std::vector<uint8_t> allowed(tree.m),built(2*tree.m,0);
        for(int u=0;u<tree.m;++u)allowed[u]=tree.should_score(u,w);
        std::vector<std::vector<int>> donors(2*tree.m);
        // Keep the reference topology/taxon order. Each donor accumulates the
        // identical subsequence of doubles, without regrouping any additions.
        for(const auto& list:evidence.records)for(const auto& entry:list){
            auto q=entry.q;std::array<int,4> taxa{taxon(q,0),taxon(q,1),taxon(q,2),taxon(q,3)};
            int mask=int(in_w[taxa[0]])|(int(in_w[taxa[1]])<<1)|(int(in_w[taxa[2]])<<2)|(int(in_w[taxa[3]])<<3);
            if(!mask||(mask&(mask-1)))continue;
            int bottom=__builtin_ctz(unsigned(mask)),rank=0,partner=-1;
            std::array<int,3> other;
            for(int i=0;i<4;++i)if(i!=bottom){
                other[rank]=taxa[i];
                if(((topology(q)>>i)&1)==((topology(q)>>bottom)&1))partner=rank;
                ++rank;
            }
            const auto sides=components.empty()?triplet_components(tree,other[0],other[1],other[2]):
                components[triples[other[2]]+pairs[other[1]]+other[0]];
            uint32_t side=sides[partner];
            if(!built[side]){
                int branch=side/2;
                for(int u=1;u<tree.m;++u)if(allowed[u]){
                    bool inside=tree.lca(branch,u)==branch;
                    if((side&1)?(!inside||u==branch):inside)donors[side].push_back(u);
                }
                built[side]=1;
            }
            Score<Value> value=Score<Value>(entry.value);
            for(int u:donors[side])scores[size_t(u)*tree.m+w]+=value;
        }
    }
    return scores;
}

} // namespace camus
