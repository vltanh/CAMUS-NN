// Dynamic programming and traceback for candidate networks.
#pragma once
#include "engine.hpp"

namespace camus {

template<class S>struct State{
    S score=0;
    std::vector<Branch> branches;
};
template<class S>bool best_split(const std::vector<State<S>>& left,const std::vector<State<S>>& right,
                                 int k,int& l,int& r){
    if(left.empty()||right.empty())throw std::runtime_error("Empty DP lookup");
    if(int(left.size()+right.size())-2<k)return false;
    bool found=false;S best=0;
    for(int i=std::max(0,k-int(right.size())+1);i<=std::min(k,int(left.size())-1);++i){
        S score=left[i].score+right[k-i].score;
        if(score>best||!found){best=score;l=i;r=k-i;found=true;}
    }
    return found;
}
template<class S>bool four_split(const std::array<const std::vector<State<S>>*,4>& lists,int k,std::array<int,4>& indices){
    int length=0;for(auto list:lists){if(list->empty())throw std::runtime_error("Empty cycle lookup");length+=list->size();}
    if(length-4<k)return false;
    std::array<std::vector<std::array<int,2>>,2> solutions;
    std::array<std::vector<State<S>>,2> scores;
    for(int part=0;part<2;++part)for(int n=0;n<=k;++n){
        int left,right;
        if(!best_split(*lists[2*part],*lists[2*part+1],n,left,right))break;
        solutions[part].push_back({left,right});
        scores[part].push_back({(*lists[2*part])[left].score+(*lists[2*part+1])[right].score,{}});
    }
    int left,right;
    if(!best_split(scores[0],scores[1],k,left,right))throw std::runtime_error("Missing expected four-way split");
    indices={solutions[0][left][0],solutions[0][left][1],solutions[1][right][0],solutions[1][right][1]};
    return true;
}
inline void append(std::vector<Branch>& out,const std::vector<Branch>& from){out.insert(out.end(),from.begin(),from.end());}

template<class S>struct DynamicProgram{
    const Tree& tree;
    const std::vector<S>& edges;
    bool capped,stop_on_plateau,scalar,anchored;
    int max_edges;
    std::vector<std::vector<State<S>>> states;
    const std::vector<State<S>> zero{{0,{}}};
    DynamicProgram(const Tree& t,const std::vector<S>& scores,bool signed_weights,int cap,bool stop,bool unrestricted=false,bool anchor=false)
        :tree(t),edges(scores),capped(signed_weights),stop_on_plateau(stop),scalar(unrestricted),anchored(anchor),
         max_edges(cap<0?t.m:cap),states(t.m){}

    bool better(const State<S>& candidate,const State<S>& incumbent)const{
        return candidate.score>incumbent.score ||
            (scalar&&candidate.score==incumbent.score&&candidate.branches.size()<incumbent.branches.size());
    }

    void update_paths(int v,int k,std::vector<std::vector<State<S>>>& paths){
        tree.preorder(v,[&](int u){
            State<S> next;
            if constexpr(std::is_same_v<S,double>)if(capped&&k>0)next.score=-std::numeric_limits<double>::infinity();
            paths[u].push_back(std::move(next));
            if(u==v||tree.parent[u]==v)return;
            int p=tree.parent[u],s=tree.sibling(u),pk,sk;
            if(!best_split(paths[p],states[s],k,pk,sk))return;
            paths[u][k].score=paths[p][pk].score+states[s][sk].score;
            paths[u][k].branches=states[s][sk].branches;
            append(paths[u][k].branches,paths[p][pk].branches);
        });
    }

    bool edges_down(int v,int k,const std::vector<std::vector<State<S>>>& paths,State<S>& best,Branch& branch){
        bool found=false;
        tree.preorder(v,[&](int w){
            if(!tree.should_score(v,w))return;
            int child=w;while(tree.parent[child]!=v)child=tree.parent[child];
            int sibling=tree.sibling(child);std::array<int,4> split;
            if(!four_split<S>({&paths[w],&states[w],&states[sibling],&zero},k,split))return;
            int path_k=split[0],down_k=split[1],sibling_k=split[2];
            S score=edges[size_t(v)*tree.m+w]+paths[w][path_k].score+states[w][down_k].score+states[sibling][sibling_k].score;
            size_t count=1+states[w][down_k].branches.size()+states[sibling][sibling_k].branches.size()+paths[w][path_k].branches.size();
            if(score>best.score||!found||(scalar&&score==best.score&&count<best.branches.size())){
                found=true;best.score=score;branch={v,w};
                best.branches=states[w][down_k].branches;best.branches.push_back(branch);
                append(best.branches,states[sibling][sibling_k].branches);append(best.branches,paths[w][path_k].branches);
            }
        });
        return found;
    }

    bool edges_across(int u,int other,int k,const std::vector<std::vector<State<S>>>& paths,State<S>& best,Branch& branch){
        bool found=false;
        tree.preorder(other,[&](int w){
            std::array<int,4> split;
            if(!four_split<S>({&paths[w],&paths[u],&states[w],&states[u]},k,split))return;
            int wp=split[0],up=split[1],wd=split[2],ud=split[3];
            S score=edges[size_t(u)*tree.m+w]+paths[w][wp].score+paths[u][up].score+states[w][wd].score+states[u][ud].score;
            size_t count=1+states[w][wd].branches.size()+states[u][ud].branches.size()+paths[u][up].branches.size()+paths[w][wp].branches.size();
            if(score>best.score||!found||(scalar&&score==best.score&&count<best.branches.size())){
                found=true;best.score=score;branch={u,w};
                best.branches=states[w][wd].branches;best.branches.push_back(branch);
                append(best.branches,states[u][ud].branches);append(best.branches,paths[u][up].branches);append(best.branches,paths[w][wp].branches);
            }
        });
        return found;
    }

    bool add_edge(int v,int k,std::vector<std::vector<State<S>>>& paths,State<S>& best){
        if(anchored&&v==0)return false; // OUT's pendant edge stays outside every cycle.
        int previous=k-1,best_length=0;bool found=false;
        update_paths(v,previous,paths);
        auto consider=[&](State<S>& candidate,const Branch& branch){
            int length=tree.cycle_length(branch[0],branch[1]);
            if(better(candidate,best)||!found||(candidate.score==best.score&&
               (!scalar||candidate.branches.size()==best.branches.size())&&length<=best_length)){
                found=true;best=std::move(candidate);best_length=length;
            }
        };
        State<S> down;Branch down_branch{};
        if(edges_down(v,previous,paths,down,down_branch))consider(down,down_branch);
        tree.across_postorder(v,[&](int u,int other){
            State<S> across;Branch branch{};
            if(edges_across(u,other,previous,paths,across,branch))consider(across,branch);
        });
        return found;
    }

    void solve(int v){
        int left=tree.nodes[v].children[0],right=tree.nodes[v].children[1];
        State<S> initial;initial.score=states[left][0].score+states[right][0].score;
        initial.branches=states[left][0].branches;append(initial.branches,states[right][0].branches);
        auto& scores=states[v];scores.push_back(std::move(initial));
        std::vector<std::vector<State<S>>> paths(tree.m);
        if(scalar){
            // Every child/path state now denotes its unrestricted optimum.
            // k=1 requests a cycle with zero cardinality indices; the same
            // geometric decomposition combines the optimal hanging subtrees.
            State<S> edge;
            if(add_edge(v,1,paths,edge)&&better(edge,scores[0]))scores[0]=std::move(edge);
            return;
        }
        for(int k=1;;++k){
            if(capped&&k>max_edges)break;
            State<S> best;bool found=false;int lk,rk;
            if(best_split(states[left],states[right],k,lk,rk)){
                found=true;best.score=states[left][lk].score+states[right][rk].score;
                best.branches=states[left][lk].branches;append(best.branches,states[right][rk].branches);
            }
            State<S> edge;
            if(add_edge(v,k,paths,edge)&&!(std::isinf(double(edge.score))&&double(edge.score)<0)&&
               (edge.score>best.score||(capped&&!found))){best=std::move(edge);found=true;}
            if(!found||((!capped||stop_on_plateau)&&scores[k-1].score>=best.score))break;
            scores.push_back(std::move(best));
            if(k==tree.m*tree.m)throw std::runtime_error("Runaway DP loop");
        }
    }
    const std::vector<State<S>>& run(){
        for(int v:tree.postorder){
            if(tree.nodes[v].children.empty())states[v].push_back({0,{}});
            else solve(v);
        }
        return states[0];
    }
};

} // namespace camus
