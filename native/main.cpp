// C++ counting, compact evidence, edge scores, and CAMUS DP.
// Counting, inference and Extended Newick serialization are self-contained.
#include "dp.hpp"
#include "serialize.hpp"
#include <sys/resource.h>

namespace camus {

struct Options{
    std::string tree,input,prefix="camus-cpp",dump;
    int threads=1,cap=-1;
    bool grouped=false,weighted=false,signed_weights=false,stop=true,scalar=false,anchored=false;
};

std::string quote_json(const std::string& text){
    std::string out="\"";
    for(unsigned char c:text){
        if(c=='"'||c=='\\'){out+='\\';out+=char(c);}
        else if(c=='\n')out+="\\n";
        else if(c=='\r')out+="\\r";
        else if(c=='\t')out+="\\t";
        else if(c<32){const char* hex="0123456789abcdef";out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=char(c);
    }
    return out+'"';
}
template<class Number>std::string number(Number value){
    if constexpr(std::is_floating_point_v<Number>){
        if(std::isnan(value))return "NaN";
        if(std::isinf(value))return value<0?"-Inf":"+Inf";
    }
    char buffer[128];auto result=std::to_chars(buffer,buffer+sizeof(buffer),value);
    if(result.ec!=std::errc())throw std::runtime_error("Cannot format score");
    return std::string(buffer,result.ptr);
}

struct Log{
    std::ofstream file;
    std::map<std::string,double> phases;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now(),previous=start;
    explicit Log(const std::string& prefix):file(prefix+".log"){
        if(!file)throw std::runtime_error("Cannot create output log");
    }
    void message(const std::string& text){
        auto now=std::chrono::system_clock::now();auto seconds=std::chrono::system_clock::to_time_t(now);
        std::tm local{};localtime_r(&seconds,&local);
        std::ostringstream out;out<<std::put_time(&local,"%Y/%m/%d %H:%M:%S")<<" "<<text<<'\n';
        std::cerr<<out.str();file<<out.str();file.flush();
    }
    void finish(const std::string& phase){
        auto now=std::chrono::steady_clock::now();phases[phase]=std::chrono::duration<double>(now-previous).count();previous=now;
    }
};

Options options(int argc,char** argv){
    Options o;std::vector<std::string> positional;
    for(int i=1;i<argc;++i){
        std::string key=argv[i],value;auto equal=key.find('=');
        if(equal!=std::string::npos){value=key.substr(equal+1);key.resize(equal);}
        auto boolean=[&](){if(equal==std::string::npos)return true;if(value=="true")return true;if(value=="false")return false;throw std::runtime_error("Invalid boolean flag");};
        if(key=="-h"||key=="--help"){
            std::cout<<"camus-native [-n threads] [-q 0] [-o prefix]\n"
                       "          [-grouped-counts | -quartet-weights -signed-weights]\n"
                       "          [-signed-weights] [-signed-max-reticulations -1|cap]\n"
                       "          [-signed-stop-on-plateau=false]\n"
                       "          [--signed-unrestricted] (exact scalar DP; no count curve)\n"
                       "          [--anchor-outgroup] (root must have leaf OUT as a child)\n"
                       "          constraint.nwk evidence.csv\n";std::exit(0);
        }
        if(key=="-grouped-counts"||key=="--grouped-counts"){o.grouped=boolean();continue;}
        if(key=="-quartet-weights"){o.weighted=boolean();continue;}
        if(key=="-signed-weights"){o.signed_weights=boolean();continue;}
        if(key=="-signed-stop-on-plateau"){o.stop=boolean();continue;}
        if(key=="--signed-unrestricted"){o.scalar=boolean();continue;}
        if(key=="--anchor-outgroup"){o.anchored=boolean();continue;}
        if(!key.empty()&&key[0]=='-'){
            if(equal==std::string::npos){if(++i>=argc)throw std::runtime_error("Missing value for "+key);value=argv[i];}
            if(key=="-n")o.threads=std::stoi(value);
            else if(key=="-q"){if(value!="0")throw std::runtime_error("Learned evidence requires -q 0");}
            else if(key=="-o")o.prefix=value;
            else if(key=="-signed-max-reticulations")o.cap=std::stoi(value);
            else if(key=="--dump-edge-scores")o.dump=value;
            else throw std::runtime_error("Unknown flag "+key);
        }else positional.push_back(key);
    }
    if(positional.size()!=2)throw std::runtime_error("Require a rooted constraint tree and evidence file");
    o.tree=std::filesystem::absolute(positional[0]);o.input=std::filesystem::absolute(positional[1]);
    if(o.threads<1||o.cap< -1)throw std::runtime_error("Invalid inference options");
    if(o.grouped==o.weighted || o.weighted!=o.signed_weights)
        throw std::runtime_error("Choose learned grouped counts or signed quartet weights");
    if(o.scalar&&(!o.signed_weights||o.cap!=-1))throw std::runtime_error("Unrestricted DP requires signed weights and no reticulation cap");
    if(o.scalar)o.anchored=true;
    return o;
}

void format(const Options& o,const Tree& tree,const std::vector<std::pair<double,std::vector<Branch>>>& candidates){
    std::ofstream output(o.prefix+".csv");
    output<<"Number of Branches,"<<(o.signed_weights?"Signed Quartet Gain":"Quartet Satisfied Percent")<<",Extended Newick\n";
    output<<"0,0,"<<csv_quote(network_newick(tree,{}))<<'\n';
    for(const auto& [score,branches]:candidates){
        char buffer[1024];auto result=std::to_chars(buffer,buffer+sizeof(buffer),score,std::chars_format::fixed);
        if(result.ec!=std::errc())throw std::runtime_error("Cannot serialize finite score");
        output<<branches.size()<<','<<std::string(buffer,result.ptr)<<','<<csv_quote(network_newick(tree,branches))<<'\n';
    }
    output.close();if(!output)throw std::runtime_error("Cannot write candidate CSV");
}

template<class Value>void run(const Options& o,Log& log){
    Tree tree(o.tree);log.finish("constraint_preparation");
    if(o.anchored){
        bool out=false;for(int c:tree.nodes[0].children)
            if(tree.nodes[c].children.empty()&&tree.nodes[c].name=="OUT")out=true;
        if(!out)throw std::runtime_error("Anchored inference requires leaf OUT as a root child");
    }
    log.message("reading learned quartet evidence");
    Evidence<Value> evidence=read_evidence<Value>(o.input,tree,o.grouped,o.signed_weights);
    log.finish("counting_or_evidence_reading");
    bool total_objective=o.signed_weights&&o.anchored;
    evidence.filter(tree,total_objective);
    log.message(std::to_string(evidence.size())+" quartet topologies not in the constraint tree");
    log.finish("filtering");
    log.message(o.weighted?"calculating weighted edge scores":"calculating edge scores");
    auto scores=edge_scores(tree,evidence,o.threads);log.finish("edge_scoring");
    if(!o.dump.empty()){
        std::ofstream file(o.dump);file<<"u,w,score\n";
        for(int u=0;u<tree.m;++u)for(int w=0;w<tree.m;++w)file<<u<<','<<w<<','<<number(scores[size_t(u)*tree.m+w])<<'\n';
        if(!file)throw std::runtime_error("Cannot write edge-score dump");
    }
    log.message("preprocessing finished, beginning dp algorithm");
    DynamicProgram<Score<Value>> dp(tree,scores,o.signed_weights,o.cap,o.stop,o.scalar,o.anchored);
    const auto& states=dp.run();log.finish("dp_and_traceback");
    std::string path=o.prefix+".branches.json";
    std::ofstream output(path);if(!output)throw std::runtime_error("Cannot write branch results");
    output<<"{\"Tree\":"<<quote_json(o.tree)<<",\"Signed\":"<<(o.signed_weights?"true":"false")
          <<",\"Unrestricted\":"<<(o.scalar?"true":"false")<<",\"Candidates\":[";
    bool emitted=false;
    std::vector<std::pair<double,std::vector<Branch>>> candidates;
    double best_gain=0;
    size_t best_k=0;
    for(size_t k=o.scalar?0:1;k<states.size();++k){
        if(o.scalar&&states[k].branches.empty())continue;
        if(emitted)output<<',';
        emitted=true;
        Score<Value> gain=0;
        for(auto branch:states[k].branches)gain+=scores[size_t(branch[0])*tree.m+branch[1]];
        if(double(gain)>best_gain){best_gain=double(gain);best_k=states[k].branches.size();}
        double statistic;
        if constexpr(std::is_same_v<Value,uint32_t>){
            double denominator=double(evidence.count_total);
            statistic=(100*double(gain))/denominator;
        }else statistic=gain;
        candidates.emplace_back(statistic,states[k].branches);
        output<<"{\"Score\":"<<quote_json(number(statistic))<<",\"Branches\":[";
        for(size_t i=0;i<states[k].branches.size();++i){auto branch=states[k].branches[i];if(i)output<<',';output<<'['<<branch[0]<<','<<branch[1]<<']';}
        output<<"]}";
    }
    output<<"]}\n";output.close();if(!output)throw std::runtime_error("Cannot finish branch results");
    uint64_t topology_count=evidence.size();
    double guide_score=evidence.guide_weight_total;
    double total_score=guide_score+best_gain;
    if(total_objective&&!std::isfinite(total_score))throw std::runtime_error("Network score is not finite");
    evidence={};
    log.message(std::to_string(o.scalar?states[0].branches.size():states.size()-1)+" edges identified");
    format(o,tree,candidates);log.finish("network_csv_serialization");
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::ofstream metrics(o.prefix+".metrics.json");
    metrics<<"{\"schema\":\"camus-cpp/v1\"";
    if(total_objective)metrics<<",\"objective\":{\"guide_score\":"<<number(guide_score)
        <<",\"added_gain\":"<<number(best_gain)<<",\"total_score\":"<<number(total_score)
        <<",\"selected_k\":"<<best_k<<"}";
    metrics<<",\"tips\":"<<tree.n<<",\"retained_topologies\":"<<topology_count
           <<",\"dp_mode\":"<<quote_json(o.scalar?"signed-unrestricted-scalar":
                total_objective&&o.cap>=0&&!o.stop?"signed-capped-cardinality":"cardinality")
           <<",\"search_cap\":"<<o.cap
           <<",\"outgroup_anchored\":"<<(o.anchored?"true":"false")
           <<",\"peak_self_rss_bytes\":"<<uint64_t(usage.ru_maxrss)*1024<<",\"threads\":"<<o.threads
           <<",\"scientific_core\":\"C++\",\"edge_scoring_backend\":\"triplet-components-v1\",\"network_csv_serializer\":\"C++\",\"phases\":{";
    bool first=true;for(auto& [name,seconds]:log.phases){if(!first)metrics<<',';first=false;metrics<<quote_json(name)<<':'<<number(seconds);}
    metrics<<"}}\n";if(!metrics)throw std::runtime_error("Cannot write inference metrics");
    log.message("done");
}

} // namespace camus

int main(int argc,char** argv)try{
    auto options=camus::options(argc,argv);
    auto parent=std::filesystem::path(options.prefix).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
    camus::Log log(options.prefix);log.message("camus-cpp: exact max-score inference");
    if(options.weighted)camus::run<double>(options,log);else camus::run<uint32_t>(options,log);
    return 0;
}catch(const std::exception& error){std::cerr<<"camus-cpp: "<<error.what()<<'\n';return 1;}
