// Exact topology counts. Tree-edge distances are an internal topology query,
// not branch-length features. Output records three topology counts per group.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

using Quartet = std::array<int, 4>;
using Counts = std::array<uint32_t, 3>;

struct Node {
    std::string name;
    std::vector<int> children;
    double support = -1;
};

struct Parser {
    const std::string& text;
    size_t pos = 0;
    std::vector<Node> nodes;
    explicit Parser(const std::string& value) : text(value) {}

    void skip() {
        while (pos < text.size()) {
            if (std::isspace(static_cast<unsigned char>(text[pos]))) { ++pos; continue; }
            if (text[pos] != '[') break;
            int depth = 1; ++pos;
            while (pos < text.size() && depth) {
                if (text[pos] == '[') ++depth;
                if (text[pos] == ']') --depth;
                ++pos;
            }
            if (depth) throw std::runtime_error("Unterminated Newick comment");
        }
    }

    std::string label() {
        skip(); std::string value;
        if (pos < text.size() && text[pos] == '\'') {
            ++pos;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '\'') {
                    if (pos < text.size() && text[pos] == '\'') ++pos;
                    else return value;
                }
                value += c;
            }
            throw std::runtime_error("Unterminated quoted Newick label");
        }
        while (pos < text.size() && std::string(",():;[]").find(text[pos]) == std::string::npos &&
               !std::isspace(static_cast<unsigned char>(text[pos]))) value += text[pos++];
        return value;
    }

    int node() {
        skip(); int index = static_cast<int>(nodes.size()); nodes.emplace_back();
        if (pos < text.size() && text[pos] == '(') {
            ++pos; int child = node(); nodes[index].children.push_back(child); skip();
            while (pos < text.size() && text[pos] == ',') {
                ++pos; int child = node(); nodes[index].children.push_back(child); skip();
            }
            if (pos >= text.size() || text[pos++] != ')') throw std::runtime_error("Expected ')' in Newick");
        }
        std::string name = label();
        if (name.find('#') != std::string::npos) throw std::runtime_error("Expected gene trees, not networks");
        if (nodes[index].children.empty()) {
            if (name.empty()) throw std::runtime_error("Empty taxon name");
            nodes[index].name = name;
        } else if (!name.empty()) {
            try {
                size_t end = 0; double support = std::stod(name, &end);
                if (end == name.size() || name[end] == '/') nodes[index].support = support;
            } catch (const std::exception&) { /* Internal node names are not supports. */ }
        }
        skip();
        while (pos < text.size() && text[pos] == ':') {
            ++pos; size_t start = pos;
            while (pos < text.size() && std::string(",):;[").find(text[pos]) == std::string::npos &&
                   !std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos == start) throw std::runtime_error("Missing branch length");
            size_t end = 0; double length = std::stod(text.substr(start, pos-start), &end);
            if (end != pos-start || !std::isfinite(length)) throw std::runtime_error("Invalid branch length");
            skip();
        }
        return index;
    }

    std::vector<Node> parse() {
        node(); skip();
        if (pos >= text.size() || text[pos++] != ';') throw std::runtime_error("Missing terminal ';'");
        skip();
        if (pos != text.size()) throw std::runtime_error("Expected one Newick tree per line");
        std::set<std::string> names;
        for (const auto& n : nodes) if (n.children.empty() && !names.insert(n.name).second)
            throw std::runtime_error("Duplicate taxon in gene tree: " + n.name);
        return std::move(nodes);
    }
};

uint64_t choose(int n, int k) {
    if (n < k) return 0;
    __uint128_t value = 1;
    for (int i = 1; i <= k; ++i) value = value * (n-i+1) / i;
    if (value > std::numeric_limits<uint64_t>::max()) throw std::runtime_error("Too many quartet groups");
    return static_cast<uint64_t>(value);
}

uint64_t rank_group(const Quartet& q) {
    return choose(q[3], 4) + choose(q[2], 3) + choose(q[1], 2) + q[0];
}

std::vector<std::string> csv_fields(const std::string& line) {
    std::vector<std::string> fields; std::string value; bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (quoted && i+1 < line.size() && line[i+1] == '"') { value += '"'; ++i; }
            else quoted = !quoted;
        } else if (c == ',' && !quoted) { fields.push_back(value); value.clear(); }
        else if (c != '\r' || quoted) value += c;
    }
    if (quoted) throw std::runtime_error("Unterminated CSV quote");
    fields.push_back(value); return fields;
}

std::string csv_quote(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) return text;
    std::string result = "\"";
    for (char c : text) { result += c; if (c == '"') result += c; }
    return result + '"';
}

// Distances count edges, with zero-cost internal edges where support collapsing
// contracts a branch. Missing taxa have distance -1.
std::vector<int> distances(const std::vector<Node>& nodes, const std::map<std::string,int>& ids,
                           int taxa, double support) {
    std::vector<std::vector<std::pair<int,int>>> edges(nodes.size());
    for (size_t u = 0; u < nodes.size(); ++u) for (int v : nodes[u].children) {
        int weight = !nodes[v].children.empty() && nodes[v].support >= 0 && nodes[v].support < support ? 0 : 1;
        edges[u].push_back({v,weight}); edges[v].push_back({static_cast<int>(u),weight});
    }
    std::vector<int> result(static_cast<size_t>(taxa)*taxa, -1);
    for (size_t source = 0; source < nodes.size(); ++source) {
        if (!nodes[source].children.empty()) continue;
        int tip = ids.at(nodes[source].name);
        std::vector<int> d(nodes.size(), -1), queue{static_cast<int>(source)}; d[source] = 0;
        for (size_t head = 0; head < queue.size(); ++head) {
            int u = queue[head];
            if (nodes[u].children.empty()) result[static_cast<size_t>(tip)*taxa + ids.at(nodes[u].name)] = d[u];
            for (auto [v,w] : edges[u]) if (d[v] < 0) { d[v] = d[u]+w; queue.push_back(v); }
        }
    }
    return result;
}

inline void add(Counts& count, const int* dist, int n, int a, int b, int c, int d) {
    int ab = dist[a*n+b], ac = dist[a*n+c], ad = dist[a*n+d];
    int bc = dist[b*n+c], bd = dist[b*n+d], cd = dist[c*n+d];
    if (ab < 0 || ac < 0 || ad < 0 || bc < 0 || bd < 0 || cd < 0) return;
    int x = ab+cd, y = ac+bd, z = ad+bc;
    if (x < y && x < z) ++count[0];
    else if (y < x && y < z) ++count[1];
    else if (z < x && z < y) ++count[2];
}

#ifndef CAMUS_QCOUNT_LIBRARY
int main(int argc, char** argv) try {
    std::string genes_path, groups_path, output = "quartets.csv"; int threads = 1; double support = 0;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (!arg.empty() && arg[0] != '-') { positional.push_back(arg); continue; }
        if (arg == "--help" || arg == "-h") {
            std::cout << "camus-count -gene-trees genes.nwk [-quartet-sets groups.csv] [-n threads] [-s support] -o counts.csv\n";
            return 0;
        }
        if (i+1 >= argc) throw std::runtime_error("Missing flag value: " + arg);
        std::string value = argv[++i];
        if (arg == "-gene-trees") genes_path = value;
        else if (arg == "-quartet-sets") groups_path = value;
        else if (arg == "-o") output = value;
        else if (arg == "-n") threads = std::stoi(value);
        else if (arg == "-s") support = std::stod(value);
        else throw std::runtime_error("Unknown flag: " + arg);
    }
    if (genes_path.empty() && positional.size() == 2) genes_path = positional[1];
    else if (!positional.empty()) throw std::runtime_error("Use -gene-trees, optionally -quartet-sets");
    if (genes_path.empty() || threads < 1 || !std::isfinite(support) || support < 0 || support > 1)
        throw std::runtime_error("Require -gene-trees, -n >= 1, and -s in [0,1]");
    std::ifstream input(genes_path); if (!input) throw std::runtime_error("Cannot open gene trees");
    std::vector<std::vector<Node>> genes; std::set<std::string> inventory; std::string line;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        auto nodes = Parser{line}.parse();
        for (const auto& node : nodes) if (node.children.empty()) inventory.insert(node.name);
        genes.push_back(std::move(nodes));
    }
    if (!input.eof()) throw std::runtime_error("Error reading gene trees");
    if (genes.empty() || inventory.size() < 4 || genes.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Require gene trees with at least four taxa and <= uint32 loci");
    if (inventory.size() >= 32768) throw std::runtime_error("CAMUS supports fewer than 32768 taxa");
    std::vector<std::string> names(inventory.begin(), inventory.end()); int n = static_cast<int>(names.size());
    std::map<std::string,int> ids; for (int i = 0; i < n; ++i) ids[names[i]] = i;
    std::vector<Quartet> groups;
    if (!groups_path.empty()) {
        std::ifstream file(groups_path); if (!file || !std::getline(file,line)) throw std::runtime_error("Cannot read groups CSV");
        auto header = csv_fields(line); std::array<size_t,4> fields;
        for (int i = 0; i < 4; ++i) {
            auto found = std::find(header.begin(),header.end(),"t"+std::to_string(i));
            if (found == header.end()) throw std::runtime_error("Groups CSV needs t0,t1,t2,t3");
            fields[i] = found-header.begin();
        }
        std::set<uint64_t> seen;
        while (std::getline(file,line)) {
            if (line.empty() || line == "\r") continue;
            auto row = csv_fields(line); if (row.size() != header.size()) throw std::runtime_error("Malformed groups CSV");
            Quartet q;
            for (int i = 0; i < 4; ++i) {
                auto found = ids.find(row[fields[i]]);
                if (found == ids.end()) throw std::runtime_error("Unknown group taxon: " + row[fields[i]]);
                q[i] = found->second;
            }
            std::sort(q.begin(),q.end());
            if (std::adjacent_find(q.begin(),q.end()) != q.end()) throw std::runtime_error("Group taxa must be distinct");
            if (!seen.insert(rank_group(q)).second) throw std::runtime_error("Duplicate quartet group");
            groups.push_back(q);
        }
        if (groups.empty()) throw std::runtime_error("Empty quartet selection");
        std::sort(groups.begin(),groups.end(),[](const auto& a,const auto& b){return rank_group(a)<rank_group(b);});
    }
    uint64_t size = groups_path.empty() ? choose(n,4) : groups.size();
    std::vector<Counts> counts(size, Counts{0,0,0}); omp_set_num_threads(threads);
    std::vector<uint64_t> starts(n); for (int d = 3; d < n; ++d) starts[d] = choose(d,4);
    for (const auto& gene : genes) {
        auto dist = distances(gene,ids,n,support); const int* matrix = dist.data();
        if (groups_path.empty()) {
            #pragma omp parallel for schedule(dynamic,1)
            for (int d = 3; d < n; ++d) {
                uint64_t index = starts[d];
                for (int c = 2; c < d; ++c) for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a)
                    add(counts[index++],matrix,n,a,b,c,d);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < groups.size(); ++i) {
                const auto& q = groups[i]; add(counts[i],matrix,n,q[0],q[1],q[2],q[3]);
            }
        }
    }
    std::ofstream out(output); if (!out) throw std::runtime_error("Cannot open output");
    out << "t0,t1,t2,t3,c0,c1,c2,n_loci,constraint\n";
    std::vector<std::string> quoted; for (const auto& name : names) quoted.push_back(csv_quote(name));
    auto emit = [&](const Quartet& q,uint64_t index) {
        const auto& c = counts[index];
        if (groups_path.empty() && c[0] == 0 && c[1] == 0 && c[2] == 0) return;
        out << quoted[q[0]] << ',' << quoted[q[1]] << ',' << quoted[q[2]] << ',' << quoted[q[3]] << ','
            << c[0] << ',' << c[1] << ',' << c[2] << ',' << genes.size() << ",-1\n";
    };
    if (groups_path.empty()) {
        uint64_t index = 0;
        for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c) for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a)
            emit(Quartet{a,b,c,d},index++);
    } else for (size_t i = 0; i < groups.size(); ++i) emit(groups[i],i);
    out.close(); if (!out) throw std::runtime_error("Error writing counts");
    std::cerr << genes.size() << " gene trees, " << names.size() << " taxa, " << size << " quartet groups\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
#endif
