// Extended Newick serialization of CAMUS branch choices.
// See LICENSE in this directory.
#pragma once
#include "engine.hpp"
#include "branch_order.hpp"

namespace camus {

inline std::string newick_label(const std::string& name) {
    if (name.find_first_of(" \t\r\n,():;[]'") == std::string::npos) return name;
    std::string out = "'";
    for (char c : name) { out += c; if (c == '\'') out += c; }
    return out + "'";
}

inline std::string network_newick(const Tree& tree, std::vector<Branch> branches) {
    auto nodes = tree.nodes;
    auto parents = tree.parent;
    // Match the reference's equal-depth ordering, including lists above 12.
    reference_sort(branches, [&](Branch a, Branch b) {
        return tree.depth[tree.lca(a[0], a[1])] < tree.depth[tree.lca(b[0], b[1])];
    });
    auto graft = [&](int child, const std::string& tag, bool donor) {
        int parent = parents.at(child);
        if (parent < 0) throw std::runtime_error("Cannot graft on the root");
        int inserted = static_cast<int>(nodes.size());
        nodes.emplace_back(); parents.push_back(parent);
        auto& children = nodes[parent].children;
        auto at = std::find(children.begin(), children.end(), child);
        if (at == children.end()) throw std::runtime_error("Missing graft edge");
        *at = inserted;
        parents[child] = inserted;
        if (donor) {
            int reference = static_cast<int>(nodes.size());
            nodes.emplace_back(); parents.push_back(inserted);
            nodes[reference].name = tag;
            nodes[inserted].children = {reference, child};
        } else {
            nodes[inserted].name = tag;
            nodes[inserted].children = {child};
        }
    };
    for (size_t i = 0; i < branches.size(); ++i) {
        auto tag = "#H" + std::to_string(i + 1);
        graft(branches[i][0], tag, true);
        graft(branches[i][1], tag, false);
    }
    std::function<std::string(int)> emit = [&](int u) {
        std::string out;
        if (!nodes[u].children.empty()) {
            out = "(";
            for (size_t i = 0; i < nodes[u].children.size(); ++i) {
                if (i) out += ',';
                out += emit(nodes[u].children[i]);
            }
            out += ')';
        }
        // Hybrid tags belong to the serializer; ordinary names are quoted.
        return out + (u >= tree.m ? nodes[u].name : newick_label(nodes[u].name));
    };
    return emit(0) + ';';
}

} // namespace camus
