/*
 * GroupClosenessLocalSwaps.hpp
 *
 *  Created on: 19.12.2019
 *      Author: Eugenio Angriman <angrimae@hu-berlin.de>
 */

// networkit-format

#ifndef NETWORKIT_CENTRALITY_GROUP_CLOSENESS_LOCAL_SWAPS_HPP_
#define NETWORKIT_CENTRALITY_GROUP_CLOSENESS_LOCAL_SWAPS_HPP_

#ifdef NETWORKIT_WITH_AVX
#include <immintrin.h>
#endif // NETWORKIT_WITH_AVX

#include <limits>
#include <unordered_map>
#include <vector>

#include <networkit/base/Algorithm.hpp>
#include <networkit/graph/Graph.hpp>

namespace NetworKit {
class GroupClosenessLocalSwaps final : public Algorithm {

public:
    /**
     * Finds a group of nodes with high group closeness centrality. This is the LS-restrict
     * algorithm presented in Angriman et al. "Local Search for Group Closeness Maximization on Big
     * Graphs" IEEE BigData 2019. The algorithm takes as input a graph and an arbitrary group of
     * nodes, and improves the group closeness of the given group by performing vertex exchanges.
     *
     * @param G A connected, undirected, and unweighted graph.
     * @param first, last A range that contains the initial group of nodes.
     */
    template <class Iter>
    GroupClosenessLocalSwaps(const Graph &graph, Iter first, Iter last)
        : G(&graph), group(first, last) {
        if (G->isDirected()) {
            std::runtime_error("Error, this algorithm does not support directed graphs.");
        }
        if (group.empty()) {
            throw std::runtime_error("Error, empty group.");
        }
        if (G->isWeighted()) {
            WARN("This algorithm does not support edge Weights, they will be ignored.");
        }
    }

    // Easier constructor to wrap in Cython.
    GroupClosenessLocalSwaps(const Graph &graph, const std::vector<node> &group)
        : GroupClosenessLocalSwaps(graph, group.begin(), group.end()) {}

    /**
     * Runs the algorithm.
     */
    void run() override;

    /**
     * Returns the computed group.
     */
    std::vector<node> groupMaxCloseness() const {
        assureFinished();
        std::vector<node> maxGroup;
        maxGroup.reserve(group.size());

        for (const auto &entry : idxMap) {
            maxGroup.push_back(entry.first);
        }

        return maxGroup;
    }

    /**
     * Returns the total number of vertex exchanges performed by the algorithm.
     */
    count numberOfSwaps() const {
        assureFinished();
        return totalSwaps;
    }

    // Maximum number of vertex exchanges allowed.
    count maxSwaps{100};

    // Random seed used for the estimation of the transitive closure of the DAG.
    uint64_t randomSeed{1};

private:
    const Graph *G;
    std::vector<node> group, stack;
    std::vector<uint32_t> distance, sumOfMins;
    std::vector<unsigned char> gamma, canSwap;
    std::unordered_map<node, index> idxMap;
    std::vector<int64_t> value, valueDecrement;

    count totalSwaps, stackSize;

    std::vector<uint8_t> visited;
    uint8_t timestamp;

    static constexpr float maxInt16 = static_cast<float>(std::numeric_limits<uint16_t>::max());

    void init();
    void bfsFromGroup();
    bool findAndSwap();
    node estimateHighestDecrement();
    void initRandomVector();
    int64_t computeFarnessDecrement(node u);

    // 16 unsigned integers
#ifdef NETWORKIT_WITH_AVX
    union RandItem {
        uint16_t items[16];
        __m256i vec;
    };
    std::vector<RandItem> randVec;
#else
    std::vector<uint16_t> randVec;
#endif // NETWORKIT_WITH_AVX

    void resetGamma(node x, index idx) {
        std::fill(gamma.begin() + group.size() * x, gamma.begin() + group.size() * (x + 1), 0);
        gamma[group.size() * x + idx] = 1;
    }

    // Increment timestamp and, if necessary, reset bfsdist vector
    void incrementTimestamp() {
        if (timestamp++ == 255) {
            timestamp = uint8_t{1};
            std::fill(visited.begin(), visited.end(), static_cast<unsigned char>(0));
        }
    }

    // Returns true if u was the only nearest node to x
    bool gammaIsU(node x, index idx) const {
        if (!gamma[group.size() * x + idx]) {
            return false;
        }

        for (count i = 0; i < group.size(); ++i) {
            if (i != idx && gamma[group.size() * x + i]) {
                return false;
            }
        }
        return true;
    }

    typedef struct {
        uint64_t state;
        uint64_t inc;
    } pcg32_random_t;

    std::vector<pcg32_random_t> rng;
    uint32_t pcg32_random_r(pcg32_random_t *rng) {
        uint64_t oldstate = rng->state;
        // Advance internal state
        rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
};

} // namespace NetworKit

#endif // NETWORKIT_CENTRALITY_GROUP_CLOSENESS_LOCAL_SWAPS_HPP_
