/*
 * GroupClosenessGrowShrink.hpp
 *
 *  Created on: 19.12.2019
 *      Author: Eugenio Angriman <angrimae@hu-berlin.de>
 */

// networkit-format

#ifndef NETWORKIT_CENTRALITY_GROUP_CLOSENESS_GROW_SHRINK_HPP_
#define NETWORKIT_CENTRALITY_GROUP_CLOSENESS_GROW_SHRINK_HPP_

#ifdef NETWORKIT_WITH_AVX
#include <immintrin.h>
#endif // NETWORKIT_WITH_AVX

#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#include <networkit/base/Algorithm.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/graph/Graph.hpp>

#include <tlx/container/d_ary_addressable_int_heap.hpp>

namespace NetworKit {

template <typename Weight>
class GroupClosenessGrowShrinkGeneral final : public Algorithm {

public:
    /**
     * Finds a group of nodes with high group closeness centrality. This is the Grow-Shrink
     * algorithm presented in Angriman et al. "Local Search for Group Closeness Maximization on Big
     * Graphs" IEEE BigData 2019. The algorithm takes as input a graph and an arbitrary group of
     * nodes, and improves the group closeness of the given group by performing vertex exchanges.
     *
     * @param G A connected undirected graph.
     * @param first, last A range that contains the initial group of nodes.
     * @param extended Set this parameter to true for the Extended Grow-Shrink algorithm (i.e.,
     * vertex exchanges are not restricted to only neighbors of the group).
     * @param insertions Number of consecutive node insertions and removal per iteration. Let this
     * parameter to zero to use Diameter(G)/sqrt(k) nodes (where k is the size of the group).
     */
    template <class Iter>
    GroupClosenessGrowShrinkGeneral(const Graph &graph, Iter first, Iter last,
                                    bool extended = false, count insertions = 0)
        : G(&graph), group(first, last), extended(extended), insertions(insertions),
          heap(CompareDistance(distance)), heap_(CompareDistance(distance_)) {

        if (G->isDirected()) {
            std::runtime_error("Error, this algorithm does not support directed graphs.");
        }

        if (group.empty()) {
            throw std::runtime_error("Error, empty group.");
        }
    }

    // Easier constructor to wrap in Cython.
    GroupClosenessGrowShrinkGeneral(const Graph &graph, const std::vector<node> &group,
                                    bool extended = false, count insertions = 0)
        : GroupClosenessGrowShrinkGeneral(graph, group.begin(), group.end(), extended, insertions) {
    }

    /**
     * Returns the computed group.
     */
    std::vector<node> groupMaxCloseness() const {
        assureFinished();
        std::vector<node> result;
        result.reserve(group.size());
        for (const auto &entry : idxMap) {
            result.push_back(entry.first);
        }

        return result;
    }

    /**
     * Returns the total number of iterations performed by the algorithm.
     */
    count numberOfIterations() const {
        assureFinished();
        return totalSwaps;
    }

    // Maximum number of iterations allowed.
    count maxIterations{100};

    // Random seed used for the estimation of the transitive closure of the DAG.
    uint64_t randomSeed{1};

    /**
     * Runs the algorithm.
     */
    void run() override {
        init();

        if (!insertions) {
            Diameter diam(*G, DiameterAlgo::estimatedRange, 0.1);
            diam.run();
            DEBUG("Diameter upper bound = ", diam.getDiameter().second);
            insertions = std::max(1., .5
                                          + static_cast<double>(diam.getDiameter().second)
                                                / std::sqrt(static_cast<double>(group.size())));
        }

        DEBUG("Using ", insertions, " consecutive insertions");

        increment.assign(group.size() + insertions, 0);
        for (count i = 0; i < insertions; ++i) {
            nextIdx.push(group.size() + i);
        }

        // Compute 1st shortest distanceances
        bfsFromGroup();

        // Compute 2nd shortest distanceances
        // Step 1: find 'bundary' nodes (sources of Dijkstra)
        incrementTimestamp();
        auto &pq = (G->isWeighted() ? heap_ : heap);
        auto process = [&](const node x, const node y, const Weight w) {
            if (timestamp != visited[y] || distance_[y] > distance[x] + w) {
                distance_[y] = distance[x] + w;
                visited[y] = timestamp;
                nearest_[y] = nearest[x];
                pq.update(y);
            }
        };

        G->forEdges([&](const node x, const node y, const edgeweight w) {
            if (nearest[x] != nearest[y]) {
                process(x, y, static_cast<Weight>(w));
                process(y, x, static_cast<Weight>(w));
            }
        });

        // Step 2: Explore rest of the graph with Dijkstra
        dijkstra();

        // The main algorithm starts here
        while (findAndSwap() && totalSwaps++ < maxIterations) {
        }

        hasRun = true;
    }

private:
    const Graph *G;
    std::vector<node> group;
    std::vector<Weight> distance, distance_;
    std::vector<uint32_t> sumOfMins;
    std::vector<uint8_t> visited;
    uint8_t timestamp;
    std::unordered_map<node, index> idxMap;

    std::vector<Weight> increment;
    std::vector<node> nearest;
    std::vector<node> nearest_;
    std::vector<node> stack;
    std::queue<index> nextIdx;

    bool extended;
    count insertions, totalSwaps, stackSize;

    static constexpr Weight infdistance = std::numeric_limits<Weight>::max();
    static constexpr float maxInt16 = static_cast<float>(std::numeric_limits<uint16_t>::max());

#ifdef NETWORKIT_WITH_AVX
    union RandItem {
        uint16_t items[16];
        __m256i vec;
    };
    std::vector<RandItem> randVec;
#else
    std::vector<uint16_t> randVec;
#endif // NETWORKIT_WITH_AVX

    // Increment timestamp and, if necessary, reset vector
    void incrementTimestamp() {
        if (timestamp++ == std::numeric_limits<uint8_t>::max()) {
            timestamp = 1;
            std::fill(visited.begin(), visited.end(), 0);
        }
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

    struct CompareDistance {
    public:
        CompareDistance(const std::vector<Weight> &distance) : distance(&distance) {}

        bool operator()(const node x, const node y) const noexcept {
            return (*distance)[x] < (*distance)[y];
        }

    private:
        const std::vector<Weight> *distance;
    };

    tlx::d_ary_addressable_int_heap<node, 2, CompareDistance> heap;
    tlx::d_ary_addressable_int_heap<node, 2, CompareDistance> heap_;

    void init() {
        const auto n = G->upperNodeIdBound();

        distance.assign(n, 0);
        distance_.assign(n, 0);
        visited.assign(n, 0);
        sumOfMins.assign(n, 0);
        heap.reserve(n);
        if (G->isWeighted()) {
            heap_.reserve(n);
        }
        nearest.assign(n, none);
        nearest_.assign(n, none);
        stack.assign(n, 0);

        rng.resize(omp_get_max_threads());
#pragma omp parallel
        { rng[omp_get_thread_num()].state = randomSeed + omp_get_thread_num(); }

        for (size_t i = 0; i < group.size(); ++i) {
            idxMap[group[i]] = i;
        }

#ifdef NETWORKIT_WITH_AVX
        randVec.resize(n);
#else
        randVec.resize(16 * n);
#endif // NETWORKIT_WITH_AVX

        timestamp = 0;
        totalSwaps = 0;

        hasRun = false;
    }

    void initRandomVec() {
        G->parallelForNodes([&](const node u) {
            // Avoid to generate numbers for nodes in the group
            if (distance[u]) {
                auto &cur_rng = rng[omp_get_thread_num()];
#ifdef NETWORKIT_WITH_AVX
                // Generating two 16-bit random integers per time
                for (int j = 0; j < 16; j += 2) {
                    const auto x = pcg32_random_r(&cur_rng);
                    randVec[u].items[j] = static_cast<uint16_t>(x);
                    randVec[u].items[j + 1] = static_cast<uint16_t>(x >> 16);
                }
                randVec[u].vec = *(__m256i *)(&randVec[u].items[0]);
#else
                // Generating two 16-bit random integers per time
                for (int j = 0; j < 16; j += 2) {
                    const auto x = pcg32_random_r(&cur_rng);
                    randVec[16 * u + j] = static_cast<uint16_t>(x);
                    randVec[16 * u + j + 1] = static_cast<uint16_t>(x >> 16);
                }
#endif // NETWORKIT_WITH_AVX
            }
        });
    }

    void dijkstra() {
        auto &pq = G->isWeighted() ? heap_ : heap;
        assert(!pq.empty());

        do {
            const auto x = pq.extract_top();

            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (timestamp != visited[y] || distance_[y] > distance_[x] + w) {
                    distance_[y] = distance_[x] + w;
                    nearest_[y] = nearest_[x];
                    pq.update(y);
                    visited[y] = timestamp;
                }
            });

        } while (!pq.empty());
    }

    bool findAndSwap() {
        // Nodes that have been inserted/removed from the group
        std::vector<node> nodesInserted, nodesRemoved;
        nodesInserted.reserve(insertions);
        nodesRemoved.reserve(insertions);

        // Total farness decrement/increment after having inserted/removed the nodes from the group
        Weight totalDecrement{0}, totalIncrement{0};

        for (count i = 0; i < insertions; ++i) {
            // Find node with highest farness decrement by estimating the size
            // of its BFS DAG
            const auto v = estimateHighestDecrement();
            nodesInserted.push_back(v);

            // Put v in the group
            idxMap[v] = nextIdx.front();
            nextIdx.pop();

            // Update farness decrement
            totalDecrement += computeFarnessDecrement(v);
        }

        for (count i = 0; i < insertions; ++i) {
            // Update the farness decrement for each node in the group
            computeFarnessIncrement();

            // Compute node with lowest farness increment
            auto u = idxMap.begin()->first;
            auto minimumIncrement = increment[idxMap.begin()->second];

            std::for_each(++idxMap.begin(), idxMap.end(), [&](const std::pair<node, index> &entry) {
                if (increment[entry.second] < minimumIncrement) {
                    u = entry.first;
                    minimumIncrement = increment[entry.second];
                }
            });

            // Update farness increment
            totalIncrement += minimumIncrement;

            // Removing u from group
            nextIdx.push(idxMap.at(u));
            idxMap.erase(u);
            nodesRemoved.push_back(u);

            // Updating (1st) shortest distances and setting 2nd shortest distances to infinity
            G->forNodes([&](const node x) {
                if (nearest[x] == u) {
                    nearest[x] = nearest_[x];
                    distance[x] = distance_[x];
                    nearest_[x] = none;
                    distance_[x] = infdistance;
                } else if (nearest_[x] == u) {
                    nearest_[x] = none;
                    distance_[x] = infdistance;
                }
            });

            auto &pq = G->isWeighted() ? heap_ : heap;

            // Update the distance of y
            auto process = [&](const node x, const node y, const Weight w) {
                if (nearest[x] != nearest[y]) {
                    if (distance_[y] > distance[x] + w) {
                        distance_[y] = distance[x] + w;
                        nearest_[y] = nearest[x];
                        pq.update(y);
                    }
                    // Checking 2nd distanceance
                } else if (distance_[x] < infdistance && distance_[y] > distance_[x] + w) {
                    distance_[y] = distance_[x] + w;
                    nearest_[y] = nearest_[x];
                    pq.update(y);
                    assert(nearest_[x] != nearest[y]);
                }
            };

            G->forEdges([&](const node x, const node y, const edgeweight w) {
                process(x, y, static_cast<Weight>(w));
                process(y, x, static_cast<Weight>(w));
            });

            do {
                const auto x = pq.extract_top();
                G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                    if (nearest[y] != nearest_[x] && distance_[y] > distance_[x] + w) {
                        distance_[y] = distance_[x] + w;
                        nearest_[y] = nearest_[x];
                        pq.update(y);
                    }
                });
            } while (!pq.empty());
        }

        if (totalDecrement <= totalIncrement) {
            // Farness could not be decreased, restore original state
            do {
                idxMap[nodesRemoved.back()] = 0;
                nodesRemoved.pop_back();
            } while (!nodesRemoved.empty());

            do {
                idxMap.erase(nodesInserted.back());
                nodesInserted.pop_back();
            } while (!nodesInserted.empty());

            assert(idxMap.size() == group.size());

            return false;
        }

        return true;
    }

    void bfsFromGroup() {
        assert(heap.empty());
        incrementTimestamp();

        // Only needed by algorithm for unweighted graphs
        std::queue<node> q;

        for (const auto &entry : idxMap) {
            if (G->isWeighted()) {
                heap.push(entry.first);
            } else {
                q.push(entry.first);
            }

            distance[entry.first] = 0;
            visited[entry.first] = timestamp;
            nearest[entry.first] = entry.first;
        }

        do {
            const auto x = G->isWeighted() ? heap.extract_top() : extractQueueTop(q);

            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (visited[y] != timestamp || (G->isWeighted() && distance[y] > distance[x] + w)) {
                    distance[y] = distance[x] + static_cast<Weight>(w);
                    nearest[y] = nearest[x];
                    visited[y] = timestamp;
                    if (G->isWeighted()) {
                        heap.update(y);
                    } else {
                        q.push(y);
                    }
                }
            });
        } while (!(G->isWeighted() ? heap.empty() : q.empty()));
    }

    // Compute increment of farness for each node in the group.
    void computeFarnessIncrement() {
        std::fill(increment.begin(), increment.end(), 0);
        G->forNodes(
            [&](const node x) { increment[idxMap.at(nearest[x])] += distance_[x] - distance[x]; });
    }

    // Do a BFS to update 1st and 2nd shortest distances after adding v to the
    // group
    Weight computeFarnessDecrement(node v) {
        distance_[v] = distance[v];
        nearest_[v] = nearest[v];
        distance[v] = Weight{0};
        nearest[v] = v;
        visited[v] = timestamp;

        // Only needed by the algorithm for unweighted graphs
        std::queue<node> q;

        if (G->isWeighted()) {
            heap.push(v);
        } else {
            q.push(v);
            incrementTimestamp();
        }

        Weight decr = G->isWeighted() ? 0 : distance[v];

        auto processWeighted = [&](const node x, const node y, const Weight w) {
            if (distance[y] > distance[x] + w) {
                // Nearest nodes could have been already updated
                if (nearest[y] != v) {
                    nearest_[y] = nearest[y];
                    nearest[y] = v;
                    distance_[y] = distance[y];
                }
                distance[y] = distance[x] + w;
                heap.update(y);
            } else if (nearest[x] == v && nearest[y] != v && distance_[y] > distance[x] + w) {
                distance_[y] = distance[x] + w;
                nearest_[y] = v;
                heap.update(y);
            } else if (nearest_[x] == v && nearest[y] != v && distance_[y] > distance_[x] + w) {
                distance_[y] = distance_[x] + w;
                nearest_[y] = v;
                heap.update(y);
            }
        };

        auto processUnweighted = [&](const node x, const node y) {
            if (timestamp == visited[y]) {
                return;
            }

            if (distance[y] > distance[x] + 1) {
                distance_[y] = distance[y];
                nearest_[y] = nearest[y];
                decr += distance[y] - (distance[x] + 1);
                distance[y] = distance[x] + 1;
                nearest[y] = v;
                q.push(y);
            } else if (nearest[x] == v && distance_[y] > distance[x] + 1) {
                distance_[y] = distance[x] + 1;
                nearest_[y] = v;
                q.push(y);
            } else if (nearest_[x] == v && distance_[y] > distance_[x] + 1) {
                distance_[y] = distance_[x] + 1;
                nearest_[y] = v;
                q.push(y);
            }
            visited[y] = timestamp;
        };

        do {
            const auto x = G->isWeighted() ? heap.extract_top() : extractQueueTop(q);

            if (G->isWeighted() && nearest[x] == v) {
                decr += distance_[x] - distance[x];
            }

            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (G->isWeighted()) {
                    processWeighted(x, y, static_cast<Weight>(w));
                } else {
                    processUnweighted(x, y);
                }
            });
        } while (!(G->isWeighted() ? heap.empty() : q.empty()));

        return decr;
    }

    // Computes real decrement of farness
    node estimateHighestDecrement() {
        assert(heap.empty());
        incrementTimestamp();

        // Needed only by the algorithm for unweighted graphs
        std::queue<node> q;
        stackSize = 0;
        for (const auto &idx : idxMap) {
            if (G->isWeighted()) {
                heap.push(idx.first);
            } else {
                q.push(idx.first);
            }
            visited[idx.first] = timestamp;
        }

        // Recompute the DAG
        do {
            const auto x = G->isWeighted() ? heap.extract_top() : extractQueueTop(q);
            bool leaf = true;
            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (timestamp != visited[y] || (G->isWeighted() && distance[y] > distance[x] + w)) {
                    visited[y] = timestamp;
                    if (G->isWeighted()) {
                        heap.update(y);
                    } else {
                        leaf = false;
                        q.push(y);
                    }
                }
            });

            if (distance[x] && (G->isWeighted() || (!leaf || distance[x] == 1))) {
                stack[stackSize++] = x;
            }

        } while (!(G->isWeighted() ? heap.empty() : q.empty()));

        // Generating all the necessary random numbers
        initRandomVec();

        auto isCandidate = [&](const node x) -> bool {
            return G->isWeighted() ? distance[x]
                                   : (distance[x] == 1 || (extended && distance[x] > 1));
        };

        // Do 16 packed repetitions at once
        for (size_t i = 0; i < stackSize; ++i) {
            const auto x = stack[stackSize - 1 - i];
#ifdef NETWORKIT_WITH_AVX
            // 16 randomly generated integers;
            __m256i x1 = randVec[x + n * it].vec;
            // Pulling leaves
            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (distance[y] == distance[x] + w) {
                    __m256i y1 = randVec[y].vec;
                    x1 = _mm256_min_epu16(x1, y1);
                }
            });
            *(__m256i *)(&randVec[x].items) = x1;
#else
            G->forNeighborsOf(x, [&](const node y, const edgeweight w) {
                if (distance[y] == distance[x] + w) {
                    for (int i = 0; i < 16; ++i) {
                        randVec[16 * x + i] = std::min(randVec[16 * x + i], randVec[16 * y + i]);
                    }
                }
            });
#endif // NETWORKIT_WITH_AVX

            if (isCandidate(x)) {
                sumOfMins[x] = 0;
                for (int j = 0; j < 16; ++j) {
#ifdef NETWORKIT_WITH_AVX
                    sumOfMins[x] += randVec[x + n * it].items[j];
#else
                    sumOfMins[x] = randVec[16 * x + j];
#endif // NETWORKIT_WITH_AVX
                }

                if (!sumOfMins[x]) {
                    sumOfMins[x] = 1;
                }
            }
        }

        node v = none;
        float bestEstimate = -1.f;
        G->forNodes([&](const node x) {
            if (isCandidate(x) && sumOfMins[x]) {
                const auto estimate =
                    static_cast<float>(distance[x])
                    * (16.f / (static_cast<float>(sumOfMins[x]) / maxInt16) - 1.f);
                if (estimate > bestEstimate) {
                    v = x;
                    bestEstimate = estimate;
                }
            }
        });

        assert(v != none);
        return v;
    }

    node extractQueueTop(std::queue<node> &q) {
        const auto u = q.front();
        q.pop();

        return u;
    }
};

// Algorithm for unweighted graphs
using GroupClosenessGrowShrink = GroupClosenessGrowShrinkGeneral<count>;

// Algorithm for weighted graphs
using GroupClosenessGrowShrinkWeighted = GroupClosenessGrowShrinkGeneral<edgeweight>;

} // namespace NetworKit

#endif // NETWORKIT_CENTRALITY_GROUP_CLOSENESS_GROW_SHRINK_HPP_
