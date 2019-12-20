/*
 * GroupClosenessLocalSwaps.cpp
 *
 *  Created on: 19.12.2019
 *      Author: Eugenio Angriman <angrimae@hu-berlin.de>
 */

// networkit-format

#include <omp.h>
#include <queue>

#include <networkit/centrality/GroupClosenessLocalSwaps.hpp>

namespace NetworKit {

void GroupClosenessLocalSwaps::init() {
    const auto n = G->upperNodeIdBound();

    distance.assign(n, 0);
    gamma.assign(n * group.size(), static_cast<unsigned char>(0));
    canSwap.assign(group.size(), static_cast<unsigned char>(0));
    visited.assign(n, 0);
    idxMap.clear();
    idxMap.reserve(group.size());
    stack.assign(n, 0);
    value.assign(group.size(), 0);
    sumOfMins.assign(n, 0);
    valueDecrement.resize(group.size(), 0);

    rng.resize(omp_get_max_threads());
#pragma omp parallel
    { rng[omp_get_thread_num()].state = randomSeed + omp_get_thread_num(); }

    for (size_t i = 0; i < group.size(); ++i) {
        const auto u = group[i];
        idxMap[u] = i;
        gamma[u * group.size() + i] = 1;
    }

#ifdef NETWORKIT_WITH_AVX
    randVec.resize(n);
#else
    randVec.resize(n * 16);
#endif

    timestamp = 0;
    totalSwaps = 0;

    hasRun = false;
}

void GroupClosenessLocalSwaps::initRandomVector() {
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
#endif
        }
    });
}

void GroupClosenessLocalSwaps::bfsFromGroup() {
    std::queue<node> q;
    incrementTimestamp();
    stackSize = 0;

    for (const auto &idx : idxMap) {
        q.push(idx.first);
        value[idx.second] = 1;
        distance[idx.first] = 0;
        visited[idx.first] = timestamp;
    }

    do {
        const auto u = q.front();
        q.pop();

        // Cannot swap with nodes in the group
        if (!distance[u]) {
            canSwap[idxMap.at(u)] = 0;
        }

        bool uIsLeaf = false;

        G->forNeighborsOf(u, [&](const node v) {
            // Whether v is in \Gamma_u i.e., the shortest path from S to v is realized only
            // by u.
            bool inGamma = true;

            // Whether the node in the group that realizes the shortest distance to v has
            // been found.
            bool nearestNodeFound = false;

            // Index of the node in the group that realizes the shortest distance to v.
            index groupIdx;

            if (timestamp != visited[v]) {
                uIsLeaf = false;
                distance[v] = distance[u] + 1;
                visited[v] = timestamp;
                q.push(v);

                for (size_t i = 0; i < group.size(); ++i) {
                    const auto curGamma = gamma[group.size() * u + i];
                    if (curGamma) {
                        if (!nearestNodeFound) {
                            nearestNodeFound = true;
                            groupIdx = i;
                        } else {
                            inGamma = false;
                        }
                    }

                    gamma[group.size() * v + i] = curGamma;
                }

                if (inGamma) {
                    ++value[groupIdx];
                }

            } else if (distance[u] + 1 == distance[v]) {
                inGamma = true;
                nearestNodeFound = false;
                bool subtract = false;

                for (size_t i = 0; i < group.size(); ++i) {
                    if (gamma[group.size() * v + i]) {
                        if (!nearestNodeFound) {
                            nearestNodeFound = true;
                            groupIdx = i;
                        } else {
                            inGamma = false;
                            break;
                        }
                    } else if (gamma[group.size() * u + i]) {
                        gamma[group.size() * v + i] = 1;
                        subtract = true;
                    }
                }
                if (inGamma && subtract) {
                    --value[groupIdx];
                }
            }

            if (!distance[u] && !distance[v]) {
                canSwap[idxMap[u]] = true;
            }
        });

        if (distance[u] && (!uIsLeaf || distance[u] == uint32_t{1})) {
            stack[stackSize++] = u;
        }
    } while (!q.empty());
}

// Computes real decrement of farness
int64_t GroupClosenessLocalSwaps::computeFarnessDecrement(node v) {
    incrementTimestamp();

    std::queue<node> q;
    q.push(v);
    distance[v] = 0;
    visited[v] = timestamp;

    int16_t decrement{1};
    std::fill(valueDecrement.begin(), valueDecrement.end(), int64_t{0});

    do {
        const auto u = q.front();
        q.pop();

        bool inGamma = false;
        index groupIdx;

        for (size_t i = 0; i < group.size(); ++i) {
            if (gamma[group.size() * u + i]) {
                if (!inGamma) {
                    inGamma = true;
                    groupIdx = i;
                } else {
                    inGamma = false;
                    break;
                }
            }
        }

        if (inGamma) {
            ++valueDecrement[groupIdx];
        }

        G->forNeighborsOf(u, [&](const node v) {
            if (timestamp != visited[v]) {
                if (distance[u] + 1 <= distance[v]) {
                    if (distance[u] + 1 < distance[v]) {
                        distance[v] = distance[u] + 1;
                        ++decrement;
                    }
                    q.push(v);
                }
                visited[v] = timestamp;
            }
        });
    } while (!q.empty());

    return decrement;
}

// Estimate node with highest farness decrement
node GroupClosenessLocalSwaps::estimateHighestDecrement() {
    initRandomVector();

    float bestEstimate = -1.f;
    node v = none;

    for (count i = 0; i < stackSize; ++i) {
        const auto x = stack[stackSize - 1 - i];
#ifdef NETWORKIT_WITH_AVX
        // 16 randomly generated integers;
        __m256i x1 = randVec[x].vec;
        // Pulling leaves
        G->forNeighborsOf(x, [&](const node y) {
            if (distance[y] == distance[x] + 1) {
                __m256i y1 = randVec[y].vec;
                x1 = _mm256_min_epu16(x1, y1);
            }
        });
        *(__m256i *)(&randVec[x].items) = x1;
#else
        // 16 random 16-bit integers are realized by 4 64-bit random integers
        G->forNeighborsOf(x, [&](const node y) {
            if (distance[y] == distance[x] + 1) {
                for (int i = 0; i < 16; ++i) {
                    randVec[16 * x + i] = std::min(randVec[16 * x + i], randVec[16 * y + i]);
                }
            }
        });
#endif
        if (distance[x] == 1) {
            sumOfMins[x] = 0;
            for (int j = 0; j < 16; ++j) {
#ifdef NETWORKIT_WITH_AVX
                sumOfMins[x] += randVec[x].items[j];
#else
                sumOfMins[x] += randVec[16 * x + j];
#endif
            }
            if (!sumOfMins[x]) {
                sumOfMins[x] = 1;
            }
        }
    }

    G->forNodes([&](const node x) {
        if (distance[x] == 1) {
            float estimate = 16.f / (static_cast<float>(sumOfMins[x]) / maxInt16) - 1.f;
            if (estimate > bestEstimate) {
                v = x;
                bestEstimate = estimate;
            }
        }
    });

    assert(v != none);
    return v;
}

bool GroupClosenessLocalSwaps::findAndSwap() {
    bfsFromGroup();
    const auto v = estimateHighestDecrement();
    const auto farnessDecrement = computeFarnessDecrement(v);
    int64_t improvement = 0;
    node u = none;

    G->forNeighborsOf(v, [&](const node y) {
        if (!distance[y]) {
            const auto idx = idxMap.at(y);
            const auto curImprovement = farnessDecrement - value[idx] + valueDecrement[idx];

            if (curImprovement > improvement) {
                improvement = curImprovement;
                u = y;
            }
        }
    });

    for (const auto &idx : idxMap) {
        if (canSwap[idx.second]) {
            int64_t curImprovement =
                farnessDecrement - value[idx.second] + valueDecrement[idx.second];
            if (curImprovement > improvement) {
                improvement = curImprovement;
                u = idx.first;
            }
        }
    }

    if (improvement <= 0) {
        return false;
    }

    const auto idxU = idxMap.at(u);
    idxMap.erase(u);
    idxMap[v] = idxU;
    resetGamma(v, idxU);

    return true;
}

void GroupClosenessLocalSwaps::run() {
    init();

    while (findAndSwap() && ++totalSwaps < maxSwaps) {
    }
    hasRun = true;
}
} // namespace NetworKit
