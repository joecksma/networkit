networkit_add_module(centrality
    ApproxBetweenness.cpp
    ApproxCloseness.cpp
    ApproxGroupBetweenness.cpp
    Betweenness.cpp
    Centrality.cpp
    Closeness.cpp
    CoreDecomposition.cpp
    DegreeCentrality.cpp
    DynApproxBetweenness.cpp
    DynBetweenness.cpp
    DynBetweennessOneNode.cpp
    DynKatzCentrality.cpp
    DynTopHarmonicCloseness.cpp
    EigenvectorCentrality.cpp
    EstimateBetweenness.cpp
    GroupCloseness.cpp
    GroupClosenessLocalSwaps.cpp
    GroupDegree.cpp
    HarmonicCloseness.cpp
    KadabraBetweenness.cpp
    KatzCentrality.cpp
    KPathCentrality.cpp
    LaplacianCentrality.cpp
    LocalClusteringCoefficient.cpp
    LocalPartitionCoverage.cpp
    PageRank.cpp
    PermanenceCentrality.cpp
    Sfigality.cpp
    SpanningEdgeCentrality.cpp
    TopCloseness.cpp
    TopHarmonicCloseness.cpp
    )

networkit_module_link_modules(centrality
        algebraic
        auxiliary
        base
        components
        distance
        dyn_distance
        dynamics
        graph
        structures
    )

add_subdirectory(test)

