/*
 * Graph.h
 *
 *  Created on: 28.11.2012
 *      Author: cls
 */

#ifndef GRAPH_H_
#define GRAPH_H_

#include <utility>
#include <cinttypes>

extern "C" {
#include "stinger.h"
}


namespace EnsembleClustering {


/** Typedefs **/


typedef int64_t node; //!< a node is an integer logical index
typedef std::pair<node, node> edge; //!< an undirected edge is a pair of nodes (indices)


/** Graph interface **/


/**
 * Graph encapsulates a STINGER graph object and provides
 * a more concise interface to it.
 *
 * The graph concept modelled is
 * - undirected
 * - weighted
 * - without self-loops (use node weights instead)
 *
 * TODO: timestamps
 *
 */
class Graph {

protected:

	stinger* stingerG;



public:
	/** default parameters ***/

	static constexpr double defaultEdgeWeight = 1.0;
	static const int64_t defaultEdgeType = 0;
	static const int64_t defaultTimeStamp = 0;

	/** methods **/


	/**
	 * Construct Graph object with new STINGER graph inside.
	 */
	Graph();

	/**
	 * Initialize with STINGER graph.
	 *
	 * @param[in]	stingerG	a STINGER graph struct
	 */
	Graph(stinger* stingerG);

	~Graph();

	/**
	 * Return the internal STINGER data structure.
	 *
	 */
	stinger* asSTINGER();

	/**
	 * Insert a weighted, undirected edge.
	 */
	void insertEdge(node u, node v, double weight=defaultEdgeWeight, int64_t type=defaultEdgeType, int64_t timestamp=defaultTimeStamp);

	/**
	 * Return node weight.
	 */
	double getWeight(node v);

	/**
	 * Return edge weight.
	 */
	double getWeight(edge uv);

	/**
	 * Return edge weight.
	 * Equivalent to getWeight(edge uv)
	 */
	double getWeight(node u, node v);


	int64_t numberOfEdges();


	int64_t numberOfNodes();





};

} /* namespace EnsembleClustering */
#endif /* GRAPH_H_ */
