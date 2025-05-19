from neo4j_graphrag.retrievers.base import RetrieverResultItem
import neo4j

# 3. Retriever
def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """
    Format the result from the database query into a RetrieverResultItem.
    The retrieved start nodes are connected to an edge (relationship) and an end node.
    The retrieved end nodes are connected to the start node with an edge (relationship).

    Args:
        record (neo4j.Record): The record returned from the database query.
    Returns:
        RetrieverResultItem: The formatted result item. This includes the content and metadata.
    The content is a string that contains the start node, the relationship, and the end node.
    The metadata includes the start node and the score.
    """
    content=""
    for i in range(len(record.get('top_triplets'))):
        content += f"{record.get('top_triplets')[i].get('subject')} {record.get('top_triplets')[i].get('relationship')} {record.get('top_triplets')[i].get('object')},"
    return RetrieverResultItem(
        content=content,
        metadata={
            "startNode": record.get('node'),
            "score": record.get("score"),
        }
    )

def prepare_pagerank_projection(driver):
    """
    Prepare the graph projection for PageRank.
    This function creates a graph projection in Neo4j for PageRank algorithm.
    """
    # Check if the graph already exists
    with driver.session() as session:
        result = session.run("""
                CALL gds.graph.list()
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                ORDER BY graphName ASC
                """)
        record = result.data()
        if not record:
            print("No graph projection found. Creating a new one.")
            
            # Create the graph projection
            with driver.session() as session:
                session.run("""
                    CALL gds.graph.project(
                        'myGraph',
                        ['StartNode', 'EndNode'],
                        {
                            RELATIONSHIP: {
                                type: 'RELATIONSHIP',
                                orientation: 'UNDIRECTED'
                            }
                        }
                    )
                    """)
            print("Graph projection created.")
    

# define the retrieval query
RETRIEVAL_QUERY_SIMILARITY = """
    // Step 1: Find neighbors of the retrieved node
    MATCH (node)-[r:RELATIONSHIP]->(e:EndNode)

    // Step 2: Compute cosine similarity manually between input and e
    WITH node, r, e,
        gds.similarity.cosine(node.embedding, e.embedding) AS e_similarity,
        score AS node_similarity // manually preserve the score for 'node'

    // Step 3: Top-k neighbors based on similarity
    ORDER BY e_similarity DESC
    WITH node, node_similarity, COLLECT(DISTINCT {entity: e, sim: e_similarity})[0..$top_k] AS top_e

    // Step 4: Combine node + top_e into one list
    WITH node, node_similarity,
        [{entity: node, sim: node_similarity}] + top_e AS nodes

    UNWIND nodes AS entity_info
    WITH node, entity_info.entity AS n, entity_info.sim AS similarity

    // Step 5: Get all outgoing edges for all relevant nodes
    MATCH (n)-[r1:RELATIONSHIP]->(e1:EndNode)

    // Step 6: Collect and rank triplets
    WITH node, n, r1, e1, similarity
    ORDER BY similarity DESC
    WITH node,
        COLLECT({subject: n.text, relationship: r1.text, object: e1.text}) AS triplets,
        AVG(similarity) AS avg_similarity

    // Step 7: Return
    RETURN
        node.text AS node,
        avg_similarity AS score,
        triplets[0..$top_k] AS top_triplets
"""

# define the retrieval query - pagerank query
RETRIEVAL_QUERY_PAGERANK = """
    MATCH (node)-[r:RELATIONSHIP]->(e:EndNode)
    WITH node, r, e
    ORDER BY node
    WITH collect({subject: node.text, relationship: r.text, object: e.text}) AS top_triplets, node AS c_node
    CALL gds.pageRank.stream('myGraph', {
        maxIterations: 20,
        dampingFactor: 0.85,
        sourceNodes: [id(c_node)]
    })
    YIELD nodeId, score
    WHERE id(c_node) = nodeId
    RETURN c_node.text, top_triplets, score
    ORDER BY score DESC
    LIMIT $k
"""