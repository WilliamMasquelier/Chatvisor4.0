# Visualizing and Debugging Your Knowledge Graph

## 1. Run the Pipeline

## 2. Locate the Knowledge Graph

## 3. Open the Graph in Gephi

## 4. Install the Leiden Algorithm Plugin

## 5. Run Statistics

## 6. Color the Graph by Clusters

## 7. Resize Nodes by Degree Centrality

## 8. Layout the Graph

## 9. Run ForceAtlas2

## 10. Add Text Labels (Optional)

The following step-by-step guide walks through the process to visualize a knowledge graph after it's been constructed by graphrag. Note that some of the settings recommended below are based on our own experience of what works well. Feel free to change and explore other settings for a better visualization experience!

Before building an index, please review your settings.yaml configuration file and ensure that graphml snapshots is enabled.
snapshots:
  graphml: true

(Optional) To support other visualization tools and exploration, additional parameters can be enabled that provide access to vector embeddings.
embed_graph:
  enabled: true # will generate node2vec embeddings for nodes
umap:
  enabled: true # will generate UMAP embeddings for nodes

After running the indexing pipeline over your data, there will be an output folder (defined by the storage.base_dir setting).

In the output folder, look for a file named merged_graph.graphml. graphml is a standard file format supported by many visualization tools. We recommend trying Gephi.











Your final graph should now be visually organized and ready for analysis!

- Output Folder: Contains artifacts from the LLM’s indexing pass.

[](#__codelineno-0-1)

[](#__codelineno-0-2)

[](#__codelineno-1-1)

[](#__codelineno-1-2)

[](#__codelineno-1-3)

[](#__codelineno-1-4)

[file format](http://graphml.graphdrawing.org)

[Gephi](https://gephi.org)

![A basic graph visualization by Gephi](https://microsoft.github.io/graphrag/../img/viz_guide/gephi-initial-graph-example.png)

![A view of Gephi's network overview settings](https://microsoft.github.io/graphrag/../img/viz_guide/gephi-network-overview-settings.png)

![A view of Gephi's appearance pane](https://microsoft.github.io/graphrag/../img/viz_guide/gephi-appearance-pane.png)

![A view of Gephi's layout pane](https://microsoft.github.io/graphrag/../img/viz_guide/gephi-layout-pane.png)

![A view of Gephi's ForceAtlas2 layout pane](https://microsoft.github.io/graphrag/../img/viz_guide/gephi-layout-forceatlas2-pane.png)

