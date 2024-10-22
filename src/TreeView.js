// treeData.js
export const initialTreeData = {
  name: 'root',
  toggled: true,
  children: [
    {
      name: 'parent',
      children: [
        { name: 'child1' },
        { name: 'child2' }
      ]
    },
    {
      name: 'loading parent',
      loading: true,
      children: []
    }
  ]
};

// TreeView.js (update nodesToTreeData)
export const nodesToTreeData = (nodes, edges) => {
  // Create a map of nodes
  const nodeMap = {};
  nodes.forEach((node) => {
    nodeMap[node.id] = { ...node, children: [] };
  });

  // Loop through the edges to build parent-child relationships
  edges.forEach((edge) => {
    const sourceNode = nodeMap[edge.source];
    const targetNode = nodeMap[edge.target];
    
    if (sourceNode && targetNode) {
      sourceNode.children.push(targetNode);
    }
  });

  // Find the root nodes (nodes that are not a target of any edge)
  const rootNodes = nodes.filter(
    (node) => !edges.some((edge) => edge.target === node.id)
  );

  // Convert root nodes to the structure required by the tree
  const treeData = rootNodes.map((node) => ({
    name: node.data.label,
    id: node.id,
    toggled: true,
    children: node.children || []
  }));

  return treeData;
};
  
export const treeStyle = {
  tree: {
    base: {
      listStyle: "none",
      backgroundColor: "white",
      margin: 0,
      padding: 0,
      color: "#00ff00",
      fontFamily: "lucida grande ,tahoma,verdana,arial,sans-serif",
      fontSize: "14px",
      height: "100%",
      width: "100%",
    },
    node: {
      base: {
        position: "relative",
      },
      link: {
        cursor: "pointer",
        position: "relative",
        padding: "0px 5px",
        display: "block",
      },
      activeLink: {
        background: "#31363F",
      },
      toggle: {
        base: {
          position: "relative",
          display: "inline-block",
          verticalAlign: "top",
          marginLeft: "-5px",
          height: "24px",
          width: "24px",
        },
        wrapper: {
          position: "absolute",
          top: "50%",
          left: "50%",
          margin: "-7px 0 0 -7px",
          height: "14px",
        },
        height: 14,
        width: 14,
        arrow: {
          fill: "#9DA5AB",
          strokeWidth: 0,
        },
      },
      header: {
        base: {
          display: "inline-block",
          verticalAlign: "top",
          color: "#9DA5AB",
        },
        connector: {
          width: "2px",
          height: "12px",
          borderLeft: "solid 2px black",
          borderBottom: "solid 2px black",
          position: "absolute",
          top: "0px",
          left: "-21px",
        },
        title: {
          lineHeight: "24px",
          verticalAlign: "middle",
        },
      },
      subtree: {
        listStyle: "none",
        paddingLeft: "19px",
      },
      loading: {
        color: "#E2C089",
      },
    },
  },
};

