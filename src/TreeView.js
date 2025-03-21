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

// Helper function to get a readable node type label
const getNodeTypeLabel = (type) => {
  const typeLabels = {
    requirements: 'Requirements',
    dataPrep: 'Data Preparation',
    modelTraining: 'Model Training',
    modelEval: 'Model Evaluation',
    modelTest: 'Model Testing',
    parameterTuning: 'Parameter Tuning',
    textUpdater: 'Text Node',
    runModel: 'Run Model'
  };
  return typeLabels[type] || type;
};

// Convert nodes and edges to tree data structure for Treebeard
export const nodesToTreeData = (nodes, edges) => {
  if (!nodes || nodes.length === 0) {
    return {
      name: 'No nodes available',
      toggled: true,
      children: []
    };
  }

  // Get root nodes (nodes without incoming edges)
  const nodeIds = new Set(nodes.map(node => node.id));
  const targetIds = new Set(edges.map(edge => edge.target));
  const rootNodeIds = [...nodeIds].filter(id => !targetIds.has(id));

  // Build the tree starting from root nodes
  const buildTree = (nodeId) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return null;

    // Find child nodes (nodes connected by outgoing edges)
    const childEdges = edges.filter(edge => edge.source === nodeId);
    const children = childEdges
      .map(edge => buildTree(edge.target))
      .filter(child => child !== null);

    // Create node name with type included
    const nodeType = getNodeTypeLabel(node.type);
    const nodeName = node.data.label || `Node ${node.id}`;
    const displayName = `${nodeType} (${node.id})`;

    // Create tree node
    return {
      id: node.id,
      name: displayName,
      title: nodeName,
      type: node.type,
      nodeTypeLabel: nodeType,
      children: children.length > 0 ? children : undefined,
      toggled: true, // Default to expanded for better visibility
    };
  };

  // Create tree from all root nodes
  const rootNodes = rootNodeIds.map(id => buildTree(id)).filter(n => n !== null);

  // Return the final tree data
  return {
    name: 'Workflow',
    toggled: true,
    active: false,
    children: rootNodes.length > 0 ? rootNodes : [
      { 
        name: 'No connected nodes', 
        type: 'info',
        toggled: true
      }
    ]
  };
};

export const treeStyle = {
  tree: {
    base: {
      listStyle: 'none',
      backgroundColor: '#ffffff',
      margin: 0,
      padding: '16px',
      color: '#333',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      fontSize: '14px',
      width: '100%',
      height: '100%',
      overflowY: 'auto'
    },
    node: {
      base: {
        position: 'relative',
        padding: '8px 0'
      },
      link: {
        cursor: 'pointer',
        position: 'relative',
        padding: '4px 6px',
        display: 'block',
        transition: 'all 0.2s',
        borderRadius: '4px'
      },
      activeLink: {
        background: '#e9ecef'
      },
      toggle: {
        base: {
          position: 'relative',
          display: 'inline-block',
          verticalAlign: 'top',
          marginLeft: '-5px',
          height: '24px',
          width: '24px',
          cursor: 'pointer'
        },
        wrapper: {
          position: 'absolute',
          top: '50%',
          left: '50%',
          margin: '-8px 0 0 -8px',
          height: '16px'
        },
        height: 8,
        width: 8,
        arrow: {
          fill: '#555',
          strokeWidth: 0
        }
      },
      header: {
        base: {
          display: 'inline-block',
          verticalAlign: 'top',
          color: '#333'
        },
        connector: {
          width: '2px',
          height: '12px',
          borderLeft: 'solid 2px #ddd',
          borderBottom: 'solid 2px #ddd',
          position: 'absolute',
          top: '0px',
          left: '-21px'
        },
        title: {
          lineHeight: '24px',
          verticalAlign: 'middle',
          fontWeight: '500'
        }
      },
      subtree: {
        listStyle: 'none',
        paddingLeft: '28px'
      }
    }
  }
};

