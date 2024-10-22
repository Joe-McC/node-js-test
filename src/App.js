import React, { useCallback, useEffect } from 'react';
import axios from 'axios';
import { Navbar, Nav, NavDropdown } from 'react-bootstrap';
import { Treebeard } from 'react-treebeard';
import 'bootstrap/dist/css/bootstrap.min.css';

import TextUpdaterNode from './TextUpdaterNode';
import { nodesToTreeData, treeStyle } from './TreeView';

import './text-updater-node.css';
import './App.css';

import ReactFlow, {
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
} from 'reactflow';

import 'reactflow/dist/style.css';

const rfStyle = {
  backgroundColor: '#B8CEFF',
};

const initialNodes = [
  { id: '1', type: 'textUpdater', position: { x: 0, y: 0 }, data: { label: 'New Node 001' } },
  { id: '2', type: 'textUpdater', position: { x: 0, y: 100 }, data: { label: 'New Node 002' } }
];

const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

const nodeTypes = { textUpdater: TextUpdaterNode };

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = React.useState(null);
  const [treeData, setTreeData] = React.useState(nodesToTreeData(initialNodes, initialEdges));

  // Recalculate tree data whenever nodes or edges change
  useEffect(() => {
    setTreeData(nodesToTreeData(nodes, edges)); // Update treeData when nodes or edges change
  }, [nodes, edges]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const saveToBackend = () => {
    const filename = window.prompt('Enter filename for saving', 'nodes.json');
    if (filename) {
      axios.post('http://127.0.0.1:5000/save_nodes', { filename, nodes })
        .then(response => {
          alert(response.data.message);
        })
        .catch(error => {
          console.error('Error saving nodes:', error);
        });
    }
  };

  const loadFromBackend = () => {
    const filename = window.prompt('Enter filename to load', 'nodes.json');
    if (filename) {
      axios.get(`http://127.0.0.1:5000/load_nodes?filename=${filename}`)
        .then(response => {
          setNodes(response.data.nodes);
        })
        .catch(error => {
          console.error('Error loading nodes:', error);
        });
    }
  };

  // Add Node function
  const addNode = () => {
    const newId = (nodes.length + 1).toString(); // New node id
    const newNode = {
      id: newId,
      type: 'textUpdater',
      position: { x: Math.random() * 250, y: Math.random() * 250 }, // Random position
      data: { label: `New Node ${newId.padStart(3, '0')}` },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  // Remove Node function
  const removeNode = () => {
    if (!selectedNode) {
      alert('No node selected to remove!');
      return;
    }

    setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
    setEdges((eds) => eds.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id));
    setSelectedNode(null); // Clear selection after removal
  };

  // Handle node click for selection
  const onNodeClick = (event, node) => {
    setSelectedNode(node);
  };

   
  const updateTreeData = (nodeToUpdate, tree) => {
    return tree.map(node => {
      if (node.id === nodeToUpdate.id) {
        // Update the node that was toggled
        return {
          ...node,
          active: true, // Toggle the active state
          toggled: nodeToUpdate.toggled, // Keep the toggled state as passed in
        };
      }
      
      // If the node has children, recursively update them
      if (node.children) {
        return {
          ...node,
          children: updateTreeData(nodeToUpdate, node.children), // Update children recursively
        };
      }
  
      return node;
    });
  };
  
  const handleTreeToggle = (node, toggled) => {
    // Set all nodes to inactive
    const resetActiveState = (tree) => {
      return tree.map(n => ({
        ...n,
        active: false, // Reset all nodes to inactive
        children: n.children ? resetActiveState(n.children) : n.children
      }));
    };
  
    // First reset the active state for all nodes
    const newTree = resetActiveState(treeData);
  
    // Update the toggled node and its children
    const updatedTree = updateTreeData(
      { ...node, toggled }, // Pass the node with the new toggled state
      newTree
    );
  
    setTreeData(updatedTree); // Update the state with the new tree
    setSelectedNode(node); // Set the selected node
  };
  
  
  return (
    <div className="App">
      <Navbar bg="light" expand="lg">
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="navbar-fixed-top">
            <Navbar.Brand href="#home">React-Bootstrap</Navbar.Brand>
            <NavDropdown title="File" id="file-nav-dropdown">
              <NavDropdown.Item onClick={saveToBackend}>Save to Backend</NavDropdown.Item>
              <NavDropdown.Item onClick={loadFromBackend}>Load from Backend</NavDropdown.Item>
            </NavDropdown>
            <NavDropdown title="Edit" id="edit-nav-dropdown">
              <NavDropdown.Item onClick={addNode}>Add Node</NavDropdown.Item>
              <NavDropdown.Item onClick={removeNode}>Remove Node</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      <div style={{ display: 'flex', height: '100vh', width: '100vw' }}>
        {/* Tree View */}
        <div style={{ width: '30%', overflow: 'auto', borderRight: '1px solid #ddd', padding: '10px' }}>
          <Treebeard
            data={treeData}
            onToggle={handleTreeToggle}
            style={treeStyle}
          />
        </div>
        {/* React Flow Diagram */}
        <div style={{ flexGrow: 1, height: '100%' }}>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              fitView
              style={rfStyle}
            >
              <Controls />
              <MiniMap />
              <Background variant="dots" gap={12} size={1} />
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  );
}

export default App;
