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
  { id: '1', type: 'textUpdater', position: { x: 0, y: 0 }, data: { label: 'New Node 001', id: '1' } },
  { id: '2', type: 'textUpdater', position: { x: 0, y: 100 }, data: { label: 'New Node 002', id: '2' } }
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
    setTreeData(nodesToTreeData(nodes, edges));
  }, [nodes, edges]);

  const saveToBackend = () => {
    const filename = window.prompt('Enter filename for saving', 'flow_data.json');
    if (filename) {
      if (nodes.length === 0 && edges.length === 0) {
        alert('No nodes or edges to save.');
        return;
      }
  
      // Sending both nodes and edges to the backend
      axios.post('http://127.0.0.1:5000/save_nodes', { filename, nodes, edges })
        .then(response => {
          alert(response.data?.message || 'Data saved successfully');
        })
        .catch(error => {
          console.error('Error saving data:', error);
          alert(`Error saving data: ${error.response?.data?.error || 'Please check the backend.'}`);
        });
    }
  };
  
  const loadFromBackend = () => {
    const filename = window.prompt('Enter filename to load', 'flow_data.json');
    if (filename) {
      axios.get(`http://127.0.0.1:5000/load_nodes?filename=${filename}`)
        .then(response => {
          const { nodes: loadedNodes, edges: loadedEdges } = response.data;
  
          const formattedNodes = loadedNodes.map(node => ({
            ...node,
            data: { ...node.data, label: node.data.label || '', id: node.data.id || node.id },
          }));
  
          setNodes(formattedNodes);
          setEdges(loadedEdges || []); // Ensure edges are set, even if empty
  
          alert('Data loaded successfully');
        })
        .catch(error => {
          console.error('Error loading data:', error);
          alert(`Error loading data: ${error.response?.data?.error || 'Please check the backend.'}`);
        });
    }
  };
  

  const addNode = () => {
    const newId = (nodes.length + 1).toString();
    const newNode = {
      id: newId,
      type: 'textUpdater',
      position: { x: Math.random() * 250, y: Math.random() * 250 },
      data: { label: `New Node ${newId.padStart(3, '0')}`, id: newId },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const removeNode = () => {
    if (!selectedNode) {
      alert('No node selected to remove!');
      return;
    }

    setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
    setEdges((eds) => eds.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id));
    setSelectedNode(null);
  };

  const updateNodeData = useCallback((id, newData) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === id
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    );
  }, [setNodes]);
  
  const updateTreeData = (nodeToUpdate, tree) => {
    return tree.map(node => {
      if (node.id === nodeToUpdate.id) {
        return { ...node, active: true, toggled: nodeToUpdate.toggled };
      }
      if (node.children) {
        return { ...node, children: updateTreeData(nodeToUpdate, node.children) };
      }
      return node;
    });
  };

  const handleTreeToggle = (node, toggled) => {
    const resetActiveState = (tree) => tree.map(n => ({
      ...n,
      active: false,
      children: n.children ? resetActiveState(n.children) : n.children
    }));

    const newTree = resetActiveState(treeData);
    const updatedTree = updateTreeData({ ...node, toggled }, newTree);

    setTreeData(updatedTree);
    setSelectedNode(node);
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
        <div style={{ width: '30%', overflow: 'auto', borderRight: '1px solid #ddd', padding: '10px' }}>
          <Treebeard
            data={treeData}
            onToggle={handleTreeToggle}
            style={treeStyle}
          />
        </div>
        <div style={{ flexGrow: 1, height: '100%' }}>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes.map(node => ({
                ...node,
                key: node.id,
                type: node.type,
                data: { ...node.data, updateNodeData },
              }))}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              onConnect={(params) => setEdges((eds) => addEdge(params, eds))}
              fitView
              style={{ backgroundColor: '#B8CEFF' }}
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
