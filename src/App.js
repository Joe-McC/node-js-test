import React, { useState, useCallback } from 'react';
import { Navbar, Nav, NavDropdown } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

//import Node from './Node';
import TextUpdaterNode from './TextUpdaterNode';

import './text-updater-node.css';

import ReactFlow, {
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
  { id: '1', type: 'textUpdater', position: { x: 0, y: 0 }, data: { label: '1' } },
  { id: '2', type: 'textUpdater', position: { x: 0, y: 100 }, data: { label: '2' } },
];
const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

const nodeTypes = { textUpdater: TextUpdaterNode };

function App() {
  /*const [nodes, setNodes] = useState([{ text: "Drag me!" }, { text: "Drag me too!" }]);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [connectors, setConnectors] = useState([]);
  */

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
 
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const addNode = () => {
    const newNode = {
      id: (nodes.length + 1).toString(),
      type: 'textUpdater',
      position: { x: 0, y: 0 },
      data: { label: 'New Node' },
    };
    setNodes((nodes) => [...nodes, newNode]);
  };


  /*const addConnector = () => {
    if (nodes.length > 1) {
      const source = nodes[0].id;
      const target = nodes[1].id;
      const newEdge = { id: `${source}-${target}`, source, target };
      setEdges((edges) => [...edges, newEdge]);
    }
  };
  // create a function to select a node
  const selectNode = (node) => {
    console.log(node);
  };*/


/*
  const addNode = () => {
    setNodes([...nodes, { text: "New Node" }]);
  };

  const selectNode = (node) => {
    setSelectedNodes([node, ...selectedNodes].slice(0, 2));
  };

  const addConnector = () => {
    if (selectedNodes.length === 2) {
      setConnectors([...connectors, selectedNodes]);
      setSelectedNodes([]);
    }
  };
*/
  return (
    <div className="App">
      <Navbar bg="light" expand="lg">
        <Navbar.Brand href="#home">React-Bootstrap</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="mr-auto">
            <NavDropdown title="File" id="basic-nav-dropdown">
              <NavDropdown.Item href="#action/3.1">New</NavDropdown.Item>
              <NavDropdown.Item href="#action/3.2">Open</NavDropdown.Item>
              <NavDropdown.Item href="#action/3.3">Save</NavDropdown.Item>
            </NavDropdown>
            <NavDropdown title="Edit" id="basic-nav-dropdown">
              <NavDropdown.Item onClick={addNode}>Add Node</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      <div style={{ width: '100vw', height: '100vh' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          style={rfStyle}
          //nodeTypes={{ editable: Node }} // The node.js type is causing the nodes to appear differntly, without the reactflow ports
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>
    </div>
  );
}

export default App;