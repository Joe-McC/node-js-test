import React, { useCallback, useEffect } from 'react';
import {useDropzone} from 'react-dropzone';
import { Navbar, Nav, NavDropdown } from 'react-bootstrap';
import { Treebeard } from 'react-treebeard';
import 'bootstrap/dist/css/bootstrap.min.css';

import TextUpdaterNode from './TextUpdaterNode';

import './text-updater-node.css';
import './App.css';

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
  { id: '1', 
    type: 'textUpdater', 
    position: { x: 0, y: 0 }, 
    data: { 
      label: 'New Node',
      id: (1).toString().padStart(3, '0')
    }
  },
  { id: '2', 
    type: 'textUpdater', 
    position: { x: 0, y: 100 }, 
    data: { 
      label: 'New Node',
      id: (2).toString().padStart(3, '0')
    }
  },
];
const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

const nodeTypes = { textUpdater: TextUpdaterNode };

function App() {

  const initialTreeData = {
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

  const nodesToTreeData = (nodes) => {
    return nodes.map((node) => ({
      name: node.data.label,
      id: node.id,
      toggled: true,
      children: [],
    }));
  };

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [treeData, setTreeData] = React.useState(nodesToTreeData(nodes));
  //const treeData = nodesToTreeData(nodes);

  useEffect(() => {
    setTreeData(nodesToTreeData(nodes));
  }, [nodes]);
 
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );
  
  const saveAsFile = () => {
    const filename = window.prompt('Enter the desired filename', 'nodes.json');
    if (filename) {
      const data = JSON.stringify(nodes);
      const blob = new Blob([data], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
  
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const onDrop = useCallback(acceptedFiles => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();
  
      reader.onabort = () => console.log('file reading was aborted');
      reader.onerror = () => console.log('file reading has failed');
      reader.onload = () => {
        const nodes = JSON.parse(reader.result);
        setNodes(nodes);
      }
      reader.readAsText(file);
    });
  }, []);
  
  const {getRootProps, getInputProps} = useDropzone({onDrop});

  const addNode = () => {
    const newNode = {
      id: (nodes.length + 1).toString().padStart(3, '0'),
      type: 'textUpdater',
      position: { x: 0, y: 0 },
      data: { 
        label: 'New Node',
        id: (nodes.length + 1).toString().padStart(3, '0')
      },
    };
    setNodes((nodes) => [...nodes, newNode]);
  };

  return (
    <div className="App">
      <Navbar bg="light" expand="lg">
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="navbar-fixed-top">
            <Navbar.Brand href="#home">React-Bootstrap</Navbar.Brand>
            <NavDropdown title="File" id="basic-nav-dropdown">
              <NavDropdown.Item href="#action/3.1">New</NavDropdown.Item>
              <NavDropdown.Item {...getRootProps()}><input {...getInputProps()} />Open</NavDropdown.Item>
              <NavDropdown.Item href="#action/3.3">Save</NavDropdown.Item>
              <NavDropdown.Item onClick={saveAsFile}>Save As</NavDropdown.Item>
            </NavDropdown>
            <NavDropdown title="Edit" id="basic-nav-dropdown">
              <NavDropdown.Item onClick={addNode}>Add Node</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      <div className="App">  {/* Add this line */}
        <div className="treeview">
          <Treebeard 
            data={treeData} 
            style={treeStyle}
          />
        </div>

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
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>
      <div className="flow">
        <ReactFlow nodes={nodes} edges={initialEdges} style={rfStyle}>
          <MiniMap />
          <Controls />
          <Background />
        </ReactFlow>
      </div>
      </div>
    </div>
  );
}


const treeStyle = {
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

export default App;