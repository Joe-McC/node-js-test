import React, { useState } from 'react';
import Draggable from 'react-draggable';
import { Navbar, Nav, NavDropdown } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const DraggableRectangle = ({ node, onClick }) => (
  <Draggable>
    <div onClick={() => onClick(node)} style={{ border: '1px solid black', width: 100, height: 100, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <input type="text" defaultValue={node.text} />
    </div>
  </Draggable>
);

function App() {
  const [nodes, setNodes] = useState([{ text: "Drag me!" }, { text: "Drag me too!" }]);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [connectors, setConnectors] = useState([]);

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
              <NavDropdown.Item onClick={addConnector}>Add Connector</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      {nodes.map((node, index) => (
        <DraggableRectangle key={index} node={node} onClick={selectNode} />
      ))}
      <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {connectors.map(([from, to], index) => (
          <line
            key={index}
            x1={from.left + 50} y1={from.top + 50}
            x2={to.left + 50} y2={to.top + 50}
            stroke="black"
            style={{ pointerEvents: 'visibleStroke' }}
          />
        ))}
      </svg>
    </div>
  );
}

export default App;