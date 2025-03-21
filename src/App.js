import React, { useCallback, useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { Navbar, Nav, NavDropdown } from 'react-bootstrap';
import { Treebeard } from 'react-treebeard';
import 'bootstrap/dist/css/bootstrap.min.css';

// Import node components
import TextUpdaterNode from './TextUpdaterNode';
import RunModelNode from './RunModelNode';
import RequirementsNode from './RequirementsNode';
import DataPrepNode from './DataPrepNode';
import ModelTrainingNode from './ModelTrainingNode';
import ModelEvalNode from './components/ModelEvalNode';
import ModelTestNode from './components/ModelTestNode';
import ParameterTuningNode from './components/ParameterTuningNode';

import { nodesToTreeData, treeStyle } from './TreeView';

// Import styles
import './styles/text-updater-node.css';
import './styles/run-model-node.css';
import './styles/node-styles.css';
import './styles/App.css';

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

// Helper function to generate unique IDs for new nodes
let id = 0;
const getId = () => `${++id}`;

const rfStyle = {
  backgroundColor: '#B8CEFF',
};

const initialNodes = [];
const initialEdges = [];

// Register all node types
const nodeTypes = {
  textUpdater: TextUpdaterNode,
  runModel: RunModelNode,
  requirements: RequirementsNode,
  dataPrep: DataPrepNode,
  modelTraining: ModelTrainingNode,
  modelEval: ModelEvalNode,
  modelTest: ModelTestNode,
  parameterTuning: ParameterTuningNode
};

function App() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = React.useState(null);
  const [treeData, setTreeData] = React.useState({
    name: 'No nodes available',
    toggled: true,
    children: []
  });
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [currentDataset, setCurrentDataset] = useState(null);
  
  // Make ReactFlow instance available to all components
  useEffect(() => {
    if (reactFlowInstance) {
      window.parentReactFlow = reactFlowInstance;
    }
  }, [reactFlowInstance]);
  
  // Handle dataset configuration from DataPrepNode
  const handleDatasetConfigured = useCallback((datasetInfo) => {
    console.log('App: handleDatasetConfigured called with:', JSON.stringify(datasetInfo));
    
    if (!datasetInfo || !datasetInfo.name) {
      console.error('App: Invalid dataset info received:', datasetInfo);
      return;
    }
    
    // Create a deep copy to ensure state updates are recognized
    const datasetInfoCopy = JSON.parse(JSON.stringify(datasetInfo));
    
    // Add a new timestamp to force recognition as a new update
    datasetInfoCopy.timestamp = new Date().getTime();
    
    // Update global dataset state
    setCurrentDataset(datasetInfoCopy);
    console.log('App: Updated currentDataset state with:', datasetInfoCopy.name);
    
    // Find and update all nodes that need this dataset info
    setNodes((nds) => {
      // Create a new array to trigger React state update
      return nds.map((node) => {
        if (node.type === 'modelTraining' || 
            node.type === 'modelEval' || 
            node.type === 'modelTest' || 
            node.type === 'parameterTuning') {
          
          console.log(`App: Updating ${node.type} node (${node.id}) with dataset: ${datasetInfoCopy.name}`);
          
          // Create a new data object with updated dataset information
          const updatedData = {
            ...node.data,
            datasetInfo: {...datasetInfoCopy},
            dataset: datasetInfoCopy.name,
            datasetName: datasetInfoCopy.name
          };
          
          // Return a new node object with the updated data
          return {
            ...node,
            data: updatedData
          };
        }
        return node;
      });
    });
    
  }, [setNodes]);
  
  // Recalculate tree data whenever nodes or edges change
  useEffect(() => {
    const newTreeData = nodesToTreeData(nodes, edges);
    console.log('Generated Tree Data:', newTreeData); // Check structure
    setTreeData(newTreeData || { name: 'No nodes available', toggled: true, children: [] });
  }, [nodes, edges]);

  // Process data flow between connected nodes
  useEffect(() => {
    // Check for connected nodes and propagate data
    edges.forEach(edge => {
      const sourceNode = nodes.find(node => node.id === edge.source);
      const targetNode = nodes.find(node => node.id === edge.target);
      
      if (sourceNode && targetNode) {
        // If source node has data that target node needs
        if (sourceNode.type === 'requirements' && 
            (targetNode.type === 'modelEval' || targetNode.type === 'modelTest')) {
          // Pass requirements to evaluation/test nodes
          const requirementsData = sourceNode.data.requirements;
          if (requirementsData) {
            setNodes(nodes => nodes.map(node => 
              node.id === targetNode.id 
                ? { ...node, data: { ...node.data, requirementsToCheck: requirementsData } }
                : node
            ));
          }
        }
        
        if (sourceNode.type === 'modelTraining' && targetNode.type === 'modelEval') {
          // Pass trained model info to evaluation node
          if (sourceNode.data.status === 'Trained') {
            setNodes(nodes => nodes.map(node => 
              node.id === targetNode.id 
                ? { 
                    ...node, 
                    data: { 
                      ...node.data, 
                      input: {
                        modelType: sourceNode.data.modelType,
                        framework: sourceNode.data.framework
                      } 
                    } 
                  }
                : node
            ));
          }
        }
        
        if (sourceNode.type === 'modelEval' && 
            (targetNode.type === 'requirements' || targetNode.type === 'parameterTuning')) {
          // Pass evaluation results to requirements or parameter tuning
          if (sourceNode.data.evaluationResults) {
            setNodes(nodes => nodes.map(node => 
              node.id === targetNode.id 
                ? { 
                    ...node, 
                    data: { 
                      ...node.data, 
                      verificationResults: sourceNode.data.evaluationResults 
                    } 
                  }
                : node
            ));
          }
        }
        
        if (sourceNode.type === 'parameterTuning' && targetNode.type === 'modelTraining') {
          // Pass tuned parameters to training
          if (sourceNode.data.bestParameters) {
            setNodes(nodes => nodes.map(node => 
              node.id === targetNode.id 
                ? { 
                    ...node, 
                    data: { 
                      ...node.data, 
                      modelParams: sourceNode.data.bestParameters 
                    } 
                  }
                : node
            ));
          }
        }
        
        if (sourceNode.type === 'dataPrep' && targetNode.type === 'modelTraining') {
          // Pass dataset info to model training
          if (sourceNode.data.isConfigured) {
            setNodes(nodes => nodes.map(node => 
              node.id === targetNode.id 
                ? { 
                    ...node, 
                    data: { 
                      ...node.data, 
                      datasetInfo: {
                        name: sourceNode.data.datasetName,
                        splitRatio: sourceNode.data.splitRatio,
                        stats: sourceNode.data.datasetStats
                      } 
                    } 
                  }
                : node
            ));
          }
        }
      }
    });
  }, [edges, nodes, setNodes]);

  const saveToBackend = () => {
    const filename = window.prompt('Enter filename for saving', 'flow_data.json');
    if (filename) {
      if (nodes.length === 0 && edges.length === 0) {
        alert('No nodes or edges to save.');
        return;
      }
  
      // Prepare nodes data for saving
      const payload = nodes.map((node) => ({
        id: node.id,
        position: node.position,
        type: node.type,
        data: { 
          ...node.data,
          // Remove circular references and functions that can't be serialized
          updateNodeData: undefined
        },
      }));
  
      axios.post('http://127.0.0.1:5000/save_nodes', { filename, nodes: payload, edges })
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
            data: {
              ...node.data,
              id: node.id,
              updateNodeData: undefined // Will be added back when rendered
            },
          }));
  
          setNodes(formattedNodes);
          setEdges(loadedEdges || []);
          alert('Data loaded successfully');
        })
        .catch(error => {
          console.error('Error loading data:', error);
          alert(`Error loading data: ${error.response?.data?.error || 'Please check the backend.'}`);
        });
    }
  };
  
  const addNode = (type = 'textUpdater') => {
    const newId = getId();
    const newNodeData = {
      label: `${getNodeLabel(type)} ${newId}`,
      id: newId,
      // Add callback for dataset configuration
      onDatasetConfigured: handleDatasetConfigured,
      // Add updateNodeData function for all nodes
      onChange: (updatedData) => {
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === newId) {
              return {
                ...node,
                data: {
                  ...updatedData,
                  onDatasetConfigured: handleDatasetConfigured, // Preserve the callback
                  onChange: node.data.onChange, // Preserve the onChange function
                },
              };
            }
            return node;
          })
        );
      },
    };
    
    // If adding a node that should inherit the current dataset
    if (['modelTraining', 'modelEval', 'modelTest', 'parameterTuning'].includes(type) && currentDataset) {
      console.log(`App: Adding new ${type} node with current dataset:`, currentDataset.name);
      newNodeData.datasetInfo = {...currentDataset};
      newNodeData.dataset = currentDataset.name;
      newNodeData.datasetName = currentDataset.name;
    }
    
    const newNode = {
      id: newId,
      type: type,
      position: { x: Math.random() * 250, y: Math.random() * 250 },
      data: newNodeData,
    };
    
    setNodes((nds) => [...nds, newNode]);
  };
  
  const getNodeLabel = (type) => {
    const labels = {
      textUpdater: 'Text Node',
      runModel: 'Run Model',
      requirements: 'Requirements',
      dataPrep: 'Data Preparation',
      modelTraining: 'Model Training',
      modelEval: 'Model Evaluation',
      modelTest: 'Model Testing',
      parameterTuning: 'Parameter Tuning'
    };
    return labels[type] || 'New Node';
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
    if (!tree) return []; // Handle undefined or null children gracefully
  
    return tree.map((node) => {
      if (node.id === nodeToUpdate.id) {
        // This is the node we want to update, toggle it
        return {
          ...node,
          toggled: nodeToUpdate.toggled,
          active: true,
          children: node.children ? updateTreeData(nodeToUpdate, node.children) : []
        };
      }
      
      // For other nodes, reset 'active' but keep 'toggled' state
      // and recursively update any children
      return {
        ...node,
        active: false,
        children: node.children ? updateTreeData(nodeToUpdate, node.children) : []
      };
    });
  };
  
  const handleTreeToggle = (node, toggled) => {
    // Clone the node and update its toggled state
    const updatedNode = { ...node, toggled: toggled };
    
    // Deep copy the tree data to avoid mutation issues
    const updatedTree = JSON.parse(JSON.stringify(treeData));
    
    // Update the root children (keep root toggled true)
    updatedTree.children = updateTreeData(updatedNode, updatedTree.children);
    
    setTreeData(updatedTree);
    
    // Find the corresponding node in the React Flow graph
    const flowNode = nodes.find(n => n.id === node.id);
    if (flowNode) {
      setSelectedNode(flowNode);
    }
  };
  
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      // Create different node data based on type
      let newNode = {
        id: getId(),
        type,
        position,
        data: { onChange: updateNodeData },
      };

      // Initialize node-specific data
      switch(type) {
        case 'modelEval':
          newNode.data = {
            ...newNode.data,
            targetColumn: '',
            referenceData: [],
            currentData: [],
            onResultsUpdate: (results) => updateNodeResults(newNode.id, results)
          };
          break;
        case 'modelTest':
          newNode.data = {
            ...newNode.data,
            modelType: 'random_forest',
            targetColumn: '',
            testData: [],
            features: [],
            onResultsUpdate: (results) => updateNodeResults(newNode.id, results)
          };
          break;
        case 'parameterTuning':
          newNode.data = {
            ...newNode.data,
            modelType: 'random_forest',
            targetColumn: '',
            trainingData: [],
            parameters: {
              n_estimators: 100,
              max_depth: 10,
              min_samples_split: 2
            },
            onResultsUpdate: (results) => updateNodeResults(newNode.id, results)
          };
          break;
        default:
          break;
      }

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const updateNodeResults = useCallback((nodeId, results) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return { 
            ...node, 
            data: { 
              ...node.data, 
              results 
            } 
          };
        }
        return node;
      })
    );
  }, [setNodes]);

  const onNodeDragStop = useCallback((event, node) => {
    // You can add additional logic here when a node stops being dragged
  }, []);

  // Example data loading (for demonstration)
  const loadSampleData = useCallback(() => {
    // Sample data for demonstration
    const sampleDataRef = [
      { feature1: 1, feature2: 2, target: 0 },
      { feature1: 2, feature2: 3, target: 1 },
      { feature1: 3, feature2: 4, target: 0 },
      { feature1: 4, feature2: 5, target: 1 },
      { feature1: 5, feature2: 6, target: 0 }
    ];
    
    const sampleDataCurrent = [
      { feature1: 1.1, feature2: 2.2, target: 0 },
      { feature1: 2.1, feature2: 3.1, target: 1 },
      { feature1: 3.3, feature2: 4.2, target: 1 },
      { feature1: 4.2, feature2: 5.1, target: 1 },
      { feature1: 5.1, feature2: 6.2, target: 0 }
    ];

    // Create a complete workflow with all node types
    const newNodes = [
      {
        id: '1',
        type: 'requirements',
        position: { x: 250, y: 50 },
        data: {
          requirements: [
            { name: 'Accuracy', threshold: 0.85, description: 'Model must achieve at least 85% accuracy' },
            { name: 'Latency', threshold: 120, description: 'Model inference time must be under 120ms' },
            { name: 'F1 Score', threshold: 0.8, description: 'F1 score must be at least 0.8' }
          ]
        }
      },
      {
        id: '2',
        type: 'dataPrep',
        position: { x: 250, y: 200 },
        data: {
          datasetName: 'iris',
          splitRatio: 0.75,
          datasetStats: {
            rows: 150,
            features: 4,
            missingValues: 0,
            target: 'species'
          },
          preprocessingSteps: ['normalization', 'encoding'],
          isConfigured: true,
          onDatasetConfigured: handleDatasetConfigured
        }
      },
      {
        id: '3',
        type: 'modelTraining',
        position: { x: 250, y: 350 },
        data: {
          modelType: 'random_forest',
          framework: 'scikit-learn',
          dataset: 'iris',
          datasetInfo: {
            name: 'iris',
            splitRatio: 0.75,
            stats: {
              rows: 150,
              features: 4,
              missingValues: 0,
              target: 'species'
            }
          },
          modelParams: {
            n_estimators: 100,
            max_depth: 10,
            min_samples_split: 2
          },
          status: 'Not Started',
          trainingLogs: [],
          trainingMetrics: null
        }
      },
      {
        id: '4',
        type: 'parameterTuning',
        position: { x: 550, y: 200 },
        data: {
          modelType: 'random_forest',
          targetColumn: 'target',
          trainingData: [...sampleDataRef, ...sampleDataCurrent],
          parameters: {
            n_estimators: { min: 50, max: 200, step: 50 },
            max_depth: { min: 3, max: 12, step: 3 },
            min_samples_split: { min: 2, max: 10, step: 2 }
          },
          bestParameters: {
            n_estimators: 150,
            max_depth: 9,
            min_samples_split: 4
          },
          results: {
            score: 0.94,
            crossValScore: 0.91
          }
        }
      },
      {
        id: '5',
        type: 'modelEval',
        position: { x: 550, y: 350 },
        data: {
          referenceData: sampleDataRef,
          currentData: sampleDataCurrent,
          targetColumn: 'target',
          metricsToCheck: ['accuracy', 'precision', 'recall', 'f1'],
          input: {
            modelType: 'random_forest',
            framework: 'scikit-learn'
          },
          evaluationResults: [
            { name: 'Accuracy', value: 0.88 },
            { name: 'F1 Score', value: 0.85 },
            { name: 'Latency', value: 95 }
          ],
          results: {
            dataQualityMetrics: {
              missingValues: 0,
              duplicateRows: 0,
              rowCount: { reference: 5, current: 5 }
            },
            driftMetrics: {
              feature1: { drift: 'low', pValue: 0.8 },
              feature2: { drift: 'low', pValue: 0.75 }
            },
            performanceMetrics: {
              accuracy: 0.88,
              precision: 0.85,
              recall: 0.87,
              f1: 0.86
            }
          }
        }
      },
      {
        id: '6',
        type: 'modelTest',
        position: { x: 550, y: 50 },
        data: {
          modelType: 'random_forest',
          targetColumn: 'species',
          testData: [...sampleDataRef, ...sampleDataCurrent],
          testDataset: 'iris',
          testSet: [
            { id: 1, name: 'DataDriftTest', description: 'Test if data drift is present', passed: true },
            { id: 2, name: 'ModelPerformanceTest', description: 'Test model accuracy metrics against benchmarks', passed: true },
            { id: 3, name: 'DataQualityTest', description: 'Test if data quality meets requirements', passed: true }
          ],
          testStatus: 'All Tests Passed',
          requirementsStatus: { passed: 3, failed: 0, total: 3 },
          results: {
            summary: { passed: 3, failed: 0, total: 3 },
            testResults: [
              { name: 'DataDriftTest', passed: true, details: 'No significant drift detected' },
              { name: 'ModelPerformanceTest', passed: true, details: 'Performance meets benchmarks' },
              { name: 'DataQualityTest', passed: true, details: 'Data quality is acceptable' }
            ],
            metrics: {
              accuracy: 0.88,
              precision: 0.85,
              recall: 0.87,
              f1: 0.86
            }
          }
        }
      }
    ];

    const newEdges = [
      { id: 'e1-2', source: '1', target: '2' },
      { id: 'e2-3', source: '2', target: '3' },
      { id: 'e3-5', source: '3', target: '5' },
      { id: 'e5-6', source: '5', target: '6' },
      { id: 'e4-3', source: '4', target: '3' },
      { id: 'e1-6', source: '1', target: '6' }
    ];

    // Set the new nodes and edges
    setNodes(newNodes.map(node => ({
      ...node,
      data: { ...node.data, updateNodeData }
    })));
    setEdges(newEdges);
    
  }, [setNodes, setEdges]);

  // We create the sidebar items that users can drag onto the canvas
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="App">
      <Navbar bg="light" expand="lg">
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="navbar-fixed-top">
            <Navbar.Brand href="#home">AI Model Verification Tool</Navbar.Brand>
            <NavDropdown title="File" id="file-nav-dropdown">
              <NavDropdown.Item onClick={saveToBackend}>Save Workflow</NavDropdown.Item>
              <NavDropdown.Item onClick={loadFromBackend}>Load Workflow</NavDropdown.Item>
            </NavDropdown>
            <NavDropdown title="Edit" id="edit-nav-dropdown">
              <NavDropdown title="Add Node" id="add-node-nav-dropdown" drop="end">
                <NavDropdown.Item onClick={() => addNode('requirements')}>Add Requirements Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('dataPrep')}>Add Data Preparation Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('modelTraining')}>Add Model Training Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('modelEval')}>Add Model Evaluation Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('modelTest')}>Add Model Testing Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('parameterTuning')}>Add Parameter Tuning Node</NavDropdown.Item>
                <NavDropdown.Divider />
                <NavDropdown.Item onClick={() => addNode('textUpdater')}>Add Text Node</NavDropdown.Item>
                <NavDropdown.Item onClick={() => addNode('runModel')}>Add Run Model Node</NavDropdown.Item>
              </NavDropdown>
              <NavDropdown.Item onClick={removeNode}>Remove Node</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>
      <div className="main-content">
        <div className="treeview-container">
          <Treebeard
            data={treeData}
            onToggle={handleTreeToggle}
            style={treeStyle}
          />
        </div>
        <div className="flow-container">
          <ReactFlowProvider>
            <aside>
              <div className="description">Drag these nodes onto the canvas to build your ML workflow.</div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'requirements')}
                draggable
              >
                Requirements
              </div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'dataPrep')}
                draggable
              >
                Data Preparation
              </div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'modelTraining')}
                draggable
              >
                Model Training
              </div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'modelEval')}
                draggable
              >
                Model Evaluation
              </div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'modelTest')}
                draggable
              >
                Model Testing
              </div>
              
              <div
                className="dndnode"
                onDragStart={(event) => onDragStart(event, 'parameterTuning')}
                draggable
              >
                Parameter Tuning
              </div>
              
              <button onClick={loadSampleData} className="load-data-button">
                Load Sample Data
              </button>
            </aside>
            
            <div className="reactflow-wrapper" ref={reactFlowWrapper}>
              <ReactFlow
                nodes={nodes.map((node) => ({
                  ...node,
                  key: node.id,
                  type: node.type,
                  data: { ...node.data, updateNodeData },
                }))}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                onConnect={onConnect}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onNodeDragStop={onNodeDragStop}
                fitView
                maxZoom={2}
                minZoom={0.2}
                defaultZoom={0.7}
                fitViewOptions={{ 
                  padding: 0.2,
                  maxZoom: 1
                }}
                style={{ backgroundColor: '#ffffff' }}
              >
                <Controls showZoom={true} showFitView={true} />
                <MiniMap />
                <Background color="#f0f0f0" variant="dots" gap={12} size={1} />
              </ReactFlow>
            </div>
          </ReactFlowProvider>
        </div>
      </div>
    </div>
  );
}

export default App;
