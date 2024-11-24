import React, { useEffect, useState } from 'react';
import { Handle, Position } from 'reactflow';
import axios from 'axios';
import './text-updater-node.css';

const RunModelNode = ({ id, data }) => {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedModel, setSelectedModel] = useState(data.model || '');
  const [selectedDataset, setSelectedDataset] = useState(data.dataset || '');

  // Fetch available models and datasets from the backend
  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/models')
      .then(response => setModels(response.data))
      .catch(error => console.error('Error fetching models:', error));

    axios.get('http://127.0.0.1:5000/api/datasets')
      .then(response => setDatasets(response.data))
      .catch(error => console.error('Error fetching datasets:', error));
  }, []);

  const executeModel = () => {
    if (!selectedModel || !selectedDataset) {
      alert('Please select both a model and a dataset before executing.');
      return;
    }

    axios.post('http://127.0.0.1:5000/api/run_model', {
      nodeId: id,
      model: selectedModel,
      dataset: selectedDataset,
    })
    .then(response => {
      alert(`Execution successful! Result: ${response.data.result}`);
    })
    .catch(error => {
      console.error('Error executing model:', error);
      alert('Error executing model. Please check the backend.');
    });
  };

  return (
    <div className="run-model-node">
      <Handle type="target" position={Position.Top} />
      <div className="node-content">
        <h4>{data.label}</h4>
        <div className="input-group">
          <label>Model:</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            <option value="">Select a model</option>
            {models.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        <div className="input-group">
          <label>Dataset:</label>
          <select value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>
            <option value="">Select a dataset</option>
            {datasets.map((dataset) => (
              <option key={dataset} value={dataset}>{dataset}</option>
            ))}
          </select>
        </div>
        <button onClick={executeModel}>Execute</button>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default RunModelNode;
