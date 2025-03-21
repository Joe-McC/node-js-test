import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import './styles/node-styles.css';

function RequirementsNode({ data, isConnectable }) {
  const [requirements, setRequirements] = useState(data.requirements || [
    { name: 'Accuracy', threshold: 0.9, description: 'Model must achieve at least 90% accuracy' },
    { name: 'Latency', threshold: 100, description: 'Model inference time must be under 100ms' }
  ]);
  const [newRequirement, setNewRequirement] = useState({ name: '', threshold: '', description: '' });
  const [verificationResults, setVerificationResults] = useState(data.verificationResults || null);

  // Update parent data when requirements change
  useEffect(() => {
    if (data.onChange) {
      data.onChange({
        ...data,
        requirements,
        verificationResults
      });
    }
  }, [data, requirements, verificationResults]);

  // Update verification results if they're passed in from another node
  useEffect(() => {
    if (data.verificationResults && data.verificationResults !== verificationResults) {
      setVerificationResults(data.verificationResults);
    }
  }, [data.verificationResults, verificationResults]);

  const addRequirement = () => {
    if (newRequirement.name && newRequirement.threshold) {
      setRequirements([...requirements, { ...newRequirement }]);
      setNewRequirement({ name: '', threshold: '', description: '' });
    }
  };

  const removeRequirement = (index) => {
    const updatedRequirements = [...requirements];
    updatedRequirements.splice(index, 1);
    setRequirements(updatedRequirements);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewRequirement({ ...newRequirement, [name]: value });
  };

  const checkRequirementStatus = (name) => {
    if (!verificationResults) return 'unknown';
    
    const result = verificationResults.find(metric => metric.name === name);
    if (!result) return 'unknown';
    
    if (name === 'Latency') {
      return result.value <= requirements.find(r => r.name === name)?.threshold ? 'met' : 'not-met';
    }
    
    return result.value >= requirements.find(r => r.name === name)?.threshold ? 'met' : 'not-met';
  };

  return (
    <div className="ml-node requirements-node">
      <div className="node-header" style={{ backgroundColor: '#00b894' }}>
        <div>Requirements</div>
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: '#555', width: '8px', height: '8px' }}
          isConnectable={isConnectable}
        />
        <Handle
          type="source"
          position={Position.Right}
          style={{ background: '#555', width: '8px', height: '8px' }}
          isConnectable={isConnectable}
        />
      </div>
      
      <div className="node-content">
        <h5>Model Requirements</h5>
        
        <div className="requirements-container">
          <table className="requirements-table">
            <thead>
              <tr>
                <th>Requirement</th>
                <th>Threshold</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {requirements.map((req, index) => (
                <tr key={index} className={`requirement-${checkRequirementStatus(req.name)}`}>
                  <td title={req.description}>{req.name}</td>
                  <td>{req.threshold}{req.name === 'Latency' ? 'ms' : ''}</td>
                  <td>
                    {checkRequirementStatus(req.name) === 'met' && '✅'}
                    {checkRequirementStatus(req.name) === 'not-met' && '❌'}
                    {checkRequirementStatus(req.name) === 'unknown' && '❓'}
                  </td>
                  <td>
                    <button 
                      className="small-button"
                      onClick={() => removeRequirement(index)}
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="add-requirement">
          <h5>Add New Requirement</h5>
          <div className="input-group">
            <label>Name</label>
            <input
              type="text"
              name="name"
              placeholder="E.g., Accuracy, F1 Score"
              value={newRequirement.name}
              onChange={handleInputChange}
            />
          </div>
          <div className="input-group">
            <label>Threshold</label>
            <input
              type="number"
              name="threshold"
              placeholder="E.g., 0.9 for accuracy"
              value={newRequirement.threshold}
              onChange={handleInputChange}
            />
          </div>
          <div className="input-group">
            <label>Description</label>
            <input
              type="text"
              name="description"
              placeholder="Optional description"
              value={newRequirement.description}
              onChange={handleInputChange}
            />
          </div>
          <button 
            className="ml-node-button"
            onClick={addRequirement}
            disabled={!newRequirement.name || !newRequirement.threshold}
          >
            Add Requirement
          </button>
        </div>
        
        {verificationResults && (
          <div className="verification-results">
            <h5>Verification Results</h5>
            <div className="results-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {verificationResults.map((result, index) => {
                    const req = requirements.find(r => r.name === result.name);
                    const passes = req 
                      ? (result.name === 'Latency' ? result.value <= req.threshold : result.value >= req.threshold)
                      : null;
                    
                    return req ? (
                      <tr key={index} className={passes ? 'passed' : 'failed'}>
                        <td>{result.name}</td>
                        <td>{typeof result.value === 'number' ? result.value.toFixed(3) : result.value}</td>
                        <td>{req.threshold}</td>
                        <td>{passes ? '✓' : '✗'}</td>
                      </tr>
                    ) : null;
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RequirementsNode; 