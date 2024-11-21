import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';

const TextUpdaterNode = React.memo(({ data, isConnectable }) => {
  const [text, setText] = useState(data.text || '');
  const [description, setDescription] = useState(data.description || '');

  useEffect(() => {
    setText(data.text || '');
    setDescription(data.description || '');
  }, [data.text, data.description]);

  const handleTextChange = (event) => {
    setText(event.target.value);
    data.updateNodeData(data.id, { text: event.target.value });
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
    data.updateNodeData(data.id, { description: event.target.value });
  };

  return (
    <div className="text-updater-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="content-container">
        <div className="input-group">
          <label>Node ID:</label>
          <input
            id="text"
            name="text"
            value={text}
            onChange={handleTextChange}
            className="nodrag"
          />
        </div>
        <div className="input-group">
          <label htmlFor="description">Description:</label>
          <input
            id="description"
            name="description"
            value={description}
            onChange={handleDescriptionChange}
            className="nodrag"
          />
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
});

export default TextUpdaterNode;
