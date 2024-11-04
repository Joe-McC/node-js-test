import React, { useState, useEffect, useCallback } from 'react';
import { Handle, Position } from 'reactflow';

const TextUpdaterNode = React.memo(({ data, isConnectable }) => {
  const [text, setText] = useState(data.text || '');

  useEffect(() => {
    setText(data.text || '');
  }, [data.text]);

  const handleChange = (event) => {
    setText(event.target.value);
    data.updateNodeData(data.id, { text: event.target.value });
  };

  return (
    <div className="text-updater-node">
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div>
        <label>Node ID:</label>
        <span>{data.id}</span>
        <label htmlFor="text">Text:</label>
        <input
          id="text"
          name="text"
          value={text}
          onChange={handleChange}
          className="nodrag"
        />
      </div>
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  );
});

export default TextUpdaterNode;
