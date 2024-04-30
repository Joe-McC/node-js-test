// Node.js
import React from 'react';
import { useDrag } from 'react-dnd';

const Node = ({ id, left, top }) => {
  const [{ isDragging }, drag] = useDrag({
    item: { id, left, top, type: 'node' },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  return (
    <div ref={drag} style={{ left, top }}>
      Node {id}
    </div>
  );
};

export default Node;