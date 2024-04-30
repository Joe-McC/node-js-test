// TreeView.js
import React, { useState } from 'react';
import { Treebeard } from 'react-treebeard';

const TreeView = ({ data }) => {
  const [cursor, setCursor] = useState(false);

  const onToggle = (node, toggled) => {
    if (cursor) {
      cursor.active = false;
    }

    node.active = true;
    if (node.children) {
      node.toggled = toggled;
    }

    setCursor(node);
  };

  return <Treebeard data={data} onToggle={onToggle} />;
};

export default TreeView;

const data = {
    name: 'Root',
    toggled: true,
    children: [
      {
        name: 'Parent',
        children: [
          { name: 'Child 1' },
          { name: 'Child 2' },
        ],
      },
      {
        name: 'Loading parent',
        loading: true,
        children: [],
      },
      {
        name: 'Parent',
        children: [
          {
            name: 'Nested child 1',
            children: [
              { name: 'Nested child 1.1' },
            ],
          },
        ],
      },
    ],
  };