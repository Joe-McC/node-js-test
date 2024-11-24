from flask import Flask, request, jsonify
import json
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for node data (this can be replaced with a database)
node_storage = {}

@app.route('/save_nodes', methods=['POST'])
def save_nodes():
    try:
        data = request.json
        filename = data.get("filename", "default_nodes.json")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not filename or not nodes:
            return jsonify({"error": "Filename or nodes data missing."}), 400

        with open(filename, 'w') as f:
            json.dump({"nodes": nodes, "edges": edges}, f)

        return jsonify({"message": "Nodes saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/load_nodes', methods=['GET'])
def load_nodes():
    try:
        filename = request.args.get("filename", "default_nodes.json")
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Ensure description field exists
        for node in data["nodes"]:
            if "description" not in node["data"]:
                node["data"]["description"] = ""

        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": f"File '{filename}' not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


models = ['Model1', 'Model2', 'Model3']  # Replace with actual models
datasets = ['Dataset1', 'Dataset2', 'Dataset3']  # Replace with actual datasets

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(models)

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify(datasets)

@app.route('/api/run_model', methods=['POST'])
def run_model():
    try:
        data = request.json
        model = data.get('model')
        dataset = data.get('dataset')
        node_id = data.get('nodeId')

        if not model or not dataset:
            return jsonify({'error': 'Model and dataset are required.'}), 400

        # Example execution logic (replace with your neural network logic)
        result = f"Model {model} executed on dataset {dataset} for node {node_id}."

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
