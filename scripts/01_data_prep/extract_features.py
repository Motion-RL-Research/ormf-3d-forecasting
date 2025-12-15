import os
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import tensorflow as tf
import torch
from torch_geometric.data import Data
from waymo_open_dataset.protos import scenario_pb2
from tqdm import tqdm

HISTORY_STEPS = 11
FUTURE_STEPS = 80
DISTANCE_THRESHOLD = 50.0  # meters

AGENT_TYPE_MAP = {0: 'unknown', 1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}
MAP_TYPE_MAP = {0: 'undefined', 1: 'lane', 2: 'road_line', 3: 'road_edge', 
                4: 'stop_sign', 5: 'crosswalk', 6: 'speed_bump'}


def load_tfrecord(filepath: str) -> List[scenario_pb2.Scenario]:
    """Load scenarios from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
    scenarios = []
    
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        scenarios.append(scenario)
    
    return scenarios


def extract_agent_features(track) -> Dict:
    """Extract features for a single agent track."""
    num_steps = len(track.states)
    
    positions = np.zeros((num_steps, 2), dtype=np.float32)
    velocities = np.zeros((num_steps, 2), dtype=np.float32)
    headings = np.zeros(num_steps, dtype=np.float32)
    valid_mask = np.zeros(num_steps, dtype=bool)
    
    for i, state in enumerate(track.states):
        if state.valid:
            positions[i] = [state.center_x, state.center_y]
            velocities[i] = [state.velocity_x, state.velocity_y]
            headings[i] = state.heading
            valid_mask[i] = True
    
    agent_type = AGENT_TYPE_MAP.get(track.object_type, 'unknown')
    
    return {
        'positions': positions,
        'velocities': velocities,
        'headings': headings,
        'valid_mask': valid_mask,
        'agent_type': agent_type,
        'track_id': track.id
    }


def extract_map_features(map_features) -> Dict:
    """Extract HD map features from scenario."""
    lanes = []
    road_edges = []
    crosswalks = []
    
    for feature in map_features:
        points = np.array([[p.x, p.y] for p in feature.polyline], dtype=np.float32)
        
        if len(points) == 0:
            continue
            
        feature_type = MAP_TYPE_MAP.get(feature.type, 'undefined')
        
        if feature_type == 'lane':
            lanes.append(points)
        elif feature_type == 'road_edge':
            road_edges.append(points)
        elif feature_type == 'crosswalk':
            crosswalks.append(points)
    
    return {
        'lane_polylines': lanes,
        'road_edge_polylines': road_edges,
        'crosswalk_polylines': crosswalks
    }


def build_pyg_data(scenario: scenario_pb2.Scenario, include_future: bool = True) -> Data:
    """Convert a WOMD scenario into a PyTorch Geometric Data object."""
    
    # Extract all agent tracks
    all_agents = []
    for track in scenario.tracks:
        agent_data = extract_agent_features(track)
        all_agents.append(agent_data)
    
    # Filter agents valid at current timestep
    current_time = HISTORY_STEPS - 1
    valid_agents = [a for a in all_agents if a['valid_mask'][current_time]]
    
    if len(valid_agents) == 0:
        return None
    
    num_agents = len(valid_agents)
    
    # Build node features
    node_features = []
    future_trajectories = []
    future_masks = []
    
    for agent in valid_agents:
        # Current state
        pos = agent['positions'][current_time]
        vel = agent['velocities'][current_time]
        heading = agent['headings'][current_time]
        
        # One-hot encode agent type
        agent_type_encoding = np.zeros(4, dtype=np.float32)
        type_idx = list(AGENT_TYPE_MAP.values()).index(agent['agent_type'])
        agent_type_encoding[type_idx] = 1.0
        
        # Concatenate features: [x, y, vx, vy, heading, type_0, type_1, type_2, type_3]
        node_feat = np.concatenate([pos, vel, [heading], agent_type_encoding])
        node_features.append(node_feat)
        
        # Ground truth future trajectory
        if include_future:
            future_pos = agent['positions'][HISTORY_STEPS:]
            future_mask = agent['valid_mask'][HISTORY_STEPS:]
            future_trajectories.append(future_pos)
            future_masks.append(future_mask)
    
    node_features = np.array(node_features, dtype=np.float32)
    
    # Build edges (spatial proximity graph)
    edge_index = []
    edge_attr = []
    
    positions = node_features[:, :2]
    
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            
            rel_pos = positions[j] - positions[i]
            distance = np.linalg.norm(rel_pos)
            
            if distance < DISTANCE_THRESHOLD:
                edge_index.append([i, j])
                angle = np.arctan2(rel_pos[1], rel_pos[0])
                edge_attr.append([distance, angle])
    
    if len(edge_index) > 0:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_attr, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 2), dtype=np.float32)
    
    # Extract map features
    map_data = extract_map_features(scenario.map_features)
    
    # Create PyG Data object
    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    
    if include_future:
        data.y = torch.from_numpy(np.array(future_trajectories, dtype=np.float32))
        data.valid_mask = torch.from_numpy(np.array(future_masks, dtype=bool))
    
    data.map_polylines = map_data['lane_polylines']
    data.road_edges = map_data['road_edge_polylines']
    data.scenario_id = scenario.scenario_id
    data.num_agents = num_agents
    
    return data


def process_tfrecord_file(input_path: str, output_dir: str, max_scenes: int = None):
    """Process a single TFRecord file and save PyG Data objects."""
    print(f"\nProcessing: {input_path}")
    print("=" * 70)
    
    scenarios = load_tfrecord(input_path)
    print(f"Loaded {len(scenarios)} scenarios from TFRecord")
    
    if max_scenes:
        scenarios = scenarios[:max_scenes]
        print(f"Limiting to first {max_scenes} scenarios for testing")
    
    processed_data = []
    skipped = 0
    
    for idx, scenario in enumerate(tqdm(scenarios, desc="Converting to PyG")):
        try:
            data = build_pyg_data(scenario, include_future=True)
            
            if data is not None:
                processed_data.append(data)
            else:
                skipped += 1
        
        except Exception as e:
            print(f"\nError processing scenario {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\nSuccessfully processed: {len(processed_data)}")
    print(f"Skipped (empty/invalid): {skipped}")
    
    output_file = Path(output_dir) / f"{Path(input_path).stem}_processed.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(processed_data, output_file)
    print(f"\nSaved to: {output_file}")
    
    if len(processed_data) > 0:
        sample = processed_data[0]
        print(f"\nSample Data Object:")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        print(f"  Node features: {sample.x.shape}")
        print(f"  Future trajectory shape: {sample.y.shape}")
        print(f"  Map polylines: {len(sample.map_polylines)}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from WOMD TFRecords')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .tfrecord file or directory')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Maximum number of scenes to process (for testing)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        process_tfrecord_file(str(input_path), args.output, args.max_scenes)
    
    elif input_path.is_dir():
        tfrecord_files = list(input_path.glob('*.tfrecord*'))
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        for tfrecord_file in tfrecord_files:
            process_tfrecord_file(str(tfrecord_file), args.output, args.max_scenes)
    
    else:
        raise ValueError(f"Input path not found: {input_path}")


if __name__ == '__main__':
    main()