import torch
from torch_geometric.data import Data
import osmnx as ox
import networkx as nx
import numpy as np
import geopandas as gpd
import momepy as mp 
from shapely.strtree import STRtree
import pickle
from tqdm import tqdm
import os, glob
from collections import Counter

def get_city_graph(lat, lon, dist, features, expand_features):
    g = ox.graph_from_point(
        (lat, lon),
        dist=dist, 
        network_type='bike', 
        simplify=True, 
        retain_all=False,
        )
    feat_dict = {i : True for i in features}
    amenities = ox.features.features_from_point((lat, lon), tags=feat_dict, dist=dist)
    amenities = amenities[amenities.geometry.notnull()]
    amenities['new_col'] = np.nan
    print('Number of amenities found:', len(amenities))

    for feat in features:
        if feat not in expand_features:
            amenities.loc[amenities[feat].notnull(), 'new_col'] = feat
    
    amenities['amenity'] = amenities['new_col']

    for feat in expand_features:
        amenities['amenity'].fillna(amenities[feat], inplace=True)
    amenities = amenities[amenities['amenity'].notnull()]

    gdf = mp.nx_to_gdf(g, points=False, lines=True, spatial_weights=True).to_crs(epsg=3857)
    gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)
    # ensure amenities and gdf have the same CRS
    amenities = gpd.GeoDataFrame(amenities, geometry='geometry', crs='EPSG:4326')
    amenities = amenities.to_crs(epsg=3857)
    # ensure gdf has the same CRS as amenities
    assert gdf.crs == amenities.crs, "CRS mismatch between gdf and amenities"
    # convert amenties geometry to centroid if it is a polygon
    amenities['geometry'] = amenities['geometry'].apply(
        lambda x: x.centroid if x.geom_type == 'Polygon' else x
    )
    # remove all amenities that are not within the gdf geometry
    xmin, ymin, xmax, ymax = gdf.total_bounds
    amenities = amenities.cx[xmin:xmax, ymin:ymax]
    print(f"Number of amenities after filtering: {len(amenities)}")
    amenities = amenities.to_crs(epsg=4326)
    return g, gdf, amenities

def create_linegraph(g):
    # g = ox.convert.to_digraph(g)
    # g = ox.convert.to_undirected(g)
    g = nx.Graph(g)
    H = nx.line_graph(g)
    H.add_nodes_from((node, g.edges[node]) for node in H)   
    for s, t in H.edges:
        H.edges[s, t]['weight'] = g.edges[s]['length'] + g.edges[t]['length']
    return H

def assign_features_to_nodes(H, amenities_gdf, geometry_col='geometry', amenity_col='amenity'):
    import tqdm
    """
    Assigns amenities from a GeoDataFrame to the nearest graph node (based on geometry),
    then stores frequency counts of amenity types as node attributes.

    Parameters:
    - H: networkx.Graph with 'geometry' in each node.
    - amenities_gdf: GeoDataFrame with amenity information and geometry.
    - geometry_col: name of the column in the GeoDataFrame containing geometry.
    - amenity_col: name of the column with amenity types.
    """

    # Extract nodes with geometry
    nodes_with_geom = [(node, data.get('geometry')) for node, data in H.nodes(data=True) if data.get('geometry') is not None]
    if not nodes_with_geom:
        raise ValueError("No nodes with 'geometry' found in the graph.")

    node_ids, linestrings = zip(*nodes_with_geom)

    # Ensure point geometries for amenity data
    amenities_gdf = amenities_gdf.copy()
    amenities_gdf[geometry_col] = amenities_gdf[geometry_col].apply(
        lambda x: x.centroid if x.geom_type == 'Polygon' else x
    )

    # Build spatial index
    tree = STRtree(linestrings)

    # Assign amenities to nearest nodes
    print('Now assigning amenities to nodes')
    for geom, amenity in tqdm.tqdm(zip(amenities_gdf[geometry_col], amenities_gdf[amenity_col]), total=len(amenities_gdf)):
        nearest_idx = tree.nearest(geom)
        nearest_node = node_ids[nearest_idx]
        H.nodes[nearest_node].setdefault('amenity', []).append(amenity)

    # Convert amenity lists to frequency counts
    print('Now converting amenity lists to frequency counts')
    for node, data in tqdm.tqdm(H.nodes(data=True), total=H.number_of_nodes()):
        if 'amenity' in data:
            for amenity_type, count in Counter(data['amenity']).items():
                H.nodes[node][amenity_type] = count
            H.nodes[node].pop('amenity', None)
    
    print('Now assigning AADT values to nodes without AADT')
    for node, value in tqdm.tqdm(H.nodes(data=True)):
        if 'aadt' not in value.keys():
            value['aadt'] = 0

def calc_bc(graph):
    ebc = dict(nx.all_pairs_dijkstra_path(graph,
                                    weight='weight',
                                    cutoff=1000,))
    bc = {i: 0 for i in graph.nodes}
    for node in tqdm(graph.nodes):
        for path in ebc[node].values():
            for node_visited in set(path):
                bc[node_visited] += 1
    total_nodes = graph.number_of_nodes() ** 2
    return {node: count / total_nodes for node, count in bc.items()}

def load_aadt(filepath, g, gdf):
    nodes, edges = mp.nx_to_gdf(g)
    gdf2 = gpd.GeoDataFrame.from_file(filepath)
    gdf2.set_crs(epsg=4326, inplace=True)
    gdf2 = gdf2.to_crs(epsg=3857)
    gdf2['geometry'] = gdf2['geometry']
    # gdf2 = gdf2[gdf2['geometry'].within(gdf['geometry'])]
    ### export only relevant columns
    gdf_new = gdf2[['id', 'vejnavn', 'geometry', 'aadt_cykler']]
    ### remove null values on aadt_cykler
    gdf_new = gdf_new[gdf_new['aadt_cykler'].notnull()]
    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf_new = gdf_new.cx[xmin:xmax, ymin:ymax]
    gdf_new.to_crs(epsg=4326, inplace=True)
    return gdf_new

def assign_aadt_to_graph_edges(g, gdf_new, H, aadt_col='aadt_cykler'):
    """
    Assigns AADT values from gdf_new to the nearest edge in graph H based on proximity.

    Parameters:
    - g: networkx graph with edge geometries.
    - gdf_new: GeoDataFrame containing points and AADT values.
    - H: networkx graph where AADT attributes will be assigned.
    - aadt_col: column name in gdf_new containing the AADT values.
    """

    edges_data = list(g.edges(data=True))
    linestrings = [attr['geometry'] if 'geometry' in attr else None for _, _, attr in edges_data]
    from_node = [u for u, _, _ in edges_data]
    to_node = [v for _, v, _ in edges_data]

    tree = STRtree(linestrings)
    over_writes = 0

    for i, row in tqdm(gdf_new.iterrows(), total=len(gdf_new)):
        point = row['geometry']
        if point is None:
            continue

        nearest_edge_idx = tree.nearest(point)
        nearest_edge = linestrings[nearest_edge_idx]
        nearest_edge_distance = nearest_edge.distance(point)

        start_node = from_node[nearest_edge_idx]
        end_node = to_node[nearest_edge_idx]

        # Ensure the edge exists in H
        if (start_node, end_node) not in H.nodes():
            if (end_node, start_node) not in H.nodes():
                continue
            else:
                start_node, end_node = end_node, start_node

        node_pair = (start_node, end_node)
        # Initialize or update AADT attributes if closer
        if 'aadt' not in H.nodes()[node_pair]:
            H.nodes()[node_pair]['aadt'] = row[aadt_col]
            H.nodes()[node_pair]['aadt_distance'] = nearest_edge_distance
        elif H.nodes()[node_pair]['aadt_distance'] > nearest_edge_distance:
            over_writes += 1
            H.nodes()[node_pair]['aadt'] = row[aadt_col]
            H.nodes()[node_pair]['aadt_distance'] = nearest_edge_distance
    print(f"How many times did we overwrite and AADT Value? {over_writes}")
    return H

def clean_and_standardize_node_features(H, remove_fields=None):
    """
    Cleans and standardizes node attributes in a graph.
    Removes specified fields, ensures all attributes are numeric floats, and fills missing features with 0.

    Parameters:
    - H: networkx graph.
    - remove_fields: list of fields to remove (default common OSM fields).
    
    Returns:
    - all_feats: list of all standardized features across nodes.
    """
    if remove_fields is None:
        remove_fields = ['geometry', 'name', 'highway', 'ref', 'aadt_dist', 'aadt_distance']

    # Clean node attributes and convert to floats where possible
    for node, data in H.nodes(data=True):
        for field in remove_fields:
            data.pop(field, None)

        # Convert to float if possible, else remove
        for key in list(data.keys()):
            if not isinstance(data[key], (int, float)):
                try:
                    data[key] = float(data[key])
                except (ValueError, TypeError):
                    data.pop(key, None)

    # Gather all unique features across all nodes
    all_feats = set()
    for _, data in H.nodes(data=True):
        all_feats.update(data.keys())
    all_feats = list(all_feats)

    # Fill missing features with 0
    for _, data in H.nodes(data=True):
        for feat in all_feats:
            data.setdefault(feat, 0)

    return all_feats

def graph_to_linegraph_data(H, all_feats, target_feat='aadt', osmid_feat='osmid'):
    """
    Converts a networkx graph H with node and edge attributes into a PyTorch Geometric Data object.
    
    Parameters:
    - H: networkx graph with node features.
    - all_feats: list of feature names to extract from nodes.
    - target_feat: feature to use as the target variable (default 'aadt').
    - osmid_feat: feature to use as osmid identifier (default 'osmid').
    
    Returns:
    - PyTorch Geometric Data object with node features, targets, osmid, and edge index.
    """
    node_list, x, y, osmid_list = [], [], [], []

    node_feat_names = [i for i in all_feats if i not in [target_feat, osmid_feat]]
    for node, feats in H.nodes(data=True):
        node_list.append(node)
        x.append([feats.get(feat, 0.0) for feat in all_feats if feat not in [target_feat, osmid_feat]])
        y.append(feats[target_feat])
        osmid_list.append(feats[osmid_feat])

    node_idx = {node: idx for idx, node in enumerate(node_list)}

    edge_index = [[node_idx[s], node_idx[t]] for s, t in H.edges()]

    data = Data()
    data.num_nodes = len(node_list)
    data.x = torch.tensor(x, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.float)
    data.osmid = torch.tensor(osmid_list, dtype=torch.long)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    # data.H = H  # Optional: Attach original H graph if needed

    return data, node_feat_names

import os
import glob
import pickle
import re

def save_graph_with_config(
    linegraph, 
    H, 
    g,
    features, 
    expand_features, 
    dist, 
    base_path='../data/graphs'
):
    """
    Save graph data, networkx graph, and node features into a structured folder.
    Uses config files to check for existing setups, creates new folders if needed.

    Parameters:
    - linegraph: PyTorch Geometric Data object.
    - H: NetworkX graph object.
    - features: list of features.
    - expand_features: list of expanded features.
    - dist: distance parameter.
    - base_path: base path to store the graphs and configs.
    
    Returns:
    - num_folder (str): The assigned folder number where data was saved.
    """

    config_folder = glob.glob(f'{base_path}/configs/*.txt')

    def config_matches(file_path):
        with open(file_path, 'r') as f:
            config = f.readlines()

        config_dict = {}
        for line in config:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'distance':
                config_dict['distance'] = int(value)
            else:
                config_dict[key] = sorted(value.split())

        match_features = sorted(features) == sorted(config_dict.get('features', []))
        match_expand = sorted(expand_features) == sorted(config_dict.get('expand_features', []))
        match_dist = dist == config_dict.get('distance', None)
        return match_features and match_expand and match_dist

    # Determine folder number
    if not config_folder:
        print('Creating initial folder structure...')
        num_folder = '1'
    else:
        num_folder = None
        for file in config_folder:
            if config_matches(file):
                num_folder = os.path.splitext(os.path.basename(file))[0]
                break
        if not num_folder:
            # Check existing graph folders to determine max folder number
            subdirs = [
                int(name) for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.isdigit()
            ]
            next_folder = max(subdirs) + 1 if subdirs else 1
            num_folder = str(next_folder)

    # Create necessary folders and save configs
    os.makedirs(f'{base_path}/{num_folder}/models', exist_ok=True)
    with open(f'{base_path}/configs/{num_folder}.txt', 'w') as f:
        f.write(f"features: {' '.join(sorted(features))}\n")
        f.write(f"expand_features: {' '.join(sorted(expand_features))}\n")
        f.write(f"distance: {dist}\n")

    # Save data
    with open(f'{base_path}/{num_folder}/linegraph_tg.pkl', 'wb') as f:
        pickle.dump(linegraph, f)

    with open(f'{base_path}/{num_folder}/linegraph_nx.pkl', 'wb') as f:
        pickle.dump(H, f)
    
    with open(f'{base_path}/{num_folder}/original_graph_nx.pkl', 'wb') as f:
            pickle.dump(g, f)
    # node_features.to_csv(f'{base_path}/{num_folder}/node_features.csv', index=False)

    print(f"Graph and data saved in folder {num_folder}")
    return num_folder


def build_value_to_column_dict():
    import osmnx as ox
    import json
    import os

    # Step 1: Define parameters
    lat, lon = 55.6867243, 12.5700724
    dist = 10000  # in meters
    features = 'aerialway aeroway amenity barrier boundary building craft emergency geological healthcare highway historic landuse leisure man_made military natural office place power public_transport railway route service shop telecom tourism water waterway'.split()

    json_path = 'osm_value_to_column.json'

    # Step 2: Load or Build value-to-column dictionary
    if os.path.exists(json_path):
        print(f"📂 Found existing '{json_path}', loading dictionary...")
        with open(json_path, 'r') as f:
            value_to_column = json.load(f)
        print(f"✅ Loaded {len(value_to_column)} value-to-column pairs from '{json_path}'.")
    else:
        print("📥 No existing dictionary found. Downloading OSM features...")

        # Download features
        tags = {feat: True for feat in features}
        amenities = ox.features.features_from_point((lat, lon), tags=tags, dist=dist)

        print(f"✅ Downloaded {len(amenities)} OSM features.")

        # Build dictionary
        value_to_column = {}

        for feature in features:
            if feature in amenities.columns:
                unique_values = amenities[feature].dropna().unique()
                for value in unique_values:
                    value = str(value).strip()
                    if value and value.lower() != 'nan':
                        value_to_column[value] = feature

        print(f"✅ Collected {len(value_to_column)} value-to-column pairs.")

        # Save it
        with open(json_path, 'w') as f:
            json.dump(value_to_column, f, indent=2)
        print(f"✅ Saved new value-to-column dictionary to '{json_path}'.")
    return value_to_column