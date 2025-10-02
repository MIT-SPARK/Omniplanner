#!/usr/bin/env python3
"""
Multi-Robot Fast Downward test with real scene graph
Follows the exact format you wanted using existing infrastructure
"""

import numpy as np
import sys
import os
import tempfile
import subprocess
import time


multi_robot_path = '/home/jaeyoun-choi/colcon_ws/src/awesome_dcist_t4/omniplanner/omniplanner/src'
if multi_robot_path not in sys.path:
    sys.path.insert(0, multi_robot_path)

from importlib.resources import as_file, files
import dsg_pddl.domains
import spark_dsg
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from dsg_pddl.dsg_pddl_grounding_Multirobot_Better import generate_multirobot_inspection_pddl, generate_multirobot_region_pddl

def extract_plan_from_wrapper(plan):
    """Extract the actual plan from OmniPlanner wrappers"""
    # Handle SymbolicContext wrapper
    if hasattr(plan, 'value'):
        inner_plan = plan.value
        # Handle RobotWrapper
        if hasattr(inner_plan, 'value'):
            return inner_plan.value
        return inner_plan
    return plan


def calculate_geometric_cost_from_plan(plan_lines, symbols, robot_poses):
    """Calculate geometric path cost from multi-robot plan actions"""
    total_cost = 0.0
    action_costs = []
    
    for action_line in plan_lines:
        if action_line.startswith('(') and action_line.endswith(')'):
            action_content = action_line[1:-1]  
            parts = action_content.split()
            
            if len(parts) >= 4 and parts[0] == "goto-poi":
                robot_id = parts[1]
                from_place = parts[2]
                to_place = parts[3]
                
                # Get positions from symbols
                if from_place in symbols and to_place in symbols:
                    from_pos = symbols[from_place].position[:2]  # Take only x,y coordinates
                    to_pos = symbols[to_place].position[:2]
                    
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
                    total_cost += distance
                    action_costs.append((action_line, distance))
                else:
                    # If symbols not found, assume unit cost
                    total_cost += 1.0
                    action_costs.append((action_line, 1.0))
            else:
                total_cost += 0.1
                action_costs.append((action_line, 0.1))
    
    return total_cost, action_costs


def examine_scene_graph_coordinates(G):
    """Examine and display coordinate information from the scene graph"""
    print("\n=== Scene Graph Coordinate Information ===")
    
    # Get all nodes by layer
    place_nodes = list(G.get_layer(spark_dsg.DsgLayers.PLACES).nodes)
    object_nodes = list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)
    
    print(f"Total places: {len(place_nodes)}")
    print(f"Total objects: {len(object_nodes)}")
    
    print("\n--- Sample Places with Coordinates ---")
    for i, node in enumerate(place_nodes):
        if i >= 100:  
            break
        position = node.attributes.position
        print(f"  Place {node.id}: position = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
    
    print("\n--- Sample Objects with Coordinates ---")
    for i, node in enumerate(object_nodes):
        if i >= 100: 
            break
        position = node.attributes.position
        print(f"  Object {node.id}: position = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

 
def print_region_kinds(G):
    """Print unique region semantic categories present in the scene graph."""
    try:
        regions_layer = G.get_layer(spark_dsg.DsgLayers.ROOMS)
        region_nodes = list(regions_layer.nodes)
        # Default ROOM layer id is 4, partition 0
        labelspace = G.get_labelspace(4, 0)
        kinds_counts = {}
        for node in region_nodes:
            if hasattr(node, "attributes") and hasattr(node.attributes, "semantic_label"):
                category = labelspace.get_category(node.attributes.semantic_label)
                kinds_counts[category] = kinds_counts.get(category, 0) + 1
        print("\n--- Region kinds present ---")
        if kinds_counts:
            for category, count in sorted(kinds_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {category}: {count}")
        else:
            print("  (none)")
    except Exception as e:
        print(f"  Region kinds error: {e}")
 
 
def print_region_names(G):
    """Print names of all regions present in the scene graph."""
    try:
        regions_layer = G.get_layer(spark_dsg.DsgLayers.ROOMS)
        region_nodes = list(regions_layer.nodes)
        print("\n--- Region names ---")
        if not region_nodes:
            print("  (none)")
            return
        # Try to use semantic labelspace as fallback if name is empty
        try:
            labelspace = G.get_labelspace(4, 0)
        except Exception:
            labelspace = None
        for node in region_nodes:
            name = getattr(node.attributes, "name", "")
            if (not name) and labelspace and hasattr(node.attributes, "semantic_label"):
                try:
                    name = labelspace.get_category(node.attributes.semantic_label)
                except Exception:
                    pass
            print(f"  {node.id}: {name}")
    except Exception as e:
        print(f"  Region names error: {e}")
 
 
def print_places_in_region(G, region_symbol: str):
    """Print all places contained in the given region symbol (e.g., 'r68')."""
    try:
        # Resolve region node by matching its canonical string form (e.g., R(68)) to input (lowercased)
        target = region_symbol.lower()
        regions_layer = G.get_layer(spark_dsg.DsgLayers.ROOMS)
        region_node = None
        for node in regions_layer.nodes:
            try:
                if node.id.str(True).lower() == target:
                    region_node = node
                    break
            except Exception:
                continue
        if region_node is None:
            print(f"\n--- Places in {region_symbol} ---\n  Region not found")
            return
 
        # Use the same place layer as PDDL (MESH_PLACES or fallback to numeric 20)
        try:
            mesh_places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
        except Exception:
            mesh_places_layer = G.get_layer(20)
 
        # Map mesh places to regions via nearest 3D place's parent (matches PDDL generation)
        places_layer_3d = G.get_layer(spark_dsg.DsgLayers.PLACES)
        place_centers = []
        place_nodes = []
        for n in places_layer_3d.nodes:
            place_centers.append(n.attributes.position)
            place_nodes.append(n)
        if len(place_centers) == 0:
            print(f"\n--- Places in {region_symbol} ---\n  No 3D places available")
            return
        place_centers = np.array(place_centers)
 
        places_in_region = []
        for mesh_place in mesh_places_layer.nodes:
            try:
                mp = mesh_place.attributes.position
                # Find nearest 3D place
                idx = int(np.argmin(np.linalg.norm(place_centers - mp, axis=1)))
                nearest_place = place_nodes[idx]
                parent_region_id = nearest_place.get_parent()
                if parent_region_id and parent_region_id == region_node.id:
                    places_in_region.append(mesh_place)
            except Exception:
                continue
 
        print(f"\n--- Places in {region_symbol} ---")
        print(f"  Count: {len(places_in_region)}")
        for p in places_in_region[:200]:  # cap to reasonable number
            pos = getattr(p.attributes, "position", None)
            if pos is not None:
                # Print up to 3 components if available
                if len(pos) >= 3:
                    print(f"  {p.id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                else:
                    print(f"  {p.id}: [{pos[0]:.2f}, {pos[1]:.2f}]")
            else:
                print(f"  {p.id}")
    except Exception as e:
        print(f"  Error listing places for {region_symbol}: {e}")

def main():
    """Main test function following your exact format"""
    print("Multi-Robot Fast Downward Test with Real Scene Graph")
    print("=" * 80)
    
    # Configuration
    # scene_graph_path = "./src/awesome_dcist_t4/omniplanner/omniplanner/examples/scenegraph/west_point_fused_map_wregions_labelspace.json"
    scene_graph_path = "/home/jaeyoun-choi/colcon_ws/assets/b45_clip_final_connected_rooms_and_labelspace.json"
    # scene_graph_path = "/home/jaeyoun-choi/colcon_ws/assets/west_point_fused_map_wregions_labelspace.json"
    # robot_ids = ["robot1","robot2"]
    robot_ids = ["robot1", "robot2", "robot3"]
    
    print(f"Scene graph: {scene_graph_path}")
    print(f"Robots: {robot_ids}")
    
    # Load scene graph
    print(f"Loading scene graph from: {scene_graph_path}")
    G = spark_dsg.DynamicSceneGraph.load(scene_graph_path)
    print(f"✓ Scene graph loaded: {G.num_nodes()} total nodes")
    
    # Examine scene graph coordinates
    examine_scene_graph_coordinates(G)
    # Print existing region kinds

    print_region_kinds(G)
    # Print region names
    print_region_names(G)
    # Print places included in specific regions of interest
    # print_places_in_region(G, "r68")
    print_places_in_region(G, "r3")
    print_places_in_region(G, "r4")
    
    # Create robot poses (following your format)
    robot_poses = {
        "robot1": np.array([-15.0, -15.1]),
        "robot2": np.array([-15.0, 0.1]),
        "robot3": np.array([0.0, 6.0])
    }
    
    print("=========================================")
    print("==== PDDL region Domain (Multi-Robot) ====")
    print("=========================================")
    print("")
    
    # goal_string ="(and (safe o2)(safe o3))"
    goal_string ="(and (explored-region r1)(explored-region r2)(visited-object o2)(visited-object o9)(visited-place p22543)(visited-place p6255))"
    goal_string ="(and (visited-object o79)(visited-object o285)(visited-object o43)(safe o79)(explored-region r2)(visited-object o2)(visited-object o9)(visited-place p22543)(visited-place p6255))"
    # goal_string ="(and (visited-poi o27))"
    # goal_string ="(and (object-in-place o5 p91) (object-in-place o85 p118) )"
    # goal_string ="(and (object-in-place o5 p91) (object-in-place o94 p2157))"
    # goal_string ="(and (safe o2))"
    # goal_string ="(and (object-in-place o5 p91) (object-in-place o85 p118) (object-in-place o94 p2157))"
    goal = PddlGoal(robot_id="robot1", pddl_goal=goal_string)
    
    # Load the multi-robot domain
    domain_path = "./src/awesome_dcist_t4/omniplanner/omniplanner/src/dsg_pddl/domains/RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
    with open(domain_path, "r") as fo:
        domain = MultiRobotPddlDomain(fo.read())
    
    # print(f"Loading domain {domain_path}")
    # print(f"Domain name: {domain.domain_name}")
    req = PlanRequest(
        domain=domain,
        goal=goal,
        robot_states=robot_poses,
    )
    # Automatically generate a multi-robot PDDL problem from DSG (first pass to discover objects)
    pddl_start_time = time.time()
    plan = full_planning_pipeline(req, G) 
    actual_plan = extract_plan_from_wrapper(plan)
 
if __name__ == "__main__":
    main() 