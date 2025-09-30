#!/usr/bin/env python
import os
import spark_dsg

region_label_dict = {
    0: "unknown",
    1: "road",
    2: "field",
    3: "shelter",
    4: "indoor",
    5: "stairs",
    6: "sidewalk",
    7: "path",
    8: "boundary",
    9: "shore",
    10: "ground",
    11: "dock",
    12: "parking",
    13: "footing"
}

def get_region_parent_of_object(object_node, scene_graph):
    # Get the parent of the object (Place)
    parent_place_id = object_node.get_parent()
    if not parent_place_id:
        return "none"
    parent_place_node = scene_graph.get_node(parent_place_id)
    parent_region_id = parent_place_node.get_parent()
    if not parent_region_id:
        return "none"
    parent_region_node = scene_graph.get_node(parent_region_id)
    return str(parent_region_node.id)

def object_to_prompt(object_node, scene_graph):
    attrs = object_node.attributes
    symbol = str(object_node.id)
    object_labelspace = scene_graph.get_labelspace(2, 0)
    if not object_labelspace:
        raise PromptingFailure("No available object labelspace")
    semantic_type = object_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]},{attrs.position[1]})"
    parent_region = get_region_parent_of_object(object_node, scene_graph)
    object_prompt = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_region={parent_region})"
    return object_prompt

def region_to_prompt(region_node, scene_graph):
    attrs = region_node.attributes
    symbol = str(region_node.id)
    region_labelspace = scene_graph.get_labelspace(4, 0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category(attrs.semantic_label)
    region_prompt = f"\n-\t(id={symbol}, type={semantic_type})"
    return region_prompt

def scene_graph_to_prompt(scene_graph):
    # Get the Region content
    region_labelspace = scene_graph.get_labelspace(4, 0)
    region_nodes = scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    regions_prompt = ""
    for region_node in region_nodes:
        attrs = region_node.attributes
        region_id = str(region_node.id)
        region_type = region_labelspace.get_category(attrs.semantic_label)
        region_pos_x = attrs.position[0]
        region_pos_y = attrs.position[1]
        tmp_str = f"(id={region_id},type={region_type},pos=({str(round(region_pos_x,3))},{str(round(region_pos_y,3))}))"
        regions_prompt += tmp_str
    regions_prompt = regions_prompt[:-1]

    object_labelspace = scene_graph.get_labelspace(2, 0)
    object_nodes = scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
    objects_prompt = ""
    for object_node in object_nodes:
        attrs = object_node.attributes
        object_id = str(object_node.id)
        object_type = object_labelspace.get_category(attrs.semantic_label)
        object_pos_x = attrs.position[0]
        object_pos_y = attrs.position[1]
        parent_region_id = get_region_parent_of_object(object_node, scene_graph)
        tmp_str = f"(id={object_id},type={object_type},pos=({str(round(object_pos_x,3))},{str(round(object_pos_y,3))}),parent_region_id={parent_region_id})"
        objects_prompt += tmp_str
    objects_prompt = objects_prompt[:-1]

    prompt_string = f"<Regions>{regions_prompt}</Regions>\n<Objects>{objects_prompt}</Objects>"
    return prompt_string

def get_scene_graph_content(scene_graph):
    regions_by_id = {}
    regions_by_semantic_class = {}
    objects_by_id = {}
    objects_by_semantic_class = {}
    region_labelspace = scene_graph.get_labelspace(4, 0)
    region_nodes = scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    for region_node in region_nodes:
        semantic_label = region_labelspace.get_category(region_node.attributes.semantic_label)
        region = {
            "semantic_label" : semantic_label,
            "center" : f"({str(round(region_node.attributes.position[0],3))}, {str(round(region_node.attributes.position[1],3))})",
            "objects" : [],
        }
        regions_by_id[str(region_node.id)] = region
        if semantic_label not in regions_by_semantic_class.keys():
            regions_by_semantic_class[semantic_label] = [] 
        regions_by_semantic_class[semantic_label].append(str(region_node.id))

    object_labelspace = scene_graph.get_labelspace(2, 0)
    object_nodes = scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
    for object_node in object_nodes:
        semantic_label = object_labelspace.get_category(object_node.attributes.semantic_label)
        parent_region = get_region_parent_of_object(object_node, scene_graph)
        o = {
            "semantic_label" : semantic_label,
            "center" : f"({str(round(object_node.attributes.position[0],3))}, {str(round(object_node.attributes.position[1],3))})",
            "parent_region" : str(parent_region),
        }
        objects_by_id[str(object_node.id)] = o
        if semantic_label not in objects_by_semantic_class.keys():
            objects_by_semantic_class[semantic_label] = []
        objects_by_semantic_class[semantic_label].append(str(object_node.id))
        if parent_region == "none": continue
        regions_by_id[str(parent_region)]["objects"].append(str(object_node.id))

    content = {}
    content["regions"] = regions_by_id
    content["regions_by_semantic_class"] = regions_by_semantic_class
    content["objects"] = objects_by_id
    content["objects_by_semantic_class"] = objects_by_semantic_class
    return content

# Load the scene graph from file into spark_dsg
scene_graph_fn = "{Your scene graph filepath goes here}"
scene_graph = spark_dsg.DynamicSceneGraph.load(scene_graph_fn)
# Get a slightly more legible info dump of scene graph content for our world model
scene_graph_content = get_scene_graph_content(scene_graph)
import json
with open("/tmp/scene_graph_content.json", 'w', encoding='utf-8') as file:                                                                        
        json.dump(scene_graph_content, file, ensure_ascii=False, indent=4)
# Contruct the prompt string for the scene graph component (this gets loaded into the prompt)
prompt = scene_graph_to_prompt(scene_graph)
print(f"prompt: {prompt}")


