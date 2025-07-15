from dataclasses import dataclass

import numpy as np
import spark_dsg


@dataclass
class DummyRobotPlanningAdaptor:
    name: str
    robot_type: str
    parent_frame: str
    child_frame: str


def build_test_dsg():
    """An example scene graph for testing"""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "p", spark_dsg.DsgLayers.PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, "B", spark_dsg.DsgLayers.BUILDINGS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)

    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([0, 0, 0])
    room.semantic_label = 0

    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 0).value, room)

    place1 = spark_dsg.PlaceNodeAttributes()
    place1.position = np.array([-1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 0).value, place1)
    place2 = spark_dsg.PlaceNodeAttributes()
    place2.position = np.array([1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 1).value, place2)

    place1_2d = spark_dsg.PlaceNodeAttributes()
    place1_2d.position = np.array([-1.1, 0, 0])
    place1_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 0).value, place1_2d
    )
    place2_2d = spark_dsg.PlaceNodeAttributes()
    place2_2d.position = np.array([1.1, 0, 0])
    place2_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 1).value, place2_2d
    )

    obj1 = spark_dsg.ObjectNodeAttributes()
    obj1.position = np.array([-1.5, 0, 0])
    obj1.semantic_label = 34  # box
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 0).value, obj1)
    obj2 = spark_dsg.PlaceNodeAttributes()
    obj2.position = np.array([1.5, 0, 0])
    obj2.semantic_label = 15  # rock
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 1).value, obj2)

    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("O", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 1).value, spark_dsg.NodeSymbol("O", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 0).value, spark_dsg.NodeSymbol("P", 1).value
    )

    labelspaces = {
        "labelspaces": {
            "_l2p0": [
                [0, "unknown"],
                [1, "sky"],
                [2, "tree"],
                [3, "water"],
                [4, "ground"],
                [5, "grass"],
                [6, "sand"],
                [7, "sidewalk"],
                [8, "dock"],
                [9, "road"],
                [10, "path"],
                [11, "vehicle"],
                [12, "building"],
                [13, "shelter"],
                [14, "signal"],
                [15, "rock"],
                [16, "fence"],
                [17, "boat"],
                [18, "sign"],
                [19, "hill"],
                [20, "bridge"],
                [21, "wall"],
                [22, "floor"],
                [23, "ceiling"],
                [24, "door"],
                [25, "stairs"],
                [26, "pole"],
                [27, "rail"],
                [28, "structure"],
                [29, "window"],
                [30, "surface"],
                [31, "flora"],
                [32, "flower"],
                [33, "bed"],
                [34, "box"],
                [35, "storage"],
                [36, "barrel"],
                [37, "bag"],
                [38, "basket"],
                [39, "seating"],
                [40, "flag"],
                [41, "decor"],
                [42, "light"],
                [43, "appliance"],
                [44, "trash"],
                [45, "bicycle"],
                [46, "food"],
                [47, "clothes"],
                [48, "thing"],
                [49, "animal"],
                [50, "human"],
            ],
            "_l4p0": [
                [0, "unknown"],
                [1, "road"],
                [2, "field"],
                [3, "shelter"],
                [4, "indoor"],
                [5, "stairs"],
                [6, "sidewalk"],
                [7, "path"],
                [8, "boundary"],
                [9, "shore"],
                [10, "ground"],
                [11, "dock"],
                [12, "parking"],
                [13, "footing"],
            ],
        }
    }

    G.metadata.add(labelspaces)

    return G
