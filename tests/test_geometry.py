"""Check sensor hits against the actual Gazebo collision world."""
from pathlib import Path
import numpy as np
from underwater_system.geometry import CollisionWorld


WORLD = Path(__file__).resolve().parents[1]/'src/underwater_system/worlds/tidewatch.sdf'


def test_downward_beam_hits_gazebo_seabed():
    world = CollisionWorld(WORLD)
    origin = np.array([0.,0.,-10.])
    direction = np.array([0.,0.,-1.])
    distance = world.first_hit(origin,direction)
    assert distance is not None and 8 < distance < 11
    assert world.surface_distance(origin+direction*distance) < 1e-8


def test_hazard_occludes_more_distant_seabed():
    world = CollisionWorld(WORLD)
    origin = np.array([11.,4.,-10.])
    direction = np.array([0.,0.,-1.])
    distance = world.first_hit(origin,direction)
    assert distance is not None and distance < 9
    assert world.surface_distance(origin+direction*distance) < 1e-8


def test_range_limit_prevents_invented_coverage():
    world = CollisionWorld(WORLD)
    assert world.first_hit([0,0,-10],[0,0,-1],maximum=2) is None
