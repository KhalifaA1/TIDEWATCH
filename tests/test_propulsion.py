import math
import numpy as np
import pytest
from underwater_system.propulsion import (ALLOCATION, ThrusterController, allocate,
                                          angular_velocity, rotation_matrix)


def test_all_six_dofs_have_thruster_authority():
    assert np.linalg.matrix_rank(ALLOCATION)==6
    for axis in range(6):
        wrench = np.zeros(6)
        wrench[axis] = 10
        assert np.allclose(ALLOCATION@allocate(wrench),wrench,atol=1e-8)


def test_extreme_requests_are_bounded_and_slew_limited():
    controller = ThrusterController()
    previous = np.zeros(8)
    for _ in range(100):
        output = controller.command([50,-50,50],[0,0,0],[0,0,0,1],[0,0,0],2,dt=.05)
        assert np.max(np.abs(output))<=80
        assert np.max(np.abs(output-previous))<=12.000001
        previous = output
    assert controller.last['saturated']
    assert np.max(np.abs(controller.integral))<.3


def test_body_rotation_keeps_world_force_direction():
    yaw = math.pi/2
    q = [0,0,math.sin(yaw/2),math.cos(yaw/2)]
    controller = ThrusterController({'thruster_time_constant_s':.001,'thrust_slew_n_s':100000})
    controller.command([.1,0,0],[0,0,0],q,[0,0,0],yaw)
    body_force = np.array(controller.last['applied_body_wrench'][:3])
    world_force = rotation_matrix(q)@body_force
    assert world_force[0]>0 and abs(world_force[1])<1e-8 and abs(world_force[2])<1e-8


def test_no_force_at_neutral_rest_and_watchdog_clears_hold():
    controller = ThrusterController()
    assert np.allclose(controller.command([0,0,0],[0,0,0],[0,0,0,1],[0,0,0],0),0)
    controller.command([1,0,0],[0,0,0],[0,0,0,1],[0,0,0],0)
    assert np.max(np.abs(controller.thrust))>0
    assert np.allclose(controller.stop(),0)


def test_quaternion_wrap_preserves_angular_velocity():
    q0 = [0,0,math.sin(3.13/2),math.cos(3.13/2)]
    q1 = [0,0,math.sin(3.15/2),math.cos(3.15/2)]
    assert np.allclose(angular_velocity(q0,q1,.1),[0,0,.2],atol=1e-8)
    assert np.allclose(angular_velocity(q0,[-v for v in q1],.1),[0,0,.2],atol=1e-8)


def test_nan_navigation_rejected_before_actuation():
    with pytest.raises(ValueError):
        ThrusterController().command([np.nan,0,0],[0,0,0],[0,0,0,1],[0,0,0],0)
