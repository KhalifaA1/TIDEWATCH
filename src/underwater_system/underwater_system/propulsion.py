"""Six-DOF feedback and allocation to eight reversible underwater thrusters.

Commands are Newtons at each native Gazebo thruster, never pose/velocity setters.
Drag, buoyancy, inertia and collisions remain Gazebo physics' responsibility.
The mass/drag coefficients are illustrative simulation parameters.
"""
import math
import numpy as np


MASS_KG = 233.781617316884
INERTIA = np.array([5.657515139068603, 46.66281081645013, 46.66281081645013])
POSITIONS = np.array([[.6,.25,0],[.6,-.25,0],[-.6,.25,0],[-.6,-.25,0],
                      [.5,.2,0],[.5,-.2,0],[-.5,.2,0],[-.5,-.2,0]])
S = math.sqrt(.5)
DIRECTIONS = np.array([[S,-S,0],[S,S,0],[S,S,0],[S,-S,0],
                       [0,0,1],[0,0,1],[0,0,1],[0,0,1]])
ALLOCATION = np.vstack((DIRECTIONS.T, np.cross(POSITIONS,DIRECTIONS).T))


def quaternion(value):
    q = np.asarray(value,dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if not np.all(np.isfinite(q)) or norm < 1e-9:
        raise ValueError('Orientation must be a finite nonzero quaternion')
    return q/norm


def rotation_matrix(value):
    x,y,z,w = quaternion(value)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def quaternion_product(a,b):
    av,aw = np.array(a[:3]),a[3]
    bv,bw = np.array(b[:3]),b[3]
    return np.r_[aw*bv+bw*av+np.cross(av,bv),aw*bw-av@bv]


def angular_velocity(previous, current, dt):
    if dt <= 0:
        return np.zeros(3)
    previous,current = quaternion(previous),quaternion(current)
    delta = quaternion_product(np.r_[-previous[:3],previous[3]],current)
    if delta[3] < 0:
        delta = -delta
    length = np.linalg.norm(delta[:3])
    if length < 1e-9:
        return np.zeros(3)
    angle = 2*math.atan2(length,max(delta[3],0))
    return delta[:3]/length*angle/dt


def attitude_error(orientation, desired_yaw):
    q = quaternion(orientation)
    desired = np.array([0,0,math.sin(desired_yaw/2),math.cos(desired_yaw/2)])
    error = quaternion_product(np.r_[-q[:3],q[3]],desired)
    if error[3] < 0:
        error = -error
    return 2*error[:3]


def allocate(wrench, limit=80.0):
    """Least-squares allocation with fixed saturated actuators and redistribution."""
    wrench = np.asarray(wrench,dtype=float).reshape(6)
    if not np.all(np.isfinite(wrench)) or limit <= 0:
        raise ValueError('Finite wrench and positive thrust limit required')
    result = np.zeros(8)
    free = np.ones(8,dtype=bool)
    for _ in range(8):
        if not free.any():
            break
        remaining = wrench-ALLOCATION[:,~free]@result[~free]
        values = np.linalg.lstsq(ALLOCATION[:,free],remaining,rcond=None)[0]
        indices = np.flatnonzero(free)
        result[indices] = values
        exceeded = np.abs(values)>limit
        if not exceeded.any():
            break
        saturated = indices[exceeded]
        result[saturated] = np.clip(result[saturated],-limit,limit)
        free[saturated] = False
    return np.clip(result,-limit,limit)


class ThrusterController:
    def __init__(self, config=None):
        self.config = config or {}
        self.limit = float(self.config.get('max_thrust_n',80))
        self.mass = float(self.config.get('mass_kg',MASS_KG))
        self.inertia = np.array(self.config.get('inertia_kg_m2',INERTIA),dtype=float)
        self.integral = np.zeros(3)
        self.thrust = np.zeros(8)
        self.last = {}

    def reset(self):
        self.integral[:] = 0
        self.thrust[:] = 0
        self.last = {}

    def command(self, desired_velocity, velocity, orientation, omega_body, desired_yaw, dt=.05):
        desired_velocity = np.asarray(desired_velocity,dtype=float).reshape(3)
        velocity = np.asarray(velocity,dtype=float).reshape(3)
        omega_body = np.asarray(omega_body,dtype=float).reshape(3)
        if not all(np.all(np.isfinite(v)) for v in (desired_velocity,velocity,omega_body)):
            raise ValueError('Navigation must be finite')
        dt = max(.001,min(float(dt),.2))
        rotation = rotation_matrix(orientation)
        error_world = desired_velocity-velocity
        old_integral = self.integral.copy()
        self.integral = np.clip(self.integral+error_world*dt,-.8,.8)
        body_velocity = rotation.T@velocity
        linear = np.array(self.config.get('linear_drag',[6,18,18]))
        quadratic = np.array(self.config.get('quadratic_drag',[9,30,30]))
        drag = linear*body_velocity+quadratic*np.abs(body_velocity)*body_velocity
        # Integral feedback estimates unknown current loading; it receives no
        # ground-truth current from the mission controller.
        force = rotation.T@(self.mass*(self.config.get('velocity_gain',1.4)*error_world+
                                      self.config.get('integral_gain',.25)*self.integral))+drag
        torque = self.inertia*(self.config.get('attitude_gain',4.0)*attitude_error(orientation,desired_yaw)-
                               self.config.get('angular_damping',3.0)*omega_body)
        torque += np.array([.4,2,2])*omega_body+np.array([.3,3,3])*np.abs(omega_body)*omega_body
        desired_wrench = np.r_[force,torque]
        target = allocate(desired_wrench,self.limit)
        achieved = ALLOCATION@target
        if np.linalg.norm(desired_wrench-achieved)>1:
            # Stop integration when actuator authority is exhausted.
            self.integral = old_integral
        tau = max(.001,self.config.get('thruster_time_constant_s',.12))
        response = target+(self.thrust-target)*math.exp(-dt/tau)
        slew = self.config.get('thrust_slew_n_s',240)*dt
        self.thrust += np.clip(response-self.thrust,-slew,slew)
        self.thrust = np.clip(self.thrust,-self.limit,self.limit)
        diameter = self.config.get('propeller_diameter_m',.12)
        ideal_power_factor = 1/math.sqrt(2*1025*math.pi*(diameter/2)**2)
        self.last = {'mode':'native_thrusters','thrust_n':self.thrust.tolist(),
                     'desired_body_wrench':desired_wrench.tolist(),
                     'applied_body_wrench':(ALLOCATION@self.thrust).tolist(),
                     'thrust_utilization':float(np.max(np.abs(self.thrust))/self.limit),
                     'saturated':bool(np.max(np.abs(target))>=self.limit-.01),
                     'power_estimate_w':float(np.sum(np.abs(self.thrust)**1.5)*ideal_power_factor),
                     'power_model':'ideal induced power; uncalibrated'}
        return self.thrust.copy()

    def stop(self):
        self.reset()
        return self.thrust.copy()
