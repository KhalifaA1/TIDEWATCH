"""Ray intersections against the actual static SDF collision primitives.

Used exclusively by simulated instruments and map evaluation. Edge nodes never
import or load the world. Supports rotated planes, boxes, spheres, cylinders,
and ellipsoids without requiring GPU rendering.
"""
import math
from xml.etree import ElementTree as ET
import numpy as np


def pose(text):
    values = [float(x) for x in (text or '0 0 0 0 0 0').split()]
    x,y,z,roll,pitch,yaw = values
    c,s = math.cos,math.sin
    rx = np.array([[1,0,0],[0,c(roll),-s(roll)],[0,s(roll),c(roll)]])
    ry = np.array([[c(pitch),0,s(pitch)],[0,1,0],[-s(pitch),0,c(pitch)]])
    rz = np.array([[c(yaw),-s(yaw),0],[s(yaw),c(yaw),0],[0,0,1]])
    result = np.eye(4)
    result[:3,:3] = rz@ry@rx
    result[:3,3] = [x,y,z]
    return result


class CollisionWorld:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.primitives = []
        for model in root.findall('./world/model'):
            if model.findtext('static') not in ('true','1'):
                continue
            mp = pose(model.findtext('pose'))
            for link in model.findall('link'):
                lp = mp@pose(link.findtext('pose'))
                for collision in link.findall('collision'):
                    transform = lp@pose(collision.findtext('pose'))
                    g = collision.find('geometry')
                    kind = next(iter(g)).tag
                    if kind == 'box':
                        extent = np.array([float(x) for x in g.findtext('box/size').split()])/2
                    elif kind == 'sphere':
                        extent = float(g.findtext('sphere/radius'))
                    elif kind == 'plane':
                        extent = np.array([float(x) for x in g.findtext('plane/size').split()])/2
                    elif kind == 'cylinder':
                        extent = [float(g.findtext('cylinder/radius')),float(g.findtext('cylinder/length'))/2]
                    elif kind == 'ellipsoid':
                        extent = np.array([float(x) for x in g.findtext('ellipsoid/radii').split()])
                    else:
                        continue
                    self.primitives.append((kind, transform[:3,3], transform[:3,:3], extent))

    def first_hit(self, origin, direction, maximum=28):
        origin, direction = np.array(origin), np.array(direction)
        candidates = []
        for kind, center, rotation, extent in self.primitives:
            p,d = rotation.T@(origin-center), rotation.T@direction
            roots = []
            if kind == 'plane':
                if abs(d[2]) > 1e-9:
                    t = -p[2]/d[2]
                    hit = p+t*d
                    if np.all(np.abs(hit[:2]) <= extent):
                        roots.append(t)
            elif kind == 'box':
                if any(abs(d[i]) < 1e-10 and abs(p[i]) > extent[i] for i in range(3)):
                    continue
                denominator = np.where(np.abs(d) < 1e-10, 1e-10, d)
                a,b = (-extent-p)/denominator, (extent-p)/denominator
                lo,hi = np.max(np.minimum(a,b)), np.min(np.maximum(a,b))
                if hi >= max(lo,0):
                    roots.append(lo if lo > 0 else hi)
            elif kind in ('sphere','ellipsoid'):
                pp,dd = p/extent,d/extent
                a,b,c = float(dd@dd),float(pp@dd),float(pp@pp)-1
                disc = b*b-a*c
                if disc >= 0:
                    roots.extend([(-b-math.sqrt(disc))/a,(-b+math.sqrt(disc))/a])
            elif kind == 'cylinder':
                radius,height = extent
                a,b,c = float(d[:2]@d[:2]),float(p[:2]@d[:2]),float(p[:2]@p[:2])-radius**2
                disc = b*b-a*c
                if a > 1e-10 and disc >= 0:
                    for t in [(-b-math.sqrt(disc))/a,(-b+math.sqrt(disc))/a]:
                        if abs(p[2]+t*d[2]) <= height:
                            roots.append(t)
                if abs(d[2]) > 1e-10:
                    for cap in (-height,height):
                        t = (cap-p[2])/d[2]
                        if np.linalg.norm((p+t*d)[:2]) <= radius:
                            roots.append(t)
            candidates.extend(t for t in roots if 1e-5 < t <= maximum)
        return min(candidates) if candidates else None

    def surface_distance(self, point):
        distances = []
        for kind,center,rotation,extent in self.primitives:
            p = rotation.T@(np.array(point)-center)
            if kind == 'plane':
                q = np.maximum(np.abs(p[:2])-extent,0)
                distance = np.sqrt(q@q+p[2]**2)
            elif kind == 'sphere':
                distance = abs(np.linalg.norm(p)-extent)
            elif kind == 'box':
                q = np.abs(p)-extent
                distance = abs(np.linalg.norm(np.maximum(q,0))+min(max(q),0))
            elif kind == 'cylinder':
                q = np.array([np.linalg.norm(p[:2])-extent[0],abs(p[2])-extent[1]])
                distance = abs(np.linalg.norm(np.maximum(q,0))+min(max(q),0))
            else:
                # Ellipsoid scaled radial residual; conservative evaluation proxy.
                distance = abs(np.linalg.norm(p/extent)-1)*min(extent)
            distances.append(float(distance))
        return min(distances) if distances else float('nan')
