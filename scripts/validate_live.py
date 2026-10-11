#!/usr/bin/env python3
"""Evaluate a running Gazebo mission through its real dashboard interface.

Exercises an acoustic outage, recovery, and optional current and mission controls.
The measurement artifact is distinct from the offline numerical harness.
"""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--output',default='results/gazebo_validation.json')
    parser.add_argument('--controls',action='store_true',help='Also exercise pause, resume and reset')
    parser.add_argument('--current',action='store_true',help='Also verify escort under a water-current disturbance')
    args = parser.parse_args()
    def state():
        return json.loads(urllib.request.urlopen(args.url+'/api/state',timeout=3).read())
    def command(value, **fields):
        req = urllib.request.Request(args.url+'/api/control',data=json.dumps({'command':value, **fields}).encode(),
                                     headers={'Content-Type':'application/json'})
        return json.loads(urllib.request.urlopen(req,timeout=3).read())
    def wait_for(predicate, timeout=20):
        start = time.monotonic()
        while time.monotonic()-start < timeout:
            current = state()
            if predicate(current):
                return current
            time.sleep(0.4)
        raise AssertionError('Live simulation did not reach the expected state')
    baseline = wait_for(lambda s:s['time']>=15 and len(s['auvs'])==3 and len(s['map']['points'])>=20)
    if baseline['mission'].get('outage'):
        command('toggle_outage')
        baseline = wait_for(lambda s:not s['mission'].get('outage'))
    start_poses = [unit['position'] for unit in baseline['auvs']]
    start_reconnects = baseline['network'].get('reconnections',0)
    start_outages = baseline['network'].get('outages',0)
    command('toggle_outage')
    outage = wait_for(lambda s:s['network'].get('outages',0)>start_outages)
    time.sleep(5)
    outage = state()
    command('toggle_outage')
    recovery = wait_for(lambda s:s['network'].get('reconnections',0)>start_reconnects)
    recovery = wait_for(lambda s:s['network'].get('last_delivery',{}).get('auv_1',0)>outage['time'],timeout=15)
    metrics = recovery['metrics']
    checks = {
        'three_independent_auvs_reporting': len(recovery['auvs'])==3,
        'auv_motion': any(math.dist(a,unit['position'])>0.1 for a,unit in zip(start_poses,recovery['auvs'])),
        'persistent_observed_map': len(recovery['map']['points'])>=len(baseline['map']['points']),
        'object_tracking': len(recovery['tracks'])>=2,
        'localization_rmse_below_half_metre': metrics['localization_rmse']<0.5,
        'map_surface_rmse_below_one_metre': metrics['mapping_rmse']<1,
        'safe_auv_separation': metrics['minimum_auv_separation']>=2.4,
        'no_oversize_packets': recovery['network']['oversize_rejected']==0,
        'acoustic_outage_and_recovery': recovery['network']['last_delivery']['auv_1']>outage['time'],
        'mqtt_surface_publisher_connected': recovery.get('mqtt',{}).get('connected',False),
        'native_force_driven_propulsion': recovery['mission'].get('propulsion')=='native_thrusters'
            and all(unit.get('propulsion',{}).get('mode')=='native_thrusters' for unit in recovery['auvs']),
        'bounded_active_thruster_commands': all(0<=unit.get('propulsion',{}).get('thrust_utilization',-1)<=1
            for unit in recovery['auvs']) and any(unit['propulsion']['thrust_utilization']>0.001 for unit in recovery['auvs']),
        'edge_classifications_evaluated': metrics.get('classification_samples',0)>50
            and bool(metrics.get('classification_confusion')),
    }
    current_test = {}
    if args.current:
        original = list(recovery['mission'].get('current',[0,0,0]))
        disturbance = [0.45,0.2,0.05]
        try:
            command('set_current',vector=disturbance)
            applied = wait_for(lambda s:s['mission'].get('current')==disturbance)
            settled = wait_for(lambda s:s['time']>=applied['time']+20, timeout=55)
            checks['current_command_applied'] = settled['mission']['current']==disturbance
            checks['escort_holds_under_current'] = settled['metrics']['formation_error']<1.5
            checks['safe_separation_under_current'] = settled['metrics']['minimum_auv_separation']>=2.4
            checks['current_run_remains_live'] = len(settled['auvs'])==3 and all(
                unit['age']<6 for unit in settled['auvs'])
            current_test = {'commanded_vector_m_s':disturbance,'original_vector_m_s':original,
                            'applied_time':applied['time'],'settled':settled}
        finally:
            command('set_current',vector=original)
        restored = wait_for(lambda s:s['mission'].get('current')==original)
        checks['original_current_restored'] = restored['mission']['current']==original
    controls = {}
    if args.controls:
        command('pause')
        paused = wait_for(lambda s:s['mission']['status']=='paused')
        time.sleep(1)
        a = state()['time']
        time.sleep(1)
        b = state()['time']
        checks['pause_stops_simulation_clock'] = abs(a-b)<0.02
        command('start')
        resumed = wait_for(lambda s:s['time']>b+0.5)
        command('reset')
        reset = wait_for(lambda s:s['time']<resumed['time']-1)
        active = wait_for(lambda s:s['time']>3 and len(s['auvs'])==3,timeout=20)
        checks['resume_advances_simulation_clock'] = resumed['time']>b
        checks['reset_restarts_running_swarm'] = active['mission']['status']=='running' and len(active['auvs'])==3
        controls = {'paused_time':paused['time'],'resumed_time':resumed['time'],
                    'reset_time':reset['time'],'active_after_reset_time':active['time']}
    result = {'source':'Live Gazebo 10.5 / ROS 2 Lyrical integration',
              'checks':checks,'all_checks_passed':all(checks.values()),
              'baseline':baseline,'outage':outage,'recovery':recovery,'controls':controls,
              'current_test':current_test,
              'scope':'Native eight-thruster AUV dynamics with buoyancy/drag; synthetic geometric sonar, '
                      'numeric echo features and local hydrophone STFT classification; emulated acoustic link.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],'checks':checks,'output':str(output)},indent=2))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
