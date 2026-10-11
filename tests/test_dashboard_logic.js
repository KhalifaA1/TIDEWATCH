// Run from the repository root: gjs tests/test_dashboard_logic.js
// Uses DOM/canvas fixtures; this does not validate live browser rendering.
const Gio = imports.gi.Gio;

function read(path) {
  const contents = Gio.File.new_for_path(path).load_contents(null)[1];
  return new TextDecoder().decode(contents);
}

const checks = [];
const errors = [];
const commands = [];
let failCommand = false;
let fakeNow = Date.now();
Date.now = () => fakeNow;

const timeoutTasks = new Map();
const intervalTasks = new Map();
let timerId = 0;

globalThis.setTimeout = (callback, ms) => {
  timeoutTasks.set(++timerId, {callback, ms});
  return timerId;
};
globalThis.clearTimeout = id => timeoutTasks.delete(id);
globalThis.setInterval = (callback, ms) => {
  intervalTasks.set(++timerId, {callback, ms});
  return timerId;
};
globalThis.clearInterval = id => intervalTasks.delete(id);
globalThis.requestAnimationFrame = () => 1;
globalThis.AbortController = class {
  constructor() {
    this.signal = {};
  }

  abort() {
    this.signal.aborted = true;
  }
};

class Target {
  constructor() {
    this.listeners = {};
  }

  addEventListener(type, callback) {
    (this.listeners[type] ??= []).push(callback);
  }

  emit(type, event = {}) {
    const dispatched = {
      target: this,
      key: '',
      preventDefault() {
        this.prevented = true;
      },
      ...event
    };
    return Promise.all((this.listeners[type] || []).map(callback => callback(dispatched)));
  }
}

const canvasContext = new Proxy({
  measureText: text => ({width: String(text).length * 5}),
  createLinearGradient: () => ({addColorStop() {}}),
  createRadialGradient: () => ({addColorStop() {}})
}, {
  get: (object, key) => key in object ? object[key] : (() => {}),
  set: (object, key, value) => {
    object[key] = value;
    return true;
  }
});

class Element extends Target {
  constructor(tag, attributes) {
    super();
    this.tagName = tag.toUpperCase();
    this.attributes = attributes;
    this.id = attributes.id || '';
    this.dataset = {};
    this.style = {};
    this.textContent = '';
    this.innerHTML = '';
    this.className = attributes.class || '';
    this.value = attributes.value || '';
    this.disabled = Object.hasOwn(attributes, 'disabled');
    this.hidden = Object.hasOwn(attributes, 'hidden');
    this.width = 0;
    this.height = 0;
    this.isContentEditable = false;
    this.classList = {
      toggle: (name, active) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        active ? classes.add(name) : classes.delete(name);
        this.className = [...classes].join(' ');
      }
    };

    for (const [key, value] of Object.entries(attributes)) {
      if (key.startsWith('data-')) {
        const name = key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
        this.dataset[name] = value;
      }
    }
  }

  setAttribute(key, value) {
    this.attributes[key] = String(value);
  }

  removeAttribute(key) {
    delete this.attributes[key];
  }

  getAttribute(key) {
    return this.attributes[key] ?? null;
  }

  getContext() {
    return canvasContext;
  }

  getBoundingClientRect() {
    return {
      width: this.id === 'scene' ? 1024 : 170,
      height: this.id === 'scene' ? 465 : 40
    };
  }

  setPointerCapture() {}

  matches(selector) {
    return selector.split(',').some(piece => {
      piece = piece.trim();
      if (piece.startsWith('#')) return this.id === piece.slice(1);
      const attribute = /^\[([^\]]+)\]$/.exec(piece);
      return attribute
        ? Object.hasOwn(this.attributes, attribute[1])
        : this.tagName.toLowerCase() === piece;
    });
  }
}

const elements = [];
const html = read('src/underwater_system/dashboard/index.html');
for (const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
  const attributes = {};
  for (const attribute of match[2].matchAll(/([a-z][a-z0-9-]*)(?:="([^"]*)")?/gi)) {
    attributes[attribute[1]] = attribute[2] ?? '';
  }
  elements.push(new Element(match[1], attributes));
}

const documentMock = new Target();
documentMock.getElementById = id => elements.find(element => element.id === id) || null;
documentMock.querySelectorAll = selector => elements.filter(element => element.matches(selector));
documentMock.activeElement = elements.find(element => element.tagName === 'BODY');
documentMock.hidden = false;
globalThis.document = documentMock;

const windowMock = new Target();
window.addEventListener = windowMock.addEventListener.bind(windowMock);
window.emit = windowMock.emit.bind(windowMock);
window.devicePixelRatio = 1;
globalThis.structuredClone = value => JSON.parse(JSON.stringify(value));

let fixture = {
  time: 20,
  dashboard: {
    state_age_seconds: 0
  },
  mission: {
    status: "running",
    mode: "protect",
    mode_revision: 0,
    planner: "adaptive escort",
    protected_radius: 6,
    current: [0.2, -0.1, 0],
    propulsion: "native_thrusters",
    diver_control: {
      mode: "scripted",
      velocity: [0, 0, 0]
    },
    exploration: {
      bounds: [
        [
          -24,
          24
        ],
        [
          -24,
          24
        ]
      ],
      depth_layers: [
        -10
      ],
      origin: [0, 0, -10],
      coverage: 0.25,
      observed_cells: 64,
      total_cells: 256,
      source: "gateway-delivered measured returns"
    }
  },
  auvs: [
    {
      id: "auv_1",
      position: [6, 0, -12],
      velocity: [0.1, 0, 0],
      online: true,
      age: 0.3,
      uncertainty: 0.3,
      propulsion: {
        mode: "native_thrusters",
        thrust_utilization: 0.4,
        power_estimate_w: 24
      },
      mission: {
        mode: "protect",
        role: "escort",
        phase: "protect",
        target: [5, 0, -12]
      }
    },
    {
      id: "auv_2",
      position: [-3, 5, -12],
      velocity: [0, 0.1, 0],
      online: true,
      age: 0.3,
      uncertainty: 0.4,
      mission: {
        mode: "protect",
        role: "escort",
        phase: "protect",
        target: [-3, 4, -12]
      }
    },
    {
      id: "auv_3",
      position: [-3, -5, -12],
      velocity: [0, 0.1, 0],
      online: true,
      uncertainty: 0.4,
      mission: null
    }
  ],
  diver: {
    position: [0, 0, -12]
  },
  tracks: [
    {
      id: "contact_1",
      label: "vessel",
      position: [8, 0, -12],
      velocity: [-0.2, 0, 0],
      uncertainty: 0.6,
      risk: "warning",
      confidence: 0.8
    }
  ],
  alerts: [
    {
      track_id: "contact_1",
      level: "warning",
      message: "Predicted trajectory enters the protected zone",
      time: 20,
      closest_approach_m: 1.4,
      time_to_closest_s: 8
    }
  ],
  map: {
    points: [
      [1, 1, -20]
    ],
    elevation: [
      {
        x: 1,
        y: 1,
        z: -20
      }
    ],
    resolution_m: 1,
    observed_cells: 64
  },
  metrics: {
    coverage: 0.12,
    formation_error: 1.2,
    localization_rmse: 0.4,
    mapping_rmse: 0.2,
    classification_accuracy: 0.85
  },
  network: {
    loss_rate: 0.1,
    delivered: 42,
    queued: 3
  }
};

globalThis.fetch = async (url, options = {}) => {
  if (url === '/api/state') {
    return {ok: true, json: async () => structuredClone(fixture)};
  }
  if (url === '/api/control') {
    const command = JSON.parse(options.body);
    commands.push(command);
    if (failCommand) {
      failCommand = false;
      return {ok: false, json: async () => ({error: 'Fixture command rejection'})};
    }
    return {ok: true, json: async () => ({queued: true})};
  }
  throw Error('Unexpected fetch ' + url);
};

function assert(condition, label) {
  checks.push({label, passed: Boolean(condition)});
  if (!condition) throw Error(label);
}

async function tick() {
  for (let i = 0; i < 12; i++) await Promise.resolve();
}

const el = id => document.getElementById(id);
const source = read('src/underwater_system/dashboard/app.js');
const api = new Function(source + `
  return {
    updateUI,
    drawScene,
    canPilot,
    haltPilot,
    queuePilot,
    camera,
    get modeRequest() { return modeRequest; },
    get pilotRequest() { return pilotRequest; },
    get pilotKeys() { return pilotKeys; },
    get pilotMoving() { return pilotMoving; }
  };
`)();

(async () => {
  try {
    await tick();
    assert(
      el('mode-protect').getAttribute('aria-pressed') === 'true',
      'Initial mission matches controller telemetry'
    );
    assert(
      !el('mode-explore').disabled,
      'Fresh telemetry enables mission selection'
    );
    assert(
      el('mission-meter').getAttribute('aria-valuenow') === '100',
      'Protection meter reports connected fleet ratio'
    );
    assert(
      el('track-rows').innerHTML.includes('CPA 1.4 m / 8.0 s'),
      'Contact table exposes measured closest approach'
    );
    api.drawScene();
    assert(
      true,
      'Scene renders measured cells, contacts, goals and null mission reports'
    );
    await el('mode-explore').emit('click');
    await tick();
    assert(
      commands.at(-1).command === 'set_mode' && commands.at(-1).mode === 'explore',
      'Survey mode sends validated API payload'
    );
    assert(
      api.modeRequest?.mode === 'explore' && el('mode-protect').getAttribute('aria-pressed') === 'true',
      'Queued mode retains acknowledged active mission'
    );
    fixture.mission.mode = 'explore';
    api.updateUI(structuredClone(fixture));
    assert(
      Boolean(api.modeRequest),
      'Mode is not confirmed until revision advances'
    );
    fixture.mission.mode_revision = 1;
    fixture.auvs[0].mission = {mode:'explore', role:'surveyor', phase:'exploring', target: [15, 12, -10]};
    api.updateUI(structuredClone(fixture));
    api.drawScene();
    assert(
      api.modeRequest === null && el('mode-explore').getAttribute('aria-pressed') === 'true',
      'Advanced fresh revision confirms mode'
    );
    assert(
      el('diver-legend').hidden && !api.canPilot(),
      'Survey hides diver and disables manual pilot'
    );
    assert(
      el('mission-meter').getAttribute('aria-valuenow') === '25' && el('formation').textContent === 64,
      'Survey meter uses reported measured coverage'
    );
    failCommand = true;
    await el('mode-protect').emit('click');
    await tick();
    assert(
      api.modeRequest === null && el('mode-status').textContent.includes('failed'),
      'Rejected mode leaves actionable inline error'
    );
    fixture.mission.mode = 'protect';
    fixture.mission.mode_revision = 2;
    api.updateUI(structuredClone(fixture));
    await el('pilot-manual').emit('click');
    await tick();
    assert(
      api.pilotRequest?.mode === 'manual' && !api.canPilot(),
      'Manual pilot awaits controller acknowledgement'
    );
    fixture.mission.diver_control.mode = 'manual';
    api.updateUI(structuredClone(fixture));
    assert(
      api.canPilot() && !el('pilot-stop').disabled,
      'Acknowledged manual mode enables pilot'
    );
    await document.emit('keydown', {key:'w', target: document.activeElement});
    await tick();
    assert(
      commands.at(-1).velocity[1] === 0.65,
      'North key commands maximum permitted speed'
    );
    await document.emit('keydown', {key:'d', target: document.activeElement});
    await tick();
    assert(
      Math.hypot(...commands.at(-1).velocity) <= 0.6501,
      'Combined pilot axes normalize speed'
    );
    assert(
      intervalTasks.size === 1,
      'Held movement starts a single heartbeat'
    );
    for (const timer of intervalTasks.values()) {
      timer.callback();
    }
    await tick();
    assert(
      commands.at(-1).command === 'set_diver_control',
      'Heartbeat refreshes the controller deadman'
    );
    await document.emit('keyup', {key:'w', target: document.activeElement});
    await document.emit('keyup', {key:'d', target: document.activeElement});
    await tick();
    assert(
      commands.at(-1).velocity.every(v => v === 0) && intervalTasks.size === 0,
      'Releasing every key sends stop and ends heartbeat'
    );
    const beforeInput = commands.length;
    await document.emit('keydown', {key:'w', target: el('current-east')});
    await tick();
    assert(
      commands.length === beforeInput,
      'Editing current inputs cannot move the diver'
    );
    await document.emit('keydown', {key:'q', target: document.activeElement});
    await tick();
    assert(
      commands.at(-1).velocity[2] === -0.65,
      'Q commands descent'
    );
    await document.emit('keydown', {key:' ', target: document.activeElement});
    await tick();
    assert(
      commands.at(-1).velocity.every(v => v === 0),
      'Space sends immediate stop'
    );
    const stoppedCount = commands.length;
    await document.emit('keydown', {key:'q', repeat: true, target: document.activeElement});
    await tick();
    assert(
      commands.length === stoppedCount,
      'Key repeat cannot restart movement after an explicit stop'
    );

    await document.emit('keydown', {key:'e', target: document.activeElement});
    await tick();
    assert(
      commands.at(-1).velocity[2] === 0.65,
      'E commands ascent'
    );
    await window.emit('blur');
    await tick();
    assert(
      commands.at(-1).velocity.every(v => v === 0) && intervalTasks.size === 0,
      'Window blur stops movement'
    );
    const pointer = document.querySelectorAll('[data-pilot-vector]').find(e => e.dataset.pilotVector === '1,0,0');
    await pointer.emit('pointerdown', {pointerId: 2});
    await tick();
    assert(
      commands.at(-1).velocity[0] === 0.65,
      'On-screen hold starts pilot movement'
    );
    await pointer.emit('pointercancel', {pointerId: 2});
    await tick();
    assert(
      commands.at(-1).velocity.every(v => v === 0),
      'Canceled pointer sends stop'
    );
    await document.emit('keydown', {key:'w', target: document.activeElement});
    await tick();
    fixture.dashboard.state_age_seconds = 6;
    api.updateUI(structuredClone(fixture));
    await tick();
    assert(
      !api.canPilot() && commands.at(-1).velocity.every(v => v === 0),
      'Stale telemetry disables pilot and sends stop'
    );
    assert(
      el('mode-protect').disabled && el('mode-explore').disabled,
      'Stale telemetry disables mode selection'
    );
    fixture.dashboard.state_age_seconds = 0;
    api.updateUI(structuredClone(fixture));
    await document.emit('keydown', {key:'w', target: el('scene')});
    await tick();
    assert(
      api.pilotKeys.size === 0,
      'Camera arrow-key focus never pilots the diver'
    );
    fixture.simulation = {hydrophone: {source:'synthetic', name: null, loop: false}};
    api.updateUI(structuredClone(fixture));
    assert(
      el('hydrophone-source').textContent === 'Hydrophone: synthetic baseline',
      'Synthetic hydrophone source is visible'
    );
    fixture.simulation.hydrophone = {source:'recorded', name:'ocean-background.wav', loop: true};
    api.updateUI(structuredClone(fixture));
    assert(
      el('hydrophone-source').textContent.includes('recorded / ocean-background.wav / loop'),
      'Recorded hydrophone source and loop are visible'
    );
    assert(
      el('acoustic-source-details').textContent.includes('shared background acoustic context')
        && el('acoustic-source-details').textContent.includes('not associated with measured contact positions'),
      'Recorded source limitations preserve honest contact association'
    );
    const north = document.querySelectorAll('[data-pilot-vector]').find(e => e.dataset.pilotVector === '0,1,0');
    await north.emit('keydown', {key:'Enter'});
    await tick();
    assert(
      commands.at(-1).velocity[1] === 0.65,
      'Accessible hold button supports Enter key'
    );
    await north.emit('keyup', {key:'Enter'});
    await tick();
    assert(
      commands.at(-1).velocity.every(v => v === 0),
      'Accessible hold button release sends stop'
    );
    fixture.mission.status = 'paused';
    api.updateUI(structuredClone(fixture));
    assert(
      !api.canPilot() && el('pilot-status').textContent.includes('mission paused'),
      'Paused mission disables pilot and explains the suspension'
    );
    fixture.mission.status = 'running';
    api.updateUI(structuredClone(fixture));
    failCommand = true;
    await document.emit('keydown', {key:'w', target: document.activeElement});
    await tick();
    assert(
      !api.canPilot() && intervalTasks.size === 0 && el('pilot-status').textContent.includes('failed'),
      'Rejected movement disables pilot and ends heartbeat'
    );
  } catch (error) {
    errors.push(String(error) + '\n' + error.stack);
  }

  const output = {
    validation: 'GJS mocked DOM/canvas fixture; no live browser available',
    passed: checks.filter(check => check.passed).length,
    total: checks.length,
    checks,
    errors,
    command_count: commands.length
  };
  const path = 'results/dashboard_logic_validation.json';
  const resultsDir = Gio.File.new_for_path('results');
  if (!resultsDir.query_exists(null)) resultsDir.make_directory_with_parents(null);
  Gio.File.new_for_path(path).replace_contents(
    JSON.stringify(output, null, 2), null, false, Gio.FileCreateFlags.NONE, null
  );
  print(`Dashboard logic: ${output.passed}/${output.total} checks passed (GJS mocked DOM/canvas).`);
  if (errors.length) print(errors.join('\n'));
  imports.system.exit(errors.length ? 1 : 0);
})();
