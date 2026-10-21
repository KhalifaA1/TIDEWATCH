'use strict';

const $ = id => document.getElementById(id);
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const number = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits);
const percent = (value, digits = 1) => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(digits)}%`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const position = object => Array.isArray(object?.position) && object.position.length >= 3 ? object.position.map(v => finite(v)) : null;
const distance = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
const colors = {mint:'#4fd8c0',blue:'#6cb4e8',amber:'#f0b45a',red:'#f0766e',map:'#7fa0b0'};
const nodeColors = ['#4fd8c0','#6cb4e8','#a99bea'];
let state = null, lastSeen = 0, lastSimTime = null, lastTableUpdate = 0, gatewayConnected = false;
const history = {coverage:[],localization:[],mapping:[],classification:[],network:[]};
const trails = new Map();
const scene = $('scene'), ctx = scene.getContext('2d');
const camera = {yaw:-.58,elevation:.56,zoom:1,target:[0,0,-12],top:false};
let width = 900, height = 478, scale = 11, dragging = false, oldPointer = [0,0], toastTimer, labelBoxes=[];
const currentInputs = ['current-east','current-north','current-up'].map($);
let currentEdited = false;
let modeRequest = null, modeError = '', pilotRequest = null, pilotError = '';
const pilotKeys = new Set(), pilotPointers = new Map();
const pilotKeyVectors = {w:[0,1,0],arrowup:[0,1,0],s:[0,-1,0],arrowdown:[0,-1,0],a:[-1,0,0],arrowleft:[-1,0,0],d:[1,0,0],arrowright:[1,0,0],q:[0,0,-1],e:[0,0,1]};
let pilotHeartbeat = null, pilotQueued = null, pilotInFlight = null, pilotMoving = false;
const activeMode = mission => mission?.mode === 'explore' ? 'explore' : ['protect','diver_escort'].includes(mission?.mode) ? 'protect' : null;
const goalPosition = auv => Array.isArray(auv.mission?.target) && auv.mission.target.length === 3 && auv.mission.target.every(v => Number.isFinite(Number(v))) ? auv.mission.target.map(Number) : null;
const isFresh = () => Boolean(gatewayConnected && state && finite(state.dashboard?.state_age_seconds) <= 5 && Date.now()-lastSeen < 5000);

function displayTime(seconds) {
  const t = Math.max(0, Math.floor(finite(seconds)));
  return [Math.floor(t/3600), Math.floor(t/60)%60, t%60].map(x => String(x).padStart(2,'0')).join(':');
}

function riskCategory(value) {
  if (typeof value === 'number') return value >= .7 ? 'danger' : value >= .3 ? 'warning' : 'info';
  const text = String(value ?? '').toLowerCase();
  return /critical|danger|high|collision|imminent/.test(text) ? 'danger' : /warn|review|medium|unknown|potential|hazard/.test(text) ? 'warning' : 'info';
}

function labelFor(value) {
  const labels = {diver:'Diver',swimmer:'Swimmer',person:'Person',marine_life:'Marine life',marine:'Marine life',fish:'Marine life',hazard:'Environmental hazard',environmental_hazard:'Environmental hazard',unknown:'Unknown object',vessel:'Vessel',boat:'Vessel',obstacle:'Obstacle',rock:'Rock',debris:'Debris',auv:'AUV',unusual_behavior:'Unusual activity',biological:'Marine life',engine:'Engine source'};
  return labels[value] || String(value || 'Unknown object').replace(/_/g,' ').replace(/^./, x => x.toUpperCase());
}

function showToast(message, error = false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').className = `visible${error?' error':''}`;
  toastTimer = setTimeout(() => $('toast').className = '', 3300);
}

async function sendCommand(command, button, fields = {}) {
  button.disabled = true;
  try {
    await haltPilot();
    const response = await fetch('/api/control', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...fields})});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || 'Command unavailable');
    showToast({start:'Start command queued',pause:'Pause command queued',reset:'Reset command queued',toggle_outage:'Link command queued',set_current:'Current command queued',set_mode:'Mission change queued; awaiting controller',set_diver_control:'Diver control change queued; awaiting controller'}[command] || 'Command queued');
    return true;
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
}

document.querySelectorAll('[data-command]').forEach(button => button.addEventListener('click', () => sendCommand(button.dataset.command,button)));
document.querySelectorAll('[data-mission]').forEach(button => button.addEventListener('click', async () => {
  const mode = button.dataset.mission;
  if (modeRequest || activeMode(state?.mission) === mode) return;
  modeError = '';
  modeRequest = {mode,revision:finite(state?.mission?.mode_revision,-1),at:Date.now(),sending:true};
  updateMissionMode(state?.mission || {},state?.auvs || []);
  const accepted = await sendCommand('set_mode',button,{mode});
  if (modeRequest?.mode === mode) {
    if (accepted) modeRequest.sending = false;
    else {modeRequest = null;modeError = 'Mission request failed. Select a mode to retry.';}
  }
  updateMissionMode(state?.mission || {},state?.auvs || []);
}));
currentInputs.forEach(input => input.addEventListener('input', () => {currentEdited = true;}));
document.querySelectorAll('[data-current]').forEach(button => button.addEventListener('click', async () => {
  const vector = button.dataset.current.split(',').map(Number);
  currentInputs.forEach((input,index) => {input.value = vector[index];});
  currentEdited = true;
  await sendCommand('set_current',button,{vector});
}));
$('current-form').addEventListener('submit', async event => {
  event.preventDefault();
  const vector = currentInputs.map(input => Number(input.value));
  if (vector.some(value => !Number.isFinite(value) || Math.abs(value) > .8)) {
    showToast('Each current component must be between -0.8 and 0.8 m/s',true);
    return;
  }
  await sendCommand('set_current',$('apply-current'),{vector});
});
function centerCamera() {
  const auvs = (state?.auvs || []).map(position).filter(Boolean);
  const origin = state?.mission?.exploration?.origin;
  camera.target = activeMode(state?.mission) === 'explore' ? Array.isArray(origin) && origin.length === 3 ? origin.map(v => finite(v)) : auvs.length ? [0,1,2].map(axis => auvs.reduce((total,p) => total+p[axis],0)/auvs.length) : [0,0,-12] : position(state?.diver) || [0,0,-12];
  camera.zoom = 1;
}
$('center').addEventListener('click',centerCamera);
$('topview').addEventListener('click', () => {camera.top = !camera.top;$('topview').textContent = camera.top ? '⌗ Perspective' : '⌗ Top view';$('topview').setAttribute('aria-pressed',String(camera.top));});
scene.addEventListener('keydown', event => {
  if (event.key === 'Home') centerCamera();
  else if (event.key === 'ArrowLeft') camera.yaw -= .08;
  else if (event.key === 'ArrowRight') camera.yaw += .08;
  else if (event.key === 'ArrowUp') camera.elevation = Math.min(1.35,camera.elevation+.06);
  else if (event.key === 'ArrowDown') camera.elevation = Math.max(.18,camera.elevation-.06);
  else if (['+','='].includes(event.key)) camera.zoom = Math.min(3,camera.zoom*1.1);
  else if (event.key === '-') camera.zoom = Math.max(.45,camera.zoom/1.1);
  else return;
  event.preventDefault();
});
scene.addEventListener('pointerdown', event => {dragging = true;oldPointer = [event.clientX,event.clientY];scene.setPointerCapture(event.pointerId);scene.style.cursor = 'grabbing';});
scene.addEventListener('pointermove', event => {if (!dragging) return;camera.yaw += (event.clientX-oldPointer[0])*.005;camera.elevation = Math.max(.18,Math.min(1.35,camera.elevation+(event.clientY-oldPointer[1])*.004));oldPointer = [event.clientX,event.clientY];});
scene.addEventListener('pointerup', () => {dragging = false;scene.style.cursor = 'grab';});
scene.addEventListener('pointercancel', () => {dragging = false;scene.style.cursor = 'grab';});
scene.addEventListener('wheel', event => {event.preventDefault();camera.zoom = Math.max(.45,Math.min(3,camera.zoom*Math.exp(-event.deltaY*.001)));},{passive:false});
scene.style.cursor = 'grab';

function canPilot() {
  return isFresh() && activeMode(state?.mission) === 'protect' && state?.mission?.diver_control?.mode === 'manual' && !pilotRequest && !modeRequest && !pilotError && String(state?.mission?.status || 'running') === 'running';
}

function pilotVector() {
  const vectors = [...pilotKeys].map(key => pilotKeyVectors[key]).concat([...pilotPointers.values()].map(button => button.dataset.pilotVector.split(',').map(Number)));
  const vector = [0,1,2].map(axis => vectors.reduce((total,v) => total+v[axis],0));
  const magnitude = Math.hypot(...vector);
  return vector.map(value => magnitude ? Number((value/magnitude*.65).toFixed(4)) : 0);
}

function updatePilotHeld() {
  document.querySelectorAll('[data-pilot-vector]').forEach(button => {
    const vector = button.dataset.pilotVector;
    button.classList.toggle('held',[...pilotPointers.values()].includes(button) || [...pilotKeys].some(key => pilotKeyVectors[key].join(',') === vector));
  });
}

function queuePilot(vector) {
  pilotQueued = vector;
  pilotMoving = vector.some(value => value !== 0);
  if (pilotInFlight) return pilotInFlight;
  pilotInFlight = (async () => {
    while (pilotQueued) {
      const requested = pilotQueued;pilotQueued = null;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(),2200);
      try {
        const response = await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'set_diver_control',mode:'manual',velocity:requested}),signal:controller.signal});
        const result = await response.json();
        if (!response.ok) throw Error(result.error || 'Diver control unavailable');
      } catch (error) {
        pilotQueued = null;pilotMoving = false;pilotKeys.clear();pilotPointers.clear();clearInterval(pilotHeartbeat);pilotHeartbeat = null;updatePilotHeld();
        pilotError = 'Diver command failed. Movement stopped sending; controller deadman will stop the diver.';
        showToast(error.name === 'AbortError' ? 'Diver control timed out' : error.message,true);
        updatePilotControls(state?.mission || {});
      } finally {clearTimeout(timeout);}
    }
  })().finally(() => {
    pilotInFlight = null;
    // A release can arrive after the drain loop exits but before this promise settles.
    // Drain that last update so a stop command cannot be left waiting for a heartbeat.
    if (pilotQueued) return queuePilot(pilotQueued);
  });
  return pilotInFlight;
}

function holdPilot() {
  updatePilotHeld();
  if (!canPilot()) return;
  queuePilot(pilotVector());
  if (!pilotHeartbeat) pilotHeartbeat = setInterval(() => {
    if (!canPilot()) {haltPilot();return;}
    queuePilot(pilotVector());
  },250);
}

function haltPilot() {
  const needsStop = pilotMoving || pilotInFlight || pilotHeartbeat;
  pilotKeys.clear();pilotPointers.clear();clearInterval(pilotHeartbeat);pilotHeartbeat = null;updatePilotHeld();
  return needsStop ? queuePilot([0,0,0]) : Promise.resolve();
}

function releasePilot() {
  updatePilotHeld();
  if (pilotKeys.size || pilotPointers.size) holdPilot();
  else haltPilot();
}

document.querySelectorAll('[data-pilot-mode]').forEach(button => button.addEventListener('click',async () => {
  const mode = button.dataset.pilotMode;
  if (pilotRequest || (state?.mission?.diver_control?.mode === mode && !pilotError)) return;
  pilotError = '';pilotRequest = {mode,at:Date.now()};
  updatePilotControls(state?.mission || {});
  const accepted = await sendCommand('set_diver_control',button,{mode,velocity:[0,0,0]});
  if (!accepted) {pilotRequest = null;pilotError = 'Control request failed. Select the control mode to retry.';}
  updatePilotControls(state?.mission || {});
}));
document.querySelectorAll('[data-pilot-vector]').forEach(button => {
  button.addEventListener('pointerdown',event => {
    if (!canPilot()) return;
    event.preventDefault();button.setPointerCapture(event.pointerId);pilotPointers.set(event.pointerId,button);holdPilot();
  });
  const release = event => {if (pilotPointers.delete(event.pointerId)) releasePilot();};
  button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);
  button.addEventListener('keydown',event => {
    const key = `keyboard-${button.dataset.pilotVector}`;
    if (event.key !== 'Enter' || !canPilot() || (event.repeat && !pilotPointers.has(key))) return;
    event.preventDefault();pilotPointers.set(key,button);holdPilot();
  });
  button.addEventListener('keyup',event => {
    if (event.key !== 'Enter') return;
    event.preventDefault();pilotPointers.delete(`keyboard-${button.dataset.pilotVector}`);releasePilot();
  });
  button.addEventListener('blur',() => {if (pilotPointers.delete(`keyboard-${button.dataset.pilotVector}`)) releasePilot();});
});
$('pilot-stop').addEventListener('click',haltPilot);
const blocksPilotKeys = target => target === scene || target?.isContentEditable || ['INPUT','TEXTAREA','SELECT','SUMMARY'].includes(target?.tagName);
document.addEventListener('keydown',event => {
  if (!canPilot() || blocksPilotKeys(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === ' ') {event.preventDefault();haltPilot();return;}
  if (!pilotKeyVectors[key]) return;
  event.preventDefault();if (event.repeat && !pilotKeys.has(key)) return;
  if (!pilotKeys.has(key)) {pilotKeys.add(key);holdPilot();}
});
document.addEventListener('keyup',event => {if (pilotKeys.delete(event.key.toLowerCase())) {event.preventDefault();releasePilot();}});
document.addEventListener('focusin',event => {if (blocksPilotKeys(event.target)) haltPilot();});
window.addEventListener('blur',haltPilot);
document.addEventListener('visibilitychange',() => {if (document.hidden) haltPilot();});

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(2,window.devicePixelRatio || 1);
  const w = Math.round(rect.width*dpr), h = Math.round(rect.height*dpr);
  if (canvas.width !== w || canvas.height !== h) {canvas.width = w;canvas.height = h;}
  const context = canvas.getContext('2d');context.setTransform(dpr,0,0,dpr,0,0);
  return [rect.width,rect.height,context];
}

function project(p) {
  const x=p[0]-camera.target[0],y=p[1]-camera.target[1],z=p[2]-camera.target[2];
  const u=x*Math.cos(camera.yaw)-y*Math.sin(camera.yaw),v=x*Math.sin(camera.yaw)+y*Math.cos(camera.yaw);
  const elevation = camera.top ? Math.PI/2 : camera.elevation;
  return [width*.5+u*scale,height*.5-(v*Math.sin(elevation)+z*Math.cos(elevation))*scale, v*Math.cos(elevation)-z*Math.sin(elevation)];
}

function line(a,b,color,lineWidth=1,dash=[]) {
  const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);
}

function polygon(points,fill,stroke,lineWidth=1) {
  if (!points.length) return;
  ctx.beginPath();points.forEach((point,i) => {const p=project(point);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]);});ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke();}
}

function circleWorld(center,radius,color,lineWidth=1,fill=null,dash=[]) {
  const points=[];for(let i=0;i<=64;i++){const theta=i/64*Math.PI*2;points.push([center[0]+Math.cos(theta)*radius,center[1]+Math.sin(theta)*radius,center[2]]);}
  ctx.setLineDash(dash);polygon(points,fill,color,lineWidth);ctx.setLineDash([]);
}

function uncertaintyHalo(p,uncertainty,color) {
  const radius=Math.min(10,Math.max(.15,finite(uncertainty,.5)));
  circleWorld(p,radius,`${color}60`,1,`${color}0a`);
  if (!camera.top) {
    const q=project(p),r=radius*scale;
    ctx.beginPath();ctx.ellipse(q[0],q[1],r*.3,r,0,0,Math.PI*2);ctx.strokeStyle=`${color}25`;ctx.lineWidth=1;ctx.stroke();
  }
}

function textLabel(p,title,subtitle,color,offset=null) {
  if(width<600)subtitle='';
  ctx.font='600 9px ui-sans-serif, sans-serif';const titleWidth=ctx.measureText(title).width;
  ctx.font='8px ui-monospace, monospace';const boxWidth=Math.max(titleWidth,subtitle?ctx.measureText(subtitle).width:0)+10;
  const q=project(p),x=q[0]+(offset?offset[0]:q[0]<width*.5?-boxWidth-10:16),initialY=q[1]+(offset?offset[1]:q[1]<height*.5?-14:20),boxHeight=subtitle?29:18;
  let y=initialY;
  for(const shift of [0,-34,34,-68,68]){const candidate={x:x-5,y:initialY+shift-11,w:boxWidth,h:boxHeight};if(!labelBoxes.some(b=>candidate.x<b.x+b.w+5&&candidate.x+candidate.w+5>b.x&&candidate.y<b.y+b.h+4&&candidate.y+candidate.h+4>b.y)){y=initialY+shift;break;}}
  labelBoxes.push({x:x-5,y:y-11,w:boxWidth,h:boxHeight});
  ctx.fillStyle='#06121adb';ctx.fillRect(x-5,y-11,boxWidth,boxHeight);ctx.strokeStyle=`${color}40`;ctx.lineWidth=1;ctx.strokeRect(x-5,y-11,boxWidth,boxHeight);
  ctx.font='600 9px ui-sans-serif, sans-serif';
  ctx.fillStyle=color;ctx.fillText(title,x,y);
  if(subtitle){ctx.font='8px ui-monospace, monospace';ctx.fillStyle='#a7b7c3';ctx.fillText(subtitle,x,y+12);}
  ctx.beginPath();ctx.moveTo(q[0]+5,q[1]-3);ctx.lineTo(x-5,y-3);ctx.strokeStyle=`${color}55`;ctx.stroke();
}

function drawAuv(auv,index) {
  const p=position(auv);if(!p)return;
  const color=auv.online===false?colors.red:nodeColors[index%nodeColors.length];
  const velocity=Array.isArray(auv.velocity)?auv.velocity:[1,0,0];
  const heading=auv.yaw!=null&&Number.isFinite(Number(auv.yaw))?Number(auv.yaw):Math.atan2(finite(velocity[1]),finite(velocity[0],1));
  const local=(x,y,z=0)=>[p[0]+x*Math.cos(heading)-y*Math.sin(heading),p[1]+x*Math.sin(heading)+y*Math.cos(heading),p[2]+z];
  const q=project(p);
  uncertaintyHalo(p,auv.uncertainty,color);
  if(auv.deterrent?.active){
    // Expanding rings show an active deterrent speaker; the outer ring is its audible range.
    const phase=(performance.now()/1400)%1;
    for(const k of [0,.33,.66]){const f=(phase+k)%1;circleWorld(p,1+f*22,`${colors.amber}${Math.round((1-f)*90).toString(16).padStart(2,'0')}`,1.5);}
    circleWorld(p,24,`${colors.amber}30`,1,null,[2,6]);
  }
  line([p[0],p[1],-20],p,`${color}24`,1,[3,5]);
  circleWorld([p[0],p[1],-20],.45,`${color}48`,1,`${color}15`);
  // Inspection ROV: an open square frame, protected motor pods and front sonar.
  const frame=[local(-.85,-.75),local(.85,-.75),local(.85,.75),local(-.85,.75)];
  polygon(frame,'#0d273970',color,2);
  polygon([local(-.85,-.75,.5),local(.85,-.75,.5),local(.85,.75,.5),local(-.85,.75,.5)],null,`${color}a0`,1.5);
  for(const [x,y] of [[-.85,-.75],[.85,-.75],[.85,.75],[-.85,.75]])line(local(x,y),local(x,y,.5),`${color}95`,1.3);
  polygon([local(-.42,-.43,.2),local(.42,-.43,.2),local(.42,.43,.2),local(-.42,.43,.2)],'#1c474d','#75bac0',1);
  line(local(-.35,0,.5),local(.35,0,.5),'#a6b09a',2.5);
  for(const [x,y] of [[-.72,-.63],[.72,-.63],[.72,.63],[-.72,.63]]){
    polygon([local(x-.22,y-.23,.1),local(x+.22,y-.23,.1),local(x+.22,y+.23,.1),local(x-.22,y+.23,.1)],'#09202c',`${color}b0`,1.2);
    const pod=project(local(x,y,.15)),r=Math.max(2.3,.14*scale);
    ctx.beginPath();ctx.arc(pod[0],pod[1],r,0,Math.PI*2);ctx.fillStyle='#183845';ctx.fill();ctx.strokeStyle='#8cc2c5';ctx.lineWidth=1;ctx.stroke();
    ctx.beginPath();ctx.moveTo(pod[0]-r*.6,pod[1]);ctx.lineTo(pod[0]+r*.6,pod[1]);ctx.moveTo(pod[0],pod[1]-r*.6);ctx.lineTo(pod[0],pod[1]+r*.6);ctx.strokeStyle=`${color}a0`;ctx.stroke();
  }
  polygon([local(.82,-.24,.22),local(1.12,-.24,.22),local(1.12,.24,.22),local(.82,.24,.22)],'#234861',colors.blue,1.4);
  line(local(1.13,-.23,.22),local(1.13,.23,.22),'#c8d5df',2);
  for(const reach of [1.8,2.5]){
    const scan=[];for(let i=0;i<=12;i++){const angle=-.42+i/12*.84;scan.push(local(.95+reach*Math.cos(angle),reach*Math.sin(angle),.22));}
    ctx.beginPath();scan.forEach((point,i)=>{const projected=project(point);i?ctx.lineTo(projected[0],projected[1]):ctx.moveTo(projected[0],projected[1]);});ctx.strokeStyle='#98adbf35';ctx.lineWidth=1;ctx.stroke();
  }
  const nodeName=String(auv.id || `auv_${index}`).replace(/^auv_/i,'AUV 0');
  const phase=activeMode(auv.mission)===activeMode(state?.mission)?String(auv.mission?.phase || auv.mission?.role || 'ONLINE').replace(/_/g,' ').toUpperCase():'AWAITING DIRECTIVE';
  textLabel(p,nodeName.toUpperCase(),`${auv.online===false?'LINK LOST':phase} · ${Math.abs(p[2]).toFixed(1)} m`,color);
}

function drawDiver(diver) {
  const p=position(diver);if(!p)return;
  const q=project(p),color=colors.blue;
  uncertaintyHalo(p,diver.uncertainty??.35,color);
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1]-8,3.1,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(q[0],q[1]-4);ctx.lineTo(q[0],q[1]+5);ctx.moveTo(q[0]-6,q[1]);ctx.lineTo(q[0]+6,q[1]);ctx.moveTo(q[0],q[1]+5);ctx.lineTo(q[0]-5,q[1]+12);ctx.moveTo(q[0],q[1]+5);ctx.lineTo(q[0]+5,q[1]+12);ctx.stroke();
  textLabel(p,width<600?'DIVER':'PROTECTED DIVER',`DEPTH ${Math.abs(p[2]).toFixed(1)} m`,color,[21,-19]);
}

function drawTrack(track) {
  const p=position(track);if(!p)return;
  const risk=riskCategory(track.risk),color=risk==='danger'?colors.red:risk==='warning'?colors.amber:'#8fb3c9';
  uncertaintyHalo(p,track.uncertainty,color);
  const q=project(p);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(q[0],q[1]-5);ctx.lineTo(q[0]+5,q[1]);ctx.lineTo(q[0],q[1]+5);ctx.lineTo(q[0]-5,q[1]);ctx.closePath();ctx.fillStyle=`${color}35`;ctx.fill();ctx.stroke();
  const velocity=track.velocity || [];if(Math.hypot(finite(velocity[0]),finite(velocity[1]))>.15)line(p,[p[0]+finite(velocity[0])*3,p[1]+finite(velocity[1])*3,p[2]+finite(velocity[2])*3],`${color}7a`,1,[4,3]);
  textLabel(p,labelFor(track.label).toUpperCase(),`${String(track.id||'TRACK').slice(0,15)} · ${percent(track.confidence,0)}`,color);
}

function drawMissionGoals(auvs) {
  const mode=activeMode(state?.mission),exploration=state?.mission?.exploration;
  const bounds=exploration?.bounds;
  if(mode==='explore'&&Array.isArray(bounds)&&bounds.length>=2&&bounds.slice(0,2).every(pair=>Array.isArray(pair)&&pair.length===2&&pair.every(v=>Number.isFinite(Number(v))))) {
    const depth=Array.isArray(exploration.depth_layers)&&exploration.depth_layers.length?finite(exploration.depth_layers[0],-10):-10;
    const corners=[[bounds[0][0],bounds[1][0],depth],[bounds[0][1],bounds[1][0],depth],[bounds[0][1],bounds[1][1],depth],[bounds[0][0],bounds[1][1],depth]];
    ctx.setLineDash([7,6]);polygon(corners,null,'#9cabbb60',1);ctx.setLineDash([]);
    const q=project(corners[3]);ctx.fillStyle='#8899a7';ctx.font='8px ui-monospace, monospace';ctx.fillText('SURVEY BOUNDS / UNOBSERVED AREA UNKNOWN',q[0]+6,q[1]-7);
  }
  auvs.forEach((auv,index)=>{
    if(activeMode(auv.mission)!==mode)return;
    const start=position(auv),target=goalPosition(auv);if(!start||!target)return;
    const color=nodeColors[index%nodeColors.length],q=project(target);
    line(start,target,`${color}85`,1,[5,6]);
    ctx.beginPath();ctx.rect(q[0]-5,q[1]-5,10,10);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.stroke();
    ctx.beginPath();ctx.moveTo(q[0]-8,q[1]);ctx.lineTo(q[0]+8,q[1]);ctx.moveTo(q[0],q[1]-8);ctx.lineTo(q[0],q[1]+8);ctx.strokeStyle=`${color}80`;ctx.stroke();
    if(width>650){ctx.font='8px ui-monospace, monospace';ctx.fillStyle=color;ctx.fillText(`${String(auv.id||index+1).toUpperCase()} GOAL`,q[0]+11,q[1]-5);}
  });
}

function drawScene() {
  [width,height]=resizeCanvas(scene);scale=Math.min(width/80,height/44)*camera.zoom;labelBoxes=[];
  const sky=ctx.createLinearGradient(0,0,0,height);sky.addColorStop(0,'#0a2a3b');sky.addColorStop(.55,'#07161f');sky.addColorStop(1,'#040d13');ctx.fillStyle=sky;ctx.fillRect(0,0,width,height);
  // Explicit visual reference grid; observed seabed points are drawn separately.
  polygon([[-35,-35,-20],[35,-35,-20],[35,35,-20],[-35,35,-20]],'#0d3a4a40','#2a6a7e55');
  for(let v=-30;v<=30;v+=5){line([v,-35,-20],[v,35,-20],v===0?'#3a7a8c70':'#25566644');line([-35,v,-20],[35,v,-20],v===0?'#3a7a8c70':'#25566644');}
  for(let v=-20;v<=-5;v+=5){line([-35,35,v],[35,35,v],'#15374233');line([-35,-35,v],[-35,35,v],'#15374233');}
  line([-35,35,-20],[-35,35,-3],'#28525f66');line([35,35,-20],[35,35,-3],'#28525f44');
  const axisOrigin=[-31,-31,-20];line(axisOrigin,[-24,-31,-20],'#98adbf88');line(axisOrigin,[-31,-24,-20],'#a0b2a188');line(axisOrigin,[-31,-31,-13],'#c5ad7677');
  [[[-23,-31,-20],'E',colors.blue],[[-31,-23,-20],'N',colors.mint],[[-31,-31,-12],'UP',colors.amber]].forEach(([p,t,c])=>{const q=project(p);ctx.fillStyle=c;ctx.font='8px monospace';ctx.fillText(t,q[0]+3,q[1]);});
  const measuredCells=state?.map?.elevation || [],resolution=finite(state?.map?.resolution_m,1),cellStride=Math.max(1,Math.ceil(measuredCells.length/1800));
  for(let i=0;i<measuredCells.length;i+=cellStride){const cell=measuredCells[i];if(![cell.x,cell.y,cell.z].every(v=>Number.isFinite(Number(v))))continue;const x=Number(cell.x),y=Number(cell.y),z=Number(cell.z),r=resolution*.48;polygon([[x-r,y-r,z],[x+r,y-r,z],[x+r,y+r,z],[x-r,y+r,z]],'#6fb7c026',null);}
  const points=state?.map?.points || [];
  const stride=Math.max(1,Math.ceil(points.length/3500));
  for(let i=0;i<points.length;i+=stride){const p=points[i];if(!Array.isArray(p)||p.length<3)continue;const q=project(p);if(q[0]<0||q[0]>width||q[1]<0||q[1]>height)continue;const elevation=Math.max(0,p[2]+20);ctx.fillStyle=elevation>1?'#9db0ad9c':'#81959d91';const r=elevation>1?2.5:1.8;ctx.fillRect(q[0]-r/2,q[1]-r/2,r,r);}
  if(state){
    const diver=activeMode(state.mission)==='explore'?null:position(state.diver),auvs=(state.auvs||[]).filter(a=>position(a));
    if(diver){
      const radius=finite(state.mission?.protected_radius,state.protected_radius || 6);
      circleWorld(diver,radius,'#98adbf88',1.3,'#27678910',[5,5]);
      const ringPoint=[diver[0]+radius,diver[1],diver[2]];const q=project(ringPoint);ctx.fillStyle='#6c9eb5';ctx.font='7px monospace';ctx.fillText('PROTECTED ZONE',q[0]+7,q[1]+15);
      if(auvs.length>=3){const ordered=[...auvs].sort((a,b)=>Math.atan2(position(a)[1]-diver[1],position(a)[0]-diver[0])-Math.atan2(position(b)[1]-diver[1],position(b)[0]-diver[0]));polygon(ordered.map(position),'#a0b2a108',auvs.every(a=>a.online!==false)?'#a0b2a145':'#cf868045',1);}
    }
    drawMissionGoals(auvs);
    trails.forEach((trail,id)=>{if(trail.length<2)return;const index=auvs.findIndex(a=>String(a.id)===id);for(let i=1;i<trail.length;i++){const alpha=Math.round(i/trail.length*80).toString(16).padStart(2,'0');line(trail[i-1],trail[i],`${index>=0?nodeColors[index%nodeColors.length]:colors.amber}${alpha}`,1);}});
    const objects=[...auvs.map((o,i)=>({o,type:'auv',i})),...(state.tracks||[]).filter(t=>position(t)).map(o=>({o,type:'track'}))];if(diver)objects.push({o:state.diver,type:'diver'});
    objects.sort((a,b)=>project(position(b.o))[2]-project(position(a.o))[2]);
    objects.forEach(item=>item.type==='auv'?drawAuv(item.o,item.i):item.type==='diver'?drawDiver(item.o):drawTrack(item.o));
  }
  $('scale-label').textContent=`${(65/scale).toFixed(0)} m`;
  requestAnimationFrame(drawScene);
}

function drawChart(id,data,color,{minimum=0,maximum=null}={}) {
  const canvas=$(id);if(!canvas)return;
  const [w,h,c]=resizeCanvas(canvas);c.clearRect(0,0,w,h);if(data.length<2)return;
  const low=minimum,high=Math.max(low+.001,maximum??Math.max(...data)*1.12);
  const points=data.map((n,i)=>[i/(Math.max(35,data.length)-1)*w,h-3-(n-low)/(high-low)*(h-6)]);
  c.beginPath();points.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.strokeStyle=color;c.lineWidth=1.5;c.stroke();
  c.lineTo(points[points.length-1][0],h);c.lineTo(points[0][0],h);c.closePath();c.fillStyle=`${color}12`;c.fill();
}

function setMetric(id,value,unit) {$(id).innerHTML=value==='—'?'—':`${escape(value)}<small>${escape(unit)}</small>`;}

function appendHistory(key,value) {if(value!=null&&Number.isFinite(Number(value))){history[key].push(Number(value));if(history[key].length>100)history[key].shift();}}

function updateMissionMode(mission,auvs) {
  const mode = activeMode(mission), fresh = isFresh();
  if (modeRequest && fresh && mode === modeRequest.mode && finite(mission.mode_revision,-1) > modeRequest.revision) {
    modeRequest = null;modeError = '';centerCamera();showToast('Mission confirmed by controller');
  } else if (modeRequest && Date.now()-modeRequest.at > 20000) {
    modeRequest = null;modeError = 'No fresh controller confirmation. Select a mode to retry.';
  }
  document.querySelectorAll('[data-mission]').forEach(button => {
    const active = mode === button.dataset.mission, pending = modeRequest?.mode === button.dataset.mission;
    button.classList.toggle('selected',active);button.classList.toggle('pending',pending);
    button.setAttribute('aria-pressed',String(active));button.disabled = !fresh || Boolean(modeRequest);
    $(`${button.dataset.mission}-mode-state`).textContent = pending ? 'PENDING' : active ? 'ACTIVE' : 'STANDBY';
  });
  const modeStatus = !mode ? 'Waiting for live mission telemetry' : modeRequest ? `${modeRequest.sending ? 'Sending' : 'Queued'}: ${modeRequest.mode === 'explore' ? 'autonomous survey' : 'diver protection'} / awaiting controller confirmation` : modeError || (!fresh ? 'Last reported mode / telemetry stale' : `Controller confirmed: ${mode === 'explore' ? 'autonomous survey' : 'diver protection'} / revision ${mission.mode_revision ?? '—'}`);
  $('mode-status').textContent = modeStatus;
  $('mode-status').classList.toggle('command-warning',Boolean(modeError || modeRequest || !fresh));
  $('mission-title').textContent = mode === 'explore' ? 'Autonomous survey' : mode === 'protect' ? 'Diver protection' : 'Dual mission swarm / Simulation';
  $('mission-description').textContent = mode === 'explore' ? 'Survey targets are allocated across the fleet. Unobserved cells remain unknown.' : mode === 'protect' ? 'Escort units follow the diver and monitor measured contact trajectories.' : 'Diver protection and autonomous survey. All positions are in local ENU meters.';
  $('scene-mission').textContent = mode === 'explore' ? 'SURVEY' : mode === 'protect' ? 'ESCORT' : 'NO DIRECTIVE';
  $('diver-legend').hidden = mode === 'explore';
  $('center').title = mode === 'explore' ? 'Recenter camera on the survey area' : 'Recenter camera on the protected diver';
  const reported = auvs.filter(a => activeMode(a.mission) === mode);
  const goals = reported.filter(goalPosition);
  const phases = [...new Set(reported.map(a => a.mission?.phase).filter(Boolean))];
  $('strategy-title').textContent = mission.planner ? labelFor(mission.planner) : 'Waiting for the mission planner';
  $('strategy-description').textContent = !mission.planner ? 'Planner responsibilities appear when the controller reports them.' : mode === 'explore' ? 'Each unit selects an unobserved frontier and shares its goal over the acoustic link. Local collision avoidance adjusts movement.' : 'Escort targets adapt to diver motion and predicted contact approach. Local collision avoidance adjusts movement.';
  $('strategy-phase').textContent = phases.length ? phases.map(labelFor).join(' / ') : 'No unit reports';
  $('strategy-phase-detail').textContent = `${reported.length} / ${auvs.length} units reporting this mission`;
  $('strategy-goals').textContent = String(goals.length);
  $('strategy-goals-detail').textContent = mode === 'explore' ? 'Assigned survey goals / acoustic reports may lag' : 'Adaptive escort targets / acoustic reports may lag';
  $('fleet-planner').textContent = mission.planner ? String(mission.planner).toUpperCase() : 'AWAITING TELEMETRY';
}

function updatePilotControls(mission) {
  const control = mission.diver_control || {}, explore = activeMode(mission) === 'explore', fresh = isFresh();
  if (pilotRequest && fresh && control.mode === pilotRequest.mode) {pilotRequest = null;pilotError = '';}
  else if (pilotRequest && Date.now()-pilotRequest.at > 20000) {pilotRequest = null;pilotError = 'No controller confirmation. Select the control mode to retry.';}
  $('pilot-status').textContent = explore ? 'Diver control inactive during autonomous survey' : pilotRequest ? `Queued ${pilotRequest.mode} control / awaiting controller` : pilotError || (!fresh ? 'Diver control unavailable / telemetry stale' : modeRequest ? 'Diver control suspended / mission change pending' : String(mission.status || 'running') !== 'running' ? `Diver control suspended / mission ${mission.status}` : control.mode === 'manual' ? 'MANUAL / hold to move' : control.mode === 'scripted' ? 'SCRIPTED / route controller active' : 'Waiting for controller telemetry');
  $('pilot-status').classList.toggle('command-warning',Boolean(pilotError || pilotRequest || !fresh));
  const p = position(state?.diver);
  $('pilot-position').textContent = p ? `E ${p[0].toFixed(1)} / N ${p[1].toFixed(1)} / UP ${p[2].toFixed(1)} m` : 'ENU —';
  document.querySelectorAll('[data-pilot-mode]').forEach(button => {
    button.setAttribute('aria-pressed',String(control.mode === button.dataset.pilotMode));
    button.disabled = !fresh || explore || Boolean(pilotRequest || modeRequest) || !control.mode;
  });
  document.querySelectorAll('[data-pilot-vector],#pilot-stop').forEach(button => {button.disabled = !canPilot();});
  $('pilot-help').textContent = explore ? 'The fleet explores independently of diver position.' : 'Hold W/A/S/D or arrows · Q descend / E ascend · Space stop';
  if (!canPilot() && (pilotHeartbeat || pilotKeys.size || pilotPointers.size)) haltPilot();
}

function updateDynamics(mission, auvs) {
  const current = Array.isArray(mission.current) && mission.current.length === 3 && mission.current.every(value => Number.isFinite(Number(value))) ? mission.current.map(Number) : null;
  $('current-speed').innerHTML = current ? `${Math.hypot(...current).toFixed(2)}<small>m/s</small>` : '—';
  $('current-vector').textContent = current ? `E ${current[0].toFixed(2)} / N ${current[1].toFixed(2)} / UP ${current[2].toFixed(2)}` : 'Awaiting current telemetry';
  document.querySelectorAll('[data-current]').forEach(button => {
    const preset = button.dataset.current.split(',').map(Number);
    button.setAttribute('aria-pressed',String(Boolean(current && preset.every((value,index) => Math.abs(value-current[index]) < .001))));
  });
  if (current) {
    const requested = currentInputs.map(input => Number(input.value));
    if (currentEdited && requested.every((value,index) => Math.abs(value-current[index]) < .001)) currentEdited = false;
    if (!currentEdited && !currentInputs.includes(document.activeElement)) currentInputs.forEach((input,index) => {input.value = current[index];});
  }
  const forceModes = ['thruster_wrench','native_thrusters'];
  const forceDriven = forceModes.includes(mission.propulsion) || auvs.some(auv => forceModes.includes(auv.propulsion?.mode));
  $('propulsion-status').textContent = forceDriven ? 'Gazebo force-driven thrusters' : mission.propulsion === 'velocity_control' ? 'Gazebo velocity control' : 'Propulsion telemetry pending';
  const utilization = auvs.map(auv => auv.propulsion?.thrust_utilization).filter(value => value != null && Number.isFinite(Number(value))).map(Number);
  const maximum = utilization.length ? Math.max(...utilization) : null;
  $('thrust-status').textContent = maximum == null ? 'Actuator status pending' : `${maximum >= .95 ? 'Actuator saturation' : 'Peak thrust'} ${percent(maximum,0)}`;
  $('thrust-status').className = maximum != null && maximum >= .95 ? 'saturated' : '';
  const hydrophone = state.simulation?.hydrophone || {};
  const recorded = hydrophone.source === 'recorded';
  $('hydrophone-source').textContent = recorded ? `Hydrophone: recorded / ${hydrophone.name || 'audio clip'}${hydrophone.loop ? ' / loop' : ''}` : hydrophone.source === 'synthetic' ? 'Hydrophone: synthetic baseline' : 'Hydrophone source pending';
  $('acoustic-source-details').textContent = recorded ? 'Sonar returns remain synthesized for the virtual scene. The recorded hydrophone clip is replayed as shared background acoustic context; it is not associated with measured contact positions. Classification confidence and accuracy on this scene do not establish field detection performance.' : 'Sonar returns and hydrophone signatures are synthesized for the virtual scene. A numeric feature classifier is a simulation baseline; trained-model status is reported separately. Classification confidence and accuracy on this scene do not establish field detection performance.';
  const backend = mission.classifier_backend || state.classifier_backend;
  $('classifier-foot').textContent = backend === 'torchscript' ? 'TorchScript acoustic model' : backend === 'numeric_feature_baseline' ? 'Numeric feature baseline / simulated scene' : state.classifier?.name || state.classifier_mode || 'Simulation sensor baseline';
  $('simulation-model-details').textContent = forceDriven ? 'AUV thruster forces are integrated by Gazebo with buoyancy, hydrodynamic drag, and the selected water current. Thrust is limited; displayed power is a model estimate. Acoustic noise, echoes, localization drift, and network outages are simulated. Physical hardware behavior still requires field validation.' : 'This monitor displays simulation telemetry. Propulsion and classifier details appear when nodes connect. Acoustic noise, echoes, localization drift, and network outages are modeled; physical hardware behavior still requires field validation.';
}

function updateUI(next) {
  const first=!state,previousMode=activeMode(state?.mission);state=next;lastSeen=Date.now();gatewayConnected=true;
  const auvs=Array.isArray(state.auvs)?state.auvs:[],tracks=Array.isArray(state.tracks)?state.tracks:[],alerts=Array.isArray(state.alerts)?state.alerts:[];
  const metrics=state.metrics || {},network=state.network || {},mission=typeof state.mission==='object'&&state.mission?state.mission:{status:state.mission||'running'};
  const diver=position(state.diver),online=auvs.filter(a=>a.online!==false).length;
  updateMissionMode(mission,auvs);updatePilotControls(mission);
  updateDynamics(mission,auvs);
  $('empty-scene').hidden=true;$('clock').textContent=displayTime(state.time);
  const age=finite(state.dashboard?.state_age_seconds);
  $('connection').className=`badge${age>5?' waiting':''}`;
  $('connection').innerHTML=`<i></i>${age>5?'TELEMETRY STALE':'GATEWAY CONNECTED'}`;
  const missionStatus=String(mission.status||'running').toLowerCase();
  $('scene-status').textContent=`${age>5?'Showing last received state':missionStatus==='paused'?'Mission paused':missionStatus==='ready'||missionStatus==='idle'?'Mission ready':'Live swarm telemetry'} · ${auvs.length} nodes`;
  const explore=activeMode(mission)==='explore';
  $('coordinates').textContent=explore?'SURVEY / GOALS FROM UNIT REPORTS':diver?`DIVER ENU  ${diver.map(x=>x.toFixed(1)).join(' / ')} m`:'DIVER POSITION —';
  if(first||previousMode!==activeMode(mission)){centerCamera();if(!first)trails.clear();}
  const radius=mission.protected_radius??state.protected_radius;
  const formation=metrics.formation_error;
  const survey=mission.exploration || {},goals=auvs.filter(a=>activeMode(a.mission)===activeMode(mission)&&goalPosition(a)).length;
  $('protection-eyebrow').textContent=explore?'SURVEY STATUS':'ESCORT STATUS';
  $('protection-title').textContent=explore?'Area observations':'Diver proximity';
  $('protection-icon').textContent=explore?'⌗':'◇';
  $('formation-label').textContent=explore?'OBSERVED CELLS':'FORMATION ERROR';
  $('nearest-label').textContent=explore?'GOALS REPORTED':'NEAREST CONTACT';
  $('zone-label').textContent=explore?'REGION CELLS':'ASSESSMENT RADIUS';
  $('formation').textContent=explore?survey.observed_cells??'—':formation!=null?`${number(formation,2)} m`:'—';
  $('zone').textContent=explore?survey.total_cells??'—':radius!=null?`${number(radius,1)} m`:'—';
  const near=diver&&tracks.length?Math.min(...tracks.filter(t=>position(t)).map(t=>distance(position(t),diver))):Infinity;
  $('nearest').textContent=explore?String(goals):Number.isFinite(near)?`${near.toFixed(1)} m`:'—';
  const risk=tracks.reduce((level,t)=>Math.max(level,{info:0,warning:1,danger:2}[riskCategory(t.risk)]),0);
  const linkDegraded=online<auvs.length;
  const protectedStatus=!isFresh()?'Telemetry stale':!auvs.length?'No unit reports':risk===2?'Proximity hazard':risk===1?'Contact review required':linkDegraded?'Unit link degraded':explore?(survey.coverage>=1?'Survey cells observed':'Survey in progress'):'Escort / monitoring';
  $('protection-status').textContent=protectedStatus;
  $('protection-status').style.color=risk===2?colors.red:risk===1||linkDegraded||!isFresh()?colors.amber:colors.mint;
  $('protection-icon').style.color=$('protection-status').style.color;
  $('protection-detail').textContent=explore?`${online} / ${auvs.length} units linked / ${survey.source || 'coverage awaiting measured returns'}`:`${online} / ${auvs.length} units linked / radius defines proximity assessment`;
  const meter=explore?survey.coverage:auvs.length?online/auvs.length:null;
  const meterPercent=meter==null?0:Math.max(0,Math.min(100,finite(meter)*100));
  $('protection-fill').style.width=`${meterPercent}%`;
  $('protection-fill').style.background=$('protection-status').style.color;
  $('mission-meter').setAttribute('aria-label',explore?'Observed survey cells':'Connected fleet');
  if(meter==null)$('mission-meter').removeAttribute('aria-valuenow');else $('mission-meter').setAttribute('aria-valuenow',String(meterPercent));
  $('meter-caption').textContent=explore?`Observed region / ${percent(meter)}`:`Linked units / ${online} of ${auvs.length}`;
  $('fleet-count').textContent=`${online} / ${auvs.length}`;
  $('fleet-list').innerHTML=auvs.map((a,i)=>{const p=position(a),speed=Array.isArray(a.velocity)?Math.hypot(...a.velocity.map(x=>finite(x))):null,propulsion=a.propulsion||{},utilization=propulsion.thrust_utilization,hasUtilization=utilization!=null&&Number.isFinite(Number(utilization)),reported=activeMode(a.mission)===activeMode(mission),goal=reported?goalPosition(a):null;return `<div class="fleet-row"><span class="node-icon${a.online===false?' offline':''}" style="color:${a.online===false?colors.red:nodeColors[i%nodeColors.length]}">◈</span><div class="node-meta"><div class="node-name">${escape(String(a.id||`AUV ${i+1}`).toUpperCase().replace('AUV_','AUV 0'))}</div><div class="node-role">${escape(reported?labelFor(a.mission?.role || a.mission?.phase):'Awaiting directive')}${goal&&p?` / ${distance(p,goal).toFixed(1)} m to goal`:''}</div><div class="node-desc">${p?`${Math.abs(p[2]).toFixed(1)} m DEPTH`:''}${speed!=null?` · ${speed.toFixed(2)} m/s`:''}${a.age!=null?` · ${number(a.age,1)} s REPORT`:''}</div>${a.deterrent?.active?`<div class="node-deterrent">SPEAKER ${number(a.deterrent.level_db,0)} dB · DETERRING ${escape(labelFor(a.deterrent.target_class).toUpperCase())}</div>`:''}${hasUtilization?`<div class="node-actuator${Number(utilization)>=.95?' saturated':''}">${Number(utilization)>=.95?'SATURATED':'THRUST'} ${percent(utilization,0)}${propulsion.power_estimate_w!=null?` · ${number(propulsion.power_estimate_w,0)} W EST.`:''}</div>`:''}</div><div class="node-status"${a.online===false?` style="color:${colors.red}"`:''}>${a.online===false?'LINK LOST':'ONLINE'}<small>±${number(a.uncertainty,2)} m</small></div></div>`;}).join('')||'<div class="empty-small">No nodes registered.</div>';
  const loss=network.loss_rate;
  $('packet-loss').textContent=percent(loss);$('packet-delivered').textContent=network.delivered??'—';$('queued').textContent=network.queued??'—';
  const outage=Boolean(network.outage??network.outage_active??mission.outage);
  $('outage').className=`outage-button${outage?' active':''}`;
  $('outage').innerHTML=`<span>⌁</span> ${outage?'Restore communication link':'Inject communication outage'}`;
  $('network-status').textContent=outage?'OUTAGE':'TDMA';$('packet-loss').style.color=finite(loss)>.2?colors.amber:colors.mint;
  const coverage=metrics.coverage??state.map?.coverage;
  setMetric('coverage',coverage==null?'—':(finite(coverage)*100).toFixed(1),'%');
  setMetric('localization',number(metrics.localization_rmse,2),'m RMSE');setMetric('mapping',number(metrics.mapping_rmse,2),'m RMSE');
  setMetric('classification',metrics.classification_accuracy==null?'—':(finite(metrics.classification_accuracy)*100).toFixed(1),'%');
  if(state.map?.observed_cells!=null)$('coverage-foot').textContent=`${state.map.observed_cells} observed cells / mission region`;
  $('track-count').textContent=String(tracks.length);
  if(Date.now()-lastTableUpdate>600||first){
    const byTrack=new Map(alerts.map(alert=>[String(alert.track_id),alert]));
    $('track-rows').innerHTML=tracks.map(track=>{const p=position(track),category=riskCategory(track.risk),assessment={info:'Monitor',warning:'Review contact',danger:'Proximity hazard'}[category],approach=byTrack.get(String(track.id));return `<tr><td><span class="track-name">${escape(labelFor(track.label))}</span><span class="track-id">${escape(track.id)}</span></td><td class="mono">${p?p.map(x=>x.toFixed(1)).join(' / '):'—'}</td><td><span class="confidence-bar"><span style="width:${Math.max(0,Math.min(100,finite(track.confidence)*100))}%"></span></span>${percent(track.confidence,0)}</td><td class="mono">±${number(track.uncertainty,2)} m</td><td><span class="risk-badge ${category}">${escape(assessment)}</span>${approach?.closest_approach_m!=null?`<span class="contact-approach">CPA ${number(approach.closest_approach_m,1)} m / ${number(approach.time_to_closest_s,1)} s</span>`:''}</td></tr>`;}).join('')||'<tr><td colspan="5" class="empty-small">No active contacts. Acoustic sensors are scanning.</td></tr>';
    $('alert-count').textContent=String(alerts.length);
    $('alert-list').innerHTML=alerts.slice(0,12).map(alert=>{const category=riskCategory(alert.level),time=alert.time??state.time;return `<div class="alert-entry ${category}"><span class="alert-symbol">${category==='info'?'ⓘ':'△'}</span><div><div class="alert-message">${escape(alert.message)}</div><div class="alert-time">${displayTime(time)} / ${escape(String(alert.level||'info').toUpperCase())}${alert.track_id?` / ${escape(alert.track_id)}`:''}${alert.time_to_zone_s!=null?`<br>Reaches protected zone in ${number(alert.time_to_zone_s,0)} s`:''}${alert.deterrable?' · deterrable':''}${alert.closest_approach_m!=null?`<br>Closest approach ${number(alert.closest_approach_m,1)} m in ${number(alert.time_to_closest_s,1)} s`:''}</div></div></div>`;}).join('')||'<div class="empty-small">No active proximity alerts.</div>';
    lastTableUpdate=Date.now();
  }
  if(state.time!==lastSimTime){
    if(lastSimTime!=null&&finite(state.time)<finite(lastSimTime)){Object.values(history).forEach(a=>a.length=0);trails.clear();}
    appendHistory('coverage',coverage);appendHistory('localization',metrics.localization_rmse);appendHistory('mapping',metrics.mapping_rmse);appendHistory('classification',metrics.classification_accuracy);appendHistory('network',network.loss_rate);
    lastSimTime=state.time;
    [...auvs,...tracks].forEach(object=>{const p=position(object);if(!p)return;const key=String(object.id);let trail=trails.get(key);if(!trail){trail=[];trails.set(key,trail);}if(!trail.length||distance(trail[trail.length-1],p)>.05){trail.push(p);if(trail.length>80)trail.shift();}});
    const active=new Set([...auvs,...tracks].map(object=>String(object.id)));for(const key of trails.keys())if(!active.has(key))trails.delete(key);
  }
  drawChart('coverage-chart',history.coverage,colors.mint,{maximum:1});drawChart('localization-chart',history.localization,colors.blue);drawChart('mapping-chart',history.mapping,'#b4acbe');drawChart('classification-chart',history.classification,colors.mint,{maximum:1});drawChart('network-chart',history.network,colors.amber,{maximum:Math.max(.3,...history.network)});
  $('last-update').textContent=`Updated ${new Date().toLocaleTimeString()} · ${finite(state.dashboard?.state_age_seconds).toFixed(1)} s data age`;
  if(state.scenario?.description)$('scenario-details').textContent=state.scenario.description;
}

async function poll() {
  try {
    const response=await fetch('/api/state',{cache:'no-store'});const next=await response.json();
    if(!response.ok)throw Error(next.error||'Telemetry unavailable');updateUI(next);
  } catch(error) {
    gatewayConnected=false;haltPilot();updateMissionMode(state?.mission || {},state?.auvs || []);updatePilotControls(state?.mission || {});
    $('connection').className='badge waiting';$('connection').innerHTML=`<i></i>${state?'GATEWAY DISCONNECTED':'WAITING FOR TELEMETRY'}`;
    $('last-update').textContent=state?`Telemetry interrupted · last received ${Math.max(0,Math.round((Date.now()-lastSeen)/1000))} s ago`:error.message;
    if(state)$('scene-status').textContent='Showing last received state';
  } finally {setTimeout(poll,450);}
}

window.addEventListener('resize',()=>{Object.entries(history).forEach(([key,data])=>drawChart(`${key==='network'?'network':key}-chart`,data,key==='network'?colors.amber:colors.mint));});
drawScene();poll();
