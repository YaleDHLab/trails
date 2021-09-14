var dpi = window.devicePixelRatio || 1;

/**
* Scene
**/

function World() {
  var start = {
    x: 0.0,
    y: 0.0,
    z: 0.5,
  }
  var size = getCanvasSize();
  this.scene = new THREE.Scene();
  this.camera = new THREE.PerspectiveCamera(75, size.w / size.h, 0.001, 1000);
  this.camera.position.x = start.x;
  this.camera.position.y = start.y;
  this.camera.position.z = start.z;
  this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  this.composer = new THREE.EffectComposer(this.renderer);
  this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));
  this.composer.addPass(new THREE.UnrealBloomPass());
  document.querySelector('#canvas-container').appendChild(this.renderer.domElement);
  this.controls = new THREE.TrackballControls(this.camera, this.renderer.domElement);
  this.controls.zoomSpeed = 0.4;
  this.controls.panSpeed = 0.4;
  this.controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
  this.controls.mouseButtons.MIDDLE = THREE.MOUSE.ZOOM;
  this.controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  this.controls.target.x = start.x;
  this.controls.target.y = start.y;
  this.controls.maxDistance = 0.5;
  this.addEventListeners();
}

World.prototype.resize = function() {
  var size = getCanvasSize(),
      w = size.w * dpi,
      h = size.h * dpi;
  world.camera.aspect = w / h;
  world.camera.updateProjectionMatrix();
  world.renderer.setSize(w, h, false);
  world.controls.handleResize();
  if (picker.initialized) picker.tex.setSize(w, h);
  if (touchtexture.initialized) touchtexture.handleResize();
  if (points.initialized) points.mesh.material.uniforms.height.value = window.innerHeight;
  if (picker.initialized) picker.mesh.material.uniforms.height.value = window.innerHeight;
  world.composer.reset();
}

World.prototype.useNightMode = function() {
  points.mesh.material.uniforms.useNightMode.value = 1.0;
  document.body.classList.add('night-mode');
}

World.prototype.useDayMode = function() {
  points.mesh.material.uniforms.useNightMode.value = 0.0;
  document.body.classList.remove('night-mode');
}

World.prototype.addEventListeners = function() {
  window.addEventListener('resize', function(e) {
    this.resize()
  }.bind(this));
  window.addEventListener('mousemove', function(e) {
    tooltip.hide();
  }.bind(this))
}

World.prototype.render = function() {
  requestAnimationFrame(this.render.bind(this));
  if (points.initialized && points.mesh.material.uniforms.useNightMode.value > 0.5) {
    this.composer.render();
  } else {
    this.renderer.render(world.scene, world.camera);
  }
  this.controls.update();
  touchtexture.update();
  stats.update();
}

/**
* Points
**/

function Points() {}

Points.prototype.init = function() {
  this.mesh = null;
  this.initialized = false;
  this.positions = [];
  this.texts = [];
  this.n = 150000;

  var tooltip = document.querySelector('#tooltip'),
      point = document.querySelector('#point'),
      selected = document.querySelector('#selected-point');

  fetch('assets/data/texts.json').then(data => data.json()).then(json => {
    this.texts = json;
  })

  fetch('assets/data/positions.json').then(data => data.json()).then(json => {
    this.positions = json.positions.slice(0, this.n);
    this.colors = json.colors.slice(0, this.n);
    var clickColor = new THREE.Color(),
        clickColors = new Float32Array(this.positions.length * 3),
        colors = new Float32Array(this.positions.length * 3),
        translations = new Float32Array(this.positions.length * 3),
        translationIterator = 0,
        colorIterator = 0,
        clickColorIterator = 0;
    for (var i=0; i<this.positions.length; i++) {
      clickColor.setHex(i + 1);
      var color = cmap((this.colors || {})[i] || 0);
      translations[translationIterator++] = this.positions[i][0];
      translations[translationIterator++] = this.positions[i][1];
      translations[translationIterator++] = 0;
      colors[colorIterator++] = color.r;
      colors[colorIterator++] = color.g;
      colors[colorIterator++] = color.b;
      clickColors[clickColorIterator++] = clickColor.r;
      clickColors[clickColorIterator++] = clickColor.g;
      clickColors[clickColorIterator++] = clickColor.b;
    }
    // create the geometry
    var geometry = new THREE.InstancedBufferGeometry();
    var position = new THREE.BufferAttribute(new Float32Array([0, 0, 0]), 3);
    var translation = new THREE.InstancedBufferAttribute(translations, 3, false, 1);
    var color = new THREE.InstancedBufferAttribute(colors, 3, false, 1);
    var clickColor = new THREE.InstancedBufferAttribute(clickColors, 3, false, 1);
    geometry.setAttribute('position', position);
    geometry.setAttribute('translation', translation);
    geometry.setAttribute('color', color);
    geometry.setAttribute('clickColor', clickColor);
    // build the mesh
    this.mesh = new THREE.Points(geometry, this.getMaterial());
    this.mesh.frustumCulled = false;
    world.scene.add(this.mesh);
    // initialize downstream layers that depend on this mesh
    picker.init();
    details.timeout = setTimeout(function() {
      details.redraw();
    }, 1000)
    // flip the initialization bool
    this.initialized = true;
  })
}

Points.prototype.getMaterial = function() {
  return new THREE.RawShaderMaterial({
    vertexShader: document.getElementById('vertex-shader').textContent,
    fragmentShader: document.getElementById('fragment-shader').textContent,
    uniforms: {
      height: {
        type: 'f',
        value: world.renderer.domElement.clientHeight,
      },
      dpi: {
        type: 'f',
        value: dpi,
      },
      grit: {
        type: 't',
        value: getTexture('assets/images/particle-texture.jpg'),
      },
      useColor: {
        type: 'f',
        value: 0.0,
      },
      touchtexture: {
        type: 't',
        value: touchtexture.texture,
      },
      useNightMode: {
        type: 'f',
        value: 0.0,
      }
    }
  });
}

/**
* Color
**/

var colors = ['#0d0887','#100788','#130789','#16078a','#19068c','#1b068d','#1d068e','#20068f','#220690','#240691','#260591','#280592','#2a0593','#2c0594','#2e0595','#2f0596','#310597','#330597','#350498','#370499','#38049a','#3a049a','#3c049b','#3e049c','#3f049c','#41049d','#43039e','#44039e','#46039f','#48039f','#4903a0','#4b03a1','#4c02a1','#4e02a2','#5002a2','#5102a3','#5302a3','#5502a4','#5601a4','#5801a4','#5901a5','#5b01a5','#5c01a6','#5e01a6','#6001a6','#6100a7','#6300a7','#6400a7','#6600a7','#6700a8','#6900a8','#6a00a8','#6c00a8','#6e00a8','#6f00a8','#7100a8','#7201a8','#7401a8','#7501a8','#7701a8','#7801a8','#7a02a8','#7b02a8','#7d03a8','#7e03a8','#8004a8','#8104a7','#8305a7','#8405a7','#8606a6','#8707a6','#8808a6','#8a09a5','#8b0aa5','#8d0ba5','#8e0ca4','#8f0da4','#910ea3','#920fa3','#9410a2','#9511a1','#9613a1','#9814a0','#99159f','#9a169f','#9c179e','#9d189d','#9e199d','#a01a9c','#a11b9b','#a21d9a','#a31e9a','#a51f99','#a62098','#a72197','#a82296','#aa2395','#ab2494','#ac2694','#ad2793','#ae2892','#b02991','#b12a90','#b22b8f','#b32c8e','#b42e8d','#b52f8c','#b6308b','#b7318a','#b83289','#ba3388','#bb3488','#bc3587','#bd3786','#be3885','#bf3984','#c03a83','#c13b82','#c23c81','#c33d80','#c43e7f','#c5407e','#c6417d','#c7427c','#c8437b','#c9447a','#ca457a','#cb4679','#cc4778','#cc4977','#cd4a76','#ce4b75','#cf4c74','#d04d73','#d14e72','#d24f71','#d35171','#d45270','#d5536f','#d5546e','#d6556d','#d7566c','#d8576b','#d9586a','#da5a6a','#da5b69','#db5c68','#dc5d67','#dd5e66','#de5f65','#de6164','#df6263','#e06363','#e16462','#e26561','#e26660','#e3685f','#e4695e','#e56a5d','#e56b5d','#e66c5c','#e76e5b','#e76f5a','#e87059','#e97158','#e97257','#ea7457','#eb7556','#eb7655','#ec7754','#ed7953','#ed7a52','#ee7b51','#ef7c51','#ef7e50','#f07f4f','#f0804e','#f1814d','#f1834c','#f2844b','#f3854b','#f3874a','#f48849','#f48948','#f58b47','#f58c46','#f68d45','#f68f44','#f79044','#f79143','#f79342','#f89441','#f89540','#f9973f','#f9983e','#f99a3e','#fa9b3d','#fa9c3c','#fa9e3b','#fb9f3a','#fba139','#fba238','#fca338','#fca537','#fca636','#fca835','#fca934','#fdab33','#fdac33','#fdae32','#fdaf31','#fdb130','#fdb22f','#fdb42f','#fdb52e','#feb72d','#feb82c','#feba2c','#febb2b','#febd2a','#febe2a','#fec029','#fdc229','#fdc328','#fdc527','#fdc627','#fdc827','#fdca26','#fdcb26','#fccd25','#fcce25','#fcd025','#fcd225','#fbd324','#fbd524','#fbd724','#fad824','#fada24','#f9dc24','#f9dd25','#f8df25','#f8e125','#f7e225','#f7e425','#f6e626','#f6e826','#f5e926','#f5eb27','#f4ed27','#f3ee27','#f3f027','#f2f227','#f1f426','#f1f525','#f0f724','#f0f921']

function hex2rgb(hex) {
  return {
    // skip # at position 0
    r: parseInt(hex.slice(1, 3), 16) / 255,
    g: parseInt(hex.slice(3, 5), 16) / 255,
    b: parseInt(hex.slice(5, 7), 16) / 255
  }
}

function zeroPadHex(hexStr) {
  return '00'.slice(hexStr.length) + hexStr
}

function rgb2hex(rgb) {
  // Map channel triplet into hex color code
  return '#' + [rgb.r, rgb.g, rgb.b]
    // Convert to hex (map [0, 1] => [0, 255] => Z => [0x0, 0xff])
    .map(function(ch) { return Math.round(ch * 255).toString(16) })
    // Make sure each channel is two digits long
    .map(zeroPadHex)
    .join('')
}

function interpolate(a, b) {
  a = hex2rgb(a)
  b = hex2rgb(b)
  var ar = a.r
  var ag = a.g
  var ab = a.b
  var br = b.r - ar
  var bg = b.g - ag
  var bb = b.b - ab
  return function(t) {
    return {
      r: ar + br * t,
      g: ag + bg * t,
      b: ab + bb * t
    }
  }
}

function interpolateArray(scaleArr) {
  var N = scaleArr.length - 2 // -1 for spacings, -1 for number of interpolate fns
  var intervalWidth = 1 / N
  var intervals = []
  for (var i = 0; i <= N; i++) {
    intervals[i] = interpolate(scaleArr[i], scaleArr[i + 1])
  }
  return function (t) {
    if (t < 0 || t > 1) throw new Error('Outside the allowed range of [0, 1]')
    var i = Math.floor(t * N)
    var intervalOffset = i * intervalWidth
    return intervals[i](t / intervalWidth - intervalOffset / intervalWidth)
  }
}

var cmap = interpolateArray(colors);

/**
* Picker
**/

function Picker() {
  this.mesh = null;
  this.scene = new THREE.Scene();
  this.scene.background = new THREE.Color(0x000000);
  this.mouse = new THREE.Vector2(); // stores last click location (unused)
  this.tex = this.getTexture();
  this.initialized = false;
  this.selectedIndex = -1;
}

// get the mesh in which to render picking elements
Picker.prototype.init = function() {
  this.addEventListeners();
  this.mesh = points.mesh.clone();
  var material = points.getMaterial();
  material.uniforms.useColor.value = 1.0;
  this.mesh.material = material;
  this.scene.add(this.mesh);
  this.initialized = true;
}

Picker.prototype.addEventListeners = function() {
  window.addEventListener('click', function(e) {
    var idx = this.select(this.getMouseOffsets(e));
    if (idx !== -1) picker.showItem(idx);
  }.bind(this));
  window.addEventListener('mousemove', function(e) {
    this.mouse.copy(this.getMouseOffsets(e))
  }.bind(this))
}

Picker.prototype.showItem = function(index) {
  tooltip.display(index, this.mouse);
}

// get the texture on which off-screen rendering will happen
Picker.prototype.getTexture = function() {
  var size = getCanvasSize();
  var tex = new THREE.WebGLRenderTarget(size.w * dpi, size.h * dpi);
  tex.texture.minFilter = THREE.LinearFilter;
  return tex;
}

// get the x, y offsets of a click within the canvas
Picker.prototype.getMouseOffsets = function(e) {
  var elem = e.target;
  var position = {
    x: e.clientX ? e.clientX : e.pageX,
    y: e.clientY ? e.clientY : e.pageY,
  };
  return position;
}

// draw an offscreen world then reset the render target so world can update
Picker.prototype.render = function() {
  world.renderer.setRenderTarget(this.tex);
  world.renderer.render(this.scene, world.camera);
  world.renderer.setRenderTarget(null);
}

Picker.prototype.select = function(obj) {
  this.render();
  // read the texture color at the current mouse pixel
  var pixelBuffer = new Uint8Array(4),
      x = obj.x * dpi,
      y = this.tex.height - obj.y * dpi;
  world.renderer.readRenderTargetPixels(this.tex, x, y, 1, 1, pixelBuffer);
  var id = (pixelBuffer[0] << 16) | (pixelBuffer[1] << 8) | (pixelBuffer[2]),
      cellIdx = id-1; // ids use id+1 as the id of null selections is 0
  return cellIdx;
}

/**
 * TouchTexture
 **/

// from https://github.com/brunoimbrizi/interactive-particles/blob/master/src/scripts/webgl/particles/TouchTexture.js
function easeOutSine (t, b, c, d) {
  return c * Math.sin(t/d * (Math.PI/2)) + b;
};

function TouchTexture() {}

TouchTexture.prototype.init = function() {
  this.initialized = false;
  this.size = 512; // smaller is more performant but more susceptible to unit quantization issues
  this.maxAge = 40; // length of trail
  this.radius = 0.05;
  this.cursorRadius = 0.01;
  this.maxForce = 0.4; // max amount of momentum in big gestures
  this.frozen = false;
  this.mouse = {x: 0, y: 0};
  this.renderCanvas = false;
  this.trail = [];
  this.setTexture();
  this.mouseMoveElem = world.renderer.domElement;
  this.addEventListeners();
}

TouchTexture.prototype.handleResize = function() {
  this.mouseMoveElem.removeEventListener('mousemove', this.handleMouseMove.bind(this));
  this.init();
}

TouchTexture.prototype.setTexture = function() {
  this.canvas = document.createElement('canvas');
  this.canvas.width = this.canvas.height = this.size;
  this.ctx = this.canvas.getContext('2d');
  this.ctx.fillStyle = 'black';
  this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  this.texture = new THREE.Texture(this.canvas);
  this.canvas.id = 'touch-texture';
  this.canvas.style.width = this.canvas.style.height = `${this.canvas.width}px`;
  if (this.renderCanvas) document.body.appendChild(this.canvas);
}

TouchTexture.prototype.update = function(delta) {
  if (this.frozen) return;
  this.clear();
  // age points
  this.trail.forEach((point, i) => {
    point.age++;
    // remove old
    if (point.age > this.maxAge) {
      this.trail.splice(i, 1);
    }
  });
  this.trail.forEach(this.drawPoint.bind(this));
  this.drawCursor();
  this.texture.needsUpdate = true;
}

TouchTexture.prototype.clear = function() {
  this.ctx.fillStyle = 'black';
  this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
}

TouchTexture.prototype.addPoint = function(point) {
  let force = 0;
  var last = this.trail[this.trail.length - 1];
  if (last) {
    var dx = last.x - point.x;
    var dy = last.y - point.y;
    var dd = dx * dx + dy * dy;
    force = Math.min(dd * 10000, this.maxForce);
  }
  this.trail.push({
    x: point.x,
    y: point.y,
    age: 0,
    force,
  });
}

TouchTexture.prototype.addEventListeners = function() {
  this.mouseMoveElem.addEventListener('mousemove', this.handleMouseMove.bind(this));
}

TouchTexture.prototype.handleMouseMove = function(e) {
  var elem = this.mouseMoveElem;
  var w = elem.clientWidth;
  var h = elem.clientHeight;
  // get the initial, unnormalized point coords
  var p = {
    x: e.clientX,
    y: e.clientY,
  }
  // convert x,y to positions within canvas (instead of positions within document/window)
  var elem = e.target;
  while (elem && elem.tag && elem.tag != 'html') {
    var box = elem.getBoundingClientRect();
    p.x -= box.left;
    p.y -= box.top;
    elem = elem.parentNode;
  }
  // normalize the point coords
  p.x = p.x / w;
  p.y = (h-p.y) / h;
  this.addPoint(p);
  this.mouse = p;
}

TouchTexture.prototype.drawPoint = function(point) {
  var pos = {
    x: point.x * this.size,
    y: (1 - point.y) * this.size
  };
  var intensity = 1;
  if (point.age < this.maxAge * 0.3) {
    intensity = easeOutSine(point.age / (this.maxAge * 0.3), 0, 1, 1);
  } else {
    intensity = easeOutSine(1 - (point.age - this.maxAge * 0.3) / (this.maxAge * 0.7), 0, 1, 1);
  }
  intensity *= point.force;
  var radius = this.size * this.radius * intensity;
  var gradient = this.ctx.createRadialGradient(pos.x, pos.y, radius * 0.25, pos.x, pos.y, radius);
  gradient.addColorStop(0, `rgba(255, 255, 255, 0.2)`);
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
  this.ctx.beginPath();
  this.ctx.fillStyle = gradient;
  this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
  this.ctx.fill();
}

TouchTexture.prototype.drawCursor = function() {
  var pos = {
    x: this.mouse.x * this.size,
    y: (1 - this.mouse.y) * this.size
  };
  var radius = this.cursorRadius * this.size;
  // create a gradient with diameter s
  var gradient = this.ctx.createRadialGradient(pos.x, pos.y, radius * 0.001, pos.x, pos.y, radius);
  gradient.addColorStop(0, `rgba(255, 255, 255, 1.0)`);
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
  this.ctx.beginPath();
  this.ctx.fillStyle = gradient;
  this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
  this.ctx.fill();
}

/**
 * Tooltip
 **/

function Tooltip() {
  this.target = document.querySelector('#tooltip');
  this.displayed = false;
  this.addEventListeners();
}

Tooltip.prototype.display = function(index, pos) {
  var pos = pos || {
    x: 50,
    y: 50,
  }
  this.displayed = true;
  this.target.style.left = pos.x + 'px';
  this.target.style.top = pos.y + 'px';
  this.target.innerHTML = this.getHtml(index);
}

Tooltip.prototype.hide = function() {
  this.displayed = false;
  this.target.innerHTML = '';
}

Tooltip.prototype.getHtml = function(index) {
  var metadata = points.texts[index]
  return _.template(document.querySelector('#tooltip-template').innerHTML)({
    index: index,
    metadata: metadata,
  });
}

Tooltip.prototype.addEventListeners = function() {
  window.addEventListener('wheel', function(e) {
    this.hide();
  }.bind(this))
  this.target.addEventListener('mousemove', function(e) {
    e.stopPropagation();
  })
}

/**
 * Details
 **/

function Details() {
  this.displayed = [];
  this.n = 30;
  this.margin = 100;
  this.size = 40;
  this.timeout = null;
  this.mouse = {
    down: false,
    dragging: false,
    x: null,
    y: null,
  };
  this.addEventListeners();
}

Details.prototype.select = function() {
  function intersects(a, b, p, s) {
    var margin = 10; // minimum margin between cells
    var a0 = a[p];
    var b0 = b[p];
    var a1 = a[p] + a[s] + margin;
    var b1 = b[p] + b[s] + margin;
    return a0 >= b0 && a0 <= b1 ||
           a1 >= b0 && a1 <= b1;
  }

  var bounds = getWorldBounds();
  for (var i=0; i<points.positions.length; i++) {
    // don't stop 'til you get enough
    if (this.displayed.length === this.n) break;
    // determine if this point has an image;
    if (points.texts[i].thumb) {
      // determine if this point is inside the frame
      var p = points.positions[i];
      if (
        p[0] >= bounds.x[0] &&
        p[0] <= bounds.x[1] &&
        p[1] >= bounds.y[0] &&
        p[1] <= bounds.y[1]
      ) {
        // create the detail object
        var screen = worldToScreenCoords({x: p[0], y: p[1]})
        var d = {
          x: screen.x,
          y: screen.y,
          width: this.size,
          height: this.size,
          idx: i,
        }
        // determine if this detail overlaps others
        var overlaps = false;
        for (var j=0; j<this.displayed.length; j++) {
          var o = this.displayed[j];
          if (
            intersects(d, o, 'x', 'width') &&
            intersects(d, o, 'y', 'height')
          ) {
            overlaps = true;
            break;
          }
        }
        if (!overlaps) {
          // add the rendered HTML content if the image can be loaded
          var url = points.texts[i].thumb;
          d.elem = document.createElement('div');
          d.elem.style.backgroundImage = `url(${url})`;
          d.elem.className = 'detail-display';
          d.elem.id = 'detail-' + d.idx;
          d.elem.style.left = d.x + 'px';
          d.elem.style.top = d.y + 'px';
          d.elem.style.height = d.height + 'px';
          d.elem.style.width = d.width + 'px';
          d.elem.style.animationDelay = Math.random() * 2.0 + 's';
          d.elem.onclick = function(i, e) {
            picker.showItem(i);
            e.stopPropagation();
          }.bind(this, i);
          // add the point to the set of rendered points
          this.displayed.push(d);
        }
      }

    }
  }
}

Details.prototype.render = function() {
  this.displayed.forEach(function(d) {
    document.body.appendChild(d.elem);
  })
}

Details.prototype.clear = function() {
  if (!this.displayed.length) return;
  this.displayed.forEach(function(d) {
    var elem = d.elem;
    elem.parentNode.removeChild(elem);
  })
  this.displayed = [];
}

Details.prototype.redraw = function() {
  console.log(' * redraw')
  this.clear();
  this.select();
  this.render();
}

Details.prototype.addEventListeners = function() {

  window.addEventListener('mousedown', function() {
    this.mouse.down = true;
  }.bind(this))

  window.addEventListener('mousemove', function() {
    if (this.mouse.down) {
      this.mouse.dragging = true;
      this.clear();
    }
  }.bind(this))

  window.addEventListener('mouseup', function() {
    if (this.mouse.dragging) {
      this.timeout = setTimeout(function() {
        this.redraw();
      }.bind(this), 250)
    }
    this.mouse.down = false;
    this.mouse.dragging = false;
  }.bind(this))

  window.addEventListener('wheel', function() {
    this.clear();
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = setTimeout(function() {
      this.redraw();
    }.bind(this), 700)
  }.bind(this))

  window.addEventListener('resize', function() {
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = setTimeout(function() {
      this.redraw();
    }.bind(this), 50)
  }.bind(this))
}

/**
 * Utils
 **/

function getCanvasSize() {
  var container = document.querySelector('#canvas-container');
  return {
    w: container.clientWidth,
    h: container.clientHeight,
  }
}

function screenToWorldCoords(pos) {
  var vector = new THREE.Vector3(),
      mouse = new THREE.Vector2(),
      // convert from screen to clip space
      x = (pos.x / world.renderer.domElement.clientWidth) * 2 - 1,
      y = -(pos.y / world.renderer.domElement.clientHeight) * 2 + 1;
  // project the screen location into world coords
  vector.set(x, y, 0.5);
  vector.unproject(world.camera);
  var direction = vector.sub(world.camera.position).normalize(),
      distance = - world.camera.position.z / direction.z,
      scaled = direction.multiplyScalar(distance),
      coords = world.camera.position.clone().add(scaled);
  return coords;
}

function worldToScreenCoords(pos) {
  var s = getCanvasSize(),
      w = s.w / 2,
      h = s.h / 2,
      vec = new THREE.Vector3(pos.x, pos.y, pos.z || 0);
  vec.project(world.camera);
  vec.x =  (vec.x * w) + w;
  vec.y = -(vec.y * h) + h;
  // add offsets that account for the negative margins in the canvas position
  var rect = world.renderer.domElement.getBoundingClientRect();
  return {
    x: vec.x + rect.left,
    y: vec.y + rect.top
  };
}

function getWorldBounds() {
  var min = screenToWorldCoords({
    x: 0,
    y: window.innerHeight,
  });
  var max = screenToWorldCoords({
    x: window.innerWidth,
    y: 0,
  })
  return {
    x: [min.x, max.x],
    y: [min.y, max.y],
  }
}

function getTexture(src) {
  var image = document.createElement('img');
  var tex = new THREE.Texture(image);
  image.addEventListener('load', function(event) {
    tex.needsUpdate = true;
  });
  image.src = src;
  return tex;
}

/**
* Main
**/

var world = new World();
var picker = new Picker();
var touchtexture = new TouchTexture();
var points = new Points();
var tooltip = new Tooltip();
var stats = new Stats();
var details = new Details();

touchtexture.init();
points.init();
world.resize();
world.render();

// add stats
document.body.appendChild(stats.dom);
stats.begin();