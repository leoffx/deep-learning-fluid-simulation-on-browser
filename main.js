const simulation = document.querySelector('#simulation');
const simulationctx = simulation.getContext('2d');

const drawing = document.querySelector('#drawing');
const drawingctx = drawing.getContext('2d');

//let [width, height] = [window.innerWidth, window.innerHeight];
let [height, width] = [100, 100];
let [scaleX, scaleY] = [500 / width, 300 / height];

[simulation.style.width, simulation.style.height] = [
  `${width * scaleX}px`,
  `${height * scaleY}px`,
];
[drawing.style.width, drawing.style.height] = [
  `${width * scaleX}px`,
  `${height * scaleY}px`,
];

[simulationctx.canvas.width, simulationctx.canvas.height] = [width, height];
[drawingctx.canvas.width, drawingctx.canvas.height] = [width, height];

let obj = tf.zeros([1, height, width, 1]);
let frame = tf.ones([1, height, width, 2]);

let encoder, decoder;
let compressed, b_mul, b_add, absVel, temp;

async function animate() {
  [absVel, compressed] = calculateNextStep();

  new Promise(() => {
    tf.browser.toPixels(absVel, simulation);
  }).then(() => tf.dispose(absVel));

  await new Promise((r) => setTimeout(r, 13));
  //console.log("numTensors (outside tidy): " + tf.memory().numTensors);
  requestAnimationFrame(animate);
}

async function loadModels() {
  const ENCODER_URL = 'encoder/model.json';
  const DECODER_URL = 'decoder/model.json';
  encoder = await tf.loadGraphModel(ENCODER_URL);
  decoder = await tf.loadGraphModel(DECODER_URL);

  [b_add, b_mul, compressed] = await encoder.predict([frame, obj]);
  animate();
}

function calculateNextStep() {
  return tf.tidy(() => {
    [temp, ans] = decoder.predict([compressed, b_add, b_mul]);
    tf.dispose(compressed);
    compressed = temp.clone();
    tf.keep(compressed);
    absVel = tf
      .abs(
        ans.slice([0, 0, 0, 1], [1, -1, -1, 1]),
        ans.slice(0, [1, -1, -1, 1])
      )
      .as2D(height, width);

    absVel = absVel;

    return [absVel, compressed];
  });
}

function getXY(drawing, event) {
  var rect = drawing.getBoundingClientRect();
  return {
    X: event.clientX - rect.left,
    Y: event.clientY - rect.top,
  };
}

function draw(e) {
  var pos = getXY(drawing, e);
  if (!painting) return;

  drawingctx.lineWidth = 3;
  drawingctx.lineCap = 'round';
  drawingctx.lineTo(pos.X / scaleX, pos.Y / scaleY);
  drawingctx.strokeStyle = 'red';
  drawingctx.stroke();
}

let painting = false;

async function finishDrawing() {
  painting = false;

  b_add.dispose();
  b_mul.dispose();
  obj.dispose();

  obj = tf.browser
    .fromPixels(drawing, 1)
    .expandDims()
    .greater(150)
    .asType('float32');

  [b_add, b_mul, _] = await encoder.predict([frame, obj]);

  tf.dispose(_);
}

drawing.addEventListener('mousedown', (e) => {
  drawingctx.beginPath();
  painting = true;
  draw(e);
});
drawing.addEventListener('mouseup', finishDrawing);

drawing.addEventListener('mousemove', draw);

loadModels();
