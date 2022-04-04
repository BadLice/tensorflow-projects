let graph = null;
let model = null;
let predictor = null;

let section = 'training';
let isTrained = false;

function preload() {
	model = new Model();
	model.setColorData(loadJSON('./colorData.json'));
}

function setup() {
	var canvas = createCanvas(800, 800);
	canvas.parent('canvas');

	graph = new Graph();
	predictor = new Predictor();

	model.readColorData();
	model.setup();
	// model.train();

	document.querySelector('#start-model').addEventListener('click', () => {
		section = 'predicting';
	});
	document.querySelector('#color-picker').addEventListener('change', (e) => {
		predictor.setColor(e.target.value);
		console.log(predictor.color);
	});

	document.querySelector('#optimizer').addEventListener('change', (e) => {
		model.optimizerIndex = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#nodes').addEventListener('change', (e) => {
		model.nodeAmount = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#activation').addEventListener('change', (e) => {
		model.activationFIndex = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#epochs').addEventListener('change', (e) => {
		model.epochsAmount = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#learning-rate').addEventListener('change', (e) => {
		model.learningRate = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#train-model').addEventListener('click', () => {
		model.train();
	});
}

function draw() {
	background(0);

	switch (section) {
		case 'training':
			graph.draw();
			break;
		case 'predicting':
			predictor.draw();

			break;
	}

	document.querySelector('#start-model').disabled = !isTrained;
}
