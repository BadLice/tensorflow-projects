let graph = null;
let model = null;
let predictor = null;

let section = 'training'; //training or predicting

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

	document.querySelector('#retrain-model').addEventListener('click', () => {
		section = 'training';
		document
			.getElementsByClassName('predictor')
			.forEach((el) => (el.style = 'display:none'));
		document.getElementsByClassName('training-params').forEach((el) => (el.style = ''));
	});

	document.querySelector('#start-model').addEventListener('click', () => {
		section = 'predicting';
		document
			.getElementsByClassName('training-params')
			.forEach((el) => (el.style = 'display:none'));
		document.getElementsByClassName('predictor').forEach((el) => (el.style = ''));
	});

	document.querySelector('#color-picker').addEventListener('change', (e) => {
		predictor.setColor(e.target.value);
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

	document.querySelector('#learning-rate').addEventListener('input', (e) => {
		document.querySelector('#lr-label').innerHTML = Number(e.target.value);
	});

	document.querySelector('#learning-rate').addEventListener('change', (e) => {
		model.learningRate = Number(e.target.value);
		model.setup();
	});

	document.querySelector('#show-batch').addEventListener('change', (e) => {
		graph.showBatch = e.target.value == 'true';
	});

	document.querySelector('#train-model').addEventListener('click', () => {
		document
			.getElementsByClassName('training-params')
			.forEach((el) => (el.disabled = true));
		document.querySelector('#is-trained').value = 'false';
		document.querySelector('#is-trained').disabled = true;
		document.querySelector('#download-model').disabled = true;
		document.querySelector('#testing-data').disabled = true;
		document.querySelector('#right-predictions').innerHTML = '';

		model.train(() => {
			document
				.getElementsByClassName('training-params')
				.forEach((el) => (el.disabled = false));
			document.querySelector('#is-trained').value = 'true';
			document.querySelector('#is-trained').disabled = true;
			document.querySelector('#download-model').disabled = false;
			document.querySelector('#testing-data').disabled = false;
		});
	});

	document.querySelector('#download-model').addEventListener('click', (e) => {
		model.download();
	});

	document.querySelector('#import-model').addEventListener('click', () => {
		const uploadJSONInput = document.getElementById('upload-json');
		const uploadWeightsInput = document.getElementById('upload-weights');
		if (uploadJSONInput.files.length !== 0 && uploadWeightsInput.files.length !== 0) {
			model.load(uploadJSONInput, uploadWeightsInput, () => {
				document.querySelector('#nodes').value = '' + model.nodeAmount;
				document.querySelector('#activation').value = '' + model.activationFIndex;
				uploadJSONInput.value = '';
				uploadWeightsInput.value = '';
				document.querySelector('#is-trained').value = 'true';
				document.querySelector('#is-trained').disabled = true;
				document.querySelector('#download-model').disabled = false;
				document.querySelector('#testing-data').disabled = false;
				document.querySelector('#right-predictions').innerHTML = '';
			});
		}
	});

	document.querySelector('#testing-data').addEventListener('click', () => {
		model.launchTest((rightPredictionsAmount, totalPredictionsAmount) => {
			document.querySelector('#right-predictions').innerHTML =
				rightPredictionsAmount + '/' + totalPredictionsAmount;
		});
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

	document.querySelector('#start-model').disabled = !model.isTrained;
}
