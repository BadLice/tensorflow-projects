//neural network that solves xor with tensorflow.js
let trainingData = [
	{ inputs: [0, 0], target: 0 },
	{ inputs: [1, 1], target: 0 },
	{ inputs: [1, 0], target: 1 },
	{ inputs: [0, 1], target: 1 },
];

let network = null;
const learningRate = 0.1;
let trainingInputs = tf.tensor2d(trainingData.map(({ inputs }) => inputs));
let trainingOutputs = tf.tensor1d(trainingData.map(({ target }) => target));
let gridInputs = null;
let resolution = 40;

function setup() {
	createCanvas(800, 800);

	network = tf.sequential();
	const hidden = tf.layers.dense({
		units: 16,
		activation: 'sigmoid',
		inputShape: [2],
	});
	const output = tf.layers.dense({
		units: 1,
		activation: 'sigmoid',
	});
	network.add(hidden);
	network.add(output);
	network.compile({
		loss: tf.losses.meanSquaredError,
		optimizer: tf.train.adam(learningRate),
	});

	startTraining();

	let inputs = [];
	cycleGrid(({ i, j, rows, cols }) => {
		inputs.push([map(i, 0, rows, 0, 1), map(j, 0, cols, 0, 1)]);
	});
	gridInputs = tf.tensor2d(inputs);
}
function draw() {
	background(0);
	drawGrid();
}

const drawGrid = () => {
	tf.tidy(() => {
		let guess = network.predict(gridInputs).dataSync();

		cycleGrid(({ i, j, index, res }) => {
			let currentGuess = guess[index];
			fill(currentGuess * 255);
			rect(i * res, j * res, res);
			fill(255, 0, 0);
			textAlign(CENTER, CENTER);
			text(nf(currentGuess, 1, 2), i * res + res / 2, j * res + res / 2);
			index++;
		});
	});
};

const cycleGrid = (callback) => {
	let res = resolution;
	let cols = width / res;
	let rows = height / res;
	let index = 0;
	for (let i = 0; i < rows; i++) {
		for (let j = 0; j < cols; j++) {
			callback({ i, j, rows, cols, index, res });
			index++;
		}
	}
};

const startTraining = () => setTimeout(train, 10);

const train = () => {
	tf.tidy(() => {
		trainModel().then(({ history }) => {
			// console.log('end training...');
			// console.log('loss', history.loss[0]);
			// console.log('start training.....');
			startTraining();
		});
	});
};

const trainModel = () => {
	// for (let i = 0; i < 15; i++) {

	// trainingInputs.print();
	// trainingOutputs.print();
	return network.fit(trainingInputs, trainingOutputs, {
		epochs: 1, //number of times to repeat the training with the same dataset
		shuffle: true, //shuffle the training data on every iteration to improve training
	});
	// }
};
