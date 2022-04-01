let colorData;
let trainingData = {
	inputs: [],
	outputs: [],
};
let testingData = {
	inputs: [],
	outputs: [],
};
let labelList = [
	'red-ish',
	'green-ish',
	'blue-ish',
	'orange-ish',
	'yellow-ish',
	'pink-ish',
	'purple-ish',
	'brown-ish',
	'grey-ish',
];
let model;
const learningRate = 0.1;

const section = 'training';
let maxLoss = Number.MIN_VALUE;
let lossGraphPoints = [];

function preload() {
	colorData = loadJSON('./colorData.json');
}

function setup() {
	createCanvas(800, 800);

	readColorData();
	setupModel();
	train();
}

function draw() {
	background(0);
	drawLossGraph();
}

const drawLossGraph = () => {
	fill(0, 255, 0);
	noStroke();
	text(maxLoss, 0, 10);
	text(0, 0, height - 5);
	lossGraphPoints.forEach(([x, y], i) => {
		if (i !== lossGraphPoints.length - 1) {
			strokeWeight(1);
			stroke(150);
			line(x, y, lossGraphPoints[i + 1][0], lossGraphPoints[i + 1][1]);
			stroke(255, 0, 0);
			point(x, y);
		}
	});
};

const updateLossGraph = async (epochs, logs) => {
	await tf.nextFrame(); //wait for tensorflow to release GPU for this frame

	console.log(epochs, logs);
	const loss = logs.loss;
	if (loss > maxLoss) {
		normalizeLossGraphPoints(maxLoss, loss);
		maxLoss = loss;
	}

	let x =
		lossGraphPoints.length === 0
			? (width / 4) * 3
			: lossGraphPoints[lossGraphPoints.length - 1][0] + 1;
	let y = map(loss, 0, maxLoss, height, 0);
	lossGraphPoints.push([x, y]);

	scrollGraph();
};

const scrollGraph = () => (lossGraphPoints = lossGraphPoints.map(([x, y]) => [x - 1, y]));

const normalizeLossGraphPoints = (oldMax, currentMax) => {
	lossGraphPoints.map(([x, y]) => {
		let oldPixelY = map(y, height, 0, 0, oldMax);
		return [x, map(oldPixelY, 0, currentMax, height, 0)];
	});
};

const getInputs = () => colorData.map(({ input }) => input);
const getOutputs = () => colorData.map(({ target }) => target);

const readColorData = () => {
	colorData = colorData.entries.map(({ r, g, b, label }) => ({
		input: [r / 255, g / 255, b / 255],
		target: labelList.indexOf(label),
	}));
	let inputs = tf.tensor2d(getInputs(), [colorData.length, 3]);
	// special type of tensor used for outputs: used to encod ethe probability that an output corresponds to a certain label
	let indices = tf.tensor1d(getOutputs(), 'int32');
	let outputs = tf.oneHot(indices, 9);
	indices.dispose();

	//split gathered data: 80% for training, 20% for testing
	let length80perc = Math.floor((colorData.length * 80) / 100);
	[trainingData.inputs, testingData.inputs] = tf.split(inputs, [
		length80perc,
		colorData.length - length80perc,
	]);
	[trainingData.outputs, testingData.outputs] = tf.split(outputs, [
		length80perc,
		colorData.length - length80perc,
	]);

	inputs.dispose();
	outputs.dispose();
};

const setupModel = () => {
	model = tf.sequential();
	const hidden = tf.layers.dense({
		units: 16,
		activation: 'sigmoid',
		inputShape: [3],
	});
	const output = tf.layers.dense({
		units: 9,
		activation: 'softmax', //is used to guarantee that the sum of all the probability guessed for every output is 100%. It's used when the output is a range of probabilities. must be used if output tensor is a oneHot
	});
	model.add(hidden);
	model.add(output);
	model.compile({
		loss: tf.metrics.categoricalCrossentropy, //function used to calculate the error betweer thwo distribution of probabilities. must be used if output tensor is a oneHot and output activation function is softmax
		optimizer: tf.train.sgd(learningRate),
	});
};

const train = () => {
	console.log('start');
	model
		.fit(trainingData.inputs, trainingData.outputs, {
			epochs: 50, //number of times to repeat the training with the same dataset
			shuffle: true, //shuffle the training data on every iteration to improve training
			validationSplit: 0.1, //Data on which to evaluate the loss and any model metrics at the end of each epoch. here im telling to use 1% of the testing data as valitadion data
			callbacks: {
				onBatchEnd: (_, logs) => updateLossGraph(null, logs),
				onEpochEnd: (epoch, logs) => updateLossGraph(epoch + 1, logs),
			},
		})
		.then(({ history }) => console.log('finish', history.loss));
};
