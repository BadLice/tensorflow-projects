class Model {
	network = null;
	learningRate = 0.1;
	colorData = null;
	nodeAmount = 16;
	epochsAmount = 50;
	isTrained = false;

	trainingData = {
		inputs: [],
		outputs: [],
	};
	testingData = {
		inputs: [],
		outputs: [],
	};

	static labelList = [
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

	static optimizerList = [
		tf.train.sgd,
		tf.train.adam,
		tf.train.adamax,
		tf.train.adadelta,
		tf.train.adagrad,
		tf.train.rmsprop,
	];
	optimizerIndex = 0;

	static activationFList = [
		'sigmoid',
		'elu',
		'hardSigmoid',
		'linear',
		'relu',
		'relu6',
		'selu',
		'softmax',
		'softplus',
		'softsign',
		'tanh',
	];
	activationFIndex = 0;

	setColorData = (data) => (this.colorData = data);
	getInputs = () => this.colorData.map(({ input }) => input);
	getOutputs = () => this.colorData.map(({ target }) => target);
	readColorData = () => {
		this.colorData = this.colorData.entries.map(({ r, g, b, label }) => ({
			input: [r / 255, g / 255, b / 255],
			target: Model.labelList.indexOf(label),
		}));
		let inputs = tf.tensor2d(this.getInputs(), [this.colorData.length, 3]);
		// special type of tensor used for outputs: used to encod ethe probability that an output corresponds to a certain label
		let indices = tf.tensor1d(this.getOutputs(), 'int32');
		let outputs = tf.oneHot(indices, 9);
		indices.dispose();

		//split gathered data: 80% for training, 20% for testing
		let length80perc = Math.floor((this.colorData.length * 80) / 100);
		[this.trainingData.inputs, this.testingData.inputs] = tf.split(inputs, [
			length80perc,
			this.colorData.length - length80perc,
		]);
		[this.trainingData.outputs, this.testingData.outputs] = tf.split(outputs, [
			length80perc,
			this.colorData.length - length80perc,
		]);

		inputs.dispose();
		outputs.dispose();
	};

	setup = () => {
		this.network = tf.sequential();
		const hidden = tf.layers.dense({
			units: 16,
			activation: Model.activationFList[this.activationFIndex], //default: sigmoid
			inputShape: [3],
		});
		const output = tf.layers.dense({
			units: 9,
			activation: 'softmax', //is used to guarantee that the sum of all the probability guessed for every output is 100%. It's used when the output is a range of probabilities. must be used if output tensor is a oneHot
		});
		this.network.add(hidden);
		this.network.add(output);
		this.compile();
	};

	compile = () => {
		this.network.compile({
			loss: tf.metrics.categoricalCrossentropy, //function used to calculate the error betweer thwo distribution of probabilities. must be used if output tensor is a oneHot and output activation function is softmax
			optimizer: Model.optimizerList[this.optimizerIndex](this.learningRate), //default optimizer: sgd
		});
	};

	train = (onTrainEnd) => {
		this.isTrained = false;

		this.network
			.fit(this.trainingData.inputs, this.trainingData.outputs, {
				epochs: this.epochsAmount, //number of times to repeat the training with the same dataset
				shuffle: true, //shuffle the training data on every iteration to improve training
				validationSplit: 0.1, //Data on which to evaluate the loss and any model metrics at the end of each epoch. here im telling to use 1% of the testing data as valitadion data
				callbacks: {
					onTrainEnd,
					onBatchEnd: (_, logs) => graph.update(null, logs),
					onEpochEnd: (epoch, logs) => graph.update(epoch + 1, logs),
				},
			})
			.then(() => {
				this.isTrained = true;
			});
	};

	download = async () => await this.network.save('downloads://my-model');

	load = async (uploadJSONInput, uploadWeightsInput, onLoadEnd) => {
		this.network = await tf.loadLayersModel(
			tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]])
		);

		this.nodeAmount = this.network.getConfig().layers[0].config.units;
		this.activationFIndex = Model.activationFList.indexOf(
			this.network.getConfig().layers[0].config.activation
		);
		this.isTrained = true;
		this.compile();
		onLoadEnd();
	};

	launchTest = (callBack) => {
		tf.tidy(() => {
			model.testingData.outputs
				.argMax(1)
				.array()
				.then((testingOutputs) => {
					let rightPredictionsAmount = Array.from(
						model.network.predict(model.testingData.inputs).argMax(1).dataSync()
					).reduce((acc, guess, i) => (guess === testingOutputs[i] ? acc + 1 : acc));
					callBack(rightPredictionsAmount, testingOutputs.length);
				});
		});
	};
}
