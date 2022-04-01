// input layer: 2 nodes, hidden layer: 4 nodes, output layer: 3 nodes
// ----- configure the model ---
//creating a neural network
const network = tf.sequential();

//create hidden and output layers. they are dense = every node of a layer is connected to every node of the next layer
const hidden = tf.layers.dense({
	units: 4, //number of nodes in the layer
	activation: 'sigmoid', //activation function
	inputShape: [2], //must always been specified for the first layer of the model, its the shape of the input layer
});
const output = tf.layers.dense({
	units: 3,
	activation: 'sigmoid',
});

// put the layers in the model
network.add(hidden);
network.add(output);

// ---- compile the model (prepare it for training and evaluation) ----

const learningRate = 0.5;
//Configures and prepares the model for training and evaluation
network.compile({
	loss: tf.losses.meanSquaredError, // function used to calculate the loss between the guess of the model and the real output
	optimizer: tf.train.sgd(learningRate), // optimizer is a thing that minimizes the loss of the guess to train the model
});

// ---- train the model ----

// retrieve the training data set (here random, but in real life there must be tons of known data with known output)
//inputs and outputs must be the same size!!! for every known input i must also know the output in order to train the model (4 in this example)
//same shape of input layer
const trainingInputs = tf.tensor2d(
	Array.from({ length: 5 }, () => Array.from({ length: 2 }, () => Math.random()))
);

//same shape of output layer
const trainingOutputs = tf.tensor2d(
	Array.from({ length: 5 }, () => Array.from({ length: 3 }, () => Math.random()))
);

const train = async () => {
	for (let i = 0; i < 100; i++) {
		const { history } = await network.fit(trainingInputs, trainingOutputs, {
			epochs: 1, //number of times to repeat the training with the same dataset
			shuffle: true, //shuffle the training data on every iteration to improve training
		});
		console.log(history.loss[0]);
	}
};

train().then(() => {
	console.log('training complete');
	// ---- take the guess ----
	// initialize a series of inputs
	const inputs = tf.tensor2d(
		Array.from({ length: 10 }, () => Array.from({ length: 2 }, () => Math.random()))
	);

	//predict the result for a series of inputs
	const outputs = network.predict(inputs);
	outputs.print();
});
