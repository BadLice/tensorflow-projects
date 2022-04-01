//PREDICT A CURVE LINE THAT APPROXIMATES ALL POINTS

//range between -1/1
// x= 0, y=0 is the venter, the top is 1 the bottom is -1, left is -1 right is 1 (like cartesian plane)
function Point(x, y) {
	const pixelX = toPixelX(x);
	const pixelY = toPixelY(y);

	const draw = () => {
		stroke(255);
		point(pixelX, pixelY);
	};

	return { x, y, pixelX, pixelY, draw };
}
const getPointsXs = () => tf.tensor1d(points.map(({ x }) => x));
const getPointsYs = () => tf.tensor1d(points.map(({ y }) => y));

function Parable() {
	// y = ax^2 + bx + c
	a = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));
	c = tf.variable(tf.scalar(random(1)));

	const draw = () => {
		tf.tidy(() => {
			stroke(255, 0, 0);
			noFill();
			beginShape();
			const parableX = tf.tensor1d(
				Array.from({ length: width }, (_, i) => toCartesianX(i))
			);
			const parableY = predict(parableX);
			const parableXList = parableX.dataSync();
			const parableYList = parableY.dataSync();
			parableXList.forEach((x, i) => vertex(toPixelX(x), toPixelY(parableYList[i])));
			endShape();
		});
	};

	/**
	 * guess the parable -> y = ax^2 + bx + c
	 * @param {Tensor} x x values
	 * @returns {Tensor} guessed y values
	 */
	const predict = (x) => tf.tidy(() => x.square().mul(a).add(x.mul(b)).add(c));

	return { a, b, c, predict, draw };
}

const toPixelX = (x) => map(x, -1, 1, 0, width);
const toPixelY = (y) => map(y, 1, -1, 0, height);
const toCartesianX = (x) => map(x, 0, width, -1, 1);
const toCartesianY = (y) => map(y, 0, height, 1, -1);

let parable;
let points = [];
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

function setup() {
	createCanvas(800, 800);

	parable = new Parable();
}

function draw() {
	train();

	background(0);
	points.forEach((p) => p.draw());
	parable.draw();
	console.log('numTensors', tf.memory().numTensors);
}

function mousePressed() {
	points.push(new Point(toCartesianX(mouseX), toCartesianY(mouseY)));
}

/**
 * calculate the error between the guessed y and the actual y using the mean squared error formula -> mean((guess - actual)^2) [mean = media dei valori nel tensor]
 * @param {Tensor} pred prediction obtained via the predict function
 * @param {Tensor} label actual y values of the points
 * @returns {Tensor} computed loss tensor
 */
const loss = (pred, label) => pred.sub(label).square().mean();

const train = () => {
	if (points.length !== 0) {
		tf.tidy(() =>
			optimizer.minimize(() => loss(parable.predict(getPointsXs()), getPointsYs()))
		);
	}
};
