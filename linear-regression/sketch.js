//PREDICT A LINE THAT APPROXIMATES ALL POINTS

//range between 0-1
// x= 0, y=0 is the bottom left corner, the top is 1
function Point(x, y) {
	return { x, y, pixelX: toPixelX(x), pixelY: toPixelY(y) };
}
const getPointsXs = () => tf.tensor1d(points.map(({ x }) => x));
const getPointsYs = () => tf.tensor1d(points.map(({ y }) => y));

function Line() {
	//i create variablwes bc they must change
	m = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));

	const draw = () =>
		line(
			toPixelX(0),
			toPixelY(resolveFor(0)),
			toPixelX(width),
			toPixelY(resolveFor(width))
		);

	/**
	 *
	 * @param {Number} x
	 * @returns {Number} y values resolving y = mx+b
	 * */
	const resolveFor = (x) => m.dataSync()[0] * x + b.dataSync()[0];

	/**
	 * guess the line -> y = mx + b
	 * @param {Tensor} x x values
	 * @returns {Tensor} guessed y values
	 */
	const predict = (x) => x.mul(lineGuess.m).add(lineGuess.b);

	return { m, b, predict, draw };
}

const toPixelX = (x) => map(x, 0, 1, 0, width);
const toPixelY = (y) => map(y, 1, 0, 0, height);

let lineGuess;
let points = [];
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

function setup() {
	createCanvas(800, 800);

	lineGuess = new Line();
}

function draw() {
	train();

	background(0);
	stroke(255);
	points.forEach((p) => point(p.pixelX, p.pixelY));
	stroke(255, 0, 0);
	lineGuess.draw();
	console.log('numTensors', tf.memory().numTensors);
}

function mousePressed() {
	points.push(new Point(map(mouseX, 0, width, 0, 1), map(mouseY, 0, height, 1, 0)));
}

/**
 * calculate the error between the guessed line and the actual line using the mean squared error formula -> mean((guess - actual)^2) [mean = media dei valori nel tensor]
 * @param {Tensor} pred prediction obtained via the predict function
 * @param {Tensor} label actual y values of the points
 * @returns {Tensor} computed loss tensor
 */
const loss = (pred, label) => pred.sub(label).square().mean();

const train = () => {
	if (points.length !== 0) {
		tf.tidy(() =>
			optimizer.minimize(() => loss(lineGuess.predict(getPointsXs()), getPointsYs()))
		);
	}
};
