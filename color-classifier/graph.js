//TODO: draw validation loss
function Graph() {
	let maxLoss = Number.MIN_VALUE;
	let lossGraphPoints = [];
	let validationLossGraphPoints = [];
	let currentEpoch = 0;
	let currentBatch = 0;
	let lastLoss = 0;
	let validationLastLoss = 0;

	const draw = () => {
		drawTraining();
		drawValidation();

		//draw x-axis
		//draw vertical line
		strokeWeight(2);
		stroke(100);
		line(10, 0, 10, height);
		//draw upper pointer line
		stroke(255, 0, 0);
		line(10, 3, 20, 3);
		//draw bottom pointer line
		line(10, height - 3, 20, height - 3);
		//draw upper and bottom pointers values
		fill(0, 255, 0);
		noStroke();
		text(nf(maxLoss, 1, 2), 25, 10);
		text(0, 25, height);

		//draw epochs and batch
		textSize(15);
		fill(255);
		text('epoch: ' + currentEpoch + ' - batch:' + currentBatch, width - 150, 15);
	};

	//training  data loss graph
	const drawTraining = () => {
		textSize(12);
		strokeWeight(3);

		//draw graph points
		lossGraphPoints.forEach(([x, y], i) => {
			if (i !== lossGraphPoints.length - 1) {
				strokeWeight(1);
				stroke(150);
				line(x, y, lossGraphPoints[i + 1][0], lossGraphPoints[i + 1][1]);
				stroke(255, 0, 0);
				point(x, y);
			}
		});

		//draw last loss value
		if (lossGraphPoints.length) {
			text(
				lastLoss,
				lossGraphPoints[lossGraphPoints.length - 1][0] + 10,
				lossGraphPoints[lossGraphPoints.length - 1][1]
			);
		}
	};

	//validation data loss graph
	const drawValidation = () => {
		textSize(12);

		//draw graph points
		validationLossGraphPoints.forEach(([x, y], i) => {
			if (i !== validationLossGraphPoints.length - 1) {
				strokeWeight(2);
				stroke(150);
				line(
					x,
					y,
					validationLossGraphPoints[i + 1][0],
					validationLossGraphPoints[i + 1][1]
				);
				stroke(0, 255, 0);
				point(x, y);
			}
		});

		//draw last loss value
		if (validationLossGraphPoints.length) {
			noStroke();
			text(
				validationLastLoss,
				validationLossGraphPoints[validationLossGraphPoints.length - 1][0] + 10,
				validationLossGraphPoints[validationLossGraphPoints.length - 1][1]
			);
		}
	};

	const update = (epochs, logs) => {
		if (epochs) {
			currentEpoch++;
		}
		if (logs.batch) {
			currentBatch = logs.batch;
		}

		// console.log(epochs, logs);
		const loss = logs.loss;
		const vloss = logs.val_loss;
		console.log(logs);
		if (loss > maxLoss || vloss > maxLoss) {
			normalizePoints(maxLoss, loss);
			normalizePoints(maxLoss, vloss);
			maxLoss = loss > vloss ? loss : vloss;
		}

		let x =
			lossGraphPoints.length === 0
				? (width / 4) * 3
				: lossGraphPoints[lossGraphPoints.length - 1][0] + 1;
		let y = map(loss, 0, maxLoss, height, 0);
		lossGraphPoints.push([x, y]);

		let vx =
			validationLossGraphPoints.length === 0
				? (width / 4) * 3
				: validationLossGraphPoints[validationLossGraphPoints.length - 1][0] + 1;
		let vy = map(loss, 0, maxLoss, height, 0);
		validationLossGraphPoints.push([vx, vy]);

		lastLoss = loss;
		validationLastLoss = vloss;

		scroll();
		return tf.nextFrame(); //wait for tensorflow to release GPU for this frame
	};

	const scroll = () => {
		lossGraphPoints = lossGraphPoints.map(([x, y]) => [x - 10, y]);
		validationLossGraphPoints = validationLossGraphPoints.map(([x, y]) => [x - 10, y]);
	};

	const normalizePoints = (oldMax, currentMax) => {
		lossGraphPoints.map(([x, y]) => {
			let oldPixelY = map(y, height, 0, 0, oldMax);
			return [x, map(oldPixelY, 0, currentMax, height, 0)];
		});
	};

	return { lossGraphPoints, draw, update, scroll, normalizePoints };
}
