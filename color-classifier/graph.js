class Graph {
	maxLoss = Number.MIN_VALUE;
	lossGraphPoints = [];
	validationLossGraphPoints = [];
	currentEpoch = 0;
	currentBatch = 0;
	lastLoss = 0;
	validationLastLoss = 0;
	pixelPerFrame = 1;
	showBatch = false;

	draw = () => {
		this.drawTraining();
		this.drawValidation();

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
		text(nf(this.maxLoss, 1, 2), 25, 10);
		text(0, 25, height);

		//draw epochs and batch
		textSize(15);
		fill(255);
		text('epoch: ' + this.currentEpoch + ' - batch:' + this.currentBatch, width - 150, 15);

		fill(255, 0, 0);
		text('training loss', width - 150, 30);
		fill(0, 255, 0);
		text('validation loss', width - 150, 45);
	};

	//training  data loss graph
	drawTraining = () => {
		textSize(12);

		//draw graph points
		this.lossGraphPoints.forEach(([x, y], i) => {
			if (i !== this.lossGraphPoints.length - 1) {
				strokeWeight(1);
				stroke(255, 0, 0);
				line(x, y, this.lossGraphPoints[i + 1][0], this.lossGraphPoints[i + 1][1]);
				strokeWeight(5);
				stroke(255, 192, 56);
				point(x, y);
			}
		});

		//draw last loss value
		if (this.lossGraphPoints.length) {
			noStroke();
			fill(255, 0, 0);
			text(
				nf(this.lastLoss, 1, 2),
				this.lossGraphPoints[this.lossGraphPoints.length - 1][0] + 10,
				this.lossGraphPoints[this.lossGraphPoints.length - 1][1]
			);
		}
	};

	//validation data loss graph
	drawValidation = () => {
		textSize(12);

		//draw graph points
		this.validationLossGraphPoints.forEach(([x, y], i) => {
			if (i !== this.validationLossGraphPoints.length - 1) {
				strokeWeight(1);
				stroke(0, 255, 0);
				line(
					x,
					y,
					this.validationLossGraphPoints[i + 1][0],
					this.validationLossGraphPoints[i + 1][1]
				);
				strokeWeight(5);
				stroke(168, 255, 175);
				point(x, y);
			}
		});

		//draw last loss value
		if (this.validationLossGraphPoints.length) {
			noStroke();
			fill(0, 255, 0);
			text(
				nf(this.validationLastLoss, 1, 2),
				this.validationLossGraphPoints[this.validationLossGraphPoints.length - 1][0] +
					10,
				this.validationLossGraphPoints[this.validationLossGraphPoints.length - 1][1]
			);
		}
	};

	update = (epochs, logs) => {
		if (!this.showBatch) {
			if (!epochs) {
				return;
			}
		}
		if (epochs) {
			this.currentEpoch++;
		}
		if (logs.batch) {
			this.currentBatch = logs.batch;
		}

		// console.log(epochs, logs);
		const loss = logs.loss;
		const vloss = logs.val_loss ?? Number.MIN_VALUE;
		// console.log(logs, loss, vloss);
		if (loss > this.maxLoss || vloss > this.maxLoss) {
			this.normalizePoints(this.maxLoss, loss);
			this.normalizePoints(this.maxLoss, vloss);
			this.maxLoss = loss > vloss ? loss : vloss;
		}

		let x = (width / 4) * 3;

		// let x =
		// 	this.lossGraphPoints.length === 0
		// 		? (width / 4) * 3
		// 		: this.lossGraphPoints[this.lossGraphPoints.length - 1][0] + this.pixelPerFrame;
		let y = map(loss, 0, this.maxLoss, height, 0);
		this.lossGraphPoints.push([x, y]);
		this.lastLoss = loss;

		if (vloss !== Number.MIN_VALUE) {
			// let vx =
			// 	this.validationLossGraphPoints.length === 0
			// 		? (width / 4) * 3
			// 		: this.validationLossGraphPoints[this.validationLossGraphPoints.length - 1][0] +
			// 		  this.pixelPerFrame;
			let vx = (width / 4) * 3;
			let vy = map(vloss, 0, this.maxLoss, height, 0);
			this.validationLossGraphPoints.push([vx, vy]);
			this.validationLastLoss = vloss;
		}

		this.scroll();
		this.clear();
		return tf.nextFrame(); //wait for tensorflow to release GPU for this frame
	};

	scroll = () => {
		this.lossGraphPoints = this.lossGraphPoints.map(([x, y]) => [
			x - this.pixelPerFrame,
			y,
		]);
		this.validationLossGraphPoints = this.validationLossGraphPoints.map(([x, y]) => [
			x - this.pixelPerFrame,
			y,
		]);
	};

	//remove points out of screen
	clear = () => {
		this.lossGraphPoints = this.lossGraphPoints.filter(([x, y]) => x > width / -2);
		this.validationLossGraphPoints = this.validationLossGraphPoints.filter(
			([x, y]) => x > width / -2
		);
	};

	normalizePoints = (oldMax, currentMax) => {
		this.lossGraphPoints.map(([x, y]) => {
			let oldPixelY = map(y, height, 0, 0, oldMax);
			return [x, map(oldPixelY, 0, currentMax, height, 0)];
		});
	};
}
