class Predictor {
	color = null;

	static hexToRgb = (hex) => {
		const c = hex;
		const r = parseInt(c.substr(1, 2), 16);
		const g = parseInt(c.substr(3, 2), 16);
		const b = parseInt(c.substr(5, 2), 16);
		return { r, g, b };
	};

	setColor = (c) => {
		if (typeof c === 'string') {
			c = Predictor.hexToRgb(c);
		}

		console.log(c);

		this.color = c;
	};

	draw = () => {
		document.querySelector('#color-picker').disabled = false;
		document.querySelector('#color-picker').hidden = false;
		if (this.color) {
			background(this.color.r, this.color.g, this.color.b);
			tf.tidy(() => {
				//predict the color lable using the neural network. Get the index with max probability using argMax along axis 1 (the second axis bc its a oneHot tensor -> 2d tensor)
				let index = model.network
					.predict(tf.tensor2d([[this.color.r, this.color.g, this.color.b]]))
					.argMax(1)
					.dataSync()[0];

				noStroke();
				fill(255 - this.color.r, 255 - this.color.g, 255 - this.color.b);
				textAlign(CENTER, CENTER);
				textSize(60);
				text(Model.labelList[index], width / 2, height / 2);
			});
		}
	};
}
