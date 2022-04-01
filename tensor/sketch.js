function setup() {
	noCanvas();
	// 3 images of 2x2 size
	// const twoXtwo = tf.tensor(
	// 	[0, 0, 127, 112.5, 0, 0, 127, 112.5, 0, 0, 127, 112.5],
	// 	[3, 2, 2],
	// 	'int32'
	// );
	// twoXtwo.print();
	// console.log(twoXtwo);

	// // 5x3 matrix filled with random values
	// const tense = tf.tensor(
	// 	Array.from({ length: 15 }, () => random(0, 255)),
	// 	[5, 3],
	// 	'int32'
	// );

	// // returns a promise bc the data takes some time to be loaded on the GPU
	// console.log(tense.data());

	// // waits until the data is loaded on the gpu and then returns (no promise)
	// console.log(tense.dataSync());

	// // now the tensor data can be changed and its mutable
	// const vTense = tf.variable(tense);
	// vTense.print();

	// console.log(tf.memory());
}

function draw() {
	const a = tf.tensor(
		Array.from({ length: 15 }, () => random(0, 255)),
		[5, 3],
		'int32'
	);
	const b = tf.tensor(
		Array.from({ length: 15 }, () => random(0, 255)),
		[5, 3],
		'int32'
	);

	const b_t = b.transpose();
	const c = a.matMul(b_t);

	//dispose -> delete and free memory of a tensor
	a.dispose();
	b.dispose();
	b_t.dispose();
	c.dispose();

	//execute a function and then dispose of all tensors, freeing memory at the end
	tf.tidy(() => {
		const a = tf.tensor(
			Array.from({ length: 15 }, () => random(0, 255)),
			[5, 3],
			'int32'
		);
		const b = tf.tensor(
			Array.from({ length: 15 }, () => random(0, 255)),
			[5, 3],
			'int32'
		);

		const b_t = b.transpose();
		const c = a.matMul(b_t);
	});

	console.log(tf.memory().numTensors);
}
