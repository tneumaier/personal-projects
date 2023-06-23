import java.util.Arrays;
public class Network {

	//the number of neurons in each layer
	private static  int[] NETWORK_LAYER_SIZES;
	//the number of neurons in the first layer
	private static  int INPUT_SIZE;
	//the number of neurons in the last layer
	private static  int OUTPUT_SIZE;
	//the number of layers
	private static  int NETWORK_SIZE;
	
	//3 dimensional matrix of layer, current neuron, and previous neuron
	private static double[][][] weights;
	//outputs of neuron at layer
	private static  double[][] outputs;
	//biases of neuron at layer
	private static  double[][] bias;
	//this contains the calculated error at each neuron for a given training data
	private static double[][] error_signal;
	//this contains the calculated derivative of the output (which is the sum all 
	//of the sums of the products of the weights of the previous layer and the current bias)
	private static double[][] output_derivative;
	
	public Network(int[] NETWORK_LAYER_SIZES)
	{
		//initializes all of the constants based on parameter
		this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
		this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
		this. INPUT_SIZE = NETWORK_LAYER_SIZES[0];
		this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1];
		
		this.weights = new double[NETWORK_SIZE][][];
		this.outputs = new double[NETWORK_SIZE][];
		this.bias = new double[NETWORK_SIZE][];
		this.error_signal = new double[NETWORK_SIZE][];
		this.output_derivative = new double[NETWORK_SIZE][];
		
		for(int i = 0; i < NETWORK_SIZE; i++)
		{
			//initializes each column of each array to be the length of a given layer
			this.outputs[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.bias[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
			
			this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], 0.0, 0.9);
			if(i>0)
			{
				//since this is in respect to the previous layer as well, can't be at 0 or it would be out of bounds
				weights[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i-1]];
				
				weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i-1], 0.0, 0.9);
			}
		}
		
		//randomizes the biases and weights for each neuron
		for(int layer = 1; layer < NETWORK_SIZE; layer++)
		{
			for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
			{
				bias[layer][neuron] = Math.random();
				for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
				{
					weights[layer][neuron][prevNeuron] = Math.random();
				}
			}
		}
		
	
	}

	//calculates the output of each neuron, as well as the output derivative
	public static double[] feedForward(double input[])
	{
		//just in case there is an invalid input
		if(input.length != INPUT_SIZE) 
		{
			return null;
		}
		
		//initializes the first layer to be the given inputs
		outputs[0] = input;
		
		//works through each layer each neuron for that layer and calculates the output.
		//output(neuron j) = bias(j) + (sum of all weights of every neuron in the previous 
		//layer times their respective output)
		//this is then put through the sigmoid (logistic) function to restrict output to >=0 and <=1
		for(int layer = 1; layer < NETWORK_SIZE; layer++)
		{
			for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
			{
				double sum = bias[layer][neuron];
				
				
				for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
				{
					sum += outputs[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
				}
				outputs[layer][neuron] = sigmoid(sum);
				output_derivative[layer][neuron] = (outputs[layer][neuron] * (1-outputs[layer][neuron]));
				
			}
		}
	return outputs[NETWORK_SIZE-1];
	}
	
	//declaration of the sigmoid (logistic curve/activation function)
	public static double sigmoid(double x)
	{
		return 1d/(1+Math.pow(Math.E, -x));
	}
	
	//calculates the error of each neuron by using equations calculated via calculus
	//and/or linear algebra.
	public static void backpropError(double[] target)
	{
		//calculates the error for each neuron in the output layer
		for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron++)
		{
			error_signal[NETWORK_SIZE-1][neuron] = (outputs[NETWORK_SIZE-1][neuron] - target[neuron]) 
					* output_derivative[NETWORK_SIZE-1][neuron];
		}
		//calculates the error for each neuron in each of the hidden layers
		for(int layer = NETWORK_SIZE-2; layer > 0; layer--)
		{
			for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
			{
				double sum = 0;
				for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++)
				{
					sum += (weights[layer+1][nextNeuron][neuron] * error_signal[layer+1][nextNeuron]);
					
				}
				error_signal[layer][neuron] = sum *  output_derivative[layer][neuron];
			}
		}
	}
	
	//changes the weights of the network based on the error of each neuron
	//parameter 'eta' is the learning rate, which is arbitrary in this case, but can be
	//optimized using 'dynamic programming'
	public static void updateWeights(double eta)
	{
		//moves through each neuron's weights to all previous neurons and adds the 
		//product of the learning rate, the output, and the error
		for(int layer = 1; layer < NETWORK_SIZE; layer++)
		{
			for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++)
			{
				for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++)
				{
					//more fancy calculus stuff
					double delta = eta * outputs[layer-1][prevNeuron] * error_signal[layer][neuron];
					weights[layer][neuron][prevNeuron] -= delta;
				}
				//this does the same for biases, except a neurons bias is not based on the previous layer,
				//so the traversing across the output or weight matrices is not needed
				double delta = eta * error_signal[layer][neuron];
				bias[layer][neuron] -= delta;
			}
		}
	}
	
	public static void train(double[] input, double[] target, double eta)
	{
		if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE)
		{
			return;
		}
		feedForward(input);
		backpropError(target);
		updateWeights(eta);
		
		
	}
	
	public static void main(String args[])
	{
		double[] inputs = new double[]{0.2, 0.8, 0.6};
		double[] target = new double[] {0.3};
		int [] layerSizes = {3, 3, 1};
		Network newNetwork = new Network(layerSizes);
		
		//trains from this data 1000 times to accurately set weights and biases
		for(int i = 0; i < 1000; i++)
		{
			newNetwork.train(inputs, target, .3);
		}
		
		double[] o = newNetwork.feedForward(inputs);
		System.out.println("Predicted Output = " + Arrays.toString(o));
	}
	
	
	
}
