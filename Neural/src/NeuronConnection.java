
public class NeuronConnection {

	double conWeight;
	
	Neuron toNeuron;
	Neuron fromNeuron;
	
	public NeuronConnection(Neuron toNeuron, Neuron fromNeuron)
	{
		this.toNeuron = toNeuron;
		this.fromNeuron = fromNeuron;	
		this.conWeight = Math.random();
	}
	
	public NeuronConnection(Neuron toNeuron, Neuron fromNeuron, int weight)
	{
		this(toNeuron, fromNeuron);
		conWeight = weight;
	}
}
