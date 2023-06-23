import java.util.ArrayList;
import java.util.List;

public class Neuron {

	private double weight;
	private double curVal;

	public List<Neuron> connections = new ArrayList<>();
	
	public Neuron(double weight, double curVal)
	{
		this.weight = weight;
		this.curVal = curVal;
	}
	public void setWeight(double newWeight)
	{
		weight = newWeight;
	}
	public double getWeight()
	{
		return weight;
	}
	
	public void setcurVal(double newVal)
	{
		curVal = newVal;
	}
	public double getVal()
	{
		return curVal;
	}
	
}

