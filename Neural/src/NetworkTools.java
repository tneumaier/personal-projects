
public class NetworkTools {

	public static double[][] createRandomArray(int sizeX, int sizeY, double lowerBound, double upperBound)
	{
		if(sizeX < 1 || sizeY < 1)
		{
			return null;
		}
		
		double[][] ar = new double[sizeX][sizeY];
		for(int i = 0; i < sizeX; i++)
		{
			ar[i] = createRandomArray(sizeY, lowerBound, upperBound);
		}
		return ar;
	}
	
	public static double[] createRandomArray(int size, double lowerBound, double upperBound)
	{
		if(size < 1)
		{
			return null;
		}
		double[] ar = new double[size];
		for(int i = 0; i < size; i++)
		{
			ar[i] = randomValue(lowerBound, upperBound);
		}
		return ar;
	}
	
	public static double randomValue(double lowerBound, double upperBound)
	{
		return Math.random() * (upperBound-lowerBound) + lowerBound;
	}
	
	public static Integer[] randomValues(int lowerBound, int upperBound, int amount)
	{
		lowerBound--;
		 
		if(amount > (upperBound-lowerBound))
		{
			return null;
		}
		
		Integer[] values = new Integer[amount];
		for(int i = 0; i < amount; i++)
		{
			int n = (int)(Math.random() * (upperBound-lowerBound+1) + lowerBound);
			while(containsValue(values, n))
			{
				n = (int)(Math.random() * (upperBound-lowerBound+1) + lowerBound);
			}
			values[i] = n;
		}
		return values;
	}
	
	public static <T extends Comparable<T>> boolean containsValue(T[] ar, T value)
	{
		for(int i = 0; i < ar.length; i++)
		{
			if(ar[i] != null)
			{
				if(value.compareTo(ar[i]) == 0)
				{
					return true;
				}
			}
		}
		return false;
	}
	
}
