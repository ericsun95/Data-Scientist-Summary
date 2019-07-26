# How to build a dataset and dataSetIterator?

### Basic knowledge for Recurrent Neural Networks DataSet
## Input into Recurrent Neural Networks (3D)
![Architectures of Deep Networks](https://www.oreilly.com/library/view/deep-learning/9781491924570/assets/dpln_0417.png)
[Link](https://www.oreilly.com/library/view/deep-learning/9781491924570/ch04.html)
**Mini-batch** Mini-batch size is the number of input records (collections of time-series points
for a single source entity) we want to model per batch. The **number of columns**
matches up to the traditional feature column count found in a normal input
vector. The **number of time-steps** is how we represent the change in the input
vector over time. 

## Uneven time-series and masking
![The time-step aspect of Recurrent Neural Network input](https://www.oreilly.com/library/view/deep-learning/9781491924570/assets/dpln_0418.png)
Every column value likely will not occur at every time-step, especially for the case in which we’re mixing descriptor data 
(e.g., columns from a static database table) with time-series data (e.g., measurements of an ICU patient’s heartrate every minute). 
For cases in which we have “jagged” time-step values, we need to use masking to let DL4J know where our real data is located in the 
vector. We do this by providing an extra matrix for the mask indicating the time-steps that have input data for at least one column, 
as demonstrated here.
![Masking specific time-steps](https://www.oreilly.com/library/view/deep-learning/9781491924570/assets/dpln_0419.png)

## Building custom DataSets from sequential data
```java
// DataSet d = new DataSet( input, labels, mask_in, mask_labels );
//Allocate space: { mini-batch size, number columns, number timesteps }
INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, inputColumnCount, maxTimestepLength });
INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, outputColumnCount, maxTimestepLength });
INDArray mask = Nd4j.zeros(new int[]{ miniBatchSize, maxTimestepLength });
for (int miniBatchIndex = 0; miniBatchIndex < miniBatchSize; miniBatchIndex++) {
    for ( int curTimestep = 0; curTimestep < endTimestep; curTimestep++ ){
        // input -> set the column index for the character id-index ->
        //at the current timestep (c)
        input.putScalar(new int[]{ miniBatchIndex, columnIndex, curTimestep }, 1.0);
        // do more column input here ....
        // now setup the masks, setting a 1.0 for the timestep wherever there is data
        mask.putScalar(new int[]{ miniBatchIndex, curTimestep }, 1.0);
        // labels -> set the column index for the next character id-index ->
        //at the current timestep (c)
        labels.putScalar(new int[]{ miniBatchIndex, nextValue, timestep }, 1.0);
    }
}
INDArray mask2 = Nd4j.zeros(new int[]{ miniBatchSize, maxLength });
Nd4j.copy(mask, mask2);
return new DataSet(input,labels, mask, mask2);

```

Obviously, by the time we’d execute this code, we’d have performed all of the
ETL and vectorization needed for the raw data. At this point, we’re just aligning
the data in the tensor data structure. Here are the main things to note about this
code:<br>
- We need to use ND4J to construct our input matrices.
- The two-loop pattern is generally a coherent way to walk through the 3D
data structure and set up our data.
- We then mostly make calls to `INDArray.setScalar()` to set values.

The trick we’ll note here is how we lay out the label matrix data. If we’re doing
character prediction, as in the Shakespeare example, we’re
putting the next character value at the current time-step in the labels matrix. If
we’re doing a sequential data classification model (e.g., classifying an anomaly
in log data, for example), we might put the class at every single time-step in the
labels matrix. How we lay out the data is a bit of an art and there are multiple
strategies for treating data that occurs at uneven time-steps. The strategy for
aligning time-steps is a decision we leave to your discretion.

> USING MASKING IN TENSORS
In DL 4J, we use masks for both training data and labels.
We set the mask time-step value to 1.0 for every time-step containing training data in the training data
mask. We set the mask time-step value to 0.0 for all other time-steps. We normally use the same mask
for both the input data structure and the label data structure.
Typically, we see this in code where we’re looping over the tensor data structure and we set the mask
entry for each time-step as we go along. This might look like the following:
```java
INDArray mask = Nd4j.zeros(new int[]{ miniBatchSize, maxLength });
...
for (...) {
...
mask.putScalar(new int[]{ miniBatchIndex, timeStep }, 1.0);
```
> Notice that we’re setting only a mini-batch index and then a time-step index in the mask data structure.
We don’t need to specify which columns have data, just that any column in this specific time-step
could have data.

## [How can I use a custom data model with Deeplearning4j?](https://stackoverflow.com/questions/48845162/how-can-i-use-a-custom-data-model-with-deeplearning4j)
In your DataSetIterator implementation you must pass your data and in the implementation of the `next()` method you should 
return a `DataSet` object comprising the next batch of your training data. It would look like this:
```java
public class MyCustomIterator implements DataSetIterator {
    private INDArray inputs, desiredOutputs;
    private int itPosition = 0; // the iterator position in the set.

    public MyCustomIterator(float[] inputsArray,
                            float[] desiredOutputsArray,
                            int numSamples,
                            int inputDim,
                            int outputDim) {
        inputs = Nd4j.create(inputsArray, new int[]{numSamples, inputDim});
        desiredOutputs = Nd4j.create(desiredOutputsArray, new int[]{numSamples, outputDim});
    }

    public DataSet next(int num) {
        // get a view containing the next num samples and desired outs.
        INDArray dsInput = inputs.get(
            NDArrayIndex.interval(itPosition, itPosition + num),
            NDArrayIndex.all());
        INDArray dsDesired = desiredOutputs.get(
            NDArrayIndex.interval(itPosition, itPosition + num),
            NDArrayIndex.all());

        itPosition += num;

        return new DataSet(dsInput, dsDesired);
    }

    // implement the remaining virtual methods...

}
```

