# Deep Learning for Java

## [ND4J](https://deeplearning4j.org/docs/latest/nd4j-overview)

## DL4J pipeline
As with training on a single machine, the final step of a data pipeline should be to produce a `DataSet` (single features arrays, single label array) 
or MultiDataSet (one or more feature arrays, one or more label arrays). In the case of DL4J on Spark, the final step of a data pipeline is data in one of 
the following formats: <br>
(a) an `RDD<DataSet>`/`JavaRDD<DataSet>` <br>
(b) an `RDD<MultiDataSet>`/`JavaRDD<MultiDataSet>` <br>
(c) a directory of serialized `DataSet/MultiDataSet` (minibatch) objects on network storage such as HDFS, S3 or Azure blob storage <br>
(d) a directory of minibatches in some other format <br>

## DataSet & MultiDataSet
`MultiDataSet` is an interface for representing complex data sets, that have (potentially) multiple inputs and outputs 
For example, some complex neural network architectures may have multiple independent inputs, and multiple independent outputs. 
These inputs and outputs need not even be the same opType of data: for example, images in and sequences out, etc


## Sequence Data
There's 2 options here for making a sequence first is make it something like 1 sequence per file and 
read using a sequence reader, second is basically "group by specified column + sort by time" sort of thing - 
[unit tests](https://github.com/deeplearning4j/deeplearning4j/blob/master/datavec/datavec-local/src/test/java/org/datavec/local/transforms/transform/sequence/TestConvertToSequence.java) here.

### How to handle the word sequence? <br>
Write a custom SequenceRecordReader that iterates over files and does the `character -> integer conversion`
each sequence is a `List<List<Writable>>`... outer list is over steps, inner list over values within each step
in this case (assuming 1-character-ahead prediction), the inner list has just 
a List`(new IntWritable(character_index), new IntWritable(next_character_index))`
then use `RecordReaderMultiDataSetIterator` to do the one-hot conversion for the features + labels


## DataVec

DataVec uses an `input/output` format system (similar in some ways to how Hadoop MapReduce uses 
InputFormat to determine InputSplits and RecordReaders, 
DataVec also provides RecordReaders to Serialize Data)

> Define the schema, define the [transformProcess](https://deeplearning4j.org/api/latest/org/datavec/api/transform/TransformProcess.Builder.html#addConstantColumn-java.lang.String-org.datavec.api.transform.ColumnType-org.datavec.api.writable.Writable-)

E.X <br>
[Basic Example](https://github.com/kogecoo/dl4j-examples-scala/blob/master/datavec-examples/src/main/scala/org/datavec/transform/basic/BasicDataVecExample.scala)
[AnalyzeSpark](https://blog.csdn.net/bewithme/article/details/84926689)
[Evaluation (MetaData)](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExampleEvaluationMetaData.java)





## RecordReader
[MapFileRecordReader](https://github.com/deeplearning4j/DataVec/blob/master/datavec-hadoop/src/main/java/org/datavec/hadoop/records/reader/mapfile/MapFileRecordReader.java)
[TestRecordReaderBytesFunction](https://github.com/deeplearning4j/DataVec/blob/master/datavec-spark/src/test/java/org/datavec/spark/functions/TestRecordReaderBytesFunction.java)


## SequenceRecordReader
`SequenceRecordReader`(From gitter) returns `List<List<Writable>>`
this is one single sequence, it's the entire sequence, no splitting, or anything like that
the sequence lengths can differ between examples. i.e., between each call of nextSequence
`SequenceRecordReader` knows literally nothing about minibatches, batch sizes, TBPTT length, masking, etc
it knows only about how to iterate over and return sequences, one at a time all of that other stuff (excluding TBPTT, 
which is net config) happens in `SequenceRecordReaderDataSetIterator / RecordReaderMultiDataSetIterator`
your sequence is just your sequence... you return all of it, in your example there, a List<Writable> is just 
one single time step within a sequence features and labels concatenated




## DataSet Iterator
[DataSet Iterator Split train test](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/DataSetIteratorSplitter.java)
[MultidatasetIterator](https://gist.github.com/bikashg/436bfa48a8803bbc25f87820b25c1833)
[MultidatasetWrapperIterator](https://github.com/eclipse/deeplearning4j/blob/bca1df607f6e58ae73baa8e684130bfa7ad8c2e3/deeplearning4j-nn/src/main/java/org/deeplearning4j/datasets/iterator/MultiDataSetWrapperIterator.java#L18)

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

### Writable to DataSet
```java
JavaRDD<String> stringList = sc.parallelize(stringData);
JavaRDD<List<Writable>> writables = stringList.map(new StringToWritablesFunction(new CSVRecordReader()));
JavaRDD<DataSet> dataSets = writables.map(new DataVecDataSetFunction(3, 5, -1, true, null, null));

```

## Neural Network
### [Storage and Loading](https://github.com/deeplearning4j/DataVec/blob/master/datavec-spark/src/main/java/org/datavec/spark/storage/SparkStorageUtils.java#L170-L191)


### Network Configurations

#### SparkDl4jMultiLayer and SparkComputationGraph
Whether you create ‘MultiLayerNetwork’ or ‘ComputationGraph’, you have to provide a network configuration to it through `‘NeuralNetConfiguration.Builder’`.`‘NeuralNetConfiguration.Builder’`, 
as the name implies, provides a Builder pattern to configure a network. To create a `MultiLayerNetwork`, we build a ‘MultiLayerConfiguraion’ and for ‘ComputationGraph’, 
it’s ‘ComputationGraphConfiguration’.
The pattern goes like this: <br>
> [High Level Configuration] -> [Configure Layers] -> [Pretraining and Backprop Configuration] -> [Build Configuration]

### LSTM
[A Beginner's Guide to LSTMs and Recurrent Neural Networks](https://skymind.ai/wiki/lstm#code)
[A Guide For Time Series Prediction Using Recurrent Neural Networks (LSTMs)](https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f)
[time-series-example](https://deeplearning4j.org/tutorials/13-clinical-lstm-time-series-example-using-skil)
[how-to-predict-when-next-event-occurs-based-on-previous-events](https://stackoverflow.com/questions/7615294/how-to-predict-when-next-event-occurs-based-on-previous-events)

## Encode & Decode
[example](https://www.javatips.net/api/dl4j-examples-master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/encdec/EncoderDecoderLSTM.java)

## Seq2Seq
[google seq2seq](https://github.com/google/seq2seq)
[pnp seq2seq](https://github.com/allenai/pnp/blob/master/src/main/scala/org/allenai/pnp/examples/Seq2Seq.scala)
[seq2seq_conversation](https://isaacchanghau.github.io/post/seq2seq_conversation/)


### Logging
```java
private static Logger log = LoggerFactory.getLogger(CSVExample.class);
log.info("Build model....");
//evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
```

### Training model

#### batch size & num Epochs
- batch size 每一步抓取的样例数量 <br>
- num Epochs 一个epoch指将给定数据集全部处理一遍的周期 - 遍历数据集的次数




#### [Visualizing Network Training with the Deeplearning4j Training UI](https://deeplearning4j.org/docs/latest/deeplearning4j-nn-visualization)
![Training Guide](https://deeplearning4j.org/images/guide/DL4J_UI_01.png)

#### Training Workflow
The training workflow usually proceeds as follows:
Prepare training code with a few components: 
a. Neural network configuration <br>
b. Data pipeline <br>
c. SparkDl4jMultiLayer/SparkComputationGraph plus Trainingmaster <br>
Create uber-JAR file (see Spark how-to guide for details)
Determine the arguments (memory, number of nodes, etc) for Spark submit
Submit the uber-JAR to Spark submit with the required arguments


#### Train on the cluster
Do conversion to RDD<DataSet> and saving that to cluster (HDFS or whatever)
then you can call fit method (there's fit(String path) overload, point it to your directory)




#### Training model on GPU
Run the program on the gpu
To run training on GPUs make sure that you are specifying the correct backend in your pom file (nd4j-cuda-x.x for GPUs vs nd4j-native backend for CPUs) 
and have set up the machines with the appropriate CUDA libraries. Refer to the 
[Deeplearning4j on Spark: How To Guides](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-howto) for more details.
[Distributed Deep Learning with DL4J and Spark](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-intro)

For a single machine with multiple GPUs or multiple physical processors, users should consider using DL4J’s Parallel-Wrapper 
implementation as shown in this example. ParallelWrapper allows for easy data parallel training of networks on a single machine 
with multiple cores. Spark has higher overheads compared to ParallelWrapper for single machine training. <br>

For a network to be large enough: here’s a rough guide. If the network takes 100ms or longer to perform one iteration
(100ms per fit operation on each minibatch), distributed training should work well with good scalability.
At 10ms per iteration, we might expect sub-linear scaling of performance vs. number of nodes. At around 1ms or 
below per iteration, the communication overhead may be too much: training on a cluster may be no faster (or perhaps even slower) 
than on a single machine. For the benefits of parallelism to outweigh the communication overhead, users should consider 
the ratio of network transfer time to computation time and ensure that the computation time is large enough to mask the
additional overhead of distributed training.


#### Examples
[CSV Example](https://github.com/deeplearning4j/dl4j-examples/blob/aeb002d151ffcf785a46aba45a722d189de2e0df/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java), 
[Quickstart with Deeplearning4J](https://www.dubs.tech/guides/quickstart-with-dl4j/#using-the-model), 
[dl4j-tutorials](https://github.com/sjsdfg/dl4j-tutorials)
[Deeplearning4J Examples for Scala](https://github.com/kogecoo/dl4j-examples-scala)
[dl4j-examples-tour](https://deeplearning4j.org/cn/examples-tour)
[word2vecsentiment](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment)
[stock prediction](https://github.com/IsaacChanghau/StockPrediction/tree/master/src/main/java/com/isaac/stock)

#### Courses
[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
