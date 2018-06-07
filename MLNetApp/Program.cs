using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace MLNetApp
{
    //http://www.csharpkit.com/2018-05-09_52160.html
    //机器学习来预测鸢尾花的类型，比如有setosa、versicolor、virginica三种，基于特征有四种：花瓣长度、花瓣宽度, 萼片长度、萼片宽度。
    class Program
    {
        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load your data

            var pipeline = new LearningPipeline();


            // If working in Visual Studio, make sure the 'Copy to Output Directory' 

            // property of iris-data.txt is set to 'Copy always'

            string dataPath = "iris-data.txt";

            //pipeline.Add(new <IrisData>(dataPath, separator: ","));


            // STEP 3: Transform your data

            // Assign numeric values to text in the "Label" column, because only

            // numbers can be processed during model training

            pipeline.Add(new Dictionarizer("Label"));


            // Puts all features into a vector

            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));


            // STEP 4: Add learner

            // Add a learning algorithm to the pipeline. 

            // This is a classification scenario (What type of iris is this?)

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());


            // Convert the Label back into original text (after converting to number in step 3)

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });


            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<IrisData, IrisPrediction>();


            // STEP 6: Use your model to make a prediction

            // You can change these numbers to test different predictions

            var prediction = model.Predict(new IrisData()

            {

                SepalLength = 3.3f,

                SepalWidth = 1.6f,

                PetalLength = 0.2f,

                PetalWidth = 5.1f,

            });


            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }
    }


    

}
