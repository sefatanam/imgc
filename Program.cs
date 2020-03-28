using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
namespace imgc
{
    class Program
    {
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb"); static void Main(string[] args)
        {

            MLContext mlContext = new MLContext();
            ITransformer model = GenerateModel(mlContext);
            ClassifySingleImage(mlContext, model);


            //var context = new MLContext();

            //var data = context.Data.LoadFromTextFile<ImageData>("./labels.csv", separatorChar: ',');

            //var preview = data.Preview();

            //var pipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
            //    .Append(context.Transforms.LoadImages("input", "images", nameof(ImageData.ImagePath)))
            //    .Append(context.Transforms.ResizeImages("input", InceptionSettings.IMageWidth, InceptionSettings.ImageHeight, "input"))
            //    .Append(context.Transforms.ExtractPixels("input", interleavePixelColors: InceptionSettings.ChannelsList,
            //        offsetImage: InceptionSettings.Mean))
            //    .Append(context.Model.LoadTensorFlowModel("./model/tensorflow_inception_graph.pb")
            //        .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, addBatchDimensionInput: true)
            //    .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey", "softmax2_pre_activation"))
            //    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
            //    .AppendCacheCheckpoint(context));

            //var model = pipeline.Fit(data);

            //var imageData = File.ReadAllLines("./labels.csv")
            //    .Select(l => l.Split(','))
            //    .Select(l => new ImageData { ImagePath = Path.Combine(Environment.CurrentDirectory, "images", l[0]) });

            //var imageDataView = context.Data.LoadFromEnumerable(imageData);

            //var predictions = model.Transform(imageDataView);

            //var imagePredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false, ignoreMissingColumns: true);

            //// Evaluate
            //Console.WriteLine("\n------------Evaluate-----------------");

            //var evalPredictions = model.Transform(data);

            //var metrics = context.MulticlassClassification.Evaluate(evalPredictions, labelColumnName: "LabelKey",
            //    predictedLabelColumnName: "PredictedLabel");

            //// Log loss should be close to 0 for accurate predictions
            //Console.WriteLine($"Log Loss - {metrics.LogLoss}");
            //Console.WriteLine($"Per class Log Loss - {String.Join(',', metrics.PerClassLogLoss.Select(l => l.ToString()))}");

            //// Predict batch
            //Console.WriteLine("\n------------Batch predictions-----------------");

            //foreach (var prediction in imagePredictions)
            //{
            //    Console.WriteLine($"Image - {prediction.ImagePath} is predicted as {prediction.PredictedLabelValue} " +
            //        $"with a score of {prediction.Score.Max()}");
            //}

            //// Predict single
            //var predictionFunc = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            //var singlePrediction = predictionFunc.Predict(new ImageData
            //{
            //    ImagePath = Path.Combine(Environment.CurrentDirectory, "images", "cup2.jpg")
            //});

            //Console.WriteLine("\n------------Single prediction-----------------");
            //Console.WriteLine($"Image {Path.GetFileName(singlePrediction.ImagePath)} was predicted as a {singlePrediction.PredictedLabelValue} " +
            //    $"with a score of {singlePrediction.Score.Max()}");

            Console.ReadLine();
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }

        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
                    return File.ReadAllLines(file)
         .Select(line => line.Split(','))
         .Select(line => new ImageData()
         {
             ImagePath = Path.Combine(folder, line[0])});
        }
        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {

            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };
            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");

        }

        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageHeight, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsList, offsetImage: InceptionSettings.Mean)).Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true)).Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
.AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
            ITransformer model = pipeline.Fit(trainingData);
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            // Create an IEnumerable for the predictions for displaying results
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);
            MulticlassClassificationMetrics metrics =
    mlContext.MulticlassClassification.Evaluate(predictions,
      labelColumnName: "LabelKey",
      predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            return model;
        }


    }
}
