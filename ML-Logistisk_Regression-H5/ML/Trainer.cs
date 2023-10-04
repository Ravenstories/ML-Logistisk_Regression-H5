using Microsoft.ML;
using ML_Logistisk_Regression_H5.ML.Base;
using ML_Logistisk_Regression_H5.ML.Objects;
using ML_Logistisk_Regression_H5.Common;

namespace ML_Logistisk_Regression_H5.ML
{
    class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file {trainingFileName}");
                return;
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<FileInput>(trainingFileName, ',');
            var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(FileInput.Label))
                .Append(mlContext.Transforms.Text.FeaturizeText("NGrams",nameof(FileInput.Strings)))
                .Append(mlContext.Transforms.Concatenate("Features", "NGrams"));


            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Constants.modelFile);
            
            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = mlContext.BinaryClassification.Evaluate(testSetTransform);

            Console.WriteLine($"Loss Function: {modelMetrics.LogLoss:0.##}{Environment.NewLine}");
            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}");
            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}");
            Console.WriteLine($"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}");
            Console.WriteLine($"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}");
            Console.WriteLine($"Positive Recall: {modelMetrics.PositiveRecall:0.##}{Environment.NewLine}");
        }
    }
}

