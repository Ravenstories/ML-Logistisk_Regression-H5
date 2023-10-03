using Microsoft.ML;
using ML_Logistisk_Regression_H5.ML.Base;
using ML_Logistisk_Regression_H5.ML.Objects;
using Newtonsoft.Json;
using Constants = ML_Logistisk_Regression_H5.Common.Constants;

namespace ML_Logistisk_Regression_H5.ML
{
    class Predictor : BaseML
    {
        public void Predict(string inputData)
        {
            if (!File.Exists(Constants.modelFile))
            {
                Console.WriteLine($"Failed to find model at {Constants.modelFile}");
                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(Constants.modelFile, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }
            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");
                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<FileInput, FilePrediction>(mlModel);
            var prediction = predictionEngine.Predict(new FileInput
            {
                Strings = GetStrings(File.ReadAllBytes(inputData))
            });


            Console.WriteLine($"Based on the file ({inputData}) the file is classified as {(prediction.IsMalicious ? "malicious" : "benign")}" 
                + $" at a confidence level of {prediction.Probability:P0}");
        }
    }
}