﻿namespace ML_Logistisk_Regression_H5.Common
{
    public class Constants
    {
        public static string sampleData = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..\\Data\\sampledata.csv");
        public static string sampleDataFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..\\Data\\testdata");
        public static string modelFile  = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..\\Data\\sampledata.trained.csv");
        public static string inputData  = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..\\Data\\inputData.json");
    }
}
