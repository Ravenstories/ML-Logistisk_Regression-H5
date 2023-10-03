using Microsoft.ML.Data;

namespace ML_Logistisk_Regression_H5.ML.Objects
{
    internal class EmploymentHistoryPrediction
    {
        [ColumnName("DurationInMonths")]
        public float DurationInMonths { get; set; }
    }
}
