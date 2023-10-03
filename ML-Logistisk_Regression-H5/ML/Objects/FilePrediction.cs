using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Logistisk_Regression_H5.ML.Objects
{
    internal class FilePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsMalicious;
        public float Probability;
        public float Score;
    }
}
