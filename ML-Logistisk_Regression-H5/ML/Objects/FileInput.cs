using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Logistisk_Regression_H5.ML.Objects
{
    public class FileInput
    {
        [LoadColumn(0)]
        public bool Label;
        [LoadColumn(1)]
        public string Strings;
    }

}
