using Microsoft.ML;
using System.Text;
using System.Text.RegularExpressions;

namespace ML_Logistisk_Regression_H5.ML.Base
{
    public class BaseML 
    {
        protected readonly MLContext mlContext;
        private static Regex _stringRex;
        protected BaseML()
        {
            mlContext = new MLContext(2);
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            _stringRex = new Regex(@"[ -~\t]{8,}", RegexOptions.Compiled);
        }
        protected string GetStrings(byte[] data)
        {
            var stringLines = new StringBuilder();

            if (data == null || data.Length == 0)
            {
                return stringLines.ToString();
            }

            using (var ms = new MemoryStream(data, false))
            {
                using (var streamReader = new StreamReader(ms, Encoding.GetEncoding(1252), false, 2048, false))
                {
                    while (!streamReader.EndOfStream)
                    {
                        var line = streamReader.ReadLine();
                        if (string.IsNullOrEmpty(line))
                        {
                            continue;
                        }
                        line = line.Replace("^", "").Replace(")", "").Replace("-", "");
                        stringLines.Append(
                            string.Join(
                                string.Empty, _stringRex.Matches(line).
                                Where(a => !string.IsNullOrEmpty(a.Value) &&
                                !string.IsNullOrWhiteSpace(a.Value)).ToList()));
                    }
                    return string.Join(string.Empty, stringLines);
                }
            }
        }
    }
}