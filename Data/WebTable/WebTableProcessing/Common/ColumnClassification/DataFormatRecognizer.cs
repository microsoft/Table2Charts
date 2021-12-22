using Microsoft.Recognizers.Text;
using Microsoft.Recognizers.Text.DateTime;
using Microsoft.Recognizers.Text.Number;
using Microsoft.Recognizers.Text.NumberWithUnit;
using Microsoft.Recognizers.Text.Sequence;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ColumnClassification.Contract;
namespace ColumnClassification.Utils
{
    class DataFormatRecognizer
    {
        public static ColumnModel GetColumnClassifiction(string header, List<string> fields, string language = "en")
        {
            var columnModel = new ColumnModel(header);

            foreach (var field in fields)
            {
                var cellModel = new CellModel(field);
                // Detect Data Format for One Cell Using Text Recognizer
                cellModel.TypeDetection = new CellType(cellModel.RawValue, language);
                columnModel.Fields.Add(cellModel);            
            }

            columnModel.ClassifyType();

            return columnModel;
        }
    }
}
