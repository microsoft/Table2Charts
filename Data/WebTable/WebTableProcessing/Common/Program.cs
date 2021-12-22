using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    class Program
    {
        public static ChartTable TableParsingResultToChartTable(TableParsingResult generalTable)
        { 
            var chartTableRes = new ChartTable();
            chartTableRes.FieldTypes = new List<FieldDataType>();
            chartTableRes.NColumns = generalTable.ColumnCount;
            chartTableRes.NRows = generalTable.RowCount;
            chartTableRes.Fields = new List<SourceField>();
            chartTableRes.Records = new List<List<Cell>>();
            chartTableRes.NumFmtIds = new List<List<uint>>();

            for (int fieldIdx = 0; fieldIdx < generalTable.FieldParsingResult.Count; fieldIdx++)
            {
                var fieldParseRes = generalTable.FieldParsingResult[fieldIdx];

                if (FieldParsingResult.HasYear(fieldParseRes.DataFormatFlags))
                {
                    chartTableRes.FieldTypes.Add(FieldDataType.Year);
                }
                else if (FieldParsingResult.IsDate(fieldParseRes.DataFormatFlags))
                {
                    chartTableRes.FieldTypes.Add(FieldDataType.DateTime);
                }
                else if (FieldParsingResult.IsNumeric(fieldParseRes.DataFormatFlags))
                {
                    chartTableRes.FieldTypes.Add(FieldDataType.Decimal);
                }
                else if (FieldParsingResult.IsUnknown(fieldParseRes.DataFormatFlags))
                {
                    chartTableRes.FieldTypes.Add(FieldDataType.Unknown);
                }
                else
                { 
                    chartTableRes.FieldTypes.Add(FieldDataType.String);
                }

                var sourceField = new SourceField();
                sourceField.Index = fieldIdx;
                sourceField.Name = fieldParseRes.Header;
                sourceField.NumberFmtId = 0;
                sourceField.DataFormatFlags = fieldParseRes.DataFormatFlags;
                sourceField.Items = new SharedItems();

                chartTableRes.Fields.Add(sourceField);
            }

            for (int rowIdx = 0; rowIdx < generalTable.RowCount; rowIdx++)
            { 
                var record = new List<Cell>();
                var recordNumFormat = new List<uint>();

                for (int fieldIdx = 0; fieldIdx < generalTable.FieldParsingResult.Count; fieldIdx++)
                {
                    var fieldParseRes = generalTable.FieldParsingResult[fieldIdx];
                    var cellRes = fieldParseRes.CellResults[rowIdx];

                    record.Add(new Cell() { Type = FieldParsingResult.IsNumeric(cellRes.DataFormatFlags) ? "n" : "s", Value = cellRes.RawValue.ToString() });

                    // Set NumFormat
                    recordNumFormat.Add(0);
                }

                chartTableRes.Records.Add(record);
                chartTableRes.NumFmtIds.Add(recordNumFormat);
            }

            chartTableRes.SharedNumFmts = new Dictionary<uint, string>();

            return chartTableRes;
        }


        static void Main(string[] args)
        {
            // Table from https://www.worldometers.info/coronavirus/coronavirus-death-toll/
            var tableHeader = new List<String>() 
            { 
                "Date", "Total Deaths", "Change in Total", "Change in Total(%)"
            };

            var tableBody = new List<List<String>>()
            {
                new List<String>() {"Aug. 9,. 2021", "4,315,628", "8,047", "0%" },
                new List<String>() {"Aug. 8,. 2021", "4,307,581", "8,321", "0%" },
                new List<String>() {"Aug. 7,. 2021", "4,299,260", "9,459", "0%" },
                new List<String>() {"Aug. 6,. 2021", "4,289,801", "10,216", "0%" },
                new List<String>() {"Aug. 5,. 2021", "4,279,585", "10,391", "0%" },
                new List<String>() {"Aug. 4,. 2021", "4,269,194", "10,086", "0%" },
                new List<String>() {"Aug. 3,. 2021", "4,259,108", "10,120", "0%" },
                new List<String>() {"Aug. 2,. 2021", "4,248,988", "8,165", "0%" },
                new List<String>() {"Aug. 1,. 2021", "4,240,823", "7,901", "0%" },
                new List<String>() {"Jul. 31,. 2021", "4,232,922", "9,139", "0%" },
                new List<String>() {"Jul. 30,. 2021", "4,223,783", "9,406", "0%" },
                new List<String>() {"Jul. 29,. 2021", "4,214,377", "10,477", "0%" },
                new List<String>() {"Jul. 28,. 2021", "4,203,900", "10,203", "0%" },
                new List<String>() {"Jul. 27,. 2021", "4,193,697", "9,492", "0%" },
                new List<String>() {"Jul. 26,. 2021", "4,184,205", "7,523", "0%" },
                new List<String>() {"Jul. 25,. 2021", "4,176,682", "7,086", "0%" },
                new List<String>() {"Jul. 24,. 2021", "4,169,596", "8,454", "0%" },
                new List<String>() {"Jul. 23,. 2021", "4,161,142", "8,751", "0%" },
                new List<String>() {"Jul. 22,. 2021", "4,152,391", "9,014", "0%" },
                new List<String>() {"Jul. 21,. 2021", "4,143,377", "8,701", "0%" },
                new List<String>() {"Jul. 20,. 2021", "4,134,676", "8,372", "0%" },
                new List<String>() {"Jul. 19,. 2021", "4,126,304", "6,909", "0%" }
            };

            var generalTable = new TableParsingResult(tableHeader, tableBody);
            var chartTable = TableParsingResultToChartTable(generalTable);
        }
    }
}
