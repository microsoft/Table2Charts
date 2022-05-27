// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DocumentFormat.OpenXml.Spreadsheet;
using Newtonsoft.Json;

namespace Common
{
    /// <summary>
    /// A Parser for parsing cell string.
    /// </summary>
    public class GeneralTableDataFormatParser
    {
        
        public static FieldParsingResult GetFieldDataFormat(string header, List<string> columnList, string language = "en") 
        {
            var columnInfo = ColumnClassification.Utils.DataFormatRecognizer.GetColumnClassifiction(header, columnList, language);

            DataFormatFlags dataFormatFlags = (DataFormatFlags)Enum.Parse(typeof(DataFormatFlags), columnInfo.FormatFlags.ToString());
            ColumnTradeOffFlags colTradeOffFlags = (ColumnTradeOffFlags)Enum.Parse(typeof(ColumnTradeOffFlags), columnInfo.TradeOffFlags.ToString());

            bool hasUnit = true;
            if (string.IsNullOrEmpty(columnInfo.Unit))
            {
                hasUnit = false;
            }
            bool hasPresuffix = false;
            bool hasPrefix = false;
            bool hasSuffix = false;
            if (!string.IsNullOrEmpty(columnInfo.prefix))
            {
                hasPrefix = true;
            }
            if (!string.IsNullOrEmpty(columnInfo.suffix))
            {
                hasSuffix = true;
            }
            if (hasPrefix || hasSuffix) {
                hasPresuffix = true;
            }

            var cells = new List<CellParsingResult>();
            foreach (var rawCell in columnInfo.Fields)
            {
                bool cellHasUnit = true;
                if (string.IsNullOrEmpty(rawCell.TypeDetection.Unit))
                {
                    cellHasUnit = false;
                }
                bool cellHasPresuffix = true;
                if (string.IsNullOrEmpty(rawCell.TypeDetection.suffix))
                {
                    cellHasPresuffix = false;
                }
                cells.Add(new CellParsingResult(dataFormatFlags, rawCell.Value, cellHasUnit, rawCell.TypeDetection.Unit, cellHasPresuffix, rawCell.TypeDetection.suffix, rawCell.RawValue));
            }

            return new FieldParsingResult(dataFormatFlags, hasUnit, columnInfo.Unit, hasPresuffix, columnInfo.suffix, hasPrefix,columnInfo.prefix,hasSuffix,columnInfo.suffix, colTradeOffFlags, cells, header, columnInfo);
        }

    }

    /// <summary>
    /// Result for parsing a data field.
    /// </summary>
    public class FieldParsingResult
    {
        /// <summary>
        /// The overall data format information of this field.
        /// </summary>
        public DataFormatFlags DataFormatFlags { get; set; }

        /// <summary>
        /// Wether we can extract unit from this field.
        /// </summary>
        public bool HasUnit { get; set; }

        /// <summary>
        /// The unit extracted from this field.
        /// </summary>
        public string Unit { get; set; }

        /// <summary>
        /// Wether we can extract prefix or suffix from this field.
        /// </summary>
        public bool HasPresuffix { get; set; }

        /// <summary>
        /// The prefix or suffix extracted from this field.
        /// </summary>
        public string Presuffix { get; set; }
        public bool HasPrefix { get; set; }
        public string Prefix { get; set; }
        public bool HasSuffix { get; set; }

        /// <summary>
        /// The prefix or suffix extracted from this field.
        /// </summary>
        public string Suffix { get; set; }

        /// <summary>
        /// The tradeoff flag of this column for debugging purpose
        /// </summary>
        public ColumnTradeOffFlags CellTradeOffFlags { get; set; }

        /// <summary>
        /// Data format information for each cells.
        /// </summary>
        public List<CellParsingResult> CellResults { get; set; }

        /// <summary>
        /// The header of this field.
        /// </summary>
        public string Header { get; set; }


        /// <summary>
        /// InternalDebugInfo
        /// </summary>
        [JsonIgnore]
        public ColumnClassification.Contract.ColumnModel ColumnProcessingInfo { get; set; }


        public FieldParsingResult(DataFormatFlags dataFormatFlags, bool hasUnit, string unit, bool hasPresuffix, string presuffix, bool hasPrefix, string prefix, bool hasSuffix, string suffix, ColumnTradeOffFlags colTradeOffFlags, List<CellParsingResult> cellResults, string header, ColumnClassification.Contract.ColumnModel columnProcessingInfo = null)
        {
            this.DataFormatFlags = dataFormatFlags;
            this.HasUnit = hasUnit;
            this.Unit = unit;
            this.HasPresuffix = hasPresuffix;
            this.Presuffix = presuffix;
            this.CellTradeOffFlags = colTradeOffFlags;
            this.CellResults = cellResults;
            this.Header = header;
            this.ColumnProcessingInfo = columnProcessingInfo;
            this.HasPrefix = HasPrefix;
            this.Prefix = prefix;
            this.HasSuffix = HasSuffix;
            this.Suffix = Suffix;
        }

        public FieldParsingResult() { }

        public static bool IsNumeric(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 0x8) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool IsSequence(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 256) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool IsOrdinal(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 512) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }


        public static bool IsDate(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 1) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool HasYear(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 128) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool HasMonth(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 64) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool HasDay(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags & 32) > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        public static bool IsUnknown(DataFormatFlags dataFormatFlags)
        {
            if (((int)dataFormatFlags) == 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    /// <summary>
    /// Result of each cell obtained by parser.
    /// </summary>
    public class CellParsingResult
    {
        /// <summary>
        /// Cell string.
        /// </summary>
        public string Text { get; }

        /// <summary>
        /// Data format information of this cell.
        /// </summary>
        public DataFormatFlags DataFormatFlags { get; }

        /// <summary>
        /// An object for value of this cell. Its type depends on the parsing results for this cell string.
        /// </summary>
        public object RawValue { get; }

        /// <summary>
        /// Whether has unit.
        /// </summary>
        public bool HasUnit { get; }

        /// <summary>
        /// Unit string.
        /// </summary>
        public string Unit { get; }

        /// <summary>
        /// Wether has prefix or suffix.
        /// </summary>
        public bool HasPresuffix { get; }

        /// <summary>
        /// The prefix or suffix.
        /// </summary>
        public string Presuffix { get; }

        public CellParsingResult(DataFormatFlags dataFormatFlags, object rawValue, bool hasUnit, string unit, bool hasPresuffix, string presuffix, string originalValue)
        {
            this.DataFormatFlags = dataFormatFlags;
            this.RawValue = rawValue;
            this.HasUnit = hasUnit;
            this.Unit = unit;
            this.HasPresuffix = hasPresuffix;
            this.Presuffix = presuffix;
            this.Text = originalValue;
        }
    }

    public enum ColumnTradeOffFlags : long
    {
        None = 0,
        // The column contains cells tagged as Numeric & NumericWithUnit 
        MixedNumericAndNumericWithUnit = 1,

        // The column contains cells which would parse out multiple numbers
        MultipleNumeric = 2,

        // The column contains cells with different units
        MultipleUnit = 4,

        // 8 multiple unit numeric
        MultipleUnitNumeric = 8,

        // Has prefix
        HasPrefix = 16,

        // Has Suffix
        HasSuffix = 32
    }


    public class TableParsingResult
    {
        public int RowCount { get; set; }

        public int ColumnCount { get; set; }

        public List<FieldParsingResult> FieldParsingResult { get; set; }

        public String DocumentUrl { get; set; }

        /// <summary>
        /// Contents of all cells.
        /// Note: Elements of the first List is all columns, and the second List is all cells in each column.
        /// </summary>
        /// 
        [JsonIgnore]
        public List<List<IGeneralCell>> Cells { get; set; }

        public IGeneralCell Cell(int rowNumber, int columnNumber)
        {
            return Cells[columnNumber][rowNumber];
        }

        public TableParsingResult(string tableHeaderChunk, string tableBodyChunk,string language = "en")
        { 
            var headerList = tableHeaderChunk.Split(new string[] { "#TAB#" }, StringSplitOptions.None).ToList();
            var rowList = Helpers.TableBodyToRowList(tableHeaderChunk, tableBodyChunk);

            this.SetTableParsingResult(headerList, rowList, language);
        }

        public TableParsingResult()
        {
            this.RowCount = 0;
            this.ColumnCount = 0;
            this.FieldParsingResult = new List<FieldParsingResult>();
            this.DocumentUrl = string.Empty;
        }

        public TableParsingResult(List<string> headerList, List<List<string>> rowList)
        {
            this.SetTableParsingResult(headerList, rowList);
        }


        public void SetTableParsingResult(List<string> headerList, List<List<string>> rowList,string language = "en")
        { 
            var columnList = Helpers.RowsToColumns(headerList.Count, rowList);

            this.RowCount = rowList.Count;
            this.ColumnCount = columnList.Count;

            this.Cells = new List<List<IGeneralCell>>();
            this.FieldParsingResult = new List<FieldParsingResult>();

            for (var colIdx = 0; colIdx < headerList.Count; colIdx++)
            {
                string header = headerList[colIdx];
                List<String> colList = columnList[colIdx];

                // var fieldParsingRes = GeneralTableDataFormatParser.Parse(header, colList);
                var fieldParsingRes = GeneralTableDataFormatParser.GetFieldDataFormat(header, colList, language);
                FieldParsingResult.Add(fieldParsingRes);


                var fieldCells = fieldParsingRes.CellResults
                    .Zip(colList, (result, text) => new GeneralCell(result, text))
                    .ToList<IGeneralCell>();
                Cells.Add(fieldCells);
            }
        }
    }
}
