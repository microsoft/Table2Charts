using System;
using System.Collections.Generic;
using System.Linq;
using DataPlatform;
using Newtonsoft.Json;

namespace Common
{
    /// <summary>
    /// The final data structure of processed Charts. This is the output of Chart Matching and will be used as
    ///   input of the downstream tasks and models.
    /// </summary>
    [Serializable]
    public class ProcessedChart : IEquatable<ProcessedChart>
    {
        // Table UID.
        public string TUid { get; set; }

        // Chart UID
        public string CUid { get; set; }

        // Chart type
        public string CType { get; set; }

        // Sheet name of the source table
        public string SheetName { get; set; }

        // The grouping method of multiple YFields
        public string Grouping { get; set; }

        // Orientation of the source table
        public DataPlatform.TableOrientation Orientation { get; set; }

        public List<Field> XFields { get; set; }
        public List<Field> YFields { get; set; }

        // Whether the data points are connected by line for each YField.
        // This property is for LineChart and ScatterChart.
        public List<bool> ValueDrawsLine { get; set; }

        // The start and end indexes of logical "row"s of header region and value region.
        public int HeaderFirstIndex { get; set; }
        public int HeaderLastIndex { get; set; }
        public int FirstIndex { get; set; }
        public int LastIndex { get; set; }
        [JsonIgnore] public List<int> MulIndexes { get; set; }

        // The corresponding Chart object. This property is used in ChartMatching and will not be dumped into JSON files.
        [JsonIgnore] public Chart Chart { get; set; }

        public ProcessedChart() { }
        public ProcessedChart(ProcessedChart other)
        {
            TUid = other.TUid;
            CUid = other.CUid;
            SheetName = other.SheetName;
            CType = other.CType;
            Grouping = other.Grouping;
            Orientation = other.Orientation;
            XFields = new List<Field>(other.XFields);
            YFields = new List<Field>(other.YFields);
            ValueDrawsLine = new List<bool>(other.ValueDrawsLine);
            HeaderFirstIndex = other.HeaderFirstIndex;
            HeaderLastIndex = other.HeaderLastIndex;
            FirstIndex = other.FirstIndex;
            LastIndex = other.LastIndex;
            MulIndexes = other.MulIndexes;
        }
        
        /// <summary>
        /// Partially construct using a Chart and some alignment info.
        /// </summary>
        public ProcessedChart(Chart chart, string sheetName, bool isColumnDirection,
            int firstIndex, int lastIndex, List<int> mulIndexes, int headerFirstIndex, int headerLastIndex)
        {
            SheetName = sheetName;
            CType = chart.ChartType;
            Grouping = chart.Grouping;
            Orientation = isColumnDirection ? TableOrientation.ColumnMajor : TableOrientation.RowMajor;
            FirstIndex = firstIndex;
            LastIndex = lastIndex;
            MulIndexes = mulIndexes;
            HeaderFirstIndex = headerFirstIndex;
            HeaderLastIndex = headerLastIndex;

            Chart = chart;
            XFields = new List<Field>();
            YFields = new List<Field>();
            ValueDrawsLine = new List<bool>();
        }

        /// <summary>
        /// Choose the minimum and maximum number of field indices as column boundary.
        /// </summary>
        public void GetPseudoColumns(out int firstPseudoColumn, out int lastPseudoColumn)
        {
            var fieldIndices = XFields.Union(YFields).Select(f => f.Index);
            firstPseudoColumn = fieldIndices.Min();
            lastPseudoColumn = fieldIndices.Max();
        }
        public void GetPseudoRows(out int firstPseudoRow, out int lastPseudoRow)
        {
            firstPseudoRow = FirstIndex;
            lastPseudoRow = LastIndex;
        }

        public bool Equals(ProcessedChart other)
        {
            return other != null &&
                Field.SetEquals(XFields, other.XFields) &&
                Field.SetEquals(YFields, other.YFields) &&
                string.Equals(Grouping, other.Grouping) &&
                CType.Equals(other.CType) &&
                Orientation.Equals(other.Orientation) &&
                FirstIndex == other.FirstIndex &&
                LastIndex == other.LastIndex &&
                SheetName == other.SheetName &&
                //Sometimes X/YFields of a new ProcessedChart are empty(They haven't been filled yet), so we need MulIndexes.
                MulIndexes.Count()==other.MulIndexes.Count() && MulIndexes.Except(other.MulIndexes).Count()==0;
        }

        public override int GetHashCode()
        {
            var hashCode = -1959644465;
            hashCode = hashCode * -1521134295 + XFields?.Count.GetHashCode() ?? 0;
            hashCode = hashCode * -1521134295 + XFields?.Select(field => field.GetHashCode()).Aggregate(0, (acc, val) => acc ^ val) ?? 0;
            hashCode = hashCode * -1521134295 + YFields?.Count.GetHashCode() ?? 0;
            hashCode = hashCode * -1521134295 + YFields?.Select(field => field.GetHashCode()).Aggregate(0, (acc, val) => acc ^ val) ?? 0;
            hashCode = hashCode * -1521134295 + Grouping?.GetHashCode() ?? 0;
            hashCode = hashCode * -1521134295 + CType.GetHashCode();
            hashCode = hashCode * -1521134295 + Orientation.GetHashCode();
            hashCode = hashCode * -1521134295 + FirstIndex.GetHashCode();
            hashCode = hashCode * -1521134295 + LastIndex.GetHashCode();
            return hashCode;
        }
    }
}
