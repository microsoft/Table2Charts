// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Spreadsheet;

namespace Common
{
    /// <summary>
    /// Data structure of source tables shared by PivotTable and Chart.
    ///   Its value properties are copied from CacheSource, but some PivotTable-only
    ///   properties are removed. This is the base class of tables.
    /// </summary>
    [Serializable]
    public class Table: IEquatable<Table>
    {
        public string TUid { get; set; }
        public string SheetName { get; set; }
        [NotNull] public Range TableRange { get; set; } // For ChartTable it means TableFlow range.
        public int NColumns { get; set; }
        public int NRows { get; set; }
        [NotNull] public List<SourceField> Fields { get; set; }
        [NotNull] public List<List<Cell>> Records { get; set; }
        [NotNull] public Dictionary<uint, string> SharedNumFmts { get; set; }

        // ------ For PivotTable ------
        [NotNull] public List<string> PUids { get; set; }

        // ------ For Chart ------
        [NotNull] public List<string> CUids { get; set; }
        [NotNull] public List<string> CTypes { get; set; }

        public Table()
        {
            TUid = null;
            PUids = new List<string>();
            CUids = new List<string>();
            CTypes = new List<string>();
            NColumns = NRows = -1;
            Fields = new List<SourceField>();
            Records = new List<List<Cell>>();
        }

        public bool Equals(Table other)
        {
            return other != null &&
                    NColumns == other.NColumns && NRows == other.NRows &&
                    Enumerable.Zip(Fields, other.Fields, (xf, yf) => xf.Equals(yf)).All(e => e) &&
                    ElementWiseEquals(Records, Fields, other.Records, other.Fields, NRows, NColumns);
        }

        public override int GetHashCode()
        {
            var hashCode = 1684631887;
            hashCode = hashCode * -1521134295 + NColumns.GetHashCode();
            hashCode = hashCode * -1521134295 + Fields.Select(f => f.GetHashCode()).Aggregate(0, (acc, val) => acc ^ val);
            hashCode = hashCode * -1521134295 + NRows.GetHashCode();
            return hashCode;
        }

        public bool ElementWiseEquals(List<List<Cell>> x, List<SourceField> xFields, List<List<Cell>> y, List<SourceField> yFields, int m, int n)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (!x[i][j].ActualRecord(xFields[j].Items).Equals(y[i][j].ActualRecord(yFields[j].Items)))
                        return false;
            return true;
        }

        protected string ToString(Type type)
        {
            StringBuilder sb = new StringBuilder($"{type.Name} object:\n");
            sb.Append($"TUid : {TUid}\n");
            sb.Append($"PUids : {string.Join(", ", PUids)}\n");
            sb.Append($"CUids : {string.Join(", ", CUids)}\n");
            sb.Append($"Columns : {NColumns}\n");
            sb.Append($"Rows : {NRows}\n");
            sb.Append($"SourceFields : ");
            foreach (var sf in Fields)
                sb.Append($"{sf.Index}_{sf.Name}_{sf.NumberFmtId}, ");
            sb.Append("\n");
            return sb.ToString();
        }

        public override string ToString()
        {
            return ToString(this.GetType());
        }
    }

    /// <summary>
    /// See https://docs.microsoft.com/en-us/dotnet/api/documentformat.openxml.spreadsheet.cachefield?view=openxml-2.8.1
    /// Some properties added for data processing.
    /// </summary>
    [Serializable]
    public class SourceField : IEquatable<SourceField>
    {
        [NotNull] public int Index { get; set; }
        [NotNull] public string Name { get; set; }
        public uint NumberFmtId { get; set; }
        [NotNull] public SharedItems Items { get; set; }

        // ------ For ChartTable ------
        public bool InHeaderRegion { get; set; }

        // ----- For PlotlyTable -----
        public string Uid { get; set; }

        public static SourceField GetInstance(CacheField cacheField, int index)
        {
            IList<PivotTableGroup> pivotGroupings = new List<PivotTableGroup>();
            PivotTableGroup pivotTableGroup = PivotTableGroup.GetInstance(cacheField, index);
            if (pivotTableGroup != null)
                pivotGroupings.Add(pivotTableGroup);

            return new SourceField()
            {
                Index = index,
                Name = cacheField.Name,
                NumberFmtId = cacheField.NumberFormatId ?? 0,  
                                // ECMA-376 Appendix B.2 Line 1284 shows that default value is "false"
                Items = SharedItems.GetInstance(cacheField),
                PivotGrouping = pivotGroupings
            };
        }

        public bool Equals(SourceField other)
        {
            return other != null
                && Index == other.Index
                && string.Equals(Name, other.Name)
                && NumberFmtId == other.NumberFmtId
                && AggregationFunc == other.AggregationFunc
                && string.Equals(AggregationFormula, other.AggregationFormula)
                && SharedItems.Equals(Items, other.Items);
        }

        public override int GetHashCode()
        {
            var hashCode = 1664113244;
            hashCode = hashCode * -1521134295 + Index.GetHashCode();
            hashCode = hashCode * -1521134295 + Name.GetHashCode();
            hashCode = hashCode * -1521134295 + NumberFmtId.GetHashCode();
            hashCode = hashCode * -1521134295 + AggregationFunc.GetHashCode();
            hashCode = hashCode * -1521134295 + AggregationFormula?.GetHashCode() ?? 0;
            hashCode = hashCode * -1521134295 + EqualityComparer<SharedItems>.Default.GetHashCode(Items);
            return hashCode;
        }

        public long GetHashCode64()
        {
            long hashCode = 1664113244L;
            hashCode = hashCode * -1521134295L + Index.GetHashCode();
            hashCode = hashCode * -1521134295L + Name.GetHashCode();
            return hashCode;
        }
    }
}
