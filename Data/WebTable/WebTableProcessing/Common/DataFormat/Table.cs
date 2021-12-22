using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Spreadsheet;


namespace Common
{
    public enum TableOrientation
    {
        RowMajor,
        ColumnMajor,
    }
    /// <summary>
    /// Data structure of source tables shared by PivotTable and Chart.
    ///   Its value properties are copied from CacheSource, but some PivotTable-only
    ///   properties are removed. This is the base class of tables.
    /// </summary>
    [Serializable]
    public class Table : IEquatable<Table>
    {
        public string TUid { get; set; }
        public string SheetName { get; set; }
        public int NColumns { get; set; }
        public int NRows { get; set; }
        public List<SourceField> Fields { get; set; }
        public List<List<Cell>> Records { get; set; }
        public Dictionary<uint, string> SharedNumFmts { get; set; }

        // ------ For PivotTable ------
        public List<string> PUids { get; set; }

        // ------ For Chart ------
        public List<string> CUids { get; set; }
        public List<string> CTypes { get; set; }

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
        public int Index { get; set; }
        public string Name { get; set; }
        public uint NumberFmtId { get; set; }
        public SharedItems Items { get; set; }

        public DataFormatFlags DataFormatFlags { get; set; }

        public bool Equals(SourceField other)
        {
            return other != null
                && Index == other.Index
                && string.Equals(Name, other.Name)
                && NumberFmtId == other.NumberFmtId
                && SharedItems.Equals(Items, other.Items);
        }

        public override int GetHashCode()
        {
            var hashCode = 1664113244;
            hashCode = hashCode * -1521134295 + Index.GetHashCode();
            hashCode = hashCode * -1521134295 + Name.GetHashCode();
            hashCode = hashCode * -1521134295 + NumberFmtId.GetHashCode();
            hashCode = hashCode * -1521134295 + EqualityComparer<SharedItems>.Default.GetHashCode(Items);
            return hashCode;
        }

        public long GetSchemaHash()
        {
            long hashCode = 1664113244L;
            hashCode = hashCode * -1521134295L + Index.GetHashCode();
            hashCode = hashCode * -1521134295L + Name.GetHashCode();
            return hashCode;
        }
    }

    /// <summary>
    /// See https://docs.microsoft.com/en-us/dotnet/api/documentformat.openxml.spreadsheet.shareditems?view=openxml-2.8.1
    /// Those bool? properties are only stored in PivotTable. For Chart and ExcelTable, only SharedCells are meaningful.
    /// </summary>
    [Serializable]
    public class SharedItems : IEquatable<SharedItems>
    {
        public IList<Cell> SharedCells = new List<Cell>();

        // ------ For PivotTable ------
        public bool? ContainsBlank { get; set; }
        public bool? ContainsDate { get; set; }
        public bool? ContainsInteger { get; set; }
        public bool? ContainsMixedTypes { get; set; }
        public bool? ContainsNonDate { get; set; }
        public bool? ContainsNumber { get; set; }
        public bool? ContainsSemiMixedTypes { get; set; }
        public bool? ContainsString { get; set; }
        public bool? LongText { get; set; }

        public static SharedItems GetInstance(CacheField cacheField)
        {
            if (cacheField.SharedItems == null)
                return null;

            return new SharedItems()
            {
                SharedCells = cacheField.SharedItems.Select(item => Cell.GetInstance(item)).ToList(),
                ContainsBlank = GetBool(cacheField.SharedItems.ContainsBlank),
                ContainsDate = GetBool(cacheField.SharedItems.ContainsDate),
                ContainsInteger = GetBool(cacheField.SharedItems.ContainsInteger),
                ContainsMixedTypes = GetBool(cacheField.SharedItems.ContainsMixedTypes),
                ContainsNonDate = GetBool(cacheField.SharedItems.ContainsNonDate),
                ContainsNumber = GetBool(cacheField.SharedItems.ContainsNumber),
                ContainsSemiMixedTypes = GetBool(cacheField.SharedItems.ContainsSemiMixedTypes),
                ContainsString = GetBool(cacheField.SharedItems.ContainsString),
                LongText = GetBool(cacheField.SharedItems.LongText)
            };
        }

        static bool? GetBool(BooleanValue b)
        {
            if (b == null)
                return null;
            return b.Value;
        }

        public bool Equals(SharedItems other)
        {
            return other != null &&
                ContainsBlank == other.ContainsBlank &&
                ContainsDate == other.ContainsDate &&
                ContainsInteger == other.ContainsInteger &&
                ContainsMixedTypes == other.ContainsMixedTypes &&
                ContainsNonDate == other.ContainsNonDate &&
                ContainsNumber == other.ContainsNumber &&
                ContainsSemiMixedTypes == other.ContainsSemiMixedTypes &&
                ContainsString == other.ContainsString &&
                LongText == other.LongText;
        }

        public static bool Equals(SharedItems x, SharedItems y)
        {
            return x != null && x.Equals(y);
        }

        public override int GetHashCode()
        {
            var hashCode = 1128399419;
            hashCode = hashCode * -1521134295 + ContainsBlank.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsDate.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsInteger.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsMixedTypes.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsNonDate.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsNumber.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsSemiMixedTypes.GetHashCode();
            hashCode = hashCode * -1521134295 + ContainsString.GetHashCode();
            hashCode = hashCode * -1521134295 + LongText.GetHashCode();
            return hashCode;
        }
    }
}
