namespace Common
{
    public class GeneralCell : IGeneralCell
    {
        public string Text { get; set; }

        public object Value { get; set; }

        public DataFormatFlags DataFormatFlags { get; }

        public bool HasUnit { get; }

        public string Unit { get; }

        public GeneralCell(CellParsingResult cellParsingResult, string text)
        {
            Text = text;
            Value = cellParsingResult.RawValue;
            DataFormatFlags = cellParsingResult.DataFormatFlags;
            HasUnit = cellParsingResult.HasUnit;
            Unit = cellParsingResult.Unit;
        }
        public GeneralCell(object RawValue,DataFormatFlags dataFormatFlags,bool hasUnit, string unit, string text)
        {
            Text = text;
            Value = RawValue;
            DataFormatFlags = dataFormatFlags;
            HasUnit = hasUnit;
            Unit = unit;
        }
    }
}
