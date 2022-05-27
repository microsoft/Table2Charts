// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using ColumnClassification.Utils;
using Microsoft.Recognizers.Text;
using Microsoft.Recognizers.Text.DateTime;
using Microsoft.Recognizers.Text.Number;
using Microsoft.Recognizers.Text.NumberWithUnit;
using Microsoft.Recognizers.Text.Sequence;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ColumnClassification.Contract
{
    public class CellModel
    {
        public string RawValue;
        public object Value;
        public CellType TypeDetection;
        public CellModel(string rawValue)
        {
            this.RawValue = rawValue;
        }
    }

    public class ColumnModel
    {
        public string Header;
        public string Unit;
        public string suffix;
        public string prefix;
        public string Type;
        public int FormatFlags;
        public long TradeOffFlags;
        public List<CellModel> Fields;
        public Regex prefixRegex = new Regex(@"^(?<prefix>[^0-9]+)", RegexOptions.Compiled);
        public Regex suffixRegex = new Regex(@"(?<suffix>[^0-9]+)$", RegexOptions.Compiled);
        public ColumnModel(string header)
        {
            this.Header = header;
            this.Fields = new List<CellModel>();
        }
        public void ClassifyType()
        {
            var CellTypeList = Fields.Select(t => t.TypeDetection).ToList();

            this.Type = TypeClassify.ClassifyType(CellTypeList, Header);

            this.Unit = TypeClassify.GetColumnUnit(CellTypeList, Type);

            SetDataFormat(CellTypeList);

            GetTradeOffAndResetType(this);

            GetPreAndSuffix(this);

            //set colum value
            SetColumnValue(this);



        }
        public void GetTradeOffAndResetType(ColumnModel columnModel)
        {
            var CellTypeList = columnModel.Fields.Select(t => t.TypeDetection).ToList();
            if (this.Type == "Text" || this.Type == "Date") this.TradeOffFlags = 0;
            this.TradeOffFlags = TypeClassify.GetTradeOff(CellTypeList);
            // SetDataFormat(CellTypeList);
            ReProcessColumnType(this);
        }

        public void ReProcessColumnType(ColumnModel columnModel)
        {
            long num = columnModel.TradeOffFlags;
            if ((num & 2) > 0)
            {
                if (columnModel.Type.Contains("Numeric"))
                {
                    columnModel.Type = "None";
                    columnModel.FormatFlags = (int)Common.DataFormatFlags.None;
                }
            }
            if ((num & 8) > 0 || (num & 4) > 0)
            {
                if (columnModel.Type.Contains("Numeric"))
                {
                    columnModel.Type = "None";
                    columnModel.FormatFlags = (int)Common.DataFormatFlags.None;
                }
            }
            if (columnModel.FormatFlags == (int)Common.DataFormatFlags.Text || columnModel.FormatFlags == (int)Common.DataFormatFlags.None)
            {
                columnModel.Unit = null;
                foreach (var i in columnModel.Fields)
                {
                    i.TypeDetection.Unit = null;
                }
            }
        }

        public void SetDataFormat(List<CellType> CellTypeList)
        {
            this.FormatFlags = 0;
            if (Type.Contains("Numeric") && Unit == "%")
            {
                FormatFlags += (int)Common.DataFormatFlags.Percent;
            }
            if (Type.Contains("Numeric"))
            {
                FormatFlags += (int)Common.DataFormatFlags.Numeric;
                var Cur = NumberWithUnitRecognizer.RecognizeCurrency("100" + Unit, Culture.English);
                if (Cur.Count != 0)
                {
                    if (Cur[0].Resolution != null && Cur[0].Resolution.ContainsKey("value"))
                    {
                        FormatFlags += (int)Common.DataFormatFlags.Currency;
                    }
                }
            }
            if (Type == "Text")
            {
                FormatFlags += (int)Common.DataFormatFlags.Text;
            }
            if (Type == "None")
            {
                FormatFlags += (int)Common.DataFormatFlags.None;
            }
            if (Type == "Sequence")
            {
                FormatFlags += (int)Common.DataFormatFlags.Sequence;
            }
            if (Type.Contains("Ordinal"))
            {
                FormatFlags += (int)Common.DataFormatFlags.Ordinal;
            }
            if (Type == "Date")
            {

                Dictionary<int, int> dateFlagDic = new Dictionary<int, int>();
                dateFlagDic[1] = 0;
                foreach (var cell in CellTypeList)
                {
                    int tempFlag = 1;
                    if (string.IsNullOrEmpty(cell.RawValue.Trim())) continue;
                    if (cell.ListOfType.ContainsKey("DateTime"))
                    {
                        var dateValue = (cell.ListOfType["DateTime"][0]).TypeValue;
                        var timeSplit = dateValue.Split('-');
                        if (timeSplit.Count() == 1)
                        {
                            if (!timeSplit[0].ToLower().Contains("t") && CommonUtil.checkDateString(timeSplit[0]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Year;
                            }
                        }
                        else if (timeSplit.Count() == 2)
                        {
                            if (CommonUtil.checkDateString(timeSplit[0]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Year;
                            }
                            if (CommonUtil.checkDateString(timeSplit[1]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Month;
                            }
                        }
                        else if (timeSplit.Count() == 3)
                        {
                            if (CommonUtil.checkDateString(timeSplit[0]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Year;
                            }
                            if (CommonUtil.checkDateString(timeSplit[1]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Month;
                            }
                            if (CommonUtil.checkDateString(timeSplit[2]))
                            {
                                tempFlag += (int)Common.DataFormatFlags.Day;
                            }
                        }
                    }

                    if (dateFlagDic.Keys.Contains(tempFlag))
                    {
                        dateFlagDic[tempFlag]++;
                    }
                    else
                    {
                        dateFlagDic[tempFlag] = 1;
                    }
                }

                var sortResult1 = from pair in dateFlagDic orderby pair.Value descending select pair;
                FormatFlags = sortResult1.First().Key;
            }
        }
        public void SetColumnValue(ColumnModel columnModel)
        {
            var type = columnModel.Type;
            if (type.Contains("Numeric"))
            {
                foreach (var cell in columnModel.Fields)
                {
                    if (cell.TypeDetection.ListOfType.Keys.Contains(CellType.ResultTypes.Numeric.ToString()))
                    {
                        cell.Value = double.Parse(cell.TypeDetection.ListOfType["Numeric"][0].TypeValue);
                    }
                    else
                    {
                        cell.Value = cell.RawValue;
                    }
                }
            }
            else if (type == "Date")
            {
                foreach (var cell in columnModel.Fields)
                {
                    cell.Value = cell.RawValue;
                    if (!cell.TypeDetection.ListOfType.ContainsKey("DateTime"))
                    {
                        continue;
                    }
                    var recResult = DateTimeRecognizer.RecognizeDateTime(cell.RawValue, Culture.English, DateTimeOptions.SkipFromToMerge);
                    if (recResult.Count > 0)
                    {
                        if (recResult[0].Resolution.ContainsKey("values"))
                        {
                            var values = recResult[0].Resolution["values"];
                            var lists = (List<Dictionary<string, string>>)(values);
                            if (lists.Count > 0)
                            {
                                var subDict = lists[0];

                                if (subDict.ContainsKey("value"))
                                {
                                    string nowDate = subDict["value"];

                                    if (!DateTime.TryParse(nowDate, out DateTime dateTime1))
                                    {
                                        dateTime1 = Convert.ToDateTime("1900-1-1");
                                    }

                                    DateTime dtStart = Convert.ToDateTime("1900-1-1");

                                    int setDays = (int)(dateTime1 - dtStart).TotalDays;
                                    cell.Value = (object)setDays;
                                }
                                else if (subDict.ContainsKey("start"))
                                {
                                    string nowDate = subDict["start"];

                                    if (!DateTime.TryParse(nowDate, out DateTime dateTime1))
                                    {
                                        dateTime1 = Convert.ToDateTime("1900-1-1");
                                    }

                                    DateTime dtStart = Convert.ToDateTime("1900-1-1");

                                    int setDays = (int)(dateTime1 - dtStart).TotalDays;
                                    cell.Value = (object)setDays;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                foreach (var cell in columnModel.Fields)
                {
                    cell.Value = cell.RawValue;
                }
            }
        }

        public void GetPreAndSuffix(ColumnModel columnModel)
        {
            var cells = columnModel.Fields;
            var strs = cells.Select(t => t.RawValue).ToList();
            Dictionary<string, string> fixDic = new Dictionary<string, string>();
            fixDic["Prefix"] = string.Empty;
            fixDic["Suffix"] = string.Empty;
            if (strs == null || strs.Count == 0)
            {
                columnModel.suffix = string.Empty;
                columnModel.prefix = string.Empty;
                return;
            }
            string pre = strs[0];
            string suf = strs[0];
            int i = 1;
            int j = 1;
            while (i < strs.Count)
            {
                while (strs[i].IndexOf(pre) != 0)
                    pre = pre.Substring(0, pre.Length - 1);
                i++;
            }
            while (j < strs.Count)
            {
                if (strs[j].Length == 0)
                {
                    suf = string.Empty;
                    break;
                }
                while ((strs[j].LastIndexOf(suf) != strs[j].Length - suf.Length || strs[j].LastIndexOf(suf)==-1) && !string.IsNullOrEmpty(suf)  )
                    suf = suf.Substring(1);
                j++;
            }
            Match Match = null;
            if ((Match = prefixRegex.Match(pre)).Success)
            {
                pre = Match.Groups["prefix"].ToString().Trim();
            }
            else {
                pre = string.Empty;
            }

            if ((Match = suffixRegex.Match(suf)).Success)
            {
                suf = Match.Groups["suffix"].ToString().Trim();
            }
            else
            {
                suf = string.Empty;
            }
            columnModel.prefix = pre;
            columnModel.suffix = suf;
            if (!string.IsNullOrEmpty(pre))
            {
                columnModel.TradeOffFlags += 16;
            }
            if (!string.IsNullOrEmpty(suf))
            {
                columnModel.TradeOffFlags += 32;
            }
        }
    }

    public class SubType
    {
        public string value { set; get; }
        public int start { set; get; }
        public string TypeValue { set; get; }
        public SubType(string m_value, int m_start, string m_typeValue)
        {
            this.value = m_value;
            this.start = m_start;
            this.TypeValue = m_typeValue;
        }
        public SubType()
        {
            this.value = "";
            this.start = -1;
            this.TypeValue = "";
        }
    }

    public class CellType
    {
        public enum ResultTypes : int
        {
            UnitNumericAge,
            UnitNumericCurrency,
            UnitNumericDimension,
            UnitNumericTemperature,
            UnitNumericNumeric,
            UnitNumericText,
            Numeric,
            NumericText,
            SequenceEmail,
            SequenceGUID,
            SequenceHashtag,
            SequenceIP,
            SequenceMention,
            SequencePhoneNumber,
            SequenceURL,
            SequenceText,
            DateTime,
            DateTimeText
        }
        public Dictionary<string, List<SubType>> ListOfType;
        public int Start { get; set; }
        public string RawValue { get; set; }
        public string Unit { get; set; }
        public string suffix { get; set; }
        public string prefix { get; set; }
        public CellType()
        {
            ListOfType = new Dictionary<string, List<SubType>>();
            this.Start = -1;
            this.Unit = "n";
        }
        public CellType(string rawData, string language)
        {
            ListOfType = DetectDataType(rawData, language);
            this.SetCellTypeStart(rawData);
            this.RawValue = rawData;
        }
        public void SetCellTypeStart(string rawData)
        {
            var cul = Culture.English;
            var result = NumberRecognizer.RecognizeNumber(rawData, cul);
            if (result.Count != 0)
            {
                this.Start = result[0].Start;
            }
            else
            {
                this.Start = -1;
            }
        }

        public Dictionary<string, List<SubType>> DetectDataType(string field, string language)
        {

            this.Unit = null;
            Dictionary<string, List<SubType>> dataTypes = new Dictionary<string, List<SubType>>();
            if (field.Length > 1000)
            {
                var valueAsType = new List<SubType>();
                valueAsType.Add(new SubType(field, -1, field));
                dataTypes[ResultTypes.UnitNumericText.ToString()] = valueAsType;
                dataTypes[ResultTypes.DateTimeText.ToString()] = valueAsType;
                dataTypes[ResultTypes.NumericText.ToString()] = valueAsType;
                dataTypes[ResultTypes.SequenceText.ToString()] = valueAsType;
                return dataTypes;
            }

            string cul = Culture.English;
            switch (language) {
                case "ar":cul = Culture.Arabic;break;
                case "hi": cul = Culture.Hindi; break;
                case "tr": cul = Culture.Turkish; break;
                case "bg": cul = Culture.Bulgarian; break;
                case "sv": cul = Culture.Swedish; break;
                case "ko": cul = Culture.Korean; break;
                case "ja": cul = Culture.Japanese; break;
                case "it": cul = Culture.Italian; break;
                case "nl": cul = Culture.Dutch; break;
                case "fr": cul = Culture.French; break;
                case "pt": cul = Culture.Portuguese; break;
                case "es": cul = Culture.Spanish; break;
                case "zh_chs": cul = Culture.Chinese; break;
                case "zh_cht": cul = Culture.Chinese; break;
                case "de": cul = Culture.German; break;
            }
            var percentUnit = NumberRecognizer.RecognizePercentage(field, cul);
            if (percentUnit.Count != 0)
            {
                this.Unit = "%";
            }
            var temp = NumberWithUnitRecognizer.RecognizeAge(field, cul);
            if (temp.Count != 0)
            {
                var valueAsType = new List<SubType>();
                foreach (var tp in temp)
                {
                    if (tp.Resolution != null && tp.Resolution.ContainsKey("value") && tp.Resolution["value"] != null)
                    {
                        valueAsType.Add(new SubType(tp.Text, tp.Start, tp.Resolution["value"].ToString()));
                        dataTypes[ResultTypes.UnitNumericAge.ToString()] = valueAsType;
                    }
                }
                if (temp[0].Resolution.ContainsKey("unit"))
                {
                    this.Unit = (string)temp[0].Resolution["unit"];
                }
            }
            var temp1 = NumberWithUnitRecognizer.RecognizeCurrency(field, cul);
            if (temp1.Count != 0)
            {
                var valueAsType = new List<SubType>();
                foreach (var tp in temp1)
                {
                    if (tp.Resolution != null && tp.Resolution.ContainsKey("value") && tp.Resolution["value"] != null)
                    {
                        valueAsType.Add(new SubType(tp.Text, tp.Start, tp.Resolution["value"].ToString()));
                        dataTypes[ResultTypes.UnitNumericCurrency.ToString()] = valueAsType;
                    }
                }
                if (temp1[0].Resolution.ContainsKey("unit"))
                {
                    this.Unit = (string)temp1[0].Resolution["unit"];
                }

            }
            var temp2 = NumberWithUnitRecognizer.RecognizeDimension(field, cul);
            if (temp2.Count != 0)
            {
                var valueAsType = new List<SubType>();
                foreach (var tp in temp2)
                {
                    if (tp.Resolution != null && tp.Resolution.ContainsKey("value") && tp.Resolution["value"] != null)
                    {
                        valueAsType.Add(new SubType(tp.Text, tp.Start, tp.Resolution["value"].ToString()));
                        dataTypes[ResultTypes.UnitNumericDimension.ToString()] = valueAsType;
                    }
                }
                if (temp2[0].Resolution.ContainsKey("unit"))
                {
                    this.Unit = (string)temp2[0].Resolution["unit"];
                }
            }
            var temp3 = NumberWithUnitRecognizer.RecognizeTemperature(field, cul);
            if (temp3.Count != 0)
            {
                var valueAsType = new List<SubType>();
                foreach (var tp in temp3)
                {
                    if (tp.Resolution != null && tp.Resolution.ContainsKey("value") && tp.Resolution["value"] != null)
                    {
                        valueAsType.Add(new SubType(tp.Text, tp.Start, tp.Resolution["value"].ToString()));
                        dataTypes[ResultTypes.UnitNumericTemperature.ToString()] = valueAsType;
                    }
                }
                if (temp3[0].Resolution.ContainsKey("unit"))
                {
                    this.Unit = (string)temp3[0].Resolution["unit"];
                }
            }
            if (temp1.Count == 0 && temp2.Count == 0 && temp3.Count == 0 && temp.Count == 0)
            {
                var valueAsType = new List<SubType>();
                valueAsType.Add(new SubType(field, -1, field));
                dataTypes[ResultTypes.UnitNumericText.ToString()] = valueAsType;
            }

            var num1 = NumberRecognizer.RecognizeNumber(field, cul);
            if (num1.Count != 0)
            {
                var valueAsType = new List<SubType>();
                foreach (var u in num1)
                {

                    if (u.Resolution != null && u.Resolution.ContainsKey("value") && u.Resolution["value"] != null && u.Resolution["value"].ToString() != "∞")
                    {

                        valueAsType.Add(new SubType(u.Text, u.Start, u.Resolution["value"].ToString()));
                        dataTypes[ResultTypes.Numeric.ToString()] = valueAsType;
                    }

                }
            }
            else
            {
                dataTypes[ResultTypes.NumericText.ToString()] = new List<SubType>() { new SubType(field, -1, field) };

            }

            var seq1 = SequenceRecognizer.RecognizeEmail(field, cul);
            if (seq1.Count != 0)
            {
                if (seq1[0].Resolution != null && seq1[0].Resolution.ContainsKey("value") && seq1[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq1[0].Text, seq1[0].Start, seq1[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceEmail.ToString()] = valueAsType;
                }
            }
            var seq2 = SequenceRecognizer.RecognizeGUID(field, cul);
            if (seq2.Count != 0)
            {
                if (seq2[0].Resolution != null && seq2[0].Resolution.ContainsKey("value") && seq2[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq2[0].Text, seq2[0].Start, seq2[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceGUID.ToString()] = valueAsType;
                }
            }
            var seq3 = SequenceRecognizer.RecognizeHashtag(field, cul);
            if (seq3.Count != 0)
            {
                if (seq3[0].Resolution != null && seq3[0].Resolution.ContainsKey("value") && seq3[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq3[0].Text, seq3[0].Start, seq3[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceHashtag.ToString()] = valueAsType;
                }
            }
            var seq4 = SequenceRecognizer.RecognizeIpAddress(field, cul);
            if (seq4.Count != 0)
            {
                if (seq4[0].Resolution != null && seq4[0].Resolution.ContainsKey("value") && seq4[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq4[0].Text, seq4[0].Start, seq4[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceIP.ToString()] = valueAsType;
                }
            }
            var seq5 = SequenceRecognizer.RecognizeMention(field, cul);
            if (seq5.Count != 0)
            {
                if (seq5[0].Resolution != null && seq5[0].Resolution.ContainsKey("value") && seq5[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq5[0].Text, seq5[0].Start, seq5[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceMention.ToString()] = valueAsType;
                }
            }
            var seq6 = SequenceRecognizer.RecognizePhoneNumber(field, cul);
            if (seq6.Count != 0)
            {
                if (seq6[0].Resolution != null && seq6[0].Resolution.ContainsKey("value") && seq6[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq6[0].Text, seq6[0].Start, seq6[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequencePhoneNumber.ToString()] = valueAsType;
                }
            }
            var seq7 = SequenceRecognizer.RecognizeURL(field, cul);
            if (seq7.Count != 0)
            {
                if (seq7[0].Resolution != null && seq7[0].Resolution.ContainsKey("value") && seq7[0].Resolution["value"] != null)
                {
                    var valueAsType = new List<SubType>();
                    valueAsType.Add(new SubType(seq7[0].Text, seq7[0].Start, seq7[0].Resolution["value"].ToString()));
                    dataTypes[ResultTypes.SequenceURL.ToString()] = valueAsType;
                }
            }
            if (seq1.Count == 0 && seq2.Count == 0 && seq3.Count == 0 && seq4.Count == 0 && seq5.Count == 0 && seq6.Count == 0 && seq7.Count == 0)
            {
                var valueAsType = new List<SubType>();
                valueAsType.Add(new SubType(field, -1, field));
                dataTypes[ResultTypes.SequenceText.ToString()] = valueAsType;
            }


            var date1 = DateTimeRecognizer.RecognizeDateTime(field, cul);

            if (date1.Count != 0)
            {
                if (date1[0].Resolution != null && date1[0].Resolution.ContainsKey("values") && date1[0].Resolution["values"] != null)
                {
                    var valueAsType = new List<SubType>();
                    var values = (List<Dictionary<string, string>>)date1[0].Resolution["values"];
                    if (values[0].ContainsKey("timex"))
                    {
                        valueAsType.Add(new SubType(date1[0].Text, date1[0].Start, values[0]["timex"]));
                        dataTypes[ResultTypes.DateTime.ToString()] = valueAsType;
                    }
                    else
                    {
                        valueAsType.Add(new SubType(field, -1, field));
                        dataTypes[ResultTypes.DateTimeText.ToString()] = valueAsType;
                    }

                }
            }
            else
            {
                var valueAsType = new List<SubType>();
                valueAsType.Add(new SubType(field, -1, field));
                dataTypes[ResultTypes.DateTimeText.ToString()] = valueAsType;
            }

            return dataTypes;
        }
    }
}
