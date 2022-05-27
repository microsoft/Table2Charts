// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using ColumnClassification.Contract;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ColumnClassification.Utils
{
    class TypeClassify
    {
        public static Regex fractionRegex = new Regex(@"^.*\d+/\d+.*$", RegexOptions.Compiled);
        public const double lengthPercentage = 0.35;
        public static string ClassifyType(List<CellType> CellTypeList, string headers) {
            
            double lineCount = 0;

            string result = string.Empty;

            lineCount = CountLines(CellTypeList);
            if (string.IsNullOrEmpty(result)&& IsDate(CellTypeList, lineCount)) {
                result = "Date";
            }
            if (string.IsNullOrEmpty(result) && IsSeq(CellTypeList, lineCount)) {
                result = "Sequence";
            }
            if (string.IsNullOrEmpty(result) && IsNumUnit(CellTypeList, lineCount)) {
                result = "NumericWithUnit";
            }
            if (string.IsNullOrEmpty(result)) {
                var numResult = IsNum(CellTypeList, lineCount);
                if ((bool)numResult["IsNum"]) {
                    if ((bool)numResult["HasFraction"]) {
                        result = "None";
                    }
                    else if (IsOrd((List<Double>)numResult["NumList"]))
                    {
                        result = "Ordinal Number";
                    }
                    else {
                        result = "Numeric";
                    }
                }
            }
            if (string.IsNullOrEmpty(result)) {
                result = "Text";
            }
            if (lineCount / CellTypeList.Count <= 0.5) {
                result = "None";
            }


            return result;
        }

        public static double CountLines(List<CellType> CellTypeList) {
            //bool emptyFlag = false;
            int lineCount = 0;
            foreach (var cell in CellTypeList)
            {
                bool skipFlag = false;
                string blankString = "—,???,na,n/a,-,/,none,null,\u2014";
                string[] blankStrings = blankString.Split(',');
                foreach (var i in blankStrings)
                {
                    if (cell.RawValue.ToLower().Contains(i) && cell.RawValue.Length < 5)
                    {
                        skipFlag = true;
                    }
                    if (cell.RawValue.Length == 0)
                    {
                        skipFlag = true;
                    }

                }
                if (skipFlag == true) continue;
                lineCount++;
            }
            return lineCount;
        }
        public static bool IsDate(List<CellType> CellTypeList, double lineCount) {
            int lengthCount = 0;
            int dateCount = 0;
            foreach (var cell in CellTypeList)
            {
                if (cell.ListOfType.ContainsKey("DateTime") && cell.ListOfType["DateTime"].Count != 0)
                {
                    var dateLength = cell.ListOfType["DateTime"][0].value.Length;
                    var dateValue = cell.ListOfType["DateTime"][0].TypeValue;
                    if (!CommonUtil.checkDateString(dateValue, true)) continue;
                    if (dateLength / (double)cell.RawValue.Length > lengthPercentage)
                    {
                        lengthCount++;
                    }
                    dateCount++;
                }
            }
            bool dateFlag = false;
            if (dateCount / lineCount > 0.9) dateFlag = true;
            if (lengthCount / lineCount < 0.9) dateFlag = false;
            return dateFlag;
        }
        public static bool IsSeq(List<CellType> CellTypeList, double lineCount) {
            int lengthCount = 0;
            int seqCount = 0;
            foreach (var cell in CellTypeList)
            {
                var content = cell.ListOfType.Keys.Where(x => x.Contains("Sequence")&&x!= "SequenceText").Select(x => cell.ListOfType[x]).ToList();
                if (content!=null &&content.Count>0&& content[0].Count != 0)
                {
                    var seqLength = content[0][0].value.Length;
                    if (seqLength / (double)cell.RawValue.Length > lengthPercentage)
                    {
                        lengthCount++;
                    }
                    seqCount++;
                }
            }
            bool seqFlag = false;
            if (seqCount / lineCount > 0.9) seqFlag = true;
            if (lengthCount / lineCount < 0.9) seqFlag = false;
            return seqFlag;
        }
        public static Dictionary<string,object> IsNum(List<CellType> CellTypeList, double lineCount) {
            List<double> numericList = new List<double>(); //record double of cell's first recognized numeric
            Dictionary<int, int> countOfStart = new Dictionary<int, int>();

            bool hasFraction = false;
            int numericCount = 0;//Count of numeric lines
            int lengthCount = 0;
            double maxNum = int.MinValue; //maxNum of this column  
            double minNum = int.MaxValue;//minNum of this column

            foreach (var cell in CellTypeList) {
                if (cell.ListOfType.ContainsKey("Numeric") && cell.ListOfType["Numeric"].Count != 0) {
                    if (!double.TryParse(cell.ListOfType["Numeric"][0].TypeValue, out double tempNum))
                    {
                        break;
                    }
                    var NumericLen = cell.ListOfType["Numeric"][0].value.Length;
                    if (NumericLen / (double)cell.RawValue.Length > lengthPercentage)
                    {
                        lengthCount++;
                    }
                    if (fractionRegex.IsMatch(cell.ListOfType["Numeric"][0].value)) {
                        hasFraction = true;
                    }
                    numericList.Add(tempNum);
                    if (tempNum > maxNum) maxNum = tempNum;
                    if (tempNum < minNum) minNum = tempNum;
                    numericCount++;
                    if (countOfStart.ContainsKey(cell.ListOfType["Numeric"][0].start))
                    {
                        countOfStart[cell.ListOfType["Numeric"][0].start] += 1;
                    }
                    else
                    {
                        countOfStart.Add(cell.ListOfType["Numeric"][0].start, 1);
                    }
                }
            }
            bool alignflag = false;
            foreach (KeyValuePair<int, int> count in countOfStart)
            {
                if (count.Value > 0.9 * lineCount)
                {
                    alignflag = true;
                }
            }

            if (numericCount / lineCount < 0.9) alignflag = false;
            if (lengthCount / lineCount < 0.9) alignflag = false;
            // Maybe add header info to trade off
            return new Dictionary<string, object>() { { "IsNum", alignflag }, {"NumList",numericList }, { "HasFraction", hasFraction} }; 

        }
        public static bool IsNumUnit(List<CellType> CellTypeList, double lineCount)
        {
            List<double> numericList = new List<double>(); //record double of cell's first recognized numeric
            Dictionary<int, int> countOfStart = new Dictionary<int, int>();

            int numericCount = 0;//Count of numeric lines
            int lengthCount = 0;
            foreach (var cell in CellTypeList)
            {
                var units = cell.ListOfType.Keys.Where(x => x.Contains("UnitNumeric")).Select(x => cell.ListOfType[x]).ToList();
                foreach (var protentialUnit in units)
                {
                    if (protentialUnit.Count > 0)
                    {
                        var NumericLen = protentialUnit[0].value.Length;
                        if (NumericLen / (double)cell.RawValue.Length > lengthPercentage)
                        {
                            lengthCount++;
                        }
                        numericCount++;
                        var start = protentialUnit[0].start;
                        if (start >= 0)
                        {
                            if (countOfStart.ContainsKey(start))
                            {
                                countOfStart[start] += 1;
                            }
                            else
                            {
                                countOfStart.Add(start, 1);
                            }
                        }
                    }
                }
            }
            // Maybe add header info to trade off
            bool alignflag = false;
            foreach (KeyValuePair<int, int> count in countOfStart)
            {
                if (count.Value > 0.9 * lineCount)
                {
                    alignflag = true;
                }
            }
            if (numericCount / lineCount < 0.9) alignflag = false;
            if (lengthCount / lineCount < 0.9) alignflag = false;
            // Maybe add header info to trade off
            return alignflag;
        }
        public static bool IsOrd(List<double> numericList) {
            bool isOrdinal = false;
            if (numericList.Count() > 1)
            {
                isOrdinal = true;
                var stanDiff = numericList[1] - numericList[0];
                for (int i = 0; i < numericList.Count() - 1; i++)
                {
                    var diff = numericList[i + 1] - numericList[i];
                    if (diff != stanDiff || Math.Abs(diff) != 1.0)
                    {
                        isOrdinal = false;
                        break;
                    }
                }
            }
            return isOrdinal;
        }

        public static string GetColumnUnit(List<CellType> CellTypeList, string Type)
        {
            if (!Type.ToLower().Contains("numeric")) {
                return null;
            }
            var countDict = new Dictionary<string, int>();
            foreach (var cellType in CellTypeList)
            {
                if (string.IsNullOrEmpty(cellType.Unit)) {
                    continue;
                }
                if (countDict.ContainsKey(cellType.Unit))
                {
                    countDict[cellType.Unit] += 1;
                }
                else
                {
                    countDict.Add(cellType.Unit, 1);
                }
            }
            var dicSort = from objDic in countDict orderby objDic.Value descending select objDic;

            foreach (KeyValuePair<string, int> kvp in dicSort)
            {
                return kvp.Key;
            }

            return null;
        }
        public static long GetTradeOff(List<CellType> CellTypeList)
        {
            // 16 32 pre/suffix
            // 8 multiple unit numeric
            // 4 miscellaneous unit
            // 2 multiple numeric result
            // 1 numeric and Unitnumeric
            

            long returnLong = 0;
            int dictCount = 0;
            var countDict = new Dictionary<string, int>();
            foreach (var cellType in CellTypeList)
            {
                if (string.IsNullOrEmpty(cellType.Unit)) {
                    continue;
                }
                if (countDict.ContainsKey(cellType.Unit))
                {
                    countDict[cellType.Unit] += 1;
                }
                else
                {
                    countDict.Add(cellType.Unit, 1);
                }
            }
            var dicSort = from objDic in countDict orderby objDic.Value descending select objDic;

            foreach (KeyValuePair<string, int> kvp in dicSort)
            {
                dictCount++;
            }
            if (dictCount > 1)
            {
                returnLong += 4;

            }

            bool numericAndUnitNumeric = false;
            bool multipleNumeric = false;
            foreach (var cell in CellTypeList)
            {
                int tempCount = 0;

                var unitList = cell.ListOfType.Keys.Where(x => x.Contains("UnitNumeric") && !x.Contains("Text")).ToList();
                tempCount = unitList.Count();

                if (cell.ListOfType.ContainsKey("Numeric")&& cell.ListOfType["Numeric"].Count > 0) tempCount++;

                if (tempCount > 1)
                {
                    numericAndUnitNumeric = true;
                    break;
                }
            }
            foreach (var cell in CellTypeList)
            {

                if (cell.ListOfType.ContainsKey("Numeric") && cell.ListOfType["Numeric"].Count > 1)
                {
                    multipleNumeric = true;
                    break;
                }
            }

            if (multipleNumeric == true) returnLong += 2;
            if (numericAndUnitNumeric == true)
            {
                returnLong += 1;
            }

            foreach (var cell in CellTypeList)
            {
                bool multipleUnitNumericFlag = false;
                var unitList = cell.ListOfType.Keys.Where(x => x.Contains("UnitNumeric") && !x.Contains("Text")).ToList();
                if (unitList.Count > 1)
                {
                    multipleUnitNumericFlag = true;
                }

                if (multipleUnitNumericFlag == true && multipleNumeric == true)
                {
                    returnLong += 8;
                    break;
                }
            }
            return returnLong;
        }
    }
}
