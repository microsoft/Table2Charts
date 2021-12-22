using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Common
{
    public partial class Helpers
    {
        /// <summary>
        /// See Section 8.5 of ECMA-376 Standard.
        /// https://www.iso.org/standard/71691.html
        /// </summary>
        public static string ColumnIndexToString(int col)
        {
            StringBuilder sb = new StringBuilder();
            while (col > 0)
            {
                col--;
                int remainer = col % 26;
                sb.Append((char)(remainer + 65));
                col /= 26;
            }
            return new string(sb.ToString().Reverse().ToArray());
        }
        public static int StringToColumnIndex(string indexStr)
        {
            if (!Regex.IsMatch(indexStr, "[a-zA-Z]+"))
            {
                Console.WriteLine($"Wrong Table Range Format: {indexStr}");
                return -1;
            }

            int index = 0;
            for (int i = 0; i < indexStr.Length; i++)
            {
                index *= 26;
                index += (indexStr[i] - 'A' + 1);
            }
            return index;
        }


        public static List<List<String>> TableBodyToRowList(string headerChunk, string bodyChunk)
        {
            // It will fail in some cases to use "#TAB#" and "#N#" separately because there are cases like:
            //   "...[something]#TAB#N#N#[something]#TAB#..."=
            var rowList = new List<List<String>>();
            string[] cellSeparator = new string[] { "#TAB#", "#N#" };
            string[] cells = bodyChunk.Split(cellSeparator, StringSplitOptions.None);
            var columnCnt = headerChunk.Split(new string[] { "#TAB#" }, StringSplitOptions.None).Length;

            if (cells.Length % columnCnt != 0)
            {
                throw new Exception("The number of records doesn't match the number of headers!");
            }

            int nRows = cells.Length / columnCnt;
            for (int i = 0; i < nRows; i++)
            {
                List<string> curRow = new List<string>();
                for (int j = 0; j < columnCnt; j++)
                {
                    curRow.Add(cells[i * columnCnt + j]);
                }
                rowList.Add(curRow);
            }

            return rowList;
        }

        public static List<List<String>> RowsToColumns(int columnCnt, List<List<String>> rowList)
        { 
            var columnList = new List<List<String>>();

            for (var colIdx = 0; colIdx < columnCnt; colIdx ++)
            {
                columnList.Add(new List<string>());
                foreach (var row in rowList)
                {
                    if (row.Count == columnCnt)
                    {
                        columnList[colIdx].Add(row[colIdx]);   
                    }
                    else
                    {
                        Console.WriteLine("Error: Col and row is not match at [{0}]", row);
                    }
                }
            }

            return columnList;
        }

    }
}
