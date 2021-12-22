using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Common
{
    public class PlotlyTable : Table, IEquatable<PlotlyTable>
    {

        public static PlotlyTable GetInstance(string fid, string filePath="")
        {
            Dictionary<string, PlotlyTableData> tableData = Helpers.LoadJson<Dictionary<string, PlotlyTableData>>(Path.Combine(filePath, $"{fid}_tabledata.json"), DataSerializer.Instance);
            
            if (tableData.Keys.Count != 1)
            {
                var logLine = string.Format("[Info][{0}] Error: {1} has more than one table or none table!", DateTime.Now.ToString("HH:mm:ss"), fid);
                Console.WriteLine(logLine);
                return null;
            }

            // Construct sourceFields of the PlotlyTable.
            int nColumns = tableData.Values.Last().Cols.Count;
            List< SourceField> fields = new List<SourceField>();

            foreach (string item in tableData.Values.Last().Cols.Keys)
            {
                fields.Add(new SourceField
                {
                    Name = item,
                    Index = tableData.Values.Last().Cols[item].Order,
                    Uid = tableData.Values.Last().Cols[item].Uid,
                });
                fields.Sort((a, b) => a.Index - b.Index);
            }

            // Construct records and numFmtIds of the PlotlyTable.
            // Since PlotlyTables have no numFmtIds, here we only assign numFmtId=14 for DateTimes
            //   and 0 for others. Note that 14 is a default DateTime format id.
            var records = new List<List<Cell>>();
            var numFmtIds = new List<List<uint>>();
            
            int nRows = 0;
            foreach (PlotlyTableDataCols item in tableData.Values.Last().Cols.Values)
            {
                nRows = Math.Max(nRows, item.Data.Count);
            }
            for (int i = 0; i < nRows; i++)
            {
                List<Cell> rowRecords = new List<Cell>();
                List<uint> rowNumFmtIds = new List<uint>();
                foreach (PlotlyTableDataCols item in tableData.Values.Last().Cols.Values)
                {
                    string value;
                    if (i < item.Data.Count())
                        value = Convert.ToString(item.Data[i]);
                    else value = "";

                    Cell cell = new Cell { Type = "s", Value = value };
                    //Cell cell = ConvertValueToCell(value, canParseDateTime);
                    rowNumFmtIds.Add(0);  // numFmtId 0 is the default format "General".
                    rowRecords.Add(cell);
                }
                numFmtIds.Add(rowNumFmtIds);
                records.Add(rowRecords);
            }

            // Decide field format ids.
            for (int j = 0; j < fields.Count; j++)
            {
                var fieldNumFmtIds = numFmtIds.Select(list => list[j]);
                var result = fieldNumFmtIds.GroupBy(id => id).OrderByDescending(g => g.Count()).ToList();
                uint mostFormatStr = result.FirstOrDefault()?.First() ?? 0;
                fields[j].NumberFmtId = mostFormatStr;
            }

            // Generate SharedItems and simplify cells in records.
            // We do not put string Cells that appear only once to SharedItems.
            for (int j = 0; j < nColumns; j++)
            {
                // Count the repeat time of the cells.
                Dictionary<Cell, int> stringCells = new Dictionary<Cell, int>();
                for (int i = 0; i < nRows; i++)
                {
                    Cell cell = records[i][j];
                    if (!cell.IsString())
                        continue;
                    if (stringCells.ContainsKey(cell))
                        stringCells[cell]++;
                    else
                        stringCells.Add(new Cell(cell), 1);
                }

                // Update SharedItems of the field.
                List<Cell> sharedStringCells = new List<Cell>();
                foreach (var pair in stringCells)
                {
                    if (pair.Value > 1)
                        sharedStringCells.Add(pair.Key);
                }
                fields[j].Items = new SharedItems { SharedCells = sharedStringCells };

                // Simplify records with shared cells.
                for (int i = 0; i < nRows; i++)
                {
                    Cell cell = records[i][j];
                    if (cell.IsString() && stringCells[cell] > 1)
                        records[i][j] = cell.SimplifyRecord(fields[j].Items);
                }
            }

            return new PlotlyTable
            {
                TUid = $"{fid}.t0",
                SheetName = fid,
                NColumns = nColumns,
                NRows = nRows,
                SharedNumFmts = new Dictionary<uint, string>(),
                Fields = fields,
                Records = records,
            };
        }

        public bool Equals(PlotlyTable other)
        {
            return base.Equals(other);
        }
        
        public class PlotlyTableDataCols
        {
            [NotNull] public string Uid { get; set; }
            [NotNull] public int Order { get; set; }
            [NotNull] public List<dynamic> Data { get; set; }
        }

        public class PlotlyTableData
        {
            [NotNull] public string Reason { get; set; }
            [NotNull] public Dictionary<string, PlotlyTableDataCols> Cols { get; set; }
        }
    }
}
