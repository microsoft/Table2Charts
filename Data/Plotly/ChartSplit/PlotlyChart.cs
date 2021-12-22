using DocumentFormat.OpenXml.Vml.Office;
using DocumentFormat.OpenXml.Wordprocessing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.DataFormat
{
    public class PlotlyChart : ProcessedChart, IEquatable<PlotlyChart>
    {
        public static List<PlotlyChart> GetInstance(string fid, PlotlyTable plotlyTable, string filePath = "")
        {
            List<OriginalPlotlyChart> chartDatas = Helpers.LoadJson<List<OriginalPlotlyChart>>(filePath + fid + @"_chartdata.json", DataSerializer.Instance);
            List<PlotlyChart> charts = new List<PlotlyChart>();
            int cUID = 0;
            int chart_num = 0;
            List<string> XsrcXaxisYaxisType = new List<string>();
            List<string> XsrcXaxisYaxis = new List<string>();
            List<string> delXsrcXaxisYaxis = new List<string>();

            foreach (OriginalPlotlyChart chartData in chartDatas)
            {
                //If it is horizontal, switch x and y 
                if (chartData?.Orientation == "h")
                {
                    string axis = chartData.Xaxis;
                    string src = chartData.Xsrc;
                    chartData.Xaxis = chartData.Yaxis;
                    chartData.Yaxis = axis;
                    chartData.Xsrc = chartData.Ysrc;
                    chartData.Ysrc = src;
                }

                //Pie charts only have valuessrc as ysrc
                if (chartData.Valuessrc != null)
                {
                    chartData.Ysrc = chartData.Valuessrc;
                }

                //Draw line
                bool line = false;
                if (chartData.Mode == null ? false : chartData.Mode.Contains("line"))
                    line = true;

                //Type
                string CType = string.Empty;
                if (chartData.Type != null)
                {
                    if (chartData.Type.Contains("scatter") && !chartData.Type.Contains("scatter3d")) CType = "scatter";
                    else if (chartData.Type.Contains("pie")) CType = "pie";
                    else if (chartData.Type.Contains("line")) CType = "line";
                    else if (chartData.Type.Contains("bar")) CType = "bar";
                    else continue;
                }
                else if (line)
                    CType = "line";
                else continue;//The chart without type and mode is incomplete

                //Add new chart or find chart_num
                string chartXsrcXaxisYaxisType = chartData.Xsrc + '-' + chartData.Xaxis + '-' + chartData.Yaxis + '-' + CType;
                string chartXsrcXaxisYaxis = chartData.Xsrc + '-' + chartData.Xaxis + '-' + chartData.Yaxis;
                if (delXsrcXaxisYaxis.Exists(t => t == chartXsrcXaxisYaxis))
                    //Delete same chart, different type with pie
                    continue;
                else if (XsrcXaxisYaxisType.Exists(t => t == chartXsrcXaxisYaxisType) && (CType == "line" || CType == "bar" || (CType == "scatter" && line)))
                {
                    //If xsrc, xaxis, yaxis and type are same, they are the same chart
                    chart_num = XsrcXaxisYaxisType.FindIndex(t => t == chartXsrcXaxisYaxisType);
                }
                else if (XsrcXaxisYaxis.Exists(t => t == chartXsrcXaxisYaxis) && (CType == "pie" || XsrcXaxisYaxisType[XsrcXaxisYaxis.FindIndex(t => t == chartXsrcXaxisYaxis)].Split('-').Last() == "pie"))
                {
                    //Delete the first same chart, different type with pie
                    chart_num = XsrcXaxisYaxis.FindIndex(t => t == chartXsrcXaxisYaxis);
                    charts.RemoveAt(chart_num);
                    XsrcXaxisYaxisType.RemoveAt(chart_num);
                    XsrcXaxisYaxis.RemoveAt(chart_num);
                    delXsrcXaxisYaxis.Add(chartXsrcXaxisYaxis);
                    continue;
                }
                else
                {
                    PlotlyChart chart = new PlotlyChart
                    {
                        CUid = $"{fid}.t0.c{cUID}",
                        SheetName = chartData.Name,
                        CType = chartData.Type,
                        XFields = new List<Field>(),
                        YFields = new List<Field>(),
                        ValueDrawsLine = new List<bool>()
                    };
                    chart.CType = CType;
                    SourceField xfield = GetField(chartData.Xsrc, plotlyTable);
                    if (xfield != null)
                    {
                        chart.XFields.Add(new Field
                        {
                            Index = xfield.Index,
                            Name = xfield.Name
                        });
                    }
                    XsrcXaxisYaxis.Add(chartXsrcXaxisYaxis);
                    XsrcXaxisYaxisType.Add(chartXsrcXaxisYaxisType);
                    charts.Add(chart);
                    chart_num = charts.Count() - 1;
                    cUID += 1;

                }

                SourceField yfield = GetField(chartData.Ysrc, plotlyTable);
                if (yfield != null)
                {
                    charts[chart_num].YFields.Add(new Field
                    {
                        Index = yfield.Index,
                        Name = yfield.Name
                    });
                }

                charts[chart_num].ValueDrawsLine.Add(line);
                if (charts[chart_num].XFields.Count() == 0 && charts[chart_num].YFields.Count() == 0)
                {
                    charts.RemoveAt(chart_num);
                    XsrcXaxisYaxisType.RemoveAt(chart_num);
                    XsrcXaxisYaxis.RemoveAt(chart_num);
                }
            }

            return charts;
        }

        /// <summary>
        /// This class is to store the information of original Plotly chart from the uid_chartdata.json.
        /// </summary>
        public class OriginalPlotlyChart
        {
            [NotNull] public string Mode { get; set; }
            [NotNull] public string Type { get; set; }
            [NotNull] public string Name { get; set; }
            [NotNull] public string Xaxis { get; set; }
            [NotNull] public string Yaxis { get; set; }
            [NotNull] public string Ysrc { get; set; }
            [NotNull] public string Xsrc { get; set; }
            [NotNull] public string Valuessrc { get; set; }
            public string Orientation { get; set; }
        }

        public static SourceField GetField (string src, PlotlyTable plotlyTable )
        {
            if (src == null) return null;
            src = src.Split(':').Last();
            foreach(SourceField field in plotlyTable.Fields)
            {
                if (field.Uid == src) return field;
            }

            return null;
        }

        public bool Equals(PlotlyChart other)
        {
            throw new NotImplementedException();
        }
    }
}
