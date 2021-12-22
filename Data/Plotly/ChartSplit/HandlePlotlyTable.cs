using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Common;
using Common.DataFormat;

namespace FeatureExtractor
{
    public class HandlePlotlyTable
    {
        public static PlotlyList LoadPlotlyTablesAll(string dataFolder)
        {
            Console.WriteLine($"Start loading Plotly tables from {dataFolder}.");
            TextReader tr = new StreamReader(dataFolder+@"\plotly_data_dedup.tsv");

            string line;
            var processedLines = 0;
            List<PlotlyTable> plotlyTables = new List<PlotlyTable>();
            List<PlotlyChart> plotlyCharts = new List<PlotlyChart>();
            line = tr.ReadLine();//The fist line
            while ((line = tr.ReadLine()) != null)
            {
                if (processedLines % 100 == 0)
                {
                    var logLine = string.Format("[Info][{0}] ================ Processed Lines: {1} ================", DateTime.Now.ToString("HH:mm:ss"), processedLines);
                    Console.WriteLine(logLine);
                }
                processedLines++;
                var items = line.Split('\t');
                PlotlyTable plotlyTable = PlotlyTable.GetInstance(items[0], dataFolder + @"\data_origin\");
                List<PlotlyChart> plotlyChart=PlotlyChart.GetInstance(items[0],plotlyTable, dataFolder + @"\data_origin\");
                if(plotlyChart.Count!=0)
                {
                    plotlyTables.Add(plotlyTable);
                    plotlyCharts = plotlyCharts.Concat(plotlyChart).ToList<PlotlyChart>();
                }
            }

            Console.WriteLine($"{processedLines} plotly tables loaded.");

            return new PlotlyList
            {
                PlotlyTables = plotlyTables,
                PlotlyCharts = plotlyCharts
            };
        }

        public static void ExtractForPlotlyTablesAll(string plotlyTableFolder)
        {
            Console.WriteLine($"Extracting PlotlyTable features from {plotlyTableFolder}.");

            PlotlyList plotlyList = LoadPlotlyTablesAll(plotlyTableFolder);
            List<PlotlyTable> plotlyTables = plotlyList.PlotlyTables;
            List<PlotlyChart> plotlyCharts = plotlyList.PlotlyCharts;

            // Initialize some models from MetadataRecoSvr to support the two Bayesian features.
            DataFeatureExtractor.InitializeMetadata("en");

            // Run features for each table and store embeddings for each plotly table.
            for (int i = 0; i < plotlyTables.Count; i++)
            {
                PlotlyTable plotlyTable = plotlyTables[i];
                Common.FileInfo plotlyTableInfo = DataFeatureExtractor.ExtractTableFeatures(plotlyTable,
                    out List<Dictionary<int, Dictionary<string, float[]>>> headerEmbeddings,
                    out SourceFeatures sf);
                string uid = plotlyTable.TUid.Substring(0, plotlyTable.TUid.Length - 3);
                Helpers.DumpJson($"{plotlyTable.TUid}.DF.json", sf, DataSerializer.Instance);
                Helpers.DumpJson($"{plotlyTable.TUid}.table.json", plotlyTable, DataSerializer.Instance);
                Helpers.DumpJson($"{uid}.EMB.json", headerEmbeddings, DataSerializer.Instance);
                Helpers.DumpJson($"{uid}.index.json", plotlyTableInfo, DataSerializer.Instance);
                Helpers.DumpCsv($"{plotlyTable.TUid}.csv", plotlyTable);
            }
            for(int i =0; i<plotlyCharts.Count; i++)
            {
                PlotlyChart plotlyChart = plotlyCharts[i];
                Helpers.DumpJson($"{plotlyChart.CUid}.json", plotlyChart, DataSerializer.Instance);
            }
        }
        
        public class PlotlyList
        {
            public List<PlotlyTable> PlotlyTables { get; set; }
            public List<PlotlyChart> PlotlyCharts { get; set; }
        }

        public static void ExtractForPlotlyTables(string fid)
        {
            Console.WriteLine(DateTime.Now);
            DataFeatureExtractor.InitializeMetadata("en");
            Console.WriteLine($"After InitializeMetadata: {DateTime.Now}");
            PlotlyTable plotlyTable = PlotlyTable.GetInstance(fid);
            List<PlotlyChart> plotlyCharts = PlotlyChart.GetInstance(fid, plotlyTable);
            Console.WriteLine($"After GetInstance: {DateTime.Now}");
            Common.FileInfo plotlyTableInfo = DataFeatureExtractor.ExtractTableFeatures(plotlyTable,
                    out List<Dictionary<int, Dictionary<string, float[]>>> headerEmbeddings,
                    out SourceFeatures sf);
            Console.WriteLine($"After FE: {DateTime.Now}");
            string uid = plotlyTable.TUid.Substring(0, plotlyTable.TUid.Length - 3);
            Helpers.DumpJson($"{plotlyTable.TUid}.DF.json", sf, DataSerializer.Instance);
            Helpers.DumpJson($"{plotlyTable.TUid}.table.json", plotlyTable, DataSerializer.Instance);
            Helpers.DumpJson($"{uid}.EMB.json", headerEmbeddings, DataSerializer.Instance);
            Helpers.DumpJson($"{uid}.index.json", plotlyTableInfo, DataSerializer.Instance);
            Helpers.DumpCsv($"{plotlyTable.TUid}.csv", plotlyTable);
            for (int i = 0; i < plotlyCharts.Count; i++)
            {
                PlotlyChart plotlyChart = plotlyCharts[i];
                Helpers.DumpJson($"{plotlyChart.CUid}.json", plotlyChart, DataSerializer.Instance);
            }
            Console.WriteLine($"After Dump: {DateTime.Now}");
        }
    }
}
