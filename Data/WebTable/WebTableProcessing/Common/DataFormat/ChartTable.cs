// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;

using Newtonsoft.Json;

namespace Common
{

    public enum FieldDataType
    {
        Unknown = 0,
        String = 1,
        DateTime = 3,
        Decimal = 5,
        Year = 7
    }

    /// <summary>
    /// Data structure for source tables of charts. There are some json-ignored properties
    ///   used when combining several source tables of one chart (i.e. an expansion
    ///   result and a TableFlow result). This class is used when chart-table matching.
    ///   Results will be converted to Table and stored in json files.
    /// Note that "row" and "column" properties refer to logic (pseudo) rows and columns,
    ///   that is, see fields as columns and entries as rows.
    /// </summary>
    [Serializable]
    public class ChartTable : Table
    {
        public List<List<uint>> NumFmtIds { get; set; }
        
        public List<FieldDataType> FieldTypes { get; set; }

    }
}
