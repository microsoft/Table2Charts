// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    /// <summary>
    /// Cell structure for general table.
    /// </summary>
    public interface IGeneralCell
    {
        /// <summary>
        /// Cell string.
        /// </summary>
        string Text { get; }

        /// <summary>
        /// Value for this cell.
        /// If the cell text can be interpreted as a number, then Value is a number. Othervise, Value is a string.
        /// </summary>
        object Value { get; }

        /// <summary>
        /// Data format information.
        /// </summary>
        DataFormatFlags DataFormatFlags { get; }
    }

    public enum DataFormatFlags : int
    {
        None = 0,
        DateTime = 1,
        Percent = 2,
        Currency = 4,
        Numeric = 8,
        Text = 16,
        Day = 32,
        Month = 64,
        Year = 128,
        Sequence = 256,
        Ordinal = 512
    }
}
