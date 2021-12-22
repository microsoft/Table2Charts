using System;
using System.Collections.Generic;

using DocumentFormat.OpenXml;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Common
{
    /// <summary>
    /// See https://docs.microsoft.com/en-us/dotnet/api/documentformat.openxml.spreadsheet.pivotcacherecord?view=openxml-2.8.1
    /// and https://docs.microsoft.com/en-us/dotnet/api/documentformat.openxml.spreadsheet.shareditems?view=openxml-2.8.1
    /// </summary>
    [Serializable]
    public class Cell : IEquatable<Cell>
    {
        public string Type { get; set; }
        public string Value { get; set; }

        public Cell()
        {
        }

        public Cell(Cell other)
        {
            Type = other.Type;
            Value = other.Value;
        }

        // See ECMA-376 Section 18.18.11
        public bool IsString()
        {
            return Type == "s" || Type == "str" || Type == "inlineStr" || Type == "e";
        }
        public bool IsNumeric()
        {
            return Type == "b" || Type == "d" || Type == "n";
        }

        public object RawValue()
        {
            switch (Type)
            {
                case "d": // Date Time
                    if (DateTime.TryParse(Value, out DateTime dateTimeValue))
                        return dateTimeValue.ToOADate();
                    else
                        throw new ArgumentException($"Could not parse {Value} as date time.");
                case "m": // No Value
                    // TODO: check if return null or empty string is better for metadata service
                    return "";
                case "b": // Boolean
                          // Currently we treat bool value as number
                case "n": // Numeric
                case null: // TODO: check why there're no other type except for "s"/"str" in sheetdata
                    if (int.TryParse(Value, out int intValue))
                        return intValue;
                    else if (double.TryParse(Value, out double doubleValue))
                        return doubleValue;
                    else
                    {
                        // TODO: investigate why there are ("n", "-")
                        Console.WriteLine($"Could not parse {Value} as numeric.");
                        return Value;
                    }
                case "e": // Error Value
                case "s": // Character Value
                case "str":
                case "inlineStr":
                    return Value;
            }
            throw new ArgumentException($"Unexpected type {Type}.");
        }

        public Cell ActualRecord(SharedItems sharedItems)
        {
            if (Type == "x")
                return sharedItems.SharedCells[int.Parse(Value)];
            else
                return this;
        }

        public Cell SimplifyRecord(SharedItems sharedItems)
        {
            int idx = sharedItems.SharedCells.IndexOf(this);
            if (idx != -1)
            {
                Type = "x";
                Value = idx.ToString();
            }
            return this;
        }

        /// <summary>
        /// This function is only used for CacheSource and PivotTable cells.
        /// </summary>
        public static Cell GetInstance(OpenXmlElement element)
        {
            string type = element.LocalName;
            string value = null;
            var attributes = element.GetAttributes();
            if (type != "m" && attributes != null && attributes.Count > 0)
            {
                // TODO: sometimes there is a "u" attribute after "v"
                if (element.GetAttributes()[0].LocalName != "v")
                    throw new FormatException("No value attribute in the element.");
                value = element.GetAttributes()[0].Value;
            }

            return new Cell()
            {
                Type = type,
                Value = value
            };
        }

        public bool Equals(Cell other)
        {
            return other != null
                && IsNumeric() == other.IsNumeric()
                && IsString() == other.IsString()
                && string.Equals(Value, other.Value);
        }

        public override int GetHashCode()
        {
            var hashCode = 1265339359;
            hashCode = hashCode * -1521134295 + Type.GetHashCode();
            hashCode = hashCode * -1521134295 + Value.GetHashCode();
            return hashCode;
        }

        public long GetSchemaHash()
        {
            long hashCode = 1265339359L;
            hashCode = hashCode * -1521134295L + Type.GetHashCode();
            hashCode = hashCode * -1521134295L + Value.GetHashCode();
            return hashCode;
        }

        public class Converter : JsonConverter<Cell>
        {
            public override Cell ReadJson(JsonReader reader, Type objectType, Cell existingValue, bool hasExistingValue, JsonSerializer serializer)
            {
                List<string> pair = JToken.Load(reader).ToObject<List<string>>(serializer);
                return new Cell()
                {
                    Type = pair[0],
                    Value = pair[1]
                };
            }

            public override void WriteJson(JsonWriter writer, Cell value, JsonSerializer serializer)
            {
                writer.WriteStartArray();
                writer.WriteValue(value.Type);
                writer.WriteValue(value.Value);
                writer.WriteEndArray();
            }
        }
    }
}
