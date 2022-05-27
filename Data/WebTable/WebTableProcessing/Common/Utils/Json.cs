// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.IO;
using System.Text;

using Newtonsoft.Json;

namespace Common
{
    public partial class Helpers
    {
        public static string DumpJsonToString(object o, JsonSerializer serializer)
        {
            MemoryStream ms = new MemoryStream();
            using (JsonWriter writer = new JsonTextWriter(new StreamWriter(ms, Encoding.GetEncoding("UTF-8"))))
            {
                serializer.Serialize(writer, o);
            }
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        public static void DumpJson(string fileName, object o, JsonSerializer serializer)
        {
            using (JsonWriter writer = new JsonTextWriter(new StreamWriter(fileName, false, Encoding.GetEncoding("UTF-8"))))
            {
                serializer.Serialize(writer, o);
            }
        }

        public static T LoadJson<T>(string fileName, JsonSerializer serializer)
        {
            using (StreamReader file = File.OpenText(fileName))
            {
                return (T)serializer.Deserialize(file, typeof(T));
            }
        }

    }
}
