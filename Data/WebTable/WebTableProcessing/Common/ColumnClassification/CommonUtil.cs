using Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ColumnClassification.Contract;
using Newtonsoft.Json;

namespace ColumnClassification.Utils
{
    public class CommonUtil
    {

        public static bool checkDateString(string input, bool includeX = false)
        {
            bool returnFlag = true;
            foreach (char character in input)
            {
                if ((character >= '0' && character <= '9') || (character == '-') || character == 'T' || character == ':' || (includeX && character == 'X')) continue;
                returnFlag = false;
                break;
            }
            return returnFlag;
        }
    }
}
