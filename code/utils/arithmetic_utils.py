"""
This file contains the code for parsing the output of the model for the base arithmetic problems.
Copied from https://github.com/ZhaofengWu/counterfactual-evaluation/blob/master/
"""

import re
import hashlib
import numpy as np

def get_label(expr, base):
    lhs, rhs = expr.split("+")
    lhs_base10 = int(lhs, base)
    rhs_base10 = int(rhs, base) 
    sum_base10 = lhs_base10 + rhs_base10
    return np.base_repr(sum_base10, base)


def unescape(str):
    placeholder = "<TMP>"
    assert placeholder not in str
    return str.replace("\\\\n", placeholder).replace("\\n", "\n").replace(placeholder, "\\n").replace("\\\\r", placeholder).replace("\\r", "\r").replace(placeholder, "\\r")


def parse_output(output):
    if len(output) == 0:
        return "FAILED"

    output_hash = hashlib.md5(output.encode("utf-8")).hexdigest()
    if output_hash in {"a7994fde4fba7d27500e6f03008abd7c"}:
        return "FAILED"

    output = output.replace(",", "").replace("С", "C")

    if (match := re.search("^[0-9A-Z]+$", output)) is not None:
        return output

    output = output.rstrip("\n$ `")
    if output.endswith("\n"):
        output = output[:-1]
    output = output.replace("\\text{", "")

    boxed_regex = r"boxed{(\\text{)?(result=)?([0-9A-Z]+(_{?[0-9]+}?)?\s*\+\s*[0-9A-Z]+(_{?[0-9]+}?)?\s*=\s*)?(0x)?([0-9A-Za-f \\.]+)(_ ?{?(base-)?([0-9]+|ten)}?)?}?(_{?([0-9]+|ten)}?)?}"
    get_result_from_boxed_regex = lambda match: match[-6].replace(" ", "").replace("\\", "")
    # match all \boxed{...} but also make sure there's only one match
    match = re.findall(boxed_regex, output)
    if len(match) >= 1 and all(get_result_from_boxed_regex(m) == get_result_from_boxed_regex(match[0]) for m in match):
        return get_result_from_boxed_regex(match[0])

    last_line = output.split("\n")[-1]
    match = re.findall(boxed_regex, last_line)
    if len(match) >= 1 and all(get_result_from_boxed_regex(m) == get_result_from_boxed_regex(match[0]) for m in match):
        return get_result_from_boxed_regex(match[0])

    last_line = output.rstrip(" .").split(".")[-1]
    match = re.findall(boxed_regex, last_line)
    if len(match) >= 1 and all(get_result_from_boxed_regex(m) == get_result_from_boxed_regex(match[0]) for m in match):
        return get_result_from_boxed_regex(match[0])

    if (match := re.search(r"\\boxed{[0-9A-Z]+}(_{?[0-9]+}?)?\s*\+\s*\\boxed{[0-9A-Z]+}(_{?[0-9]+}?)?\s*=\s*\\boxed{([0-9A-Z]+)}(_{?[0-9]+}?)?\.?$", last_line)) is not None:
        return match.groups()[-2]

    if (match := re.search(r"\\boxed{([0-9A-Z]+)_{?[0-9]+}?\s*=\s*[0-9A-Z]+_{?10}?}\$?\.?$", last_line)) is not None:
        return match.groups()[0]

    if (match := re.search(r"\$?[0-9A-Z]+(_{?[0-9]+}?)\s*\+\s*[0-9A-Z]+(_{?[0-9]+}?)\s*=\s*(0x)?([0-9A-Z]+)(_{?[0-9]+}?)\$?( in base-[0-9]+)?\.?$", output)) is not None:
        return match.groups()[-3]

    if (match := re.search(r"(=|is):?\s*\$?\\boxed{(0x)?([0-9A-Z]+)}\$? \(?in base-[0-9]+\)?,?( and| or| =) \$?\\boxed{(0x)?[0-9A-Z]+}\$? \(?in (base-10|decimal)\)?\.?$", output)) is not None:
        return match.groups()[2]
    if (match := re.search(r"(=|is):?\s*\$?\\boxed{(0x)?[0-9A-Z]+}\$? \(?in (base-10|decimal)\)?,?( and| or| =) \$?\\boxed{(0x)?([0-9A-Z]+)}\$? \(?in base-[0-9]+\)?\.?$", output)) is not None:
        return match.groups()[-1]
    # \boxed{207}_{10}$ which in base-11 is $\boxed{18A}$.
    if (match := re.search(r"\\boxed{[0-9A-Z]+}_\{10\}\$? which in base-[0-9]+ is \$?\\boxed{(0x)?([0-9A-Z]+)}\$?\.?$", output)) is not None:
        return match.groups()[-1]
    # 39 + 31 = 5A\boxed{}
    if (match := re.search(r"[0-9]+\s*\+\s*[0-9]+\s*=\s*([0-9A-Z]+)\\boxed\{\}\.?$", output)) is not None:
        return match.groups()[-1]

    # \boxed{result}\n62
    if (match := re.search(r"\\boxed{result}\s*(\n|=)?\s*([0-9A-Z+*^. ]+=\s*)?([0-9A-Z.]+)\$?\.?\**}?$", output)) is not None:
        return match.groups()[-1]

    # \boxed{result: 62}
    if (match := re.search(r"\\boxed{result: ([0-9A-Z]+)}$", output)) is not None:
        return match.groups()[0]

    match = re.findall(r"[0-9A-Z]+\s*\+\s*[0-9A-Z]+\s*=\s*(0x)?([0-9A-Z]+)", last_line)
    if len(match) == 1:
        return match[0][1]

    match_after_semicolon = r"\s+((\n|[ 0-9A-Z*^])+(\+(\n|[ 0-9A-Z*^])+)+(=|-+|_+)\s*)*([0-9A-Z]+)\s*(\(?(in )?base-[0-9]+\)?)?(, which [^,.]+)?(\s*\([^()]+\))?\.?$"
    if (match := re.search(r"\n([0-9A-Z]+)$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r" in base-[0-9]+ is (equal to )?\"?(0x)?([0-9A-Z]+)\"?( base-[0-9]+)?(, (or|since) [^.]+)?( \([^()]+\))?\.$", output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r" in base-[0-9]+: \$?([0-9A-Z]+)\$?\.$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r" the base-[0-9]+ sum: ([0-9A-Z]+)\.$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"the result in base-[0-9]+ is ([0-9A-Z]+), which is equal to [0-9 *^+()]+\.$", output)) is not None:
        return match[1]
    if (match := re.search(r"the sum of [0-9A-Z]+ and [0-9A-Z]+ (in base-[0-9]+ )?(is|as):?" + match_after_semicolon, output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r"the result of [0-9A-Z]+\s*\+\s*[0-9A-Z]+ (in base-[0-9]+ )?(is|as):?" + match_after_semicolon, output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r"[0-9A-Z]+\s*\+\s*[0-9A-Z]+( in base-[0-9]+)?,? (which )?(equals|is equal to|as):? \$?([0-9A-Z]+)\$?(, written as [0-9A-Z]+)?\.?$", output)) is not None:
        return match.groups()[-2]
    if (match := re.search(r"in base-10 is \$?[0-9]+\$?,? (which )?(equals|is equal to|as):? \$?([0-9A-Z]+)\$?(, written as [0-9A-Z]+)?\.?$", output)) is not None:
        return match.groups()[-2]
    if (match := re.search(r"[0-9A-Z]+\s*\+\s*[0-9A-Z]+\s*=\s*([0-9A-Z]+)( in base-[0-9]+)?\.?$", output)) is not None:
        return match.groups()[-2]
    if (match := re.search(r"we can simply write the result as ([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"which can be written as ([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"(which gives|giving) us the( base-[0-9]+)? number ([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"the final result is simply the sum of the tens and ones places: ([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"the result is simply the combination of these two sums: ([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"we have ([0-9A-Z]+) in base-[0-9]+ as the (final answer for|result of|sum of) [0-9A-Z]+ (\+|and) [0-9A-Z]+\.$", output)) is not None:
        return match[1]
    if (match := re.search(r"we (have|get|end up with) ([0-9A-Z]+)( in base-[0-9]+)? as the( final)? (result|answer|sum)( in base-[0-9]+)?\.$", output)) is not None:
        return match.groups()[1]
    if (match := re.search(r"(=| is) \"?([0-9A-Z]+)\"?\s*(\s+\(?(in )?base-[0-9]+\)?)?\.?$", output)) is not None:
        return match.groups()[1]
    if (match := re.search(r"( final)?( base-[0-9]+)? (result|answer|sum)( in base-[0-9]+)?( is)?( simply)?( of)?( as)?:?" + match_after_semicolon, output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r"we get:" + match_after_semicolon, output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r"we can add the two numbers in base-[0-9]+:" + match_after_semicolon, output)) is not None:
        return match.groups()[-5]
    if (match := re.search(r"[tT]he combination of these sums:\s+([0-9A-Z]+)(\(in base-[0-9]+\))?\.?$", output)) is not None:
        return match.groups()[-2]
    if (match := re.search(r"(Result|Answer)( is)?:?\s+([0-9A-Z]+)\.?$", output)) is not None:
        return match.groups()[-1]
    if (match := re.search(r"The decimal equivalent of \$?([0-9A-Z]+)\$? is therefore \$?[0-9A-Z]+\$?\.?$", output)) is not None:
        return match.groups()[0]
    if (match := re.search(r"(T|t)he final (result|answer) is:?\s+([0-9A-Z ]+\s*\+\s*[0-9A-Z ]+\s*(=|-+)+\s*)?([0-9A-Z ]+)(\(in base-[0-9]+\))?\.?\**$", output)) is not None:
        return match.groups()[-2].replace(" ", "")
    if (match := re.search(r" in base-[0-9]+ is (equal to )?\"?(0x)?([0-9A-Z ]+)\"?(, or [^,.]+)?\.$", output)) is not None:
        return match.groups()[-2].replace(" ", "")
    if (match := re.search(r"( |(\n))([0-9A-Z]+) \(?in base-[0-9]+\)?\.$", output)) is not None:
        return match.groups()[-1]

    #print("Failed to parse output:", output)
    #print(output_hash)
    return "FAILED"