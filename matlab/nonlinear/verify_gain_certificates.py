"""Exactly verify declared decimal bounded-real certificates, not the whole simulator."""
import argparse
from fractions import Fraction
import json
from pathlib import Path


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def multiply(left, right):
    return [[sum(a * b for a, b in zip(row, column)) for column in zip(*right)] for row in left]


def add(*matrices):
    return [[sum(matrix[i][j] for matrix in matrices) for j in range(len(matrices[0][0]))]
            for i in range(len(matrices[0]))]


def determinant(matrix):
    matrix = [row[:] for row in matrix]
    result = Fraction(1)
    for column in range(len(matrix)):
        pivot = next((row for row in range(column, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            result = -result
        value = matrix[column][column]
        result *= value
        for row in range(column + 1, len(matrix)):
            ratio = matrix[row][column] / value
            for index in range(column + 1, len(matrix)):
                matrix[row][index] -= ratio * matrix[column][index]
    return result


def positive_minors(matrix):
    assert matrix == transpose(matrix), "The declared matrix is not exactly symmetric."
    minors = [determinant([row[:size] for row in matrix[:size]]) for size in range(1, len(matrix) + 1)]
    return all(value > 0 for value in minors), [str(value) for value in minors]


def verify(source, destination):
    data = json.loads(source.read_text(encoding="utf-8"))
    results = []
    for entry in data["gainCertificates"]:
        convert = lambda matrix: [[Fraction(value) for value in row] for row in matrix]
        state = convert(entry["ADecimal"])
        storage = convert(entry["XDecimal"])
        channel = [[Fraction(value)] for value in entry["BDecimal"]]
        output = [[Fraction(value) for value in entry["CDecimal"]]]
        gamma = Fraction(entry["gammaDecimal"])
        top = add(multiply(transpose(state), storage), multiply(storage, state),
                  multiply(transpose(output), output))
        cross = multiply(storage, channel)
        lmi = [row + cross[index] for index, row in enumerate(top)]
        lmi.append(transpose(cross)[0] + [-gamma * gamma])
        positive_x, x_minors = positive_minors(storage)
        negative_lmi, lmi_minors = positive_minors([[-value for value in row] for row in lmi])
        assert positive_x and negative_lmi, f"Certificate {entry['outputIndex']}/{entry['inputIndex']} failed."
        results.append({"output": entry["outputIndex"], "input": entry["inputIndex"],
                        "gamma": str(gamma), "X_leading_minors": x_minors,
                        "negative_LMI_leading_minors": lmi_minors, "exactly_verified": True})
    report = {"scope": "Exact rational bounded-real LMIs for the declared rounded coefficients; not a full software guarantee",
              "verified_count": len(results), "certificates": results}
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified_count": len(results), "all_exact": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    verify(arguments.source, arguments.destination)
