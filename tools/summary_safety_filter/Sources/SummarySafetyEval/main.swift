import Foundation
import Darwin
import SummarySafetyFilter

enum CLIError: Error, CustomStringConvertible {
    case missingArgument(String)
    case invalidJSONLine(Int)
    case missingRequiredField(Int, String)

    var description: String {
        switch self {
        case .missingArgument(let name):
            return "Missing required argument \(name)"
        case .invalidJSONLine(let line):
            return "Invalid JSON object on line \(line)"
        case .missingRequiredField(let line, let field):
            return "Missing required field \(field) on line \(line)"
        }
    }
}

struct Arguments {
    let generationsJSONL: String
    let outputDir: String
}

func parseArguments(_ arguments: [String]) throws -> Arguments {
    var generationsJSONL: String?
    var outputDir: String?
    var index = 1
    while index < arguments.count {
        switch arguments[index] {
        case "--generations-jsonl":
            guard index + 1 < arguments.count else { throw CLIError.missingArgument("--generations-jsonl") }
            generationsJSONL = arguments[index + 1]
            index += 2
        case "--output-dir":
            guard index + 1 < arguments.count else { throw CLIError.missingArgument("--output-dir") }
            outputDir = arguments[index + 1]
            index += 2
        case "--help", "-h":
            print("Usage: summary-safety-eval --generations-jsonl <path> --output-dir <dir>")
            exit(0)
        default:
            throw CLIError.missingArgument(arguments[index])
        }
    }
    guard let generationsJSONL else { throw CLIError.missingArgument("--generations-jsonl") }
    guard let outputDir else { throw CLIError.missingArgument("--output-dir") }
    return Arguments(generationsJSONL: generationsJSONL, outputDir: outputDir)
}

func stringField(_ row: [String: Any], _ key: String, lineNumber: Int) throws -> String {
    if let value = row[key] as? String {
        return value
    }
    throw CLIError.missingRequiredField(lineNumber, key)
}

func optionalStringField(_ row: [String: Any], _ key: String) -> String? {
    row[key] as? String
}

func jsonLine(_ object: [String: Any]) throws -> Data {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    var output = data
    output.append(0x0A)
    return output
}

func increment(_ dictionary: inout [String: Int], _ key: String) {
    dictionary[key, default: 0] += 1
}

func incrementNested(_ dictionary: inout [String: [String: Int]], _ outer: String, _ inner: String) {
    var bucket = dictionary[outer, default: [:]]
    bucket[inner, default: 0] += 1
    dictionary[outer] = bucket
}

func openFileForWriting(_ url: URL) throws -> FileHandle {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    return try FileHandle(forWritingTo: url)
}

func run() throws {
    let args = try parseArguments(CommandLine.arguments)
    let inputURL = URL(fileURLWithPath: args.generationsJSONL)
    let outputURL = URL(fileURLWithPath: args.outputDir, isDirectory: true)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

    let decisionsURL = outputURL.appendingPathComponent("filter_decisions.jsonl")
    let allowedURL = outputURL.appendingPathComponent("allowed_generations.jsonl")
    let suppressedURL = outputURL.appendingPathComponent("suppressed_generations.jsonl")
    let summaryURL = outputURL.appendingPathComponent("filter_summary.json")

    let decisionsHandle = try openFileForWriting(decisionsURL)
    let allowedHandle = try openFileForWriting(allowedURL)
    let suppressedHandle = try openFileForWriting(suppressedURL)
    defer {
        try? decisionsHandle.close()
        try? allowedHandle.close()
        try? suppressedHandle.close()
    }

    let evaluator = SummarySafetyEvaluator()
    var rowCount = 0
    var byStatus: [String: Int] = [:]
    var byReason: [String: Int] = [:]
    var bySplitStatus: [String: [String: Int]] = [:]
    var bySplitReason: [String: [String: Int]] = [:]

    let contents = try String(contentsOf: inputURL, encoding: .utf8)
    for (offset, rawLine) in contents.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
        let lineNumber = offset + 1
        let data = Data(rawLine.utf8)
        guard let row = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CLIError.invalidJSONLine(lineNumber)
        }
        let paperID = try stringField(row, "paper_id", lineNumber: lineNumber)
        let title = try stringField(row, "title", lineNumber: lineNumber)
        let abstract = try stringField(row, "abstract", lineNumber: lineNumber)
        let summary = try stringField(row, "summary", lineNumber: lineNumber)
        let split = optionalStringField(row, "split") ?? "unknown"
        let evalID = optionalStringField(row, "eval_id") ?? "row_\(lineNumber)"

        let evaluation = evaluator.evaluate(
            summary: summary,
            arxivID: paperID,
            title: title,
            abstract: abstract
        )

        rowCount += 1
        increment(&byStatus, evaluation.status)
        incrementNested(&bySplitStatus, split, evaluation.status)
        if let reason = evaluation.reason {
            increment(&byReason, reason)
            incrementNested(&bySplitReason, split, reason)
        }

        var decisionRow: [String: Any] = [
            "eval_id": evalID,
            "paper_id": paperID,
            "split": split,
            "status": evaluation.status,
            "summary": evaluation.displaySummary,
            "title": title,
        ]
        if let reason = evaluation.reason {
            decisionRow["reason"] = reason
        }
        if let topics = row["edge_topics"] {
            decisionRow["edge_topics"] = topics
        }
        if let variant = row["adversarial_variant"] {
            decisionRow["adversarial_variant"] = variant
        }
        try decisionsHandle.write(contentsOf: jsonLine(decisionRow))

        var outputRow = row
        outputRow["filter_status"] = evaluation.status
        outputRow["filter_reason"] = evaluation.reason ?? NSNull()
        outputRow["filter_summary"] = evaluation.displaySummary
        if evaluation.isAllowed {
            try allowedHandle.write(contentsOf: jsonLine(outputRow))
        } else {
            try suppressedHandle.write(contentsOf: jsonLine(outputRow))
        }
    }

    let summary: [String: Any] = [
        "input": args.generationsJSONL,
        "filter_version": SummarySafetyEvaluator.filterVersion,
        "row_count": rowCount,
        "by_status": byStatus,
        "by_reason": byReason,
        "by_split_status": bySplitStatus,
        "by_split_reason": bySplitReason,
        "outputs": [
            "decisions_jsonl": decisionsURL.lastPathComponent,
            "allowed_generations_jsonl": allowedURL.lastPathComponent,
            "suppressed_generations_jsonl": suppressedURL.lastPathComponent,
            "summary_json": summaryURL.lastPathComponent,
        ],
    ]
    let summaryData = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    try summaryData.write(to: summaryURL)
    print(String(data: summaryData, encoding: .utf8) ?? "")
}

do {
    try run()
} catch {
    fputs("\(error)\n", stderr)
    exit(1)
}
