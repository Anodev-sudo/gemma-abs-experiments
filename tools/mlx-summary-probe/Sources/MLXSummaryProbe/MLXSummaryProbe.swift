import Foundation
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

private let defaultModelDir = "hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx"
private let defaultPromptSpec = "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
private let defaultInputMode = "title_abstract"

private struct ProbeArguments {
    var modelDir = defaultModelDir
    var promptSpec = defaultPromptSpec
    var inputMode = defaultInputMode
    var example: ExampleFixture?
    var title: String?
    var abstract: String?
    var inputJSONL: String?
    var outputJSONL: String?
    var limit: Int?
    var skipCount = 0
    var warmupCount = 0
    var repeatCount = 1
    var json = false
}

private enum ExampleFixture: String {
    case cablecar

    var title: String {
        switch self {
        case .cablecar:
            "CAbLECAR: efficiently scheduling QLDPC codes on a tileable spin qubit chip with shuttling"
        }
    }

    var abstract: String {
        switch self {
        case .cablecar:
            """
            Semiconductor spin qubits are a promising platform for large-scale quantum computing, but have yet to take full advantage of the broad class of quantum low-density parity check (QLDPC) codes, which promise high encoding rates and efficient logic but require nonlocal connectivity between physical qubits. In this work, we investigate the implementation of QLDPC codes on a tileable, shuttling-based spin qubit architecture. By tailoring syndrome extraction circuits to the shuttling noise model, we significantly improve on previous surface code proposals and extend the feasible shuttling range of the architecture by 5-10x, enabling the implementation of more complex codes with long-range interactions. Taking inspiration from the field of robotics, we develop a coordinated shuttle scheduling algorithm that supports arbitrary codes and use it to benchmark the logical performance of a variety of promising code families. We find that the optimized schedules are up to 86% faster than hand-optimized schedules for certain code families.
            Through detailed circuit-level simulations, we identify specific QLDPC codes that improve upon prior surface code implementations by orders of magnitude, increasing encoding efficiency and reducing logical error rates. This work demonstrates the potential of shuttling-based spin qubit hardware platforms for scalable and efficient fault-tolerant quantum computation.
            """
        }
    }
}

private struct PromptSpec: Decodable {
    let promptVersion: String
    let systemPrompt: String
    let userPromptTemplates: [String: String]

    enum CodingKeys: String, CodingKey {
        case promptVersion = "prompt_version"
        case systemPrompt = "system_prompt"
        case userPromptTemplates = "user_prompt_templates"
    }
}

private struct ProbeResult: Encodable {
    let ok: Bool
    let modelDir: String
    let promptSpec: String
    let promptVersion: String?
    let inputMode: String
    let title: String?
    let loadTimeMs: Int?
    let warmupCount: Int
    let repeatCount: Int
    let generationTimeMs: Int?
    let generationTimesMs: [Int]
    let rawOutput: String?
    let normalizedOutput: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case modelDir = "model_dir"
        case promptSpec = "prompt_spec"
        case promptVersion = "prompt_version"
        case inputMode = "input_mode"
        case title
        case loadTimeMs = "load_time_ms"
        case warmupCount = "warmup_count"
        case repeatCount = "repeat_count"
        case generationTimeMs = "generation_time_ms"
        case generationTimesMs = "generation_times_ms"
        case rawOutput = "raw_output"
        case normalizedOutput = "normalized_output"
        case error
    }
}

private struct BatchInputRow: Decodable {
    let paperID: String
    let title: String
    let abstract: String
    let evalID: String?

    enum CodingKeys: String, CodingKey {
        case paperID = "paper_id"
        case title
        case abstract
        case evalID = "eval_id"
    }
}

private struct BatchOutputRow: Encodable {
    let ok: Bool
    let paperID: String
    let evalID: String?
    let title: String
    let rawOutput: String?
    let normalizedOutput: String?
    let generationTimeMs: Int?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case paperID = "paper_id"
        case evalID = "eval_id"
        case title
        case rawOutput = "raw_output"
        case normalizedOutput = "normalized_output"
        case generationTimeMs = "generation_time_ms"
        case error
    }
}

private struct BatchResult: Encodable {
    let ok: Bool
    let modelDir: String
    let promptSpec: String
    let promptVersion: String?
    let inputMode: String
    let inputJSONL: String
    let outputJSONL: String?
    let rowCount: Int
    let loadTimeMs: Int?
    let generationTimesMs: [Int]
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case modelDir = "model_dir"
        case promptSpec = "prompt_spec"
        case promptVersion = "prompt_version"
        case inputMode = "input_mode"
        case inputJSONL = "input_jsonl"
        case outputJSONL = "output_jsonl"
        case rowCount = "row_count"
        case loadTimeMs = "load_time_ms"
        case generationTimesMs = "generation_times_ms"
        case error
    }
}

@main
private enum MLXSummaryProbe {
    static func main() async {
        let arguments: ProbeArguments
        do {
            arguments = try parseArguments(Array(CommandLine.arguments.dropFirst()))
        } catch {
            fputs("\(error.localizedDescription)\n\n\(usage())\n", stderr)
            Foundation.exit(2)
        }

        if arguments.inputJSONL != nil {
            let result = await runBatchProbe(arguments)
            printBatchResult(result, asJSON: arguments.json)
            Foundation.exit(result.ok ? 0 : 1)
        } else {
            let result = await runProbe(arguments)
            printResult(result, asJSON: arguments.json)
            Foundation.exit(result.ok ? 0 : 1)
        }
    }
}

private func parseArguments(_ rawArguments: [String]) throws -> ProbeArguments {
    var arguments = ProbeArguments()
    var index = 0

    func value(after flag: String) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < rawArguments.count else {
            throw ProbeError.argument("Missing value for \(flag).")
        }
        return rawArguments[valueIndex]
    }

    while index < rawArguments.count {
        let argument = rawArguments[index]
        switch argument {
        case "--model-dir":
            arguments.modelDir = try value(after: argument)
            index += 2
        case "--prompt-spec":
            arguments.promptSpec = try value(after: argument)
            index += 2
        case "--input-mode":
            arguments.inputMode = try value(after: argument)
            index += 2
        case "--example":
            let rawExample = try value(after: argument)
            guard let example = ExampleFixture(rawValue: rawExample) else {
                throw ProbeError.argument("Unknown example '\(rawExample)'. Supported examples: cablecar.")
            }
            arguments.example = example
            index += 2
        case "--title":
            arguments.title = try value(after: argument)
            index += 2
        case "--abstract":
            arguments.abstract = try value(after: argument)
            index += 2
        case "--input-jsonl":
            arguments.inputJSONL = try value(after: argument)
            index += 2
        case "--output-jsonl":
            arguments.outputJSONL = try value(after: argument)
            index += 2
        case "--limit":
            arguments.limit = try parsePositiveInt(try value(after: argument), flag: argument)
            index += 2
        case "--skip":
            arguments.skipCount = try parseNonNegativeInt(try value(after: argument), flag: argument)
            index += 2
        case "--warmup":
            arguments.warmupCount = try parseNonNegativeInt(try value(after: argument), flag: argument)
            index += 2
        case "--repeat":
            arguments.repeatCount = try parsePositiveInt(try value(after: argument), flag: argument)
            index += 2
        case "--json":
            arguments.json = true
            index += 1
        case "--help", "-h":
            throw ProbeError.argument(usage())
        default:
            throw ProbeError.argument("Unknown argument '\(argument)'.")
        }
    }

    if let example = arguments.example {
        arguments.title = arguments.title ?? example.title
        arguments.abstract = arguments.abstract ?? example.abstract
    }

    if arguments.inputJSONL == nil {
        guard arguments.title?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            throw ProbeError.argument("Provide --example cablecar, both --title and --abstract, or --input-jsonl.")
        }
        guard arguments.abstract?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            throw ProbeError.argument("Provide --example cablecar, both --title and --abstract, or --input-jsonl.")
        }
    }

    return arguments
}

private func parseNonNegativeInt(_ rawValue: String, flag: String) throws -> Int {
    guard let value = Int(rawValue), value >= 0 else {
        throw ProbeError.argument("\(flag) must be a non-negative integer.")
    }
    return value
}

private func parsePositiveInt(_ rawValue: String, flag: String) throws -> Int {
    guard let value = Int(rawValue), value > 0 else {
        throw ProbeError.argument("\(flag) must be a positive integer.")
    }
    return value
}

private func runProbe(_ arguments: ProbeArguments) async -> ProbeResult {
    let modelURL = resolvePath(arguments.modelDir)
    let promptSpecURL = resolvePath(arguments.promptSpec)
    let title = arguments.title ?? ""
    let abstract = arguments.abstract ?? ""
    var promptVersion: String?

    do {
        guard FileManager.default.fileExists(atPath: modelURL.path(percentEncoded: false)) else {
            throw ProbeError.runtime("Model directory not found: \(arguments.modelDir)")
        }
        guard FileManager.default.fileExists(atPath: promptSpecURL.path(percentEncoded: false)) else {
            throw ProbeError.runtime("Prompt spec not found: \(arguments.promptSpec)")
        }

        let promptSpec = try loadPromptSpec(from: promptSpecURL)
        promptVersion = promptSpec.promptVersion
        let systemPrompt = promptSpec.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let userPrompt = try renderUserPrompt(
            promptSpec: promptSpec,
            inputMode: arguments.inputMode,
            title: title,
            abstract: abstract
        )

        let parameters = GenerateParameters(
            maxTokens: 64,
            temperature: 0,
            topP: 1,
            topK: 1
        )
        let loadStart = ContinuousClock.now
        let model = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        let loadTimeMs = milliseconds(from: loadStart)

        for _ in 0..<arguments.warmupCount {
            let session = ChatSession(
                model,
                instructions: systemPrompt,
                generateParameters: parameters
            )
            _ = try await session.respond(to: userPrompt)
        }

        var generationTimesMs: [Int] = []
        var rawOutput = ""
        for _ in 0..<arguments.repeatCount {
            let session = ChatSession(
                model,
                instructions: systemPrompt,
                generateParameters: parameters
            )
            let generationStart = ContinuousClock.now
            rawOutput = try await session.respond(to: userPrompt)
            generationTimesMs.append(milliseconds(from: generationStart))
        }
        let normalizedOutput = normalizeWhitespace(rawOutput)

        return ProbeResult(
            ok: !normalizedOutput.isEmpty,
            modelDir: arguments.modelDir,
            promptSpec: arguments.promptSpec,
            promptVersion: promptVersion,
            inputMode: arguments.inputMode,
            title: title,
            loadTimeMs: loadTimeMs,
            warmupCount: arguments.warmupCount,
            repeatCount: arguments.repeatCount,
            generationTimeMs: generationTimesMs.last,
            generationTimesMs: generationTimesMs,
            rawOutput: rawOutput,
            normalizedOutput: normalizedOutput,
            error: normalizedOutput.isEmpty ? "Model returned an empty response." : nil
        )
    } catch {
        return ProbeResult(
            ok: false,
            modelDir: arguments.modelDir,
            promptSpec: arguments.promptSpec,
            promptVersion: promptVersion,
            inputMode: arguments.inputMode,
            title: title.isEmpty ? nil : title,
            loadTimeMs: nil,
            warmupCount: arguments.warmupCount,
            repeatCount: arguments.repeatCount,
            generationTimeMs: nil,
            generationTimesMs: [],
            rawOutput: nil,
            normalizedOutput: nil,
            error: error.localizedDescription
        )
    }
}

private func runBatchProbe(_ arguments: ProbeArguments) async -> BatchResult {
    let modelURL = resolvePath(arguments.modelDir)
    let promptSpecURL = resolvePath(arguments.promptSpec)
    let inputJSONL = arguments.inputJSONL ?? ""
    let inputURL = resolvePath(inputJSONL)
    let outputURL = arguments.outputJSONL.map(resolvePath)
    var promptVersion: String?
    var outputHandle: FileHandle?
    var generationTimesMs: [Int] = []
    var rowCount = 0

    do {
        guard FileManager.default.fileExists(atPath: modelURL.path(percentEncoded: false)) else {
            throw ProbeError.runtime("Model directory not found: \(arguments.modelDir)")
        }
        guard FileManager.default.fileExists(atPath: promptSpecURL.path(percentEncoded: false)) else {
            throw ProbeError.runtime("Prompt spec not found: \(arguments.promptSpec)")
        }
        guard FileManager.default.fileExists(atPath: inputURL.path(percentEncoded: false)) else {
            throw ProbeError.runtime("Input JSONL not found: \(inputJSONL)")
        }

        let promptSpec = try loadPromptSpec(from: promptSpecURL)
        promptVersion = promptSpec.promptVersion
        let systemPrompt = promptSpec.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let rows = try loadBatchRows(
            from: inputURL,
            skip: arguments.skipCount,
            limit: arguments.limit
        )

        if let outputURL {
            let directory = outputURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: outputURL.path(percentEncoded: false), contents: nil)
            outputHandle = try FileHandle(forWritingTo: outputURL)
        }
        defer {
            try? outputHandle?.close()
        }

        let parameters = GenerateParameters(
            maxTokens: 64,
            temperature: 0,
            topP: 1,
            topK: 1
        )
        let loadStart = ContinuousClock.now
        let model = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        let loadTimeMs = milliseconds(from: loadStart)

        for row in rows {
            let outputRow: BatchOutputRow
            do {
                let userPrompt = try renderUserPrompt(
                    promptSpec: promptSpec,
                    inputMode: arguments.inputMode,
                    title: row.title,
                    abstract: row.abstract
                )
                let session = ChatSession(
                    model,
                    instructions: systemPrompt,
                    generateParameters: parameters
                )
                let generationStart = ContinuousClock.now
                let rawOutput = try await session.respond(to: userPrompt)
                let generationTimeMs = milliseconds(from: generationStart)
                let normalizedOutput = normalizeWhitespace(rawOutput)
                generationTimesMs.append(generationTimeMs)
                outputRow = BatchOutputRow(
                    ok: !normalizedOutput.isEmpty,
                    paperID: row.paperID,
                    evalID: row.evalID,
                    title: row.title,
                    rawOutput: rawOutput,
                    normalizedOutput: normalizedOutput,
                    generationTimeMs: generationTimeMs,
                    error: normalizedOutput.isEmpty ? "Model returned an empty response." : nil
                )
            } catch {
                outputRow = BatchOutputRow(
                    ok: false,
                    paperID: row.paperID,
                    evalID: row.evalID,
                    title: row.title,
                    rawOutput: nil,
                    normalizedOutput: nil,
                    generationTimeMs: nil,
                    error: error.localizedDescription
                )
            }
            rowCount += 1
            try writeBatchOutput(outputRow, to: outputHandle)
            if outputHandle == nil {
                printJSONLine(outputRow)
            }
            if rowCount % 10 == 0 {
                fputs("[mlx-summary-probe] generated \(rowCount)/\(rows.count) rows\n", stderr)
            }
        }

        return BatchResult(
            ok: true,
            modelDir: arguments.modelDir,
            promptSpec: arguments.promptSpec,
            promptVersion: promptVersion,
            inputMode: arguments.inputMode,
            inputJSONL: inputJSONL,
            outputJSONL: arguments.outputJSONL,
            rowCount: rowCount,
            loadTimeMs: loadTimeMs,
            generationTimesMs: generationTimesMs,
            error: nil
        )
    } catch {
        return BatchResult(
            ok: false,
            modelDir: arguments.modelDir,
            promptSpec: arguments.promptSpec,
            promptVersion: promptVersion,
            inputMode: arguments.inputMode,
            inputJSONL: inputJSONL,
            outputJSONL: arguments.outputJSONL,
            rowCount: rowCount,
            loadTimeMs: nil,
            generationTimesMs: generationTimesMs,
            error: error.localizedDescription
        )
    }
}

private func loadBatchRows(from url: URL, skip: Int, limit: Int?) throws -> [BatchInputRow] {
    let text = try String(contentsOf: url, encoding: .utf8)
    let decoder = JSONDecoder()
    var rows: [BatchInputRow] = []
    for (index, rawLine) in text.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
        if index < skip {
            continue
        }
        if let limit, rows.count >= limit {
            break
        }
        guard let data = String(rawLine).data(using: .utf8) else {
            throw ProbeError.runtime("Failed to encode input JSONL line.")
        }
        rows.append(try decoder.decode(BatchInputRow.self, from: data))
    }
    return rows
}

private func writeBatchOutput(_ row: BatchOutputRow, to handle: FileHandle?) throws {
    guard let handle else {
        return
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(row)
    handle.write(data)
    handle.write(Data("\n".utf8))
}

private func printJSONLine<T: Encodable>(_ value: T) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    if let data = try? encoder.encode(value),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}

private func loadPromptSpec(from url: URL) throws -> PromptSpec {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(PromptSpec.self, from: data)
}

private func printBatchResult(_ result: BatchResult, asJSON: Bool) {
    if asJSON {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(result),
           let text = String(data: data, encoding: .utf8) {
            print(text)
        } else {
            print("{\"ok\":false,\"error\":\"Failed to encode batch result.\"}")
        }
        return
    }

    if result.ok {
        print("ok: true")
        print("model_dir: \(result.modelDir)")
        if let promptVersion = result.promptVersion {
            print("prompt_version: \(promptVersion)")
        }
        if let loadTimeMs = result.loadTimeMs {
            print("load_time_ms: \(loadTimeMs)")
        }
        print("row_count: \(result.rowCount)")
        if !result.generationTimesMs.isEmpty {
            print("generation_times_ms: \(result.generationTimesMs)")
        }
    } else {
        print("ok: false")
        print("error: \(result.error ?? "Unknown error.")")
    }
}

private func renderUserPrompt(
    promptSpec: PromptSpec,
    inputMode: String,
    title: String,
    abstract: String
) throws -> String {
    guard let template = promptSpec.userPromptTemplates[inputMode] else {
        let modes = promptSpec.userPromptTemplates.keys.sorted().joined(separator: ", ")
        throw ProbeError.runtime("Input mode '\(inputMode)' not found in prompt spec. Available modes: \(modes).")
    }

    return template
        .replacingOccurrences(of: "{title}", with: title.trimmingCharacters(in: .whitespacesAndNewlines))
        .replacingOccurrences(of: "{abstract}", with: abstract.trimmingCharacters(in: .whitespacesAndNewlines))
}

private func resolvePath(_ path: String) -> URL {
    let expanded = NSString(string: path).expandingTildeInPath
    if expanded.hasPrefix("/") {
        return URL(fileURLWithPath: expanded)
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: expanded)
}

private func normalizeWhitespace(_ value: String) -> String {
    value
        .split(whereSeparator: { $0.isWhitespace })
        .joined(separator: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private func milliseconds(from start: ContinuousClock.Instant) -> Int {
    let duration = start.duration(to: .now)
    return Int(duration.components.seconds * 1000 + duration.components.attoseconds / 1_000_000_000_000_000)
}

private func printResult(_ result: ProbeResult, asJSON: Bool) {
    if asJSON {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(result),
           let text = String(data: data, encoding: .utf8) {
            print(text)
        } else {
            print("{\"ok\":false,\"error\":\"Failed to encode probe result.\"}")
        }
        return
    }

    if result.ok {
        print("ok: true")
        print("model_dir: \(result.modelDir)")
        if let promptVersion = result.promptVersion {
            print("prompt_version: \(promptVersion)")
        }
        if let loadTimeMs = result.loadTimeMs {
            print("load_time_ms: \(loadTimeMs)")
        }
        print("warmup_count: \(result.warmupCount)")
        print("repeat_count: \(result.repeatCount)")
        if let generationTimeMs = result.generationTimeMs {
            print("generation_time_ms: \(generationTimeMs)")
        }
        if !result.generationTimesMs.isEmpty {
            print("generation_times_ms: \(result.generationTimesMs)")
        }
        print("normalized_output: \(result.normalizedOutput ?? "")")
    } else {
        print("ok: false")
        print("error: \(result.error ?? "Unknown error.")")
    }
}

private func usage() -> String {
    """
    Usage:
      mlx-summary-probe --example cablecar [--json]
      mlx-summary-probe --title <title> --abstract <abstract> [--json]
      mlx-summary-probe --input-jsonl <rows.jsonl> [--output-jsonl <outputs.jsonl>] [--json]

    Options:
      --model-dir <path>      Local MLX model directory.
      --prompt-spec <path>    Prompt spec JSON.
      --input-mode <mode>     Prompt input mode. Default: title_abstract.
      --example cablecar      Run the CAbLECAR deployment-failure fixture.
      --input-jsonl <path>    Generate for JSONL rows with paper_id, title, and abstract.
      --output-jsonl <path>   Write batch generation rows to this JSONL path.
      --limit <count>         Limit batch generation to the first count rows.
      --skip <count>          Skip count rows before batch generation.
      --warmup <count>        Warmup generations after load. Default: 0.
      --repeat <count>        Timed generations after warmup. Default: 1.
      --json                  Print a compact JSON result.
    """
}

private enum ProbeError: LocalizedError {
    case argument(String)
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case let .argument(message), let .runtime(message):
            message
        }
    }
}
