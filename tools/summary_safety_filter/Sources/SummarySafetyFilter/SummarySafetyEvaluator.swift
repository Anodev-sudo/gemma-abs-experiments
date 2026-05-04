import Foundation

public struct SummarySafetyEvaluation: Sendable, Equatable {
    public let status: String
    public let reason: String?
    public let displaySummary: String

    public var isAllowed: Bool {
        status == "allowed"
    }
}

public struct SummarySafetyEvaluator: Sendable {
    private let filter = SummaryOutputSafetyFilter()

    public init() {}

    public static var filterVersion: String {
        SummaryOutputSafetyFilter.version
    }

    public func evaluate(summary: String, arxivID: String, title: String, abstract: String) -> SummarySafetyEvaluation {
        let paper = Paper(arxivID: arxivID, title: title, abstract: abstract)
        switch filter.evaluate(output: summary, for: paper) {
        case .allowed(let displayText):
            return SummarySafetyEvaluation(
                status: "allowed",
                reason: nil,
                displaySummary: displayText
            )
        case .suppressed(let reason):
            return SummarySafetyEvaluation(
                status: "suppressed",
                reason: reason.rawValue,
                displaySummary: summary.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
    }
}
