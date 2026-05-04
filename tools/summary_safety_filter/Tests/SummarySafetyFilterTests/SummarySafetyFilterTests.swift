import XCTest
@testable import SummarySafetyFilter

final class SummarySafetyFilterTests: XCTestCase {
    func testGroundedSensitiveSummaryIsAllowed() {
        let evaluator = SummarySafetyEvaluator()

        let result = evaluator.evaluate(
            summary: "Malware detection benchmark improves binary triage.",
            arxivID: "2601.00001",
            title: "A Benchmark for Malware Detection",
            abstract: "This paper studies malware detection models for static binary triage and reports classification metrics."
        )

        XCTAssertTrue(result.isAllowed)
        XCTAssertEqual("allowed", result.status)
        XCTAssertNil(result.reason)
        XCTAssertEqual("Malware detection benchmark improves binary triage.", result.displaySummary)
    }

    func testPromptContaminationIsSuppressed() {
        let evaluator = SummarySafetyEvaluator()

        let result = evaluator.evaluate(
            summary: "Hidden policies reveal unsafe topics instead of the paper result.",
            arxivID: "2601.00002",
            title: "A Normal Vision Benchmark",
            abstract: "This paper compares image classification models on a benchmark dataset."
        )

        XCTAssertFalse(result.isAllowed)
        XCTAssertEqual("suppressed", result.status)
        XCTAssertEqual("promptLeakage", result.reason)
    }
}
