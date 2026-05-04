// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SummarySafetyFilter",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SummarySafetyFilter", targets: ["SummarySafetyFilter"]),
        .executable(name: "summary-safety-eval", targets: ["SummarySafetyEval"]),
    ],
    targets: [
        .target(name: "SummarySafetyFilter"),
        .executableTarget(
            name: "SummarySafetyEval",
            dependencies: ["SummarySafetyFilter"]
        ),
        .testTarget(
            name: "SummarySafetyFilterTests",
            dependencies: ["SummarySafetyFilter"]
        ),
    ]
)
